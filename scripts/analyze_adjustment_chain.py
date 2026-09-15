"""
Mide que le hace la cadena de ajustes post-prediccion a la calidad del modelo.

Contexto: live_pull calcula el umbral sobre `calibrated_proba` (el percentil 78)
y despues etiqueta comparando la probabilidad YA ajustada contra ese umbral.
Como los cuatro ajustes son noisy-OR —solo pueden subir la probabilidad— la
tasa de positivos resultante es por construccion mayor al 22% buscado.

Este script no asume nada de eso: compara las dos probabilidades que quedan
guardadas en cada fila de `predictions` contra el resultado real.

    proba_raw    probabilidad calibrada, antes de cualquier ajuste
    proba_delay  probabilidad final, despues de GDP + dep_delay + ADS-B

Si `proba_raw` gana en Brier y AUC, la cadena esta restando y el arreglo no es
recalibrar las bandas sino apagarlas.

Uso:
    GCS_BUCKET=ontimeai-prod-live-db python3 scripts/analyze_adjustment_chain.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", "/tmp/analyze_live_data.db"))
DELAY_THRESHOLD_MIN = 15.0


def download() -> None:
    bucket = os.getenv("GCS_BUCKET", "")
    if not bucket or DB_PATH.exists():
        return
    from google.cloud import storage

    blob = storage.Client().bucket(bucket).blob(os.getenv("DB_OBJECT", "live_data.db"))
    print(f"descargando gs://{bucket}/{blob.name} ...", flush=True)
    blob.download_to_filename(str(DB_PATH))
    print(f"  {DB_PATH.stat().st_size / 1e6:.0f} MB\n", flush=True)


def build_sample(con: sqlite3.Connection) -> int:
    """Una fila por vuelo: la ultima prediccion ACCIONABLE, con el resultado real.

    Accionable = anterior al aterrizaje. Tomar la ultima prediccion a secas
    sesga la muestra: para un vuelo ya aterrizado esa fila es de fase
    POST_LANDING, y justo ahi `dep_delay_map` esta vacio porque su consulta pide
    `actual_in_utc IS NULL`. Medido asi, el ajuste por demora de salida parecia
    no existir (8 filas en 3.980). Es la prediccion que nadie mira: el operador
    decide mientras el vuelo esta en el aire.
    """
    con.executescript(
        """
        DROP TABLE IF EXISTS temp.sample;
        CREATE TEMP TABLE sample AS
        SELECT p.stable_id,
               p.proba_raw,
               p.proba_delay,
               p.threshold_used,
               p.prediction_phase,
               p.intermediate_dep_delay_min AS dep_delay,
               COALESCE(p.gdp_orig_delay_min, 0) AS gdp_orig,
               COALESCE(p.gdp_dest_delay_min, 0) AS gdp_dest,
               p.adsb_eta_delay_min,
               p.adsb_holding_min,
               a.source_provider,
               a.arr_delay_min,
               a.departure_delay_min,
               CASE WHEN a.arr_delay_min > 15.0 THEN 1 ELSE 0 END AS y
          FROM predictions p
          JOIN actuals a ON a.stable_id = p.stable_id
         WHERE a.arr_delay_min IS NOT NULL
           AND p.proba_raw IS NOT NULL
           AND p.threshold_used IS NOT NULL
           AND COALESCE(a.cancelled, 0) = 0
           AND COALESCE(a.diverted, 0) = 0
           AND p.prediction_phase IS NOT 'POST_LANDING'
           AND p.predicted_at_utc = (
                 SELECT MAX(p2.predicted_at_utc) FROM predictions p2
                  WHERE p2.stable_id = p.stable_id AND p2.proba_raw IS NOT NULL
                    AND p2.prediction_phase IS NOT 'POST_LANDING');
        CREATE INDEX temp.idx_sample ON sample(stable_id);
        """
    )
    return con.execute("SELECT COUNT(*) FROM temp.sample").fetchone()[0]


def _auc(pairs: list[tuple[float, int]]) -> float:
    """AUC por el estadistico de rangos de Mann-Whitney, con empates promediados."""
    pairs = sorted(pairs)
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(1 for _, y in pairs if y == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def _ece(pairs: list[tuple[float, int]], bins: int = 10) -> float:
    """Expected Calibration Error con bins de ancho fijo."""
    total = len(pairs)
    if total == 0:
        return float("nan")
    acc = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        chunk = [(p, y) for p, y in pairs if (p >= lo and p < hi) or (b == bins - 1 and p == 1.0)]
        if not chunk:
            continue
        conf = sum(p for p, _ in chunk) / len(chunk)
        freq = sum(y for _, y in chunk) / len(chunk)
        acc += (len(chunk) / total) * abs(conf - freq)
    return acc


def score(con: sqlite3.Connection, column: str) -> dict:
    rows = con.execute(
        f"SELECT {column}, y, threshold_used FROM temp.sample WHERE {column} IS NOT NULL"
    ).fetchall()
    pairs = [(float(p), int(y)) for p, y, _ in rows]
    n = len(pairs)
    return {
        "n": n,
        "brier": sum((p - y) ** 2 for p, y in pairs) / n,
        "auc": _auc(pairs),
        "ece": _ece(pairs),
        "pos_rate": sum(1 for p, _, t in rows if p >= t) / n,
        "mean_p": sum(p for p, _ in pairs) / n,
    }


def main() -> int:
    download()
    if not DB_PATH.exists():
        print(f"no existe {DB_PATH}", file=sys.stderr)
        return 1
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    n = build_sample(con)
    base = con.execute("SELECT AVG(y) FROM temp.sample").fetchone()[0]
    print("=" * 66)
    print(f"MUESTRA: {n} vuelos con resultado real | tasa real de retraso {base:.1%}")
    print("=" * 66)

    print("\n[1] La cadena de ajustes, ¿suma o resta?\n")
    print(f"    {'':<14}{'Brier':>9}{'AUC':>9}{'ECE':>9}{'% positivos':>13}{'p media':>10}")
    for label, col in (("calibrada", "proba_raw"), ("ajustada", "proba_delay")):
        s = score(con, col)
        print(
            f"    {label:<14}{s['brier']:>9.4f}{s['auc']:>9.4f}{s['ece']:>9.4f}"
            f"{s['pos_rate']:>12.1%}{s['mean_p']:>10.3f}"
        )
    print(f"\n    objetivo de la estrategia quantile@0.22: 22,0% de positivos")

    print("\n[2] Sesgo de departure_delay_min por proveedor (issue #11)\n")
    print(f"    {'proveedor':<12}{'n':>9}{'mediana dep-arr':>18}{'mediana dep':>14}")
    # SQLite no trae MEDIAN: se saca la fila del medio con window functions.
    for prov, cnt, med_diff, med_dep in con.execute(
        """
        WITH base AS (
          SELECT COALESCE(source_provider,'(nulo)') AS prov,
                 departure_delay_min - arr_delay_min AS diff,
                 departure_delay_min AS dep
            FROM actuals
           WHERE departure_delay_min IS NOT NULL AND arr_delay_min IS NOT NULL
        ), ranked AS (
          SELECT prov, diff, dep,
                 ROW_NUMBER() OVER (PARTITION BY prov ORDER BY diff) AS rd,
                 ROW_NUMBER() OVER (PARTITION BY prov ORDER BY dep)  AS rp,
                 COUNT(*)   OVER (PARTITION BY prov)                 AS n
            FROM base
        )
        SELECT prov, MAX(n),
               MAX(CASE WHEN rd = (n+1)/2 THEN diff END),
               MAX(CASE WHEN rp = (n+1)/2 THEN dep  END)
          FROM ranked GROUP BY prov ORDER BY MAX(n) DESC
        """
    ).fetchall():
        print(f"    {prov:<12}{cnt:>9,}{med_diff:>+18.1f}{med_dep:>+14.1f}")

    print("\n[3] Bandas de intermediate_dep_delay_adjust contra la realidad\n")
    print("    La banda asigna p_from_dep; al lado, la tasa real observada.\n")
    print(f"    {'banda':<12}{'p asignada':>12}{'proveedor':>12}{'n':>8}{'tasa real':>12}")
    bands = (("5-15", 5, 15, 0.25), ("15-30", 15, 30, 0.75),
             ("30-60", 30, 60, 0.90), ("60+", 60, 1e9, 0.97))
    for name, lo, hi, assigned in bands:
        for prov, cnt, rate in con.execute(
            """
            SELECT COALESCE(source_provider,'(nulo)'), COUNT(*), AVG(y)
              FROM temp.sample
             WHERE dep_delay >= ? AND dep_delay < ?
             GROUP BY source_provider ORDER BY COUNT(*) DESC
            """, (lo, hi),
        ).fetchall():
            print(f"    {name:<12}{assigned:>12.2f}{prov:>12}{cnt:>8,}{rate:>12.1%}")

    print("\n[4] Vuelos que la cadena empujo por encima del umbral\n")
    row = con.execute(
        """
        SELECT COUNT(*), AVG(y), AVG(proba_delay), AVG(proba_raw)
          FROM temp.sample
         WHERE proba_raw < threshold_used AND proba_delay >= threshold_used
        """
    ).fetchone()
    cnt, rate, p_adj, p_raw = row
    if cnt:
        print(f"    n                        : {cnt:,}  ({cnt/n:.1%} de la muestra)")
        print(f"    probabilidad que muestra : {p_adj:.1%}   (antes del ajuste: {p_raw:.1%})")
        print(f"    tasa real de retraso     : {rate:.1%}")
        print(f"\n    Son los falsos positivos que agrega la cadena por si sola.")

    print("\n[5] Que etapa de la cadena empuja\n")
    print("    `estimated` no se guarda como columna; se deduce por descarte:")
    print("    la probabilidad subio y ninguna otra etapa tenia senal.\n")
    print(f"    {'etapa':<28}{'n':>8}{'p media':>11}{'tasa real':>12}")
    etapas = (
        ("GDP (origen o destino)", "(gdp_orig > 0 OR gdp_dest > 0)"),
        ("demora de salida real",  "dep_delay >= 5"),
        ("ADS-B ETA",              "adsb_eta_delay_min > 5"),
        ("ADS-B holding",          "adsb_holding_min >= 5"),
        ("estimada (por descarte)",
         "gdp_orig = 0 AND gdp_dest = 0 AND COALESCE(dep_delay,0) < 5"
         " AND COALESCE(adsb_eta_delay_min,0) <= 5 AND COALESCE(adsb_holding_min,0) < 5"),
    )
    for nombre, cond in etapas:
        cnt, p_adj, rate = con.execute(
            f"""SELECT COUNT(*), AVG(proba_delay), AVG(y) FROM temp.sample
                 WHERE proba_delay > proba_raw + 1e-6 AND {cond}"""
        ).fetchone()
        if cnt:
            print(f"    {nombre:<28}{cnt:>8,}{p_adj:>10.1%}{rate:>12.1%}")

    print("\n[6] Por fase de la prediccion\n")
    print(f"    {'fase':<16}{'n':>8}{'Brier cal':>12}{'Brier ajus':>12}{'tasa real':>12}")
    for fase, cnt, b_raw, b_adj, rate in con.execute(
        """SELECT prediction_phase, COUNT(*),
                  AVG((proba_raw   - y) * (proba_raw   - y)),
                  AVG((proba_delay - y) * (proba_delay - y)), AVG(y)
             FROM temp.sample GROUP BY prediction_phase ORDER BY COUNT(*) DESC"""
    ).fetchall():
        print(f"    {str(fase):<16}{cnt:>8,}{b_raw:>12.4f}{b_adj:>12.4f}{rate:>12.1%}")

    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

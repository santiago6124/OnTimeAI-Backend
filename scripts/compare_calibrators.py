"""
Compara `4year_v9` contra `4year_v9_recal` sobre la cohorte live actual.

Los dos artefactos comparten el mismo `model.lgb` —mismo tamano, misma fecha— y
difieren solo en el calibrador: `Calibrator` contra `StackedCalibrator`. Asi que
comparar no exige volver a inferir: alcanza con tomar la probabilidad cruda del
booster, que queda guardada en cada snapshot del training store, y pasarla por
uno y por otro.

Por que no alcanza con el recal_report de agosto: se midio sobre 53.750 muestras
anteriores al switch a Fase 4, cuando la fuente de inferencia era AeroAPI y no
FR24, y con una prevalencia cercana al 20% contra el 7,7% de hoy. Un calibrador
ajustado sobre otra tasa base no tiene por que transferir. Es el punto 1 de los
criterios de aceptacion del issue #15.

Uso:
    GCS_BUCKET=ontimeai-prod-live-db python3 scripts/compare_calibrators.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np

DB_PATH = Path(os.getenv("DB_PATH", "/tmp/analyze_live_data.db"))
ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
SNAPSHOT_EVENT_TYPE = "prediction_snapshots"


def download() -> None:
    bucket = os.getenv("GCS_BUCKET", "")
    if not bucket or DB_PATH.exists():
        return
    from google.cloud import storage

    blob = storage.Client().bucket(bucket).blob(os.getenv("DB_OBJECT", "live_data.db"))
    print(f"descargando gs://{bucket}/{blob.name} ...", flush=True)
    blob.download_to_filename(str(DB_PATH))
    print(f"  {DB_PATH.stat().st_size / 1e6:.0f} MB\n", flush=True)


def load_cohort(con: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray, int]:
    """Probabilidad cruda del booster y resultado real, una fila por vuelo."""
    rows = con.execute(
        """
        WITH snap AS (
          SELECT json_extract(payload_json, '$.stable_id')           AS stable_id,
                 json_extract(payload_json, '$.booster_probability') AS booster_p,
                 json_extract(payload_json, '$.predicted_at_utc')    AS at_utc,
                 json_extract(payload_json, '$.prediction_phase')    AS phase
            FROM training_export_outbox
           WHERE event_type = ?
        ), ultima AS (
          SELECT stable_id, booster_p,
                 ROW_NUMBER() OVER (PARTITION BY stable_id ORDER BY at_utc DESC) AS rn
            FROM snap
           WHERE booster_p IS NOT NULL AND phase IS NOT 'POST_LANDING'
        )
        SELECT u.booster_p,
               CASE WHEN a.arr_delay_min > 15.0 THEN 1 ELSE 0 END AS y
          FROM ultima u
          JOIN actuals a ON a.stable_id = u.stable_id
         WHERE u.rn = 1
           AND a.arr_delay_min IS NOT NULL
           AND COALESCE(a.cancelled, 0) = 0
           AND COALESCE(a.diverted, 0) = 0
        """,
        (SNAPSHOT_EVENT_TYPE,),
    ).fetchall()
    if not rows:
        return np.array([]), np.array([]), 0
    p = np.array([float(r[0]) for r in rows])
    y = np.array([int(r[1]) for r in rows])
    return p, y, len(rows)


def auc(p: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos, neg = int(y.sum()), int((1 - y).sum())
    if not pos or not neg:
        return float("nan")
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def curva(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[tuple]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append((edges[b], edges[b + 1], int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return out


def main() -> int:
    download()
    if not DB_PATH.exists():
        print(f"no existe {DB_PATH}", file=sys.stderr)
        return 1
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    booster_p, y, n = load_cohort(con)
    con.close()

    if n == 0:
        print("Sin snapshots del training store con probabilidad del booster.")
        print("Sin esa columna no se puede comparar calibradores sin volver a inferir.")
        return 2

    print("=" * 70)
    print(f"COHORTE: {n:,} vuelos post-Fase 4 | tasa real de retraso {y.mean():.1%}")
    print("=" * 70)

    print(f"\n{'artefacto':<20}{'calibrador':<20}{'Brier':>9}{'AUC':>9}{'ECE':>9}{'p media':>10}")
    resultados = {}
    for name in ("4year_v9", "4year_v9_recal"):
        meta = joblib.load(ARTIFACTS / name / "meta.joblib")
        cal = meta.get("calibrator")
        p = cal.transform(booster_p.copy()) if cal is not None else booster_p.copy()
        resultados[name] = p
        print(
            f"{name:<20}{type(cal).__name__:<20}"
            f"{float(np.mean((p - y) ** 2)):>9.4f}{auc(p, y):>9.4f}"
            f"{ece(p, y):>9.4f}{p.mean():>10.3f}"
        )
    print(f"{'(sin calibrar)':<20}{'-':<20}"
          f"{float(np.mean((booster_p - y) ** 2)):>9.4f}{auc(booster_p, y):>9.4f}"
          f"{ece(booster_p, y):>9.4f}{booster_p.mean():>10.3f}")

    print("\nCurva de calibracion por bin (lo que pide el criterio 1 de #15)\n")
    print(f"    {'bin':<14}{'n':>7}{'v9 dice':>11}{'recal dice':>13}{'real':>9}")
    c9 = {(a, b): (n_, m, r) for a, b, n_, m, r in curva(resultados["4year_v9"], y)}
    cr = {(a, b): (n_, m, r) for a, b, n_, m, r in curva(resultados["4year_v9_recal"], y)}
    for key in sorted(set(c9) | set(cr)):
        a, b = key
        n9, m9, r9 = c9.get(key, (0, float("nan"), float("nan")))
        nr, mr, rr = cr.get(key, (0, float("nan"), float("nan")))
        real = r9 if n9 >= nr else rr
        print(f"    {a:.1f}-{b:.1f}{'':<8}{max(n9, nr):>7,}{m9:>10.1%}{mr:>13.1%}{real:>9.1%}")

    print(json.dumps({"n": n, "base_rate": round(float(y.mean()), 4)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

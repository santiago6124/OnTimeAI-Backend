"""
Compara `4year_v9` contra `4year_v9_recal` sobre la cohorte live actual.

Los dos artefactos comparten el mismo `model.lgb` y difieren solo en el
calibrador. El de recal es un `StackedCalibrator`, que aplica dos etapas:

    transform(p) = cal2.transform(cal1.transform(p))

y `cal1` resulta ser identico al calibrador de v9 —verificado numericamente
sobre una grilla, no por inspeccion—. Eso vuelve la comparacion barata: la
columna `predictions.proba_raw` ya guarda `cal1(booster)`, asi que la salida
de v9_recal se obtiene aplicandole `cal2` encima. No hace falta volver a
inferir ni recuperar la probabilidad cruda del booster.

Hizo falta llegar hasta aca porque los snapshots del training store, que si la
guardan, se vacian del outbox al entregarse a GCS: quedan 187 filas, todas
demasiado recientes para tener resultado.

Por que no alcanza con el recal_report de agosto: se midio sobre 53.750
muestras anteriores al switch a Fase 4, cuando la fuente de inferencia era
AeroAPI y no FR24, y con una prevalencia cercana al 20% contra el 7,7% de hoy.
Un calibrador ajustado sobre otra tasa base no tiene por que transferir. Es el
punto 1 de los criterios de aceptacion del issue #15.

Uso:
    GCS_BUCKET=ontimeai-prod-live-db python3 scripts/compare_calibrators.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np

DB_PATH = Path(os.getenv("DB_PATH", "/tmp/analyze_live_data.db"))
ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))


def download() -> None:
    bucket = os.getenv("GCS_BUCKET", "")
    if not bucket or DB_PATH.exists():
        return
    from google.cloud import storage

    blob = storage.Client().bucket(bucket).blob(os.getenv("DB_OBJECT", "live_data.db"))
    print(f"descargando gs://{bucket}/{blob.name} ...", flush=True)
    blob.download_to_filename(str(DB_PATH))
    print(f"  {DB_PATH.stat().st_size / 1e6:.0f} MB\n", flush=True)


def load_cohort(con: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray]:
    """Salida de v9 y resultado real, una fila por vuelo.

    Se toma la ultima prediccion anterior al aterrizaje: la posterior es la que
    nadie mira, y ademas cae en una fase donde varias senales no existen.
    """
    rows = con.execute(
        """
        SELECT p.proba_raw,
               CASE WHEN a.arr_delay_min > 15.0 THEN 1 ELSE 0 END AS y
          FROM predictions p
          JOIN actuals a ON a.stable_id = p.stable_id
         WHERE a.arr_delay_min IS NOT NULL
           AND p.proba_raw IS NOT NULL
           AND COALESCE(a.cancelled, 0) = 0
           AND COALESCE(a.diverted, 0) = 0
           AND p.prediction_phase IS NOT 'POST_LANDING'
           AND p.predicted_at_utc = (
                 SELECT MAX(p2.predicted_at_utc) FROM predictions p2
                  WHERE p2.stable_id = p.stable_id AND p2.proba_raw IS NOT NULL
                    AND p2.prediction_phase IS NOT 'POST_LANDING')
        """
    ).fetchall()
    if not rows:
        return np.array([]), np.array([])
    return (np.array([float(r[0]) for r in rows]),
            np.array([int(r[1]) for r in rows]))


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
    idx = np.clip(np.digitize(p, np.linspace(0.0, 1.0, bins + 1)[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def main() -> int:
    download()
    if not DB_PATH.exists():
        print(f"no existe {DB_PATH}", file=sys.stderr)
        return 1
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    p_v9, y = load_cohort(con)
    con.close()
    if p_v9.size == 0:
        print("sin cohorte", file=sys.stderr)
        return 2

    recal = joblib.load(ARTIFACTS / "4year_v9_recal" / "meta.joblib")["calibrator"]
    v9 = joblib.load(ARTIFACTS / "4year_v9" / "meta.joblib")["calibrator"]

    # Se verifica el supuesto en vez de confiar en el: si cal1 dejara de ser el
    # calibrador de v9, `proba_raw` no seria su entrada y todo esto mentiria.
    grilla = np.linspace(0.0, 1.0, 101)
    if not np.allclose(v9.transform(grilla.copy()), recal.cal1.transform(grilla.copy())):
        print("cal1 ya no coincide con el calibrador de v9; la comparacion no aplica",
              file=sys.stderr)
        return 3

    p_recal = recal.cal2.transform(p_v9.copy())

    print("=" * 68)
    print(f"COHORTE: {len(y):,} vuelos | tasa real de retraso {y.mean():.1%}")
    print("=" * 68)
    print(f"\n{'artefacto':<20}{'Brier':>10}{'AUC':>10}{'ECE':>10}{'p media':>11}")
    for nombre, p in (("4year_v9", p_v9), ("4year_v9_recal", p_recal)):
        print(f"{nombre:<20}{float(np.mean((p - y) ** 2)):>10.4f}"
              f"{auc(p, y):>10.4f}{ece(p, y):>10.4f}{p.mean():>11.4f}")

    print("\nCurva de calibracion por bin\n")
    print(f"    {'bin':<12}{'n':>7}{'v9 dice':>11}{'recal dice':>13}{'real':>9}")
    bins = 10
    idx = np.clip(np.digitize(p_v9, np.linspace(0.0, 1.0, bins + 1)[1:-1]), 0, bins - 1)
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        print(f"    {b/bins:.1f}-{(b+1)/bins:.1f}{'':<6}{int(m.sum()):>7,}"
              f"{p_v9[m].mean():>10.1%}{p_recal[m].mean():>13.1%}{y[m].mean():>9.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

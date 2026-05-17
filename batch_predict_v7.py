"""Batch prediction sobre vuelos históricos en la DB para evaluar v7.

Selecciona vuelos que tienen actuals pero no tienen predicciones v7,
corre inferencia y guarda en la tabla predictions para que live_metrics.py
pueda evaluar el modelo.

Uso:
    python3 batch_predict_v7.py
    python3 batch_predict_v7.py --days 7 --batch-size 200
    python3 batch_predict_v7.py --artifact artifacts/4year_v7_full
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ontimeai.live import open_db, build_inference_frame, stable_id
from ontimeai.model import load_artifact, predict_label, predict_proba
from ontimeai.lineage_fallback import load_lookups
from ontimeai.config import ARTIFACTS_DIR
from predict import prepare_inference_frame


def _fix_flow_atl(df):
    import pandas as pd
    mask = ~df["ORIGIN"].eq("ATL") & ~df["DEST"].eq("ATL")
    df.loc[mask, "FLOW_ATL"] = "NON_ATL"
    df.loc[mask, "PAR_AIRPORT"] = ""
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", default=str(ARTIFACTS_DIR / "4year_v7_recal"))
    p.add_argument("--days", type=int, default=7,
                   help="Evaluar vuelos de los últimos N días")
    p.add_argument("--batch-size", type=int, default=300,
                   help="Vuelos por batch de inferencia")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-predecir aunque ya exista predicción")
    args = p.parse_args()

    print(f"Artifact: {args.artifact}")
    print(f"Ventana:  últimos {args.days} días")

    meta = load_artifact(args.artifact)
    fallback = load_lookups(PROJECT_ROOT / "artifacts" / "lineage_fallback.joblib")

    conn = open_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()

    # Vuelos con actuals en la ventana, sin predicción v7 (o todos si --overwrite)
    if args.overwrite:
        rows = conn.execute(
            """SELECT DISTINCT f.fa_flight_id
               FROM flights f
               JOIN actuals a ON a.stable_id = f.stable_id
               WHERE f.scheduled_off_utc >= ?
               ORDER BY f.scheduled_off_utc""",
            (cutoff,),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT DISTINCT f.fa_flight_id
               FROM flights f
               JOIN actuals a ON a.stable_id = f.stable_id
               LEFT JOIN predictions p ON p.stable_id = f.stable_id
               WHERE f.scheduled_off_utc >= ?
                 AND p.fa_flight_id IS NULL
               ORDER BY f.scheduled_off_utc""",
            (cutoff,),
        ).fetchall()

    all_ids = [r[0] for r in rows]
    print(f"Vuelos con actuals sin predicción: {len(all_ids)}")

    if not all_ids:
        print("Nada que predecir — todos tienen predicción o no hay actuals.")
        return

    # Predecir en batches
    total_predicted = 0
    pred_now = datetime.now(timezone.utc).isoformat()

    for i in range(0, len(all_ids), args.batch_size):
        batch = all_ids[i: i + args.batch_size]
        print(f"  Batch {i//args.batch_size + 1}: {len(batch)} vuelos...", end=" ", flush=True)

        df = build_inference_frame(conn, batch, history_days=7)
        if df.empty:
            print("vacío")
            continue

        df = _fix_flow_atl(df)

        # Solo los vuelos target (sin actual en el frame — ARR_DELAY=NaN)
        target_mask = df["fa_flight_id"].isin(batch) & df["ARR_DELAY"].isna()
        if target_mask.sum() == 0:
            print("sin targets")
            continue

        try:
            X = prepare_inference_frame(
                df, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=fallback,
            )
            cat_cols_set = set(meta.get("cat_cols", []))
            for c in X.columns:
                if c not in cat_cols_set and X[c].dtype == object:
                    import pandas as pd
                    X[c] = pd.to_numeric(X[c], errors="coerce")

            proba = predict_proba(meta["booster"], X)
            if meta.get("calibrator") is not None and meta["target"] == "binary":
                proba = meta["calibrator"].transform(proba)

            threshold = meta["threshold"]
            labels = predict_label(proba, threshold, "binary")

            pred_rows = [
                (
                    df.loc[idx, "fa_flight_id"],
                    stable_id(df.loc[idx, "fa_flight_id"]),
                    pred_now,
                    float(proba[pos]),
                    int(labels[pos]),
                    float(threshold),
                    "artifact_batch_v7",
                )
                for pos, idx in enumerate(df.index)
                if target_mask.iloc[pos]
            ]

            conn.executemany(
                """INSERT OR REPLACE INTO predictions
                   (fa_flight_id, stable_id, predicted_at_utc, proba_delay, predicted_delay,
                    threshold_used, threshold_strategy)
                   VALUES (?,?,?,?,?,?,?)""",
                pred_rows,
            )
            conn.commit()
            total_predicted += len(pred_rows)
            print(f"OK ({len(pred_rows)} predicciones)")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    print(f"\nTotal predicciones escritas: {total_predicted}")
    print("Ahora corrés: python3 live_metrics.py --since 7d")


if __name__ == "__main__":
    main()

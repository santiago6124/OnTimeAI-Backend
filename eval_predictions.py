"""Evaluates predictions.csv against the ground-truth ARR_DELAY in the input master.

Usage:
    python3 eval_predictions.py --master dataset_maestro_ATL_2021_*.csv --preds preds_2021.csv
    python3 eval_predictions.py --master vuelos_aviationstack.csv --preds preds_live.csv \
        --threshold-min 15 --label "Live AviationStack 2026-04-25/26"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss, confusion_matrix,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--master", type=Path, required=True, help="Input CSV used for prediction (must contain ARR_DELAY).")
    p.add_argument("--preds", type=Path, required=True, help="predict.py output (proba_delay, predicted_delay).")
    p.add_argument("--threshold-min", type=float, default=15.0)
    p.add_argument("--label", type=str, default="Evaluation")
    p.add_argument("--out", type=Path, default=None, help="Optional JSON to write metrics.")
    p.add_argument("--breakdown", choices=["none", "month", "quarter"], default="none")
    args = p.parse_args(argv)

    print(f"== {args.label} ==")
    if Path(args.master).suffix == ".parquet":
        master = pd.read_parquet(args.master)
        master["FL_DATE"] = pd.to_datetime(master["FL_DATE"], errors="coerce")
    else:
        master = pd.read_csv(args.master, parse_dates=["FL_DATE"], low_memory=False)
    preds = pd.read_csv(args.preds)
    if len(master) != len(preds):
        print(f"WARN: master rows ({len(master)}) != preds rows ({len(preds)}); reindexing by row order.")
    n = min(len(master), len(preds))
    master = master.iloc[:n].reset_index(drop=True)
    preds = preds.iloc[:n].reset_index(drop=True)

    # Same filter as training: drop cancelled / diverted / null ARR_DELAY
    cancelled = pd.to_numeric(master.get("CANCELLED"), errors="coerce").fillna(0).astype(int)
    diverted = pd.to_numeric(master.get("DIVERTED"), errors="coerce").fillna(0).astype(int)
    arr = pd.to_numeric(master["ARR_DELAY"], errors="coerce")
    valid = (cancelled == 0) & (diverted == 0) & arr.notna()

    master = master.loc[valid].reset_index(drop=True)
    preds = preds.loc[valid].reset_index(drop=True)
    arr = arr.loc[valid].reset_index(drop=True)
    print(f"Valid flights with ground truth: {len(master):,}")
    if len(master) == 0:
        print("Nothing to evaluate.")
        return 1

    y_true = (arr > args.threshold_min).astype(int).values
    y_pred = preds["predicted_delay"].astype(int).values
    y_proba = preds["proba_delay"].astype(float).values

    metrics = {
        "label": args.label,
        "n": int(len(y_true)),
        "positive_rate_truth": float(y_true.mean()),
        "positive_rate_pred": float(y_pred.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else None,
        "brier": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-7, 1 - 1e-7))) if len(np.unique(y_true)) > 1 else None,
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }

    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    print(f"Confusion matrix [tn fp / fn tp]: {cm.tolist()}")

    if args.breakdown != "none":
        freq = "Q" if args.breakdown == "quarter" else "M"
        period = master["FL_DATE"].dt.to_period(freq)
        print(f"\n--- per {args.breakdown} ---")
        for key in sorted(period.dropna().unique()):
            mask = (period == key).values
            yt = y_true[mask]
            yp = y_pred[mask]
            ypr = y_proba[mask]
            try:
                auc = float(roc_auc_score(yt, ypr)) if len(np.unique(yt)) > 1 else float("nan")
            except Exception:
                auc = float("nan")
            print(f"  {key} n={mask.sum():,} acc={accuracy_score(yt, yp):.3f} "
                  f"f1={f1_score(yt, yp, zero_division=0):.3f} auc={auc:.3f} "
                  f"pos_rate={yt.mean():.3f}")

    if args.out:
        args.out.write_text(json.dumps(metrics, indent=2))
        print(f"\nMetrics -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

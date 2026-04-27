"""Compute running accuracy/AUC over predictions that have settled actuals.

Usage:
    python3 live_metrics.py                    # all-time
    python3 live_metrics.py --since 24h
    python3 live_metrics.py --since 7d --threshold-min 15
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
)

from ontimeai.live import open_db


def parse_since(s: str | None) -> datetime | None:
    if not s:
        return None
    m = re.match(r"^(\d+)\s*([hdw])$", s.strip().lower())
    if not m:
        raise ValueError(f"Invalid --since: {s} (use e.g. 24h, 7d, 2w)")
    n = int(m.group(1))
    unit = m.group(2)
    delta = {"h": timedelta(hours=n), "d": timedelta(days=n), "w": timedelta(weeks=n)}[unit]
    return datetime.now(timezone.utc) - delta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--since", default=None, help="e.g. 24h, 7d, 2w")
    p.add_argument("--threshold-min", type=float, default=15.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    conn = open_db()
    since = parse_since(args.since)

    sql = """
        SELECT p.fa_flight_id, p.predicted_at_utc, p.proba_delay, p.predicted_delay,
               a.arr_delay_min, a.cancelled, a.diverted, f.op_carrier, f.origin, f.dest,
               f.fl_date, f.scheduled_off_utc
        FROM predictions p
        JOIN actuals a ON a.fa_flight_id = p.fa_flight_id
        JOIN flights f ON f.fa_flight_id = p.fa_flight_id
        WHERE a.cancelled = 0 AND a.diverted = 0 AND a.arr_delay_min IS NOT NULL
    """
    params: tuple = ()
    if since:
        sql += " AND p.predicted_at_utc >= ?"
        params = (since.isoformat(),)

    df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        print(f"No settled predictions"
              + (f" since {since.isoformat()}" if since else "") + ".")
        return 1

    # If multiple predictions per flight, keep the last (latest)
    df = df.sort_values("predicted_at_utc").groupby("fa_flight_id", as_index=False).last()

    y_true = (df["arr_delay_min"] > args.threshold_min).astype(int).values
    y_pred = df["predicted_delay"].astype(int).values
    y_proba = df["proba_delay"].astype(float).values

    metrics = {
        "since": args.since or "all",
        "n": int(len(df)),
        "first_prediction": df["predicted_at_utc"].min(),
        "last_prediction": df["predicted_at_utc"].max(),
        "positive_rate_truth": float(y_true.mean()),
        "positive_rate_pred": float(y_pred.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else None,
        "brier": float(brier_score_loss(y_true, y_proba)),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                                   "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}

    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    print(f"Confusion matrix [tn fp / fn tp]: {cm.tolist()}")

    # Per-carrier breakdown
    print("\n--- per carrier (top 10) ---")
    car = df.groupby("op_carrier").apply(lambda g: pd.Series({
        "n": len(g),
        "acc": accuracy_score((g["arr_delay_min"] > args.threshold_min).astype(int), g["predicted_delay"]),
        "f1": f1_score((g["arr_delay_min"] > args.threshold_min).astype(int), g["predicted_delay"], zero_division=0),
        "pos_rate_true": (g["arr_delay_min"] > args.threshold_min).mean(),
    }), include_groups=False).sort_values("n", ascending=False).head(10)
    print(car.to_string())

    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(metrics, indent=2))
        print(f"\nMetrics → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compute running accuracy/AUC over predictions that have settled actuals.

Usage:
    python3 live_metrics.py                    # all-time
    python3 live_metrics.py --since 24h
    python3 live_metrics.py --since 7d --threshold-min 15
    python3 live_metrics.py --psi             # add PSI drift report (7d vs all-time)
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


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two 1-D probability distributions.

    PSI < 0.10  → no significant shift (stable)
    PSI 0.10-0.25 → moderate shift (monitor)
    PSI > 0.25  → significant shift (recalibrate)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)
    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    # Replace zeros to avoid log(0)
    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--since", default=None, help="e.g. 24h, 7d, 2w")
    p.add_argument("--threshold-min", type=float, default=15.0)
    p.add_argument("--out", default=None)
    p.add_argument("--psi", action="store_true",
                   help="Compute PSI drift: compare last-7d proba distribution vs all-time baseline")
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

    if args.psi:
        print("\n--- PSI drift report (7d window vs all-time baseline) ---")
        cutoff_7d = datetime.now(timezone.utc) - timedelta(days=7)
        sql_all = "SELECT proba_delay FROM predictions"
        sql_7d = "SELECT proba_delay FROM predictions WHERE predicted_at_utc >= ?"
        df_all = pd.read_sql_query(sql_all, conn)
        df_7d = pd.read_sql_query(sql_7d, conn, params=(cutoff_7d.isoformat(),))
        if df_all.empty or df_7d.empty:
            print("  Not enough data for PSI (need all-time + 7d window predictions).")
        else:
            proba_all = df_all["proba_delay"].astype(float).to_numpy()
            proba_7d = df_7d["proba_delay"].astype(float).to_numpy()
            psi_val = _psi(proba_all, proba_7d)
            mean_all = float(proba_all.mean())
            mean_7d = float(proba_7d.mean())
            status = (
                "STABLE" if psi_val < 0.10
                else "MONITOR" if psi_val < 0.25
                else "⚠ RECALIBRATE"
            )
            print(f"  n_all={len(proba_all):,}  n_7d={len(proba_7d):,}")
            print(f"  mean_proba_all={mean_all:.4f}  mean_proba_7d={mean_7d:.4f}")
            print(f"  PSI={psi_val:.4f}  → {status}")
            if psi_val >= 0.25:
                print("  ACTION: threshold drift detected — rerun tune_threshold on recent 1000+ actuals")
            elif psi_val >= 0.10:
                print("  ACTION: watch closely — consider re-calibrating Platt scaling")

    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(metrics, indent=2))
        print(f"\nMetrics → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

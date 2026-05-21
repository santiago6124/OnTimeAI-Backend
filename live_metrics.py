"""Compute running accuracy/AUC over predictions that have settled actuals.

Usage:
    python3 live_metrics.py                    # all-time
    python3 live_metrics.py --since 24h
    python3 live_metrics.py --since 7d --threshold-min 15
    python3 live_metrics.py --psi             # add PSI drift report (7d vs all-time)
    python3 live_metrics.py --per-day          # add per-day AUC breakdown
    python3 live_metrics.py --calibration      # add calibration curve
    python3 live_metrics.py --min-actuals 100  # filter days with too few actuals
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
    p.add_argument("--per-day", action="store_true",
                   help="Show per-day AUC breakdown")
    p.add_argument("--calibration", action="store_true",
                   help="Show calibration curve")
    p.add_argument("--min-actuals", type=int, default=0,
                   help="Filter days with fewer than N settled actuals from aggregate")
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

    # Per-day breakdown
    if args.per_day and "fl_date" in df.columns:
        print("\n--- per-day breakdown ---")
        print(f"{'Date':<12} {'n':>6} {'AUC':>8} {'Brier':>8} {'Delay%':>8} {'PredPos%':>8}")
        print("-" * 55)
        for date_val in sorted(df["fl_date"].unique()):
            mask = df["fl_date"] == date_val
            n_d = int(mask.sum())
            if n_d < 10:
                continue
            y_d = y_true[mask.to_numpy()]
            p_d = y_proba[mask.to_numpy()]
            l_d = y_pred[mask.to_numpy()]
            try:
                auc_d = float(roc_auc_score(y_d, p_d)) if len(np.unique(y_d)) > 1 else None
            except Exception:
                auc_d = None
            brier_d = float(brier_score_loss(y_d, p_d))
            auc_str = f"{auc_d:.3f}" if auc_d is not None else "N/A"
            quality = "✅" if n_d >= max(args.min_actuals, 50) else "⚠"
            print(f"  {quality} {str(date_val):<10} {n_d:>6} {auc_str:>8} {brier_d:>8.3f} "
                  f"{y_d.mean():>8.1%} {l_d.mean():>8.1%}")

    # Quality-filtered aggregate (exclude incomplete days)
    if args.min_actuals > 0 and "fl_date" in df.columns:
        good_dates = []
        for date_val in df["fl_date"].unique():
            if (df["fl_date"] == date_val).sum() >= args.min_actuals:
                good_dates.append(date_val)
        if good_dates:
            mask_good = df["fl_date"].isin(good_dates)
            y_good = y_true[mask_good.to_numpy()]
            p_good = y_proba[mask_good.to_numpy()]
            try:
                auc_good = float(roc_auc_score(y_good, p_good))
            except Exception:
                auc_good = None
            brier_good = float(brier_score_loss(y_good, p_good))
            print(f"\n--- quality-filtered (>={args.min_actuals} actuals/day, {len(good_dates)} days) ---")
            print(f"  n: {len(y_good):,}")
            print(f"  AUC: {auc_good:.4f}" if auc_good is not None else "  AUC: N/A")
            print(f"  Brier: {brier_good:.4f}")

    # Calibration curve
    if args.calibration:
        print("\n--- calibration curve ---")
        bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
        print(f"{'Bin':<12} {'n':>7} {'Pred':>7} {'Actual':>7} {'Gap':>7}")
        print("-" * 45)
        for lo, hi in bins:
            mask = (y_proba >= lo) & (y_proba < hi)
            n_bin = int(mask.sum())
            if n_bin > 0:
                mean_p = float(y_proba[mask].mean())
                actual = float(y_true[mask].mean())
                gap = actual - mean_p
                inv = " ← INVERTED" if gap < -0.10 else ""
                print(f"  [{lo:.1f},{hi:.1f}) {n_bin:>7,} {mean_p:>7.3f} {actual:>7.3f} {gap:>+7.3f}{inv}")

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

"""Backtest threshold-selection strategies against resolved live actuals.

Motivation: production uses `quantile@0.22` (force ~22% of every batch positive),
which over-flags on calm days (live ATL base rate ~4-8%) destroying precision.
This script compares candidate decision rules on the held-out resolved set so we
can pick one with evidence instead of assumption.

Decision rules compared (all operate on the SAME stored `proba_delay`, i.e. the
post-adjustment probability the live threshold actually sees):
  - quantile@X : per-day quantile so ~X% flagged (current prod = 0.22)
  - abs@T      : fixed absolute probability cutoff
  - abs@bestF1 : the F1-optimal fixed cutoff found by sweep (in-sample upper bound)

Usage:
    python scripts/threshold_backtest.py --db /path/live_data.db --since 2026-05-19
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def load_resolved(db: str, since: str | None) -> pd.DataFrame:
    con = sqlite3.connect(db)
    where = f"AND p.predicted_at_utc >= '{since}'" if since else ""
    df = pd.read_sql(
        f"""
        SELECT p.fa_flight_id, p.proba_delay, p.predicted_delay,
               p.predicted_at_utc,
               CASE WHEN a.arr_delay_min > 15 THEN 1 ELSE 0 END AS delayed
        FROM predictions p
        JOIN actuals a ON p.fa_flight_id = a.fa_flight_id
        WHERE a.arr_delay_min IS NOT NULL AND a.cancelled = 0 {where}
        ORDER BY p.predicted_at_utc
        """,
        con,
    )
    con.close()
    # Latest prediction per flight (most info before resolution), matches eval_live.
    df = df.sort_values("predicted_at_utc").groupby("fa_flight_id", as_index=False).last()
    df["date"] = df["predicted_at_utc"].str[:10]
    return df


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(y) if len(y) else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": prec, "recall": rec, "F1": f1, "accuracy": acc,
            "flag_rate": (tp + fp) / len(y) if len(y) else 0.0}


def pred_quantile_perday(df: pd.DataFrame, target: float) -> np.ndarray:
    """Per-day quantile threshold so ~target fraction flagged (proxy for per-batch)."""
    out = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby("date").groups.items():
        sub = df.loc[idx, "proba_delay"].to_numpy()
        t = float(np.quantile(sub[np.isfinite(sub)], 1.0 - target)) if np.isfinite(sub).any() else 0.5
        out[df.index.get_indexer(idx)] = (df.loc[idx, "proba_delay"].to_numpy() >= t).astype(int)
    return out


def pred_abs(df: pd.DataFrame, t: float) -> np.ndarray:
    return (df["proba_delay"].to_numpy() >= t).astype(int)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--since", default="2026-05-19")
    args = ap.parse_args(argv)

    df = load_resolved(args.db, args.since).reset_index(drop=True)
    y = df["delayed"].to_numpy()
    n = len(df)
    base = y.mean()
    print(f"Resolved flights: {n:,}  ({df.date.min()} -> {df.date.max()})")
    print(f"Actual delay base rate: {base:.3f}  |  mean proba: {df.proba_delay.mean():.3f}\n")

    # F1-optimal absolute threshold (in-sample sweep, upper bound reference)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1s = [(t, confusion(y, pred_abs(df, t))["F1"]) for t in grid]
    best_t = max(f1s, key=lambda kv: kv[1])[0]

    strategies = {
        "quantile@0.22 (prod)": pred_quantile_perday(df, 0.22),
        "quantile@0.10":        pred_quantile_perday(df, 0.10),
        "quantile@0.05":        pred_quantile_perday(df, 0.05),
        "abs@0.32 (artifact)":  pred_abs(df, 0.32),
        "abs@0.50":             pred_abs(df, 0.50),
        f"abs@{best_t} (bestF1)": pred_abs(df, best_t),
    }

    # Sanity: reproduce stored production decisions
    stored = confusion(y, df["predicted_delay"].to_numpy())

    hdr = f"{'strategy':24s} {'flag%':>6s} {'prec':>6s} {'rec':>6s} {'F1':>6s} {'acc':>6s}  {'TP':>4s} {'FP':>5s} {'FN':>4s}"
    print("=" * len(hdr))
    print("OVERALL (v9 era)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    print(f"{'STORED predicted_delay':24s} {stored['flag_rate']*100:6.1f} "
          f"{stored['precision']:6.3f} {stored['recall']:6.3f} {stored['F1']:6.3f} "
          f"{stored['accuracy']:6.3f}  {stored['TP']:4d} {stored['FP']:5d} {stored['FN']:4d}")
    for name, pred in strategies.items():
        m = confusion(y, pred)
        print(f"{name:24s} {m['flag_rate']*100:6.1f} {m['precision']:6.3f} "
              f"{m['recall']:6.3f} {m['F1']:6.3f} {m['accuracy']:6.3f}  "
              f"{m['TP']:4d} {m['FP']:5d} {m['FN']:4d}")

    # Per-day precision for the two front-runners vs prod
    print("\n" + "=" * 78)
    print("PER-DAY PRECISION  (prod quantile@0.22  vs  abs@bestF1)  — lower FP = fewer false alarms")
    print("=" * 78)
    print(f"{'date':12s} {'n':>5s} {'actual%':>8s} | {'q22 flag%':>9s} {'q22 prec':>9s} | {'abs flag%':>9s} {'abs prec':>9s}")
    print("-" * 78)
    for date, sub in df.groupby("date"):
        ys = sub["delayed"].to_numpy()
        if len(ys) < 20:
            continue
        q = confusion(ys, pred_quantile_perday(sub.reset_index(drop=True), 0.22))
        a = confusion(ys, pred_abs(sub, best_t))
        print(f"{date:12s} {len(ys):5d} {ys.mean()*100:7.1f}% | "
              f"{q['flag_rate']*100:8.1f}% {q['precision']:9.3f} | "
              f"{a['flag_rate']*100:8.1f}% {a['precision']:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

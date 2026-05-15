"""Live model evaluation against resolved actuals in live_data.db.

Computes AUC, Brier, precision/recall per model version and per day,
so we can track whether v9 is genuinely better than v7_recal on live data.

Usage:
    python3 eval_live.py                  # full comparison, all days
    python3 eval_live.py --days 7         # last 7 days only
    python3 eval_live.py --out metrics_live.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss, f1_score, precision_score,
    recall_score, roc_auc_score,
)

# Artifact timestamps — add new entries whenever a model is deployed
MODEL_VERSIONS: list[tuple[str, str]] = [
    ("v7_recal",  "2000-01-01 00:00"),   # everything before v9
    ("v9",        "2026-05-15 01:43"),   # artifacts/4year_v9 deployed
]

DB_PATH = Path(__file__).parent / "live_data.db"


def label_model(ts: str) -> str:
    model = MODEL_VERSIONS[0][0]
    for name, start in MODEL_VERSIONS:
        if ts >= start:
            model = name
    return model


def load_resolved(db: Path, days: int | None) -> pd.DataFrame:
    con = sqlite3.connect(db)
    where_days = (
        f"AND p.predicted_at_utc >= datetime('now', '-{days} days')" if days else ""
    )
    df = pd.read_sql(f"""
        SELECT p.proba_delay,
               p.threshold_used,
               p.threshold_strategy,
               p.predicted_at_utc,
               a.arr_delay_min,
               CASE WHEN a.arr_delay_min > 15 THEN 1 ELSE 0 END AS delayed
        FROM predictions p
        JOIN actuals a ON p.fa_flight_id = a.fa_flight_id
        WHERE a.arr_delay_min IS NOT NULL
          AND a.cancelled = 0
          {where_days}
        ORDER BY p.predicted_at_utc
    """, con)
    con.close()
    df["model"] = df["predicted_at_utc"].apply(label_model)
    df["date"]  = df["predicted_at_utc"].str[:10]
    return df


def metrics_for(g: pd.DataFrame, threshold: float = 0.22) -> dict:
    if len(g) < 30 or g["delayed"].nunique() < 2:
        return {}
    y = g["delayed"].to_numpy()
    p = g["proba_delay"].to_numpy()
    # Use quantile threshold matching 22% positive rate prediction
    t = float(np.quantile(p, 1 - threshold))
    pred = (p >= t).astype(int)
    return {
        "n":           int(len(y)),
        "actual_rate": float(y.mean()),
        "avg_proba":   float(p.mean()),
        "AUC":         float(roc_auc_score(y, p)),
        "Brier":       float(brier_score_loss(y, p)),
        "precision":   float(precision_score(y, pred, zero_division=0)),
        "recall":      float(recall_score(y, pred, zero_division=0)),
        "F1":          float(f1_score(y, pred, zero_division=0)),
    }


def fmt(d: dict) -> str:
    if not d:
        return "  (insufficient data)"
    return (
        f"  n={d['n']:5d}  AUC={d['AUC']:.4f}  Brier={d['Brier']:.4f}  "
        f"P={d['precision']:.3f}  R={d['recall']:.3f}  F1={d['F1']:.3f}  "
        f"actual={d['actual_rate']:.3f}  proba={d['avg_proba']:.4f}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",   default=str(DB_PATH))
    ap.add_argument("--days", type=int, default=None,
                    help="Only consider predictions in the last N days")
    ap.add_argument("--out",  default=None,
                    help="Write JSON results to this file")
    args = ap.parse_args(argv)

    df = load_resolved(Path(args.db), args.days)
    if df.empty:
        print("No resolved predictions found.")
        return 1

    print(f"Resolved predictions: {len(df):,}  "
          f"({df['date'].min()} → {df['date'].max()})")
    print()

    results: dict = {}

    # ── Overall per model ──────────────────────────────────────────────────
    print("=" * 70)
    print("OVERALL PER MODEL")
    print("=" * 70)
    for model in [v[0] for v in MODEL_VERSIONS]:
        g = df[df["model"] == model]
        m = metrics_for(g)
        results[model] = m
        if m:
            print(f"{model}:")
            print(fmt(m))
    print()

    # ── Delta summary ──────────────────────────────────────────────────────
    v7m = results.get("v7_recal", {})
    v9m = results.get("v9", {})
    if v7m and v9m:
        print("DELTA  v9 vs v7_recal")
        print(f"  ΔAUC   = {v9m['AUC']   - v7m['AUC']:+.4f}")
        print(f"  ΔBrier = {v9m['Brier'] - v7m['Brier']:+.4f}")
        print(f"  ΔF1    = {v9m['F1']    - v7m['F1']:+.4f}")
        print()

    # ── Per day ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("PER DAY")
    print("=" * 70)
    print(f"{'date':12s} {'model':10s} {'n':>5s}  {'AUC':>6s}  "
          f"{'Brier':>6s}  {'F1':>5s}  {'act':>5s}  {'proba':>5s}")
    print("-" * 70)
    daily_results = []
    for (date, model), g in df.groupby(["date", "model"]):
        m = metrics_for(g)
        if not m:
            continue
        daily_results.append({"date": date, "model": model, **m})
        print(f"{date:12s} {model:10s} {m['n']:5d}  "
              f"{m['AUC']:.4f}  {m['Brier']:.4f}  {m['F1']:.3f}  "
              f"{m['actual_rate']:.3f}  {m['avg_proba']:.4f}")

    print()

    # ── Rolling 7-day AUC (if enough data) ────────────────────────────────
    day_df = pd.DataFrame(daily_results)
    if not day_df.empty:
        v9_days = day_df[day_df["model"] == "v9"]
        if len(v9_days) >= 3:
            avg_auc = v9_days["AUC"].mean()
            print(f"v9 avg daily AUC over {len(v9_days)} day(s): {avg_auc:.4f}")

    if args.out:
        out = {
            "overall": results,
            "daily":   daily_results,
        }
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nResults written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

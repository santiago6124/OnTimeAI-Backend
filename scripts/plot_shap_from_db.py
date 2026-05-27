"""Aggregate SHAP values from prediction_shap table and plot top features.

Uses real production SHAP values persisted by live_pull.py since Tier 1 fix D,
no need to re-run SHAP on offline data. Fast (seconds, not minutes).

Two plots:
  shap_live_mean_abs.png  - mean(|shap|) across all predictions (overall importance)
  shap_live_push_to_delay.png - mean(shap) restricted to positive contributions
                                 (which features actually push toward "delay")
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "live_data.db"
OUT_DIR = REPO / "artifacts" / "live_period_plots"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2026-05-22")
    parser.add_argument("--until", default="2026-05-26")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    since_iso = f"{args.since}T00:00:00"
    until_iso = f"{args.until}T23:59:59"

    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        """SELECT feature_name, shap_value
           FROM prediction_shap
           WHERE predicted_at_utc >= ? AND predicted_at_utc <= ?""",
        conn, params=(since_iso, until_iso),
    )
    conn.close()
    if df.empty:
        print(f"No SHAP rows in [{args.since}, {args.until}]")
        return 1
    print(f"Loaded {len(df)} SHAP entries across "
          f"{df['feature_name'].nunique()} unique features")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # === Plot 1: mean(|shap|) - overall feature importance ===
    abs_agg = df.assign(abs_shap=df["shap_value"].abs()).groupby("feature_name").agg(
        mean_abs_shap=("abs_shap", "mean"),
        n_occurrences=("shap_value", "size"),
        mean_shap=("shap_value", "mean"),
    ).sort_values("mean_abs_shap", ascending=False).head(args.top)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#d73027" if r["mean_shap"] > 0 else "#4575b4"
              for _, r in abs_agg.iterrows()]
    y_pos = np.arange(len(abs_agg))
    ax.barh(y_pos, abs_agg["mean_abs_shap"], color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(abs_agg.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(
        f"Top {args.top} features by mean(|SHAP|) - LIVE production\n"
        f"{args.since} to {args.until}  (n_preds with SHAP: "
        f"{df.groupby(['feature_name']).size().max()})  "
        f"red=pushes toward delay, blue=pushes toward on-time",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)
    for i, (_, r) in enumerate(abs_agg.iterrows()):
        ax.text(r["mean_abs_shap"], i,
                f"  n={int(r['n_occurrences'])}", va="center", fontsize=8)
    plt.tight_layout()
    out1 = OUT_DIR / "shap_live_mean_abs.png"
    fig.savefig(out1, dpi=110)
    plt.close(fig)
    print(f"Wrote {out1}")

    # === Plot 2: mean(shap) where positive - features pushing toward delay ===
    pos_only = df[df["shap_value"] > 0]
    pos_agg = pos_only.groupby("feature_name").agg(
        mean_pos_shap=("shap_value", "mean"),
        count_pos=("shap_value", "size"),
    ).sort_values("mean_pos_shap", ascending=False).head(args.top)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.arange(len(pos_agg)), pos_agg["mean_pos_shap"],
            color="#c73020", alpha=0.85)
    ax.set_yticks(np.arange(len(pos_agg)))
    ax.set_yticklabels(pos_agg.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean SHAP value (when positive only)")
    ax.set_title(
        f"Top {args.top} features pushing toward DELAY (positive SHAP only)\n"
        f"{args.since} to {args.until}",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)
    for i, (_, r) in enumerate(pos_agg.iterrows()):
        ax.text(r["mean_pos_shap"], i,
                f"  n={int(r['count_pos'])}", va="center", fontsize=8)
    plt.tight_layout()
    out2 = OUT_DIR / "shap_live_push_to_delay.png"
    fig.savefig(out2, dpi=110)
    plt.close(fig)
    print(f"Wrote {out2}")

    # === Stdout summary ===
    print("\nTop 15 features (mean |SHAP|):")
    print(abs_agg.head(15).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Slim feature coverage plot - bypasses LightGBM booster load (avoids OOM on Windows).

Reads `meta.joblib` directly to obtain feature_cols + cat_mapping, then walks
the live DB to compute daily coverage (1 - NaN rate) for key features. Same
output as plot_feature_coverage.py but without instantiating the booster.

Usage:
    python scripts/plot_feature_coverage_slim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ontimeai.live import open_db, build_inference_frame
from predict import prepare_inference_frame


def main() -> int:
    conn = open_db()

    print("Fetching predicted flights...")
    rows = conn.execute(
        """SELECT DISTINCT p.fa_flight_id, f.fl_date
           FROM predictions p
           JOIN flights f ON f.fa_flight_id = p.fa_flight_id
           ORDER BY f.fl_date"""
    ).fetchall()
    if not rows:
        print("No predicted flights in DB.")
        return 1
    fa_ids = [r[0] for r in rows]
    fl_dates = {r[0]: r[1] for r in rows}
    print(f"Found {len(fa_ids):,} predicted flights")

    # Load metadata only - skip the booster entirely
    meta_path = PROJECT_ROOT / "artifacts" / "4year_v9" / "meta.joblib"
    meta = joblib.load(meta_path)
    feature_cols = meta["feature_cols"]
    cat_mapping = meta["cat_mapping"]
    cat_cols_set = set(meta.get("cat_cols", []))
    print(f"Loaded meta.joblib (skipping booster): {len(feature_cols)} features")

    # Build features in chunks
    chunk_size = 2000
    dfs = []
    for i in range(0, len(fa_ids), chunk_size):
        chunk_ids = fa_ids[i:i + chunk_size]
        df_chunk = build_inference_frame(conn, chunk_ids, history_days=7)
        if df_chunk.empty:
            continue
        target_mask = df_chunk["fa_flight_id"].isin(chunk_ids) & df_chunk["ARR_DELAY"].isna()
        df_target = df_chunk[target_mask]
        if df_target.empty:
            continue
        X_raw = prepare_inference_frame(
            df_chunk, feature_cols, cat_mapping, fallback_lookup=None
        )
        X_raw_target = X_raw.loc[df_target.index].copy()
        X_raw_target["fl_date"] = df_target["fa_flight_id"].map(fl_dates)
        dfs.append(X_raw_target)

    if not dfs:
        print("No feature frames built.")
        return 1
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total target rows prepared: {len(df_all):,}")

    for c in df_all.columns:
        if c == "fl_date":
            continue
        if c not in cat_cols_set:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    features_to_plot = {
        "prev_arr_delay_tail": "Lineage (prev_arr_delay)",
        "carrier_delay_rate_24h": "Rolling Delay Rates (carrier_24h)",
        "absorb_score_origin": "Absorb Score (origin)",
        "ORIG_WX_TMPC": "METAR Weather (temperature)",
        "PREV_ACTUAL_BLOCK_MIN": "ADS-B Features (block_min)",
        "ERA5_U_KT": "ERA5 Wind Features (U component)",
    }

    grouped = df_all.groupby("fl_date")
    dates, coverage_data = [], {f: [] for f in features_to_plot}
    for name, group in grouped:
        if len(group) < 10:
            continue
        dates.append(name)
        for feat in features_to_plot:
            if feat in group.columns:
                coverage_data[feat].append(float(1.0 - group[feat].isna().mean()))
            else:
                coverage_data[feat].append(0.0)
    plot_df = pd.DataFrame(coverage_data, index=dates).sort_index()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ef4444", "#6b7280"]
    markers = ["o", "s", "^", "D", "v", "x"]
    for idx, (feat, label) in enumerate(features_to_plot.items()):
        ax.plot(plot_df.index, plot_df[feat] * 100,
                label=label, color=colors[idx], marker=markers[idx],
                markersize=5, linewidth=2, alpha=0.85)
    ax.set_title("Feature Coverage Over Time (Live Pipeline May 2026)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date (2026)", fontsize=11, labelpad=10)
    ax.set_ylabel("Data Coverage (% non-NaN)", fontsize=11, labelpad=10)
    ax.set_ylim(-5, 105)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(np.arange(0, 101, 10), fontsize=9)
    ax.legend(loc="lower left", frameon=True, facecolor="white",
              edgecolor="#e5e7eb", framealpha=0.9, fontsize=10,
              title="Feature Families", title_fontsize=11)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color("#e5e7eb")
    plt.tight_layout()

    out_dir = PROJECT_ROOT / "artifacts" / "live_period_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_coverage_over_time.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

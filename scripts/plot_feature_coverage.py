"""Plot feature coverage over time for the live period in May 2026.

Queries live_data.db for all predicted flights, rebuilds their raw features
(before cold-deck fallback), computes the daily coverage (1 - NaN rate)
for key features, and generates a beautiful line chart.

Usage:
    python3 scripts/plot_feature_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ontimeai.config import ARTIFACTS_DIR
from ontimeai.live import open_db, build_inference_frame
from ontimeai.model import load_artifact
from predict import prepare_inference_frame


def main() -> int:
    conn = open_db()

    # 1. Fetch all predicted flights with their date
    print("Fetching predicted flights...")
    rows = conn.execute(
        """SELECT DISTINCT p.fa_flight_id, f.fl_date
           FROM predictions p
           JOIN flights f ON f.fa_flight_id = p.fa_flight_id
           ORDER BY f.fl_date"""
    ).fetchall()

    if not rows:
        print("ERROR: No predicted flights found in database.")
        return 1

    fa_ids = [r[0] for r in rows]
    fl_dates = {r[0]: r[1] for r in rows}
    print(f"Rebuilding inference frame for {len(fa_ids):,} predicted flights...")

    # Build features in chunks to avoid memory issues
    chunk_size = 2000
    dfs = []
    meta = load_artifact(ARTIFACTS_DIR / "4year_v9")

    for i in range(0, len(fa_ids), chunk_size):
        chunk_ids = fa_ids[i:i + chunk_size]
        df_chunk = build_inference_frame(conn, chunk_ids, history_days=7)
        if df_chunk.empty:
            continue
        
        # Only keep target flights (where we want predictions, i.e. ARR_DELAY is NaN)
        target_mask = df_chunk["fa_flight_id"].isin(chunk_ids) & df_chunk["ARR_DELAY"].isna()
        df_target = df_chunk[target_mask]
        if df_target.empty:
            continue

        # Prepare raw features (without fallback lookup)
        X_raw = prepare_inference_frame(
            df_chunk, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=None
        )
        
        # Keep only the target indices
        X_raw_target = X_raw.loc[df_target.index].copy()
        
        # Re-attach the flight date for grouping
        X_raw_target["fl_date"] = df_target["fa_flight_id"].map(fl_dates)
        dfs.append(X_raw_target)

    if not dfs:
        print("ERROR: Failed to build features for any flights.")
        return 1

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Total target rows prepared: {len(df_all):,}")

    # Ensure numeric columns are parsed as numeric for accurate NaN checks
    cat_cols_set = set(meta.get("cat_cols", []))
    for c in df_all.columns:
        if c == "fl_date":
            continue
        if c not in cat_cols_set:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    # 2. Select key features to plot
    features_to_plot = {
        "prev_arr_delay_tail": "Lineage (prev_arr_delay)",
        "carrier_delay_rate_24h": "Rolling Delay Rates (carrier_24h)",
        "absorb_score_origin": "Absorb Score (origin)",
        "ORIG_WX_TMPC": "METAR Weather (temperature)",
        "PREV_ACTUAL_BLOCK_MIN": "ADS-B Features (block_min)",
        "ERA5_U_KT": "ERA5 Wind Features (U component)",
    }

    # Group by date and calculate coverage (1.0 - NaN rate)
    grouped = df_all.groupby("fl_date")
    dates = []
    coverage_data = {feat: [] for feat in features_to_plot}

    for name, group in grouped:
        # Ignore dates with very low flight count (outages)
        if len(group) < 10:
            continue
        dates.append(name)
        for feat in features_to_plot:
            if feat in group.columns:
                cov = float(1.0 - group[feat].isna().mean())
                coverage_data[feat].append(cov)
            else:
                coverage_data[feat].append(0.0)

    # Convert to pandas DataFrame for plotting
    plot_df = pd.DataFrame(coverage_data, index=dates)
    # Sort dates chronologically
    plot_df = plot_df.sort_index()

    # 3. Create the plot
    # Set up styling for a modern, clean, premium look
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)

    # Use a custom, harmonious color palette
    colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ef4444", "#6b7280"]
    markers = ["o", "s", "^", "D", "v", "x"]

    for idx, (feat, label) in enumerate(features_to_plot.items()):
        ax.plot(
            plot_df.index,
            plot_df[feat] * 100,
            label=label,
            color=colors[idx],
            marker=markers[idx],
            markersize=5,
            linewidth=2,
            alpha=0.85
        )

    # Format Axes
    ax.set_title("Feature Coverage Over Time (Live Pipeline May 2026)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date (2026)", fontsize=11, labelpad=10)
    ax.set_ylabel("Data Coverage (% non-NaN)", fontsize=11, labelpad=10)
    ax.set_ylim(-5, 105)
    
    # Tick formatting
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(np.arange(0, 101, 10), fontsize=9)

    # Legend
    ax.legend(
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#e5e7eb",
        framealpha=0.9,
        fontsize=10,
        title="Feature Families",
        title_fontsize=11
    )

    # Clean borders
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color("#e5e7eb")

    plt.tight_layout()

    # Create output directory and save
    out_dir = PROJECT_ROOT / "artifacts" / "live_period_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_coverage_over_time.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

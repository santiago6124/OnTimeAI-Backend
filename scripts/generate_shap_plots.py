"""Generate SHAP summary plots comparing offline test data vs. live test data.

Queries the historical dataset for offline test flights and the live database
for live predictions. Computes SHAP values using the trained v9 LightGBM model
and creates comparison plots showing feature importance shift under data degradation.
"""
from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ontimeai.config import ARTIFACTS_DIR, DATA_PATH, TrainConfig
from ontimeai.data import load_master
from ontimeai.pipeline import prepare_dataset
from ontimeai.model import load_artifact
from ontimeai.features import build_feature_matrix
from ontimeai.live import open_db, build_inference_frame
from ontimeai.lineage_fallback import load_lookups
from predict import prepare_inference_frame

def main() -> int:
    # 1. Load the v9 model
    artifact_path = ARTIFACTS_DIR / "4year_v9"
    meta = load_artifact(artifact_path)
    booster = meta["booster"]
    feature_cols = meta["feature_cols"]
    cat_mapping = meta["cat_mapping"]
    
    print(f"Loaded v9 model with {len(feature_cols)} features.")

    # 2. Rebuild offline test features (healthy/complete dataset)
    print("\n[Step 1/4] Preparing offline test dataset sample...")
    df_raw = load_master(DATA_PATH, optimize=True, valid_only=True)
    # Take a contiguous chunk from the end of the historical dataset (late 2025)
    slice_size = min(300000, len(df_raw))
    df_slice = df_raw.iloc[-slice_size:].copy().sort_values(["FL_DATE", "CRS_DEP_MIN"]).reset_index(drop=True)
    del df_raw
    
    cfg = TrainConfig()
    df_ready, _ = prepare_dataset(df_slice, cfg, already_filtered=True)
    X_offline, _, _ = build_feature_matrix(df_ready)
    
    # We want a sample of 1,000 flights for the SHAP analysis
    np.random.seed(42)
    sample_size = min(1000, len(X_offline))
    X_offline_sample = X_offline.sample(n=sample_size, random_state=42).copy()
    print(f"Prepared {len(X_offline_sample)} offline test rows.")

    # 3. Rebuild live test features (production data with cold-deck fallback)
    print("\n[Step 2/4] Preparing live test dataset sample...")
    conn = open_db()
    rows = conn.execute(
        """SELECT DISTINCT p.fa_flight_id
           FROM predictions p
           JOIN flights f ON f.fa_flight_id = p.fa_flight_id
           ORDER BY p.predicted_at_utc DESC
           LIMIT 2000"""
    ).fetchall()
    
    if not rows:
        print("ERROR: No predicted flights found in live database.")
        conn.close()
        return 1
        
    fa_ids = [r[0] for r in rows]
    df_live = build_inference_frame(conn, fa_ids, history_days=7)
    conn.close()
    
    fallback_path = ARTIFACTS_DIR / "lineage_fallback.joblib"
    fallback = load_lookups(fallback_path) if fallback_path.exists() else None
    
    # Build with fallback as in the live prediction path
    X_live = prepare_inference_frame(
        df_live, feature_cols, cat_mapping, fallback_lookup=fallback
    )
    
    # Keep only target rows (flights where we predicted)
    target_mask = df_live["fa_flight_id"].isin(fa_ids) & df_live["ARR_DELAY"].isna()
    target_idx = df_live.index[target_mask]
    X_live_target = X_live.loc[target_idx] if not target_idx.empty else X_live.head(0)
    
    live_sample_size = min(1000, len(X_live_target))
    X_live_sample = X_live_target.sample(n=live_sample_size, random_state=42).copy()
    print(f"Prepared {len(X_live_sample)} live test rows.")

    # 4. Initialize TreeExplainer
    print("\n[Step 3/4] Initializing SHAP TreeExplainer and computing values...")
    explainer = shap.TreeExplainer(booster)
    
    # Run SHAP on offline
    shap_offline = explainer.shap_values(X_offline_sample)
    # TreeExplainer might return a list of arrays for binary classifiers in some shap versions,
    # or a single 2D array. Let's force it to represent the positive class contribution.
    if isinstance(shap_offline, list) and len(shap_offline) == 2:
        shap_offline = shap_offline[1]
        
    # Run SHAP on live
    shap_live = explainer.shap_values(X_live_sample)
    if isinstance(shap_live, list) and len(shap_live) == 2:
        shap_live = shap_live[1]

    print("SHAP computation completed successfully.")

    # 5. Generate and save summary plots
    print("\n[Step 4/4] Plotting SHAP summaries...")
    out_dir = PROJECT_ROOT / "artifacts" / "live_period_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # A. Offline Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    shap.summary_plot(
        shap_offline,
        X_offline_sample,
        plot_size=None,
        show=False,
        max_display=15
    )
    plt.title("SHAP Feature Importance: Offline Test (Healthy/Complete Data)", fontsize=13, fontweight="bold", pad=15)
    out_path_offline = out_dir / "shap_offline_summary.png"
    plt.savefig(out_path_offline, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved offline SHAP plot to {out_path_offline}")
    
    # B. Live Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    shap.summary_plot(
        shap_live,
        X_live_sample,
        plot_size=None,
        show=False,
        max_display=15
    )
    plt.title("SHAP Feature Importance: Live Test (Production / Fallbacks Active)", fontsize=13, fontweight="bold", pad=15)
    out_path_live = out_dir / "shap_live_summary.png"
    plt.savefig(out_path_live, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved live SHAP plot to {out_path_live}")

    return 0

if __name__ == "__main__":
    sys.exit(main())

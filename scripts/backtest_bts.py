"""BTS Retrospective Backtest — proper out-of-time evaluation.

Uses the existing training dataset's test split (2025-Q2 onward) to evaluate
the model under controlled conditions (all features available, no chain-walk
gaps), then compares to ATL-only slices to quantify bias.

This separates model quality from infrastructure quality:
  - If AUC drops here too → model problem (overfitting, shift)
  - If AUC is fine here but bad live → infrastructure problem (missing features)

Usage:
    python3 scripts/backtest_bts.py
    python3 scripts/backtest_bts.py --artifact artifacts/4year_v5_lineage_fixed
    python3 scripts/backtest_bts.py --sample-size 100000  # faster
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ontimeai.config import ARTIFACTS_DIR, ARR_DELAY_COL, DATA_PATH, TrainConfig
from ontimeai.data import filter_valid_flights, load_master
from ontimeai.model import load_artifact, predict_proba
from predict import prepare_inference_frame


def compute_metrics(y_true, proba, threshold=0.5):
    """Compute a full set of classification metrics."""
    pred = (proba >= threshold).astype(int)
    m = {
        "n": int(len(y_true)),
        "pos_rate_truth": float(y_true.mean()),
        "pos_rate_pred": float(pred.mean()),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
    }
    if y_true.nunique() >= 2:
        m["roc_auc"] = float(roc_auc_score(y_true, proba))
    else:
        m["roc_auc"] = None
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", type=Path,
                   default=ARTIFACTS_DIR / "4year_v5_lineage_fixed")
    p.add_argument("--sample-size", type=int, default=0,
                   help="If >0, subsample the test set for speed (0 = use all)")
    p.add_argument("--test-months", type=str, default="2025-06,2025-07,2025-08",
                   help="Comma-separated YYYY-MM months to use as OOT test set")
    p.add_argument("--output", type=Path,
                   default=ARTIFACTS_DIR / "backtest_results.json")
    args = p.parse_args()

    print("=" * 70)
    print("  BTS RETROSPECTIVE BACKTEST")
    print(f"  Data: {DATA_PATH.name}")
    print(f"  Artifact: {args.artifact.name}")
    print("=" * 70)

    # Load model
    meta = load_artifact(args.artifact)
    threshold = float(meta["threshold"])
    print(f"\nModel threshold: {threshold:.4f}")
    print(f"Feature count: {len(meta['feature_cols'])}")

    # Parse test months
    test_months = [m.strip() for m in args.test_months.split(",")]
    print(f"  Test months: {test_months}")

    # Load only the months we need (test + 1 month history) to save memory
    print(f"\nLoading dataset from {DATA_PATH} (filtering to needed months)...")
    from datetime import datetime
    # Parse earliest test month for history range
    earliest_test = min(datetime.strptime(m, "%Y-%m") for m in test_months)
    history_months = []
    for offset in range(1, 3):  # 2 months of history
        hm = earliest_test - pd.DateOffset(months=offset)
        history_months.append(hm.strftime("%Y-%m"))
    needed_months = set(test_months + history_months)
    print(f"  Loading months: {sorted(needed_months)}")

    if DATA_PATH.suffix == ".parquet":
        df_all = pd.read_parquet(DATA_PATH)
        df_all["FL_DATE"] = pd.to_datetime(df_all["FL_DATE"], errors="coerce")
    else:
        df_all = pd.read_csv(DATA_PATH, parse_dates=["FL_DATE"], low_memory=False)

    # Filter immediately to needed months
    df_all["_ym"] = df_all["FL_DATE"].dt.strftime("%Y-%m")
    df_all = df_all[df_all["_ym"].isin(needed_months)].reset_index(drop=True)
    print(f"  Filtered to {len(df_all):,} rows")

    from ontimeai.data import optimize_dtypes
    df_all = optimize_dtypes(df_all)
    df_test = df_all[df_all["_ym"].isin(test_months)].copy()
    print(f"  Test set: {len(df_test):,} flights")

    if args.sample_size > 0 and len(df_test) > args.sample_size:
        df_test = df_test.sample(n=args.sample_size, random_state=42)
        print(f"  Subsampled to: {len(df_test):,}")

    if df_test.empty:
        print("ERROR: No test data found for specified months")
        print(f"  Available months: {sorted(df_all['_ym'].unique())}")
        return 1

    # We need history data for lineage features
    # Get 30 days before earliest test date
    min_test_date = df_test["FL_DATE"].min()
    history_start = min_test_date - pd.Timedelta(days=30)
    df_history = df_all[(df_all["FL_DATE"] >= history_start) &
                        (df_all["FL_DATE"] < min_test_date)]
    print(f"  History for lineage: {len(df_history):,} flights "
          f"({history_start.date()} to {min_test_date.date()})")

    # Combine test + history for lineage computation
    df_test["_role"] = "target"
    df_history["_role"] = "history"
    df_combined = pd.concat([df_history, df_test], ignore_index=True)

    # Sort by date for lineage computation
    df_combined = df_combined.sort_values("FL_DATE").reset_index(drop=True)

    # Prepare features
    print("\nPreparing features (lineage + weather + cyclical)...")
    fallback_path = ARTIFACTS_DIR / "lineage_fallback.joblib"
    from ontimeai.lineage_fallback import load_lookups
    fallback = load_lookups(fallback_path) if fallback_path.exists() else None
    if fallback:
        print(f"  Loaded cold-deck fallback")

    X = prepare_inference_frame(df_combined, meta["feature_cols"],
                                meta["cat_mapping"], fallback_lookup=fallback)

    # Filter to only test rows
    test_mask = df_combined["_role"] == "target"
    X_test = X.loc[test_mask].reset_index(drop=True)
    df_test_final = df_combined.loc[test_mask].reset_index(drop=True)

    # Coerce non-categorical columns to numeric
    cat_cols_set = set(meta.get("cat_cols", []))
    for c in X_test.columns:
        if c in cat_cols_set:
            continue
        if X_test[c].dtype == object:
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    # Create target
    y_test = (df_test_final[ARR_DELAY_COL] > 15).astype(int)

    print(f"\nScoring {len(X_test):,} test flights...")
    proba = predict_proba(meta["booster"], X_test)
    if meta.get("calibrator") is not None and meta["target"] == "binary":
        proba = meta["calibrator"].transform(proba)

    # ── Overall metrics ──────────────────────────────────────────────────
    results = {"overall": compute_metrics(y_test, proba, threshold)}
    print(f"\n{'='*70}")
    print(f"  OVERALL BACKTEST RESULTS")
    print(f"{'='*70}")
    for k, v in results["overall"].items():
        print(f"  {k:<20} {v}")

    # ── By ATL flow ──────────────────────────────────────────────────────
    results["by_atl_flow"] = {}
    if "FLOW_ATL" in df_test_final.columns:
        print(f"\n{'='*70}")
        print(f"  BY ATL FLOW")
        print(f"{'='*70}")
        print(f"  {'Flow':<20} {'n':>8} {'AUC':>8} {'F1':>8} {'Brier':>8} {'pos_rate':>10}")
        print("-" * 60)
        for flow in sorted(df_test_final["FLOW_ATL"].dropna().unique()):
            mask = df_test_final["FLOW_ATL"] == flow
            if mask.sum() < 50:
                continue
            m = compute_metrics(y_test[mask], proba[mask.to_numpy()], threshold)
            results["by_atl_flow"][str(flow)] = m
            auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] else "n/a"
            print(f"  {str(flow):<20} {m['n']:>8} {auc_str:>8} "
                  f"{m['f1']:>8.4f} {m['brier']:>8.4f} {m['pos_rate_truth']:>10.3f}")

        # ATL-touching only
        atl_mask = df_test_final["FLOW_ATL"].isin(["DEP_FROM_ATL", "ARR_TO_ATL"])
        if atl_mask.sum() >= 50:
            m = compute_metrics(y_test[atl_mask], proba[atl_mask.to_numpy()], threshold)
            results["by_atl_flow"]["ATL_TOUCHING"] = m
            print(f"  {'ATL_TOUCHING':<20} {m['n']:>8} {m['roc_auc']:.4f} "
                  f"{m['f1']:>8.4f} {m['brier']:>8.4f} {m['pos_rate_truth']:>10.3f}")
    elif "ORIGIN" in df_test_final.columns:
        # Derive ATL flows from ORIGIN/DEST
        print(f"\n{'='*70}")
        print(f"  BY ATL RELATIONSHIP")
        print(f"{'='*70}")
        is_atl_origin = df_test_final["ORIGIN"].astype(str).str.contains("ATL", na=False)
        is_atl_dest = df_test_final["DEST"].astype(str).str.contains("ATL", na=False)
        atl_mask = is_atl_origin | is_atl_dest
        non_atl_mask = ~atl_mask

        for label, mask in [("ATL-touching", atl_mask), ("Non-ATL", non_atl_mask)]:
            if mask.sum() < 50:
                continue
            m = compute_metrics(y_test[mask], proba[mask.to_numpy()], threshold)
            results["by_atl_flow"][label] = m
            auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] else "n/a"
            print(f"  {label:<20} {m['n']:>8} {auc_str:>8} "
                  f"{m['f1']:>8.4f} {m['brier']:>8.4f} {m['pos_rate_truth']:>10.3f}")

    # ── By carrier ───────────────────────────────────────────────────────
    results["by_carrier"] = {}
    carrier_col = "OP_CARRIER"
    if carrier_col in df_test_final.columns:
        print(f"\n{'='*70}")
        print(f"  BY CARRIER (n ≥ 100)")
        print(f"{'='*70}")
        print(f"  {'Carrier':<10} {'n':>8} {'AUC':>8} {'F1':>8} {'pos_rate':>10} {'Verdict':>10}")
        print("-" * 55)
        for carrier in sorted(df_test_final[carrier_col].dropna().unique()):
            mask = df_test_final[carrier_col] == carrier
            if mask.sum() < 100:
                continue
            m = compute_metrics(y_test[mask], proba[mask.to_numpy()], threshold)
            results["by_carrier"][str(carrier)] = m
            if m["roc_auc"]:
                verdict = "✅" if m["roc_auc"] > 0.75 else ("⚠️" if m["roc_auc"] > 0.60 else "❌")
                print(f"  {str(carrier):<10} {m['n']:>8} {m['roc_auc']:>8.4f} "
                      f"{m['f1']:>8.4f} {m['pos_rate_truth']:>10.3f} {verdict:>10}")

    # ── By delay rate bucket ─────────────────────────────────────────────
    results["by_delay_rate_day"] = {}
    if "FL_DATE" in df_test_final.columns:
        print(f"\n{'='*70}")
        print(f"  BY DAILY DELAY RATE")
        print(f"{'='*70}")
        df_test_final["_date"] = df_test_final["FL_DATE"].dt.date
        daily = df_test_final.groupby("_date").apply(
            lambda g: pd.Series({
                "n": len(g),
                "delay_rate": (g[ARR_DELAY_COL] > 15).mean(),
            })
        ).reset_index()

        # Bucket days by delay rate
        daily["bucket"] = pd.cut(daily["delay_rate"],
                                  bins=[0, 0.10, 0.20, 0.30, 0.50, 1.0],
                                  labels=["<10%", "10-20%", "20-30%", "30-50%", ">50%"])
        print(f"  {'Bucket':<10} {'n_days':>8} {'n_flights':>10} {'AUC':>8} {'pos_rate':>10}")
        print("-" * 50)

        for bucket in ["<10%", "10-20%", "20-30%", "30-50%", ">50%"]:
            bucket_dates = daily[daily["bucket"] == bucket]["_date"].values
            mask = df_test_final["_date"].isin(bucket_dates)
            if mask.sum() < 100:
                continue
            m = compute_metrics(y_test[mask], proba[mask.to_numpy()], threshold)
            results["by_delay_rate_day"][bucket] = m
            n_days = len(bucket_dates)
            auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] else "n/a"
            print(f"  {bucket:<10} {n_days:>8} {m['n']:>10} {auc_str:>8} {m['pos_rate_truth']:>10.3f}")

    # ── Calibration check ────────────────────────────────────────────────
    results["calibration"] = {}
    print(f"\n{'='*70}")
    print(f"  CALIBRATION CHECK")
    print(f"{'='*70}")
    print(f"  {'Bin':<12} {'n':>6} {'mean_proba':>12} {'actual_pos':>12} {'Gap':>8}")
    print("-" * 55)
    for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]:
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() < 5:
            continue
        mean_p = proba[mask].mean()
        actual = y_test[mask].mean()
        gap = actual - mean_p
        results["calibration"][f"[{lo:.1f},{hi:.1f})"] = {
            "n": int(mask.sum()), "mean_proba": float(mean_p),
            "actual_pos": float(actual), "gap": float(gap)
        }
        status = "✅" if abs(gap) < 0.05 else ("⚠️" if abs(gap) < 0.15 else "❌")
        print(f"  [{lo:.1f}, {hi:.1f}) {mask.sum():>6} {mean_p:>12.4f} "
              f"{actual:>12.4f} {gap:>+8.3f} {status}")

    # ── Probability distribution ─────────────────────────────────────────
    results["proba_distribution"] = {
        "mean": float(proba.mean()),
        "std": float(proba.std()),
        "p05": float(np.quantile(proba, 0.05)),
        "p25": float(np.quantile(proba, 0.25)),
        "p50": float(np.quantile(proba, 0.50)),
        "p75": float(np.quantile(proba, 0.75)),
        "p95": float(np.quantile(proba, 0.95)),
    }
    print(f"\n  Proba distribution: mean={proba.mean():.4f} std={proba.std():.4f}")
    print(f"  p05={np.quantile(proba,0.05):.4f} p50={np.quantile(proba,0.5):.4f} "
          f"p95={np.quantile(proba,0.95):.4f}")

    # ── Comparison to live ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BACKTEST vs. LIVE COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {'Backtest':>12} {'Live (May 4-8)':>15} {'Gap':>8}")
    print("-" * 60)
    live = {"roc_auc": 0.6013, "brier": 0.201, "f1": 0.286, "pos_rate_truth": 0.233}
    for metric in ["roc_auc", "f1", "brier", "pos_rate_truth"]:
        bt = results["overall"].get(metric)
        lv = live.get(metric)
        if bt is not None and lv is not None:
            gap = bt - lv
            print(f"  {metric:<20} {bt:>12.4f} {lv:>15.4f} {gap:>+8.3f}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

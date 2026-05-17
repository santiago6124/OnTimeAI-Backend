"""Comprehensive diagnosis of live AUC degradation.

Analyzes exported live snapshots to identify root causes of the
0.85 (offline) → 0.59 (live) AUC gap.

Usage:
    python3 scripts/diagnose_live_auc_gap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, accuracy_score

REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "artifacts"

# ─── Load data ───────────────────────────────────────────────────────────────

def load_live_data():
    """Load and join the most complete prediction + actuals snapshot."""
    preds = pd.read_csv(ARTIFACTS / "live_snapshots" / "predictions_2026-05-08_final.csv")
    actuals = pd.read_csv(ARTIFACTS / "live_snapshots" / "actuals_2026-05-08_final.csv")

    # Normalize join key
    preds["fa_flight_id"] = preds["fa_flight_id"].astype(str).str.strip()
    actuals["fa_flight_id"] = actuals["fa_flight_id"].astype(str).str.strip()

    # Also try stable_id join for better matching
    if "stable_id" in preds.columns and "stable_id" in actuals.columns:
        preds["stable_id"] = preds["stable_id"].astype(str).str.strip()
        actuals["stable_id"] = actuals["stable_id"].astype(str).str.strip()

    # Join predictions to actuals
    merged = preds.merge(actuals, on="fa_flight_id", how="inner", suffixes=("_pred", "_act"))
    print(f"Predictions: {len(preds)}, Actuals: {len(actuals)}, Matched: {len(merged)}")

    # If match is too low, try stable_id
    if len(merged) < len(preds) * 0.5 and "stable_id_pred" in merged.columns:
        merged2 = preds.merge(actuals, left_on="stable_id", right_on="stable_id",
                              how="inner", suffixes=("_pred", "_act"))
        if len(merged2) > len(merged):
            print(f"  (stable_id join improved: {len(merged)} → {len(merged2)})")
            merged = merged2

    return preds, actuals, merged


def load_flights_context():
    """Try to get flight-level context from run exports."""
    # Check if predictions have carrier/route info
    p = ARTIFACTS / "live_snapshots" / "predictions_2026-05-08_final.csv"
    df = pd.read_csv(p)
    return df


# ─── Analysis functions ──────────────────────────────────────────────────────

def analyze_prediction_distribution(preds):
    """How are probabilities distributed? Compressed = feature degradation."""
    proba = preds["proba_delay"].dropna()
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: PROBABILITY DISTRIBUTION")
    print("=" * 70)
    print(f"  n={len(proba)}")
    print(f"  mean={proba.mean():.4f}, std={proba.std():.4f}")
    print(f"  min={proba.min():.4f}, max={proba.max():.4f}")
    print(f"  quantiles:")
    for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        print(f"    p{int(q*100):02d} = {proba.quantile(q):.4f}")

    # Compare to what offline should look like
    print(f"\n  DIAGNOSIS:")
    if proba.std() < 0.12:
        print(f"  ❌ Probabilities are EXTREMELY compressed (std={proba.std():.3f})")
        print(f"     → Features are severely degraded, model can't discriminate")
    elif proba.std() < 0.18:
        print(f"  ⚠️  Probabilities are moderately compressed (std={proba.std():.3f})")
        print(f"     → Some features missing, partial discrimination ability")
    else:
        print(f"  ✅ Probability spread looks healthy (std={proba.std():.3f})")

    # Fraction of predictions in extreme bins
    low = (proba < 0.1).mean()
    high = (proba > 0.5).mean()
    print(f"\n  Low confidence (<0.1): {low*100:.1f}%")
    print(f"  High confidence (>0.5): {high*100:.1f}%")
    if low > 0.60:
        print(f"  ❌ {low*100:.0f}% of predictions are < 0.1 — model is not using features")


def analyze_threshold_strategies(preds, merged):
    """Compare AUC across threshold strategies."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: THRESHOLD STRATEGY COMPARISON")
    print("=" * 70)

    if "threshold_strategy" not in preds.columns:
        print("  No threshold_strategy column found")
        return

    # We need matched data with ground truth
    if "arr_delay_min" not in merged.columns:
        print("  No arr_delay_min in merged data")
        return

    merged["y_true"] = (merged["arr_delay_min"] > 15).astype(int)
    strategies = merged["threshold_strategy"].dropna().unique()

    for strat in sorted(strategies):
        mask = merged["threshold_strategy"] == strat
        sub = merged[mask]
        if sub["y_true"].nunique() < 2 or len(sub) < 10:
            continue
        auc = roc_auc_score(sub["y_true"], sub["proba_delay"])
        brier = brier_score_loss(sub["y_true"], sub["proba_delay"])
        pos_rate = sub["y_true"].mean()
        print(f"  {strat:<30} n={len(sub):>5}  AUC={auc:.4f}  Brier={brier:.4f}  pos_rate={pos_rate:.3f}")


def analyze_by_carrier(merged):
    """Check if specific carriers drag down AUC."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: CARRIER-LEVEL DISCRIMINATION")
    print("=" * 70)

    # Try to find carrier column
    carrier_col = None
    for c in ["op_carrier", "op_carrier_pred", "OP_CARRIER"]:
        if c in merged.columns:
            carrier_col = c
            break

    if carrier_col is None:
        # Load from predictions directly if available
        print("  No carrier column in merged data — checking predictions CSV...")
        # Read the raw predictions to get carrier info
        preds_raw = pd.read_csv(ARTIFACTS / "live_snapshots" / "predictions_2026-05-08_final.csv")
        if "op_carrier" not in preds_raw.columns:
            print("  No carrier data available in exports")
            return
        return

    merged["y_true"] = (merged["arr_delay_min"] > 15).astype(int)

    print(f"{'Carrier':<10} {'n':>6} {'AUC':>8} {'pos_rate':>10} {'pred_pos':>10} {'Verdict':>12}")
    print("-" * 60)

    for carrier in sorted(merged[carrier_col].dropna().unique()):
        sub = merged[merged[carrier_col] == carrier]
        if sub["y_true"].nunique() < 2 or len(sub) < 10:
            continue
        auc = roc_auc_score(sub["y_true"], sub["proba_delay"])
        pos_truth = sub["y_true"].mean()
        pos_pred = (sub["proba_delay"] > 0.3).mean()
        verdict = "❌ BROKEN" if auc < 0.55 else ("⚠️  POOR" if auc < 0.65 else "✅ OK")
        print(f"  {carrier:<8} {len(sub):>6} {auc:>8.4f} {pos_truth:>10.3f} {pos_pred:>10.3f} {verdict:>12}")


def analyze_calibration_inversion(merged):
    """Check if high-confidence predictions are actually wrong."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: CALIBRATION INVERSION CHECK")
    print("=" * 70)

    merged["y_true"] = (merged["arr_delay_min"] > 15).astype(int)
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]

    print(f"  {'Bin':<12} {'n':>6} {'mean_proba':>12} {'actual_pos':>12} {'Gap':>8} {'Status':>10}")
    print("-" * 65)

    for lo, hi in bins:
        mask = (merged["proba_delay"] >= lo) & (merged["proba_delay"] < hi)
        sub = merged[mask]
        if len(sub) < 5:
            continue
        mean_p = sub["proba_delay"].mean()
        actual = sub["y_true"].mean()
        gap = actual - mean_p
        status = "✅" if abs(gap) < 0.05 else ("⚠️" if abs(gap) < 0.15 else "❌ INVERTED")
        print(f"  [{lo:.1f}, {hi:.1f}) {len(sub):>6} {mean_p:>12.4f} {actual:>12.4f} {gap:>+8.3f} {status:>10}")


def analyze_temporal_pattern(merged):
    """Check if AUC degrades with delay rate."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: DELAY RATE vs. AUC CORRELATION")
    print("=" * 70)

    if "predicted_at_utc" not in merged.columns:
        # Try to derive from other columns
        print("  No temporal column available for daily analysis")
        return

    merged["y_true"] = (merged["arr_delay_min"] > 15).astype(int)
    merged["_date"] = pd.to_datetime(merged["predicted_at_utc"], format="mixed", utc=True).dt.date

    print(f"  {'Date':<12} {'n':>6} {'AUC':>8} {'delay_rate':>12} {'mean_proba':>12}")
    print("-" * 55)

    for date in sorted(merged["_date"].unique()):
        sub = merged[merged["_date"] == date]
        if sub["y_true"].nunique() < 2 or len(sub) < 10:
            continue
        auc = roc_auc_score(sub["y_true"], sub["proba_delay"])
        dr = sub["y_true"].mean()
        mp = sub["proba_delay"].mean()
        flag = "✅" if auc > 0.70 else ("⚠️" if auc > 0.55 else "❌")
        print(f"  {flag} {str(date):<10} {len(sub):>6} {auc:>8.4f} {dr:>12.3f} {mp:>12.3f}")

    # Correlation
    daily_auc = []
    daily_dr = []
    for date in sorted(merged["_date"].unique()):
        sub = merged[merged["_date"] == date]
        if sub["y_true"].nunique() < 2 or len(sub) < 10:
            continue
        daily_auc.append(roc_auc_score(sub["y_true"], sub["proba_delay"]))
        daily_dr.append(sub["y_true"].mean())

    if len(daily_auc) >= 3:
        corr = np.corrcoef(daily_auc, daily_dr)[0, 1]
        print(f"\n  Correlation(AUC, delay_rate) = {corr:.3f}")
        if corr < -0.5:
            print(f"  ❌ Strong negative correlation — model FAILS on high-delay days")
            print(f"     → Lineage features are likely missing when they're needed most")
        elif corr < -0.2:
            print(f"  ⚠️  Moderate negative correlation")
        else:
            print(f"  ✅ No clear correlation between AUC and delay rate")


def analyze_prediction_vs_actuals_timing(merged):
    """Check: are predictions being matched to correct actuals?"""
    print("\n" + "=" * 70)
    print("  ANALYSIS 6: PREDICTION ↔ ACTUAL MATCHING QUALITY")
    print("=" * 70)

    n_total = len(merged)
    n_with_delay = merged["arr_delay_min"].notna().sum()
    n_cancelled = merged.get("cancelled", merged.get("cancelled_act", pd.Series())).sum() if any(
        c in merged.columns for c in ["cancelled", "cancelled_act"]) else 0
    n_diverted = merged.get("diverted", merged.get("diverted_act", pd.Series())).sum() if any(
        c in merged.columns for c in ["diverted", "diverted_act"]) else 0

    print(f"  Total matched: {n_total}")
    print(f"  With arr_delay_min: {n_with_delay} ({n_with_delay/n_total*100:.1f}%)")
    print(f"  Cancelled: {n_cancelled}")
    print(f"  Diverted: {n_diverted}")

    # Check delay distribution
    delays = merged["arr_delay_min"].dropna()
    print(f"\n  ARR_DELAY distribution:")
    print(f"    mean={delays.mean():.1f} min, median={delays.median():.1f} min")
    print(f"    % delayed (>15): {(delays > 15).mean()*100:.1f}%")
    print(f"    % on-time (≤15): {(delays <= 15).mean()*100:.1f}%")
    print(f"    % early (<0): {(delays < 0).mean()*100:.1f}%")

    # Check for suspicious patterns
    if delays.mean() > 20:
        print(f"\n  ⚠️  Mean delay is {delays.mean():.1f} min — higher than historical average (~5-10 min)")
        print(f"     → May 2026 test period had abnormally high delays")


def main():
    print("=" * 70)
    print("  LIVE AUC GAP DIAGNOSIS")
    print("  Offline AUC: 0.849 → Live AUC: 0.589 (gap: -0.260)")
    print("=" * 70)

    preds, actuals, merged = load_live_data()

    if len(merged) < 50:
        print(f"\nERROR: Only {len(merged)} matched records — insufficient for analysis")
        print("Falling back to predictions-only analysis...")
        analyze_prediction_distribution(preds)
        return 1

    analyze_prediction_distribution(preds)
    analyze_threshold_strategies(preds, merged)
    analyze_by_carrier(merged)
    analyze_calibration_inversion(merged)
    analyze_temporal_pattern(merged)
    analyze_prediction_vs_actuals_timing(merged)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: ROOT CAUSE RANKING")
    print("=" * 70)
    merged["y_true"] = (merged["arr_delay_min"] > 15).astype(int)
    overall_auc = roc_auc_score(merged["y_true"], merged["proba_delay"]) if merged["y_true"].nunique() >= 2 else float("nan")
    print(f"\n  Overall matched AUC: {overall_auc:.4f}")
    print(f"  True positive rate: {merged['y_true'].mean():.3f}")
    print(f"  Proba std: {merged['proba_delay'].std():.4f}")
    print(f"\n  See per-analysis verdicts above for root cause identification.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

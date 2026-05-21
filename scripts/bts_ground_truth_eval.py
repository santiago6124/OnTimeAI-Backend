"""Evaluate v9 model against BTS ground truth (Jan-Feb 2026).

This provides an unbiased AUC benchmark by running the full prediction pipeline
on BTS data that was NOT in the training set, with complete features (no cold-start).

This establishes the "best possible" AUC for the model on recent data, isolating
the data quality gap from the model quality gap:

    AUC_gap = AUC_offline - AUC_live
    where AUC_offline comes from this script (BTS ground truth)
    and AUC_live comes from live_metrics.py

If AUC_offline ≈ AUC_train_test, the model hasn't degraded — the gap is data quality.
If AUC_offline << AUC_train_test, the model has degraded — retrain needed.

Usage:
    python3 scripts/bts_ground_truth_eval.py
    python3 scripts/bts_ground_truth_eval.py --input dataset_maestro_FULL_US_2026_BTS_IEM.parquet
    python3 scripts/bts_ground_truth_eval.py --atl-only    # evaluate only ATL-touching flights
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ontimeai.config import ARTIFACTS_DIR, ARR_DELAY_COL
from ontimeai.data import filter_valid_flights
from ontimeai.model import load_artifact, predict_label, predict_proba
from ontimeai.evaluation import binary_metrics
from predict import prepare_inference_frame
from ontimeai.lineage_fallback import load_lookups


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate v9 on BTS ground truth")
    p.add_argument("--artifact", type=Path, default=ARTIFACTS_DIR / "4year_v9")
    p.add_argument("--input", type=Path, default=None,
                   help="Path to BTS master parquet/CSV for 2025-2026")
    p.add_argument("--atl-only", action="store_true",
                   help="Only evaluate ATL-touching flights")
    p.add_argument("--out", type=Path, default=Path("artifacts/bts_ground_truth_eval.json"))
    p.add_argument("--sample", type=int, default=None,
                   help="Subsample size (for faster testing)")
    args = p.parse_args()

    # Auto-detect input file
    if args.input is None:
        candidates = list(PROJECT_ROOT.glob("dataset_maestro_FULL_US_2025*")) + \
                     list(PROJECT_ROOT.glob("dataset_maestro_FULL_US_2026*"))
        if not candidates:
            print("ERROR: No BTS master dataset found for 2025-2026.")
            print("Run: python3 scripts/download_bts_2026.py")
            return 1
        args.input = candidates[0]
        print(f"Auto-detected input: {args.input.name}")

    # Load model
    meta = load_artifact(args.artifact)
    print(f"Model: {args.artifact.name}")
    print(f"  target: {meta['target']}, threshold: {meta['threshold']:.4f}")
    print(f"  features: {len(meta['feature_cols'])}")

    # Load BTS data
    print(f"\nLoading {args.input.name}...")
    if args.input.suffix == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input, low_memory=False)
    if "FL_DATE" in df.columns:
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    print(f"  {len(df):,} rows, {len(df.columns)} cols")

    # Filter
    df = filter_valid_flights(df)
    print(f"  {len(df):,} after filtering cancelled/diverted/null")

    if args.atl_only:
        atl_mask = df["ORIGIN"].eq("ATL") | df["DEST"].eq("ATL")
        df = df[atl_mask].reset_index(drop=True)
        print(f"  {len(df):,} ATL-touching flights")

    if args.sample and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"  subsampled to {len(df):,}")

    # Ground truth
    y_true = (df[ARR_DELAY_COL] > 15).astype(int).to_numpy()
    pos_rate = float(y_true.mean())
    print(f"  ground truth delay rate: {pos_rate:.3f} ({y_true.sum():,} delayed)")

    # Build features
    print("\nBuilding features...")
    fallback_path = ARTIFACTS_DIR / "lineage_fallback.joblib"
    fallback = None
    if fallback_path.exists():
        try:
            fallback = load_lookups(fallback_path)
        except Exception as e:
            print(f"  ⚠ fallback load failed (pickle/pandas version mismatch): {e}")
            print("  proceeding without lineage fallback — raw NaN rates visible")

    X = prepare_inference_frame(
        df, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=fallback,
    )

    # Coerce non-categorical object columns
    cat_cols_set = set(meta.get("cat_cols", []))
    for c in X.columns:
        if c in cat_cols_set:
            continue
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Predict
    print("Predicting...")
    proba = predict_proba(meta["booster"], X)
    if meta.get("calibrator") is not None and meta["target"] == "binary":
        proba = meta["calibrator"].transform(proba)

    labels = predict_label(proba, meta["threshold"], "binary")

    # Evaluate
    metrics = binary_metrics(y_true, labels, proba)
    print(f"\n{'='*60}")
    print("BTS GROUND TRUTH EVALUATION")
    print(f"{'='*60}")
    print(f"  n:         {len(y_true):,}")
    print(f"  AUC:       {metrics['roc_auc']:.4f}")
    print(f"  Brier:     {metrics['brier']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")

    # Carrier breakdown
    if "OP_CARRIER" in df.columns:
        print(f"\n{'Carrier':<10} {'n':>8} {'AUC':>8} {'Delay%':>8}")
        print("-" * 40)
        carrier_metrics = {}
        for carrier in sorted(df["OP_CARRIER"].unique()):
            mask = df["OP_CARRIER"] == carrier
            n_c = int(mask.sum())
            if n_c < 50:
                continue
            y_c = y_true[mask.to_numpy()]
            p_c = proba[mask.to_numpy()]
            l_c = labels[mask.to_numpy()]
            try:
                m_c = binary_metrics(y_c, l_c, p_c)
                print(f"  {carrier:<8} {n_c:>8,} {m_c['roc_auc']:>8.3f} {y_c.mean():>8.1%}")
                carrier_metrics[carrier] = {
                    "n": n_c,
                    "auc": m_c["roc_auc"],
                    "delay_rate": float(y_c.mean()),
                }
            except Exception:
                print(f"  {carrier:<8} {n_c:>8,}     N/A {y_c.mean():>8.1%}")

    # ATL vs non-ATL
    if "FLOW_ATL" in df.columns:
        print(f"\n{'Flow':<15} {'n':>8} {'AUC':>8}")
        print("-" * 35)
        flow_metrics = {}
        for flow in sorted(df["FLOW_ATL"].unique()):
            mask = df["FLOW_ATL"] == flow
            n_f = int(mask.sum())
            if n_f < 50:
                continue
            y_f = y_true[mask.to_numpy()]
            p_f = proba[mask.to_numpy()]
            l_f = labels[mask.to_numpy()]
            try:
                m_f = binary_metrics(y_f, l_f, p_f)
                print(f"  {flow:<13} {n_f:>8,} {m_f['roc_auc']:>8.3f}")
                flow_metrics[flow] = {"n": n_f, "auc": m_f["roc_auc"]}
            except Exception:
                pass

    # Calibration
    print("\nCalibration:")
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    cal = {}
    for lo, hi in bins:
        mask = (proba >= lo) & (proba < hi)
        n_bin = int(mask.sum())
        if n_bin > 0:
            mean_p = float(proba[mask].mean())
            actual = float(y_true[mask].mean())
            gap = actual - mean_p
            cal[f"[{lo},{hi})"] = {"n": n_bin, "mean_proba": mean_p, "actual_pos": actual, "gap": gap}
            print(f"  [{lo:.1f},{hi:.1f}) n={n_bin:>7,} pred={mean_p:.3f} actual={actual:.3f} gap={gap:+.3f}")

    # Save results
    result = {
        "input": str(args.input.name),
        "artifact": str(args.artifact.name),
        "atl_only": args.atl_only,
        "n": len(y_true),
        "pos_rate": pos_rate,
        "metrics": metrics,
        "calibration": cal,
    }
    if "OP_CARRIER" in df.columns:
        result["by_carrier"] = carrier_metrics
    if "FLOW_ATL" in df.columns:
        result["by_flow"] = flow_metrics

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResults → {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

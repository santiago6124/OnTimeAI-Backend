"""CLI: Train and compare ensemble methods for OnTimeAI.

Runs three approaches on the same data split:
  - v1 Baseline: single LightGBM (existing)
  - v2 Stacking: LightGBM + XGBoost + ExtraTrees → LogReg meta-learner
  - v3 Blending: LightGBM + XGBoost + HistGBT → optimized weight blend

Usage:
    python3 train_ensemble.py --subsample 300000
    python3 train_ensemble.py                      # full dataset
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ontimeai.config import ARTIFACTS_DIR, DATA_PATH, TARGET_COL, TrainConfig
from ontimeai.data import load_master
from ontimeai.ensemble import train_blending, train_stacking
from ontimeai.evaluation import binary_metrics, confusion_df
from ontimeai.features import build_feature_matrix
from ontimeai.model import predict_proba, train_booster, tune_threshold, predict_label
from ontimeai.pipeline import prepare_dataset
from ontimeai.split import temporal_split


def _run_baseline(
    X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, cfg
) -> dict:
    """v1 — Single LightGBM baseline."""
    print("\n" + "=" * 70)
    print("  v1 BASELINE — Single LightGBM")
    print("=" * 70)
    t0 = time.time()
    booster = train_booster(X_train, y_train, X_val, y_val, cat_cols, cfg)
    elapsed = time.time() - t0

    proba_val = predict_proba(booster, X_val)
    proba_test = predict_proba(booster, X_test)
    threshold = tune_threshold(proba_val, y_val, metric="f1")
    y_pred_test = predict_label(proba_test, threshold, "binary")

    test_m = binary_metrics(y_test, y_pred_test, proba_test)
    test_m["elapsed_s"] = round(elapsed, 1)
    test_m["threshold"] = threshold
    print(f"  Train time: {elapsed:.1f}s")
    print(f"  Test AUC={test_m['roc_auc']:.4f}  Acc={test_m['accuracy']:.4f}  "
          f"F1={test_m['f1']:.4f}  Brier={test_m['brier']:.4f}")
    return test_m


def _run_stacking(
    X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, cfg
) -> dict:
    """v2 — Stacking Ensemble."""
    print("\n" + "=" * 70)
    print("  v2 STACKING — LightGBM + XGBoost + ExtraTrees → LogReg")
    print("=" * 70)
    t0 = time.time()
    result = train_stacking(X_train, y_train, X_val, y_val, X_test, y_test,
                            cat_cols, cfg, n_folds=3)
    elapsed = time.time() - t0

    threshold = tune_threshold(result.proba_val, y_val, metric="f1")
    y_pred_test = predict_label(result.proba_test, threshold, "binary")
    test_m = binary_metrics(y_test, y_pred_test, result.proba_test)
    test_m["elapsed_s"] = round(elapsed, 1)
    test_m["threshold"] = threshold

    # Also report individual base model AUCs
    for name, proba in result.base_probas_test.items():
        test_m[f"base_{name}_auc"] = float(
            np.round(np.float64(
                __import__("sklearn.metrics", fromlist=["roc_auc_score"])
                .roc_auc_score(y_test, proba)
            ), 4)
        )

    # Meta-learner coefficients
    test_m["meta_coefs"] = {
        k: round(float(v), 4)
        for k, v in zip(["lgb", "xgb", "et"], result.meta_model.coef_[0])
    }

    print(f"  Train time: {elapsed:.1f}s")
    print(f"  Test AUC={test_m['roc_auc']:.4f}  Acc={test_m['accuracy']:.4f}  "
          f"F1={test_m['f1']:.4f}  Brier={test_m['brier']:.4f}")
    print(f"  Base AUCs: LGB={test_m['base_lgb_auc']:.4f}  "
          f"XGB={test_m['base_xgb_auc']:.4f}  ET={test_m['base_et_auc']:.4f}")
    print(f"  Meta coefs: {test_m['meta_coefs']}")
    return test_m


def _run_blending(
    X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, cfg
) -> dict:
    """v3 — Weighted Blending."""
    print("\n" + "=" * 70)
    print("  v3 BLENDING — LightGBM + XGBoost + HistGBT → Optimized Weights")
    print("=" * 70)
    t0 = time.time()
    result = train_blending(X_train, y_train, X_val, y_val, X_test, y_test,
                            cat_cols, cfg)
    elapsed = time.time() - t0

    threshold = tune_threshold(result.proba_val, y_val, metric="f1")
    y_pred_test = predict_label(result.proba_test, threshold, "binary")
    test_m = binary_metrics(y_test, y_pred_test, result.proba_test)
    test_m["elapsed_s"] = round(elapsed, 1)
    test_m["threshold"] = threshold

    for name, proba in result.base_probas_test.items():
        from sklearn.metrics import roc_auc_score as _auc
        test_m[f"base_{name}_auc"] = round(float(_auc(y_test, proba)), 4)

    test_m["blend_weights"] = {
        k: round(float(v), 4)
        for k, v in zip(["lgb", "xgb", "hgb"], result.weights)
    }

    print(f"  Train time: {elapsed:.1f}s")
    print(f"  Test AUC={test_m['roc_auc']:.4f}  Acc={test_m['accuracy']:.4f}  "
          f"F1={test_m['f1']:.4f}  Brier={test_m['brier']:.4f}")
    print(f"  Base AUCs: LGB={test_m['base_lgb_auc']:.4f}  "
          f"XGB={test_m['base_xgb_auc']:.4f}  HGB={test_m['base_hgb_auc']:.4f}")
    print(f"  Blend weights: {test_m['blend_weights']}")
    return test_m


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and compare ensemble methods.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--num-boost-round", type=int, default=1500)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "ensemble_comparison.json")
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        target="binary",
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping,
        balance_classes=not args.no_balance,
        random_state=args.seed,
    )

    # Load and prepare data
    print("Loading data...")
    df_raw = load_master(args.data)
    if args.subsample > 0:
        df_raw = (df_raw.sort_values(["FL_DATE", "CRS_DEP_MIN"])
                  .head(args.subsample).reset_index(drop=True))
        print(f"  Subsampled to {len(df_raw)} rows.")

    print("Preparing features (lineage + rolling + cyclical + congestion)...")
    df_ready, _ = prepare_dataset(df_raw, cfg)
    train_idx, val_idx, test_idx = temporal_split(
        df_ready, train_frac=cfg.train_frac, val_frac=cfg.val_frac
    )
    X_full, cat_cols, cat_mapping = build_feature_matrix(df_ready)
    y_full = df_ready[TARGET_COL].to_numpy()

    X_train, X_val, X_test = X_full.iloc[train_idx], X_full.iloc[val_idx], X_full.iloc[test_idx]
    y_train, y_val, y_test = y_full[train_idx], y_full[val_idx], y_full[test_idx]

    print(f"\nSplit sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"Positive rate: train={y_train.mean():.3f}, val={y_val.mean():.3f}, test={y_test.mean():.3f}")
    print(f"Features: {X_full.shape[1]}, Categoricals: {len(cat_cols)}")

    # Run all three approaches
    results = {}
    results["v1_baseline"] = _run_baseline(X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, cfg)
    results["v2_stacking"] = _run_stacking(X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, cfg)
    results["v3_blending"] = _run_blending(X_train, y_train, X_val, y_val, X_test, y_test, cat_cols, cfg)

    # Summary comparison
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    header = f"{'Method':<25} {'AUC':>8} {'Acc':>8} {'F1':>8} {'Brier':>8} {'Time(s)':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<25} {m['roc_auc']:>8.4f} {m['accuracy']:>8.4f} "
              f"{m['f1']:>8.4f} {m['brier']:>8.4f} {m['elapsed_s']:>8.1f}")

    # Save results
    results["config"] = {
        "subsample": args.subsample,
        "num_boost_round": cfg.num_boost_round,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "n_features": X_full.shape[1],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

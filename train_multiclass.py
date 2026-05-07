"""CLI: Train and compare multiclass approaches for delay severity.

Approach A — Standard Multiclass LightGBM (baseline)
Approach B — Multiclass Stacking (LightGBM + XGBoost + HistGBT → LogReg)
Approach C — Chained Binary Classifiers (ordinal-aware cumulative thresholds)

Usage:
    python3 train_multiclass.py --subsample 300000
    python3 train_multiclass.py --subsample 300000 --no-balance
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, roc_auc_score,
    classification_report, confusion_matrix,
)

from ontimeai.config import ARTIFACTS_DIR, DATA_PATH, TARGET_COL, MULTICLASS_LABELS, TrainConfig
from ontimeai.data import load_master
from ontimeai.features import build_feature_matrix
from ontimeai.multiclass_ensemble import (
    train_multiclass_baseline,
    train_multiclass_stacking, predict_stacking,
    train_chained_binary, predict_chained,
)
from ontimeai.pipeline import prepare_dataset
from ontimeai.split import temporal_split


def _eval_multiclass(y_true, proba, name):
    """Compute multiclass metrics from probability matrix."""
    labels = list(MULTICLASS_LABELS)
    y_pred = np.argmax(proba, axis=1).astype(np.int8)

    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    ll = log_loss(y_true, np.clip(proba, 1e-9, 1.0), labels=labels)

    # Per-class F1
    f1_per = {
        f"C{c}": round(float(f1_score(y_true, y_pred, labels=[c], average="macro", zero_division=0)), 4)
        for c in labels
    }

    # ROC AUC macro OVR
    try:
        y_1h = np.eye(len(labels))[y_true.astype(int)]
        auc = roc_auc_score(y_1h, proba, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_dict = {
        f"true_{i}": {f"pred_{j}": int(cm[i][j]) for j in range(len(labels))}
        for i in range(len(labels))
    }

    # Per-class report
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=[f"C{c}" for c in labels],
                                   output_dict=True, zero_division=0)

    m = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_w, 4),
        "roc_auc_macro_ovr": round(auc, 4),
        "log_loss": round(ll, 4),
        "f1_per_class": f1_per,
        "confusion_matrix": cm_dict,
    }

    print(f"  [{name}] Acc={acc:.4f}  F1_macro={f1_mac:.4f}  "
          f"F1_weighted={f1_w:.4f}  AUC_macro={auc:.4f}  LogLoss={ll:.4f}")
    print(f"    Per-class F1: {f1_per}")
    # Print support per class
    support = {f"C{c}": int(np.sum(y_true == c)) for c in labels}
    print(f"    Support: {support}")
    return m


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare multiclass approaches.")
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--subsample", type=int, default=0)
    p.add_argument("--num-boost-round", type=int, default=1500)
    p.add_argument("--early-stopping", type=int, default=100)
    p.add_argument("--no-balance", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR / "multiclass_comparison")
    args = p.parse_args(argv)

    cfg = TrainConfig(
        target="multiclass",
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping,
        balance_classes=not args.no_balance,
        random_state=args.seed,
    )

    print("Loading data...")
    df_raw = load_master(args.data)
    if args.subsample > 0:
        df_raw = (df_raw.sort_values(["FL_DATE", "CRS_DEP_MIN"])
                  .head(args.subsample).reset_index(drop=True))
        print(f"  Subsampled to {len(df_raw)} rows.")

    print("Preparing features...")
    df_ready, _ = prepare_dataset(df_raw, cfg)
    tr, va, te = temporal_split(df_ready, train_frac=cfg.train_frac, val_frac=cfg.val_frac)
    X_full, cat_cols, _ = build_feature_matrix(df_ready)
    y_full = df_ready[TARGET_COL].to_numpy()

    X_tr, X_va, X_te = X_full.iloc[tr], X_full.iloc[va], X_full.iloc[te]
    y_tr, y_va, y_te = y_full[tr], y_full[va], y_full[te]

    print(f"\nSplit: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")
    for c in MULTICLASS_LABELS:
        print(f"  C{c}: train={int((y_tr==c).sum())} val={int((y_va==c).sum())} test={int((y_te==c).sum())}")

    results = {}

    # --- A: Standard Multiclass LightGBM ---
    print("\n" + "=" * 70)
    print("  APPROACH A — Standard Multiclass LightGBM")
    print("=" * 70)
    t0 = time.time()
    bst_a = train_multiclass_baseline(X_tr, y_tr, X_va, y_va, cat_cols, cfg)
    elapsed_a = time.time() - t0
    proba_a = bst_a.predict(X_te)
    m_a = _eval_multiclass(y_te, proba_a, "A")
    m_a["elapsed_s"] = round(elapsed_a, 1)
    results["A_standard_multiclass"] = m_a

    # --- B: Multiclass Stacking ---
    print("\n" + "=" * 70)
    print("  APPROACH B — Multiclass Stacking (LGB + XGB + HGB → LogReg)")
    print("=" * 70)
    t0 = time.time()
    res_b = train_multiclass_stacking(X_tr, y_tr, X_va, y_va, cat_cols, cfg, n_folds=3)
    elapsed_b = time.time() - t0
    proba_b = predict_stacking(res_b, X_te)
    m_b = _eval_multiclass(y_te, proba_b, "B")
    m_b["elapsed_s"] = round(elapsed_b, 1)
    results["B_multiclass_stacking"] = m_b

    # --- C: Chained Binary Classifiers ---
    print("\n" + "=" * 70)
    print("  APPROACH C — Chained Binary (Ordinal-Aware)")
    print("=" * 70)
    t0 = time.time()
    res_c = train_chained_binary(X_tr, y_tr, X_va, y_va, cat_cols, cfg)
    elapsed_c = time.time() - t0
    proba_c = predict_chained(res_c, X_te)
    m_c = _eval_multiclass(y_te, proba_c, "C")
    m_c["elapsed_s"] = round(elapsed_c, 1)
    results["C_chained_binary"] = m_c

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    hdr = f"{'Approach':<30} {'AUC':>8} {'Acc':>8} {'F1mac':>8} {'F1wgt':>8} {'LogL':>8} {'Time':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in results.items():
        print(f"{name:<30} {m['roc_auc_macro_ovr']:>8.4f} {m['accuracy']:>8.4f} "
              f"{m['f1_macro']:>8.4f} {m['f1_weighted']:>8.4f} "
              f"{m['log_loss']:>8.4f} {m['elapsed_s']:>7.1f}s")

    # Per-class F1 comparison
    print(f"\n{'Approach':<30} {'C0':>8} {'C1':>8} {'C2':>8} {'C3':>8}")
    print("-" * 62)
    for name, m in results.items():
        fc = m["f1_per_class"]
        print(f"{name:<30} {fc['C0']:>8.4f} {fc['C1']:>8.4f} {fc['C2']:>8.4f} {fc['C3']:>8.4f}")

    # Save
    results["config"] = {
        "subsample": args.subsample,
        "num_boost_round": cfg.num_boost_round,
        "train_size": len(X_tr), "val_size": len(X_va), "test_size": len(X_te),
        "n_features": X_full.shape[1],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "multiclass_comparison.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

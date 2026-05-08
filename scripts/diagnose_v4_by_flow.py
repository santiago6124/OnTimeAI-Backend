"""Offline diagnostic: compute v4's AUC, F1, Brier on the test split,
broken down by FLOW_ATL subsets.

Hypothesis: v4 trained on 91% NON_ATL flights and may rank ATL-touching
flights worse than the headline 0.847 AUC suggests. This script confirms
or refutes that hypothesis without spending any AeroAPI budget.

Usage:
    python3 scripts/diagnose_v4_by_flow.py
    python3 scripts/diagnose_v4_by_flow.py --save-prepared    # cache for reuse
    python3 scripts/diagnose_v4_by_flow.py --use-cached       # skip prepare_dataset

Outputs:
    artifacts/v4_by_flow_atl.json          # full numbers
    (optional) dataset_prepared_v4.parquet # cached prepared dataset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ontimeai.config import TrainConfig
from ontimeai.data import load_master
from ontimeai.features import apply_categorical_mapping, build_feature_matrix
from ontimeai.model import load_artifact, predict_proba
from ontimeai.pipeline import prepare_dataset
from ontimeai.split import temporal_split


def _metrics(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    out = {
        "n": int(len(y)),
        "pos_rate_truth": float(y.mean()),
        "pos_rate_pred": float(pred.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, proba)),
    }
    if pd.Series(y).nunique() == 2:
        out["roc_auc"] = float(roc_auc_score(y, proba))
        out["log_loss"] = float(log_loss(y, np.clip(proba, 1e-9, 1 - 1e-9)))
    else:
        out["roc_auc"] = None
        out["log_loss"] = None
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--save-prepared", action="store_true",
                   help="Cache prepared dataset for future re-runs")
    p.add_argument("--use-cached", action="store_true",
                   help="Load dataset_prepared_v4.parquet instead of running prepare_dataset")
    p.add_argument("--artifact", default="artifacts/4year_v4_full")
    p.add_argument("--out", type=Path, default=REPO / "artifacts" / "v4_by_flow_atl.json")
    args = p.parse_args()

    cached_path = REPO / "dataset_prepared_v4.parquet"

    if args.use_cached:
        if not cached_path.exists():
            print(f"ERROR: --use-cached requested but {cached_path} doesn't exist.")
            print("Run without --use-cached first (with --save-prepared) to create it.")
            return 1
        print(f"[1/4] Loading cached prepared dataset {cached_path.name}...")
        t0 = time.time()
        df_ready = pd.read_parquet(cached_path)
        print(f"      {len(df_ready):,} rows in {time.time()-t0:.0f}s")
    else:
        print("[1/4] Loading master...")
        t0 = time.time()
        df_raw = load_master()
        print(f"      {len(df_raw):,} rows in {time.time()-t0:.0f}s")

        print("[2/4] Running prepare_dataset (lineage + cyclical + congestion + weather)...")
        print("      This is the slow step — 20-45 min on 27M rows.")
        t0 = time.time()
        cfg = TrainConfig(target="binary")
        df_ready, _ = prepare_dataset(df_raw, cfg)
        print(f"      {len(df_ready):,} rows after filter in {time.time()-t0:.0f}s")
        del df_raw

        if args.save_prepared:
            print(f"[2.5/4] Caching to {cached_path.name}...")
            t0 = time.time()
            df_ready.to_parquet(cached_path, index=False, compression="zstd")
            print(f"      saved in {time.time()-t0:.0f}s")

    print("[3/4] Splitting + applying v4 model...")
    t0 = time.time()
    tr, va, te = temporal_split(df_ready, train_frac=0.6, val_frac=0.2)
    X_full, _, _ = build_feature_matrix(df_ready)
    y_full = df_ready["TARGET"].to_numpy()

    X_test = X_full.iloc[te]
    y_test = y_full[te]
    flow_atl_test = df_ready["FLOW_ATL"].iloc[te].astype(str).to_numpy()
    fl_date_test = pd.to_datetime(df_ready["FL_DATE"].iloc[te]).dt.date.to_numpy()

    meta = load_artifact(args.artifact)
    X_test = apply_categorical_mapping(X_test[meta["feature_cols"]], meta["cat_mapping"])

    proba = predict_proba(meta["booster"], X_test)
    if meta.get("calibrator"):
        proba = meta["calibrator"].transform(proba)

    threshold = meta["threshold"]
    print(f"      n_test={len(y_test):,}  threshold={threshold:.4f}  in {time.time()-t0:.0f}s")

    print()
    print("[4/4] Computing metrics by FLOW_ATL subset...")
    print()
    header = f"{'Subset':<22} {'n':>10} {'AUC':>8} {'F1':>8} {'Brier':>8} {'pos_rate':>10}"
    print(header)
    print("-" * len(header))

    subsets = {
        "All test": np.ones(len(y_test), dtype=bool),
        "NON_ATL only": flow_atl_test == "NON_ATL",
        "DEP_FROM_ATL only": flow_atl_test == "DEP_FROM_ATL",
        "ARR_TO_ATL only": flow_atl_test == "ARR_TO_ATL",
        "ATL-touching all": flow_atl_test != "NON_ATL",
    }

    results = {}
    for name, mask in subsets.items():
        n = int(mask.sum())
        if n < 100 or len(set(y_test[mask])) < 2:
            print(f"{name:<22} {n:>10,}  (skipped: too small or single class)")
            continue
        m = _metrics(y_test[mask], proba[mask], threshold)
        results[name] = m
        auc = m["roc_auc"]
        print(f"{name:<22} {n:>10,} "
              f"{auc:>8.4f} {m['f1']:>8.4f} {m['brier']:>8.4f} {m['pos_rate_truth']:>10.4f}")

    print()
    print(f"Test set date range: {fl_date_test.min()} → {fl_date_test.max()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": args.artifact,
        "threshold": float(threshold),
        "test_size": int(len(y_test)),
        "test_date_range": [str(fl_date_test.min()), str(fl_date_test.max())],
        "by_flow_atl": results,
    }
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {args.out}")

    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE:")
    print("=" * 60)
    auc_atl = results.get("ATL-touching all", {}).get("roc_auc")
    auc_non = results.get("NON_ATL only", {}).get("roc_auc")
    if auc_atl and auc_non:
        gap = auc_non - auc_atl
        print(f"NON_ATL AUC:      {auc_non:.4f}")
        print(f"ATL-touching AUC: {auc_atl:.4f}")
        print(f"Gap:              {gap:+.4f}")
        print()
        if auc_atl >= 0.80:
            print("→ v4 ranks ATL-touching well offline (>=0.80).")
            print("  The live AUC 0.58 is then NOT a model issue — it's a pipeline issue.")
            print("  Action: debug live pipeline (cold-deck, weather match, lineage history).")
        elif auc_atl >= 0.70:
            print("→ v4 ranks ATL-touching moderately well offline (0.70-0.80).")
            print("  Some degradation in live (0.58) is consistent with both shift + pipeline issues.")
            print("  Action: investigate live + consider v5 ATL-only (modest expected gain).")
        else:
            print("→ v4 ranks ATL-touching POORLY offline (<0.70).")
            print("  v4 was trained on a sub-distribution that doesn't match production.")
            print("  Action: train v5 ATL-only is justified (expect significant lift).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

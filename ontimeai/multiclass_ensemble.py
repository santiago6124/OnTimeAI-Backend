"""Multiclass ensemble methods for flight delay severity prediction.

Approach A — Standard Multiclass LightGBM (baseline)
Approach B — Multiclass Stacking (LightGBM + XGBoost + HistGBT → LogReg meta)
Approach C — Chained Binary Classifiers (ordinal-aware)
    Stage 1: C0 vs {C1,C2,C3}  (on-time vs any delay)
    Stage 2: {C0,C1} vs {C2,C3}  (minor vs major)
    Stage 3: {C0,C1,C2} vs C3  (severe delay)
    Final class = derived from cumulative thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from ontimeai.config import TrainConfig, MULTICLASS_LABELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cats_to_codes(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for c in out.select_dtypes("category").columns:
        out[c] = out[c].cat.codes.replace(-1, np.nan).astype("float32")
    return out


def _lgb_multi_train(X_tr, y_tr, X_va, y_va, cat_cols, cfg, n_class=4):
    sw = compute_sample_weight("balanced", y_tr) if cfg.balance_classes else None
    params = dict(cfg.lgb_params)
    params.update({
        "objective": "multiclass", "num_class": n_class,
        "metric": ["multi_logloss"], "verbose": -1, "seed": cfg.random_state,
    })
    ds_tr = lgb.Dataset(X_tr, label=y_tr, weight=sw,
                        categorical_feature=cat_cols, free_raw_data=False)
    ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols,
                        reference=ds_tr, free_raw_data=False)
    return lgb.train(params, ds_tr, num_boost_round=cfg.num_boost_round,
                     valid_sets=[ds_va], valid_names=["val"],
                     callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                                lgb.log_evaluation(0)])


def _lgb_binary_train(X_tr, y_tr, X_va, y_va, cat_cols, cfg):
    sw = compute_sample_weight("balanced", y_tr) if cfg.balance_classes else None
    params = dict(cfg.lgb_params)
    params.update({
        "objective": "binary", "metric": ["auc", "binary_logloss"],
        "verbose": -1, "seed": cfg.random_state,
    })
    ds_tr = lgb.Dataset(X_tr, label=y_tr, weight=sw,
                        categorical_feature=cat_cols, free_raw_data=False)
    ds_va = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols,
                        reference=ds_tr, free_raw_data=False)
    return lgb.train(params, ds_tr, num_boost_round=cfg.num_boost_round,
                     valid_sets=[ds_va], valid_names=["val"],
                     callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                                lgb.log_evaluation(0)])


def _xgb_multi_train(X_tr, y_tr, X_va, y_va, cfg, n_class=4):
    sw = compute_sample_weight("balanced", y_tr) if cfg.balance_classes else None
    X_tr_n, X_va_n = _cats_to_codes(X_tr), _cats_to_codes(X_va)
    dtrain = xgb.DMatrix(X_tr_n, label=y_tr, weight=sw)
    dval = xgb.DMatrix(X_va_n, label=y_va)
    params = {
        "objective": "multi:softprob", "num_class": n_class,
        "eval_metric": "mlogloss", "max_depth": 8, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.9, "min_child_weight": 200,
        "seed": cfg.random_state, "verbosity": 0,
    }
    return xgb.train(params, dtrain, num_boost_round=cfg.num_boost_round,
                     evals=[(dval, "val")],
                     early_stopping_rounds=cfg.early_stopping_rounds,
                     verbose_eval=False)


# ---------------------------------------------------------------------------
# Approach A — Standard Multiclass LightGBM
# ---------------------------------------------------------------------------

def train_multiclass_baseline(X_tr, y_tr, X_va, y_va, cat_cols, cfg):
    """Returns (booster, proba_val, proba_test_fn)."""
    print("  [A] Training Standard Multiclass LightGBM...")
    bst = _lgb_multi_train(X_tr, y_tr, X_va, y_va, cat_cols, cfg)
    return bst


# ---------------------------------------------------------------------------
# Approach B — Multiclass Stacking
# ---------------------------------------------------------------------------

@dataclass
class MulticlassStackingResult:
    meta_model: LogisticRegression
    lgb_booster: Any
    xgb_booster: Any
    hgb_model: Any
    n_class: int = 4


def train_multiclass_stacking(X_tr, y_tr, X_va, y_va, cat_cols, cfg, n_folds=3):
    """Stacking: LightGBM + XGBoost + HistGBT → Logistic Regression meta."""
    n = len(X_tr)
    fold_size = n // n_folds
    n_class = len(MULTICLASS_LABELS)
    oof_lgb = np.zeros((n, n_class), dtype=np.float64)
    oof_xgb = np.zeros((n, n_class), dtype=np.float64)
    oof_hgb = np.zeros((n, n_class), dtype=np.float64)

    print(f"  [B] Generating OOF predictions with {n_folds} temporal folds...")
    for fi in range(n_folds):
        start = fi * fold_size
        end = (fi + 1) * fold_size if fi < n_folds - 1 else n
        mask = np.zeros(n, dtype=bool)
        mask[start:end] = True

        Xf_tr, yf_tr = X_tr.iloc[~mask], y_tr[~mask]
        Xf_oof = X_tr.iloc[mask]

        # LightGBM
        bst = _lgb_multi_train(Xf_tr, yf_tr, Xf_oof, y_tr[mask], cat_cols, cfg, n_class)
        oof_lgb[mask] = bst.predict(Xf_oof)

        # XGBoost
        bst_x = _xgb_multi_train(Xf_tr, yf_tr, Xf_oof, y_tr[mask], cfg, n_class)
        oof_xgb[mask] = bst_x.predict(xgb.DMatrix(_cats_to_codes(Xf_oof)))

        # HistGBT
        Xf_tr_n = _cats_to_codes(Xf_tr).fillna(-999)
        Xf_oof_n = _cats_to_codes(Xf_oof).fillna(-999)
        hgb = HistGradientBoostingClassifier(
            max_iter=cfg.num_boost_round, learning_rate=0.05, max_depth=8,
            max_leaf_nodes=63, min_samples_leaf=200, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=cfg.early_stopping_rounds,
            random_state=cfg.random_state,
            class_weight="balanced" if cfg.balance_classes else None,
        )
        hgb.fit(Xf_tr_n, yf_tr)
        oof_hgb[mask] = hgb.predict_proba(Xf_oof_n)

        print(f"    Fold {fi+1}/{n_folds} done.")

    # Train final base models on full train
    print("  [B] Training final base models...")
    final_lgb = _lgb_multi_train(X_tr, y_tr, X_va, y_va, cat_cols, cfg, n_class)
    final_xgb = _xgb_multi_train(X_tr, y_tr, X_va, y_va, cfg, n_class)

    X_tr_n = _cats_to_codes(X_tr).fillna(-999)
    final_hgb = HistGradientBoostingClassifier(
        max_iter=cfg.num_boost_round, learning_rate=0.05, max_depth=8,
        max_leaf_nodes=63, min_samples_leaf=200, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=cfg.early_stopping_rounds,
        random_state=cfg.random_state,
        class_weight="balanced" if cfg.balance_classes else None,
    )
    final_hgb.fit(X_tr_n, y_tr)

    # Meta-learner
    print("  [B] Fitting meta-learner...")
    meta_X = np.hstack([oof_lgb, oof_xgb, oof_hgb])  # (N, 12)
    meta = LogisticRegression(max_iter=1000,
                              random_state=cfg.random_state)
    meta.fit(meta_X, y_tr)

    return MulticlassStackingResult(
        meta_model=meta, lgb_booster=final_lgb,
        xgb_booster=final_xgb, hgb_model=final_hgb,
    )


def predict_stacking(result: MulticlassStackingResult, X: pd.DataFrame):
    """Generate stacked predictions for a dataset."""
    X_n = _cats_to_codes(X).fillna(-999)
    p_lgb = result.lgb_booster.predict(X)
    p_xgb = result.xgb_booster.predict(xgb.DMatrix(X_n))
    p_hgb = result.hgb_model.predict_proba(X_n)
    meta_X = np.hstack([p_lgb, p_xgb, p_hgb])
    return result.meta_model.predict_proba(meta_X)


# ---------------------------------------------------------------------------
# Approach C — Chained Binary Classifiers (ordinal-aware)
# ---------------------------------------------------------------------------

@dataclass
class ChainedBinaryResult:
    stage1_booster: Any  # C0 vs {C1,C2,C3}
    stage2_booster: Any  # {C0,C1} vs {C2,C3}
    stage3_booster: Any  # {C0,C1,C2} vs C3


def train_chained_binary(X_tr, y_tr, X_va, y_va, cat_cols, cfg):
    """Three cumulative binary classifiers that respect ordinal structure.

    Stage 1: P(delay > 15 min)  → P(class >= 1)
    Stage 2: P(delay > 30 min)  → P(class >= 2)
    Stage 3: P(delay > 60 min)  → P(class >= 3)

    Final class probabilities derived from cumulative probabilities:
      P(C0) = 1 - P(>=1)
      P(C1) = P(>=1) - P(>=2)
      P(C2) = P(>=2) - P(>=3)
      P(C3) = P(>=3)
    """
    print("  [C] Training Stage 1: on-time vs any-delay (C0 vs C1+C2+C3)...")
    y1_tr = (y_tr >= 1).astype(np.int8)
    y1_va = (y_va >= 1).astype(np.int8)
    bst1 = _lgb_binary_train(X_tr, y1_tr, X_va, y1_va, cat_cols, cfg)

    print("  [C] Training Stage 2: minor vs major delay (C0+C1 vs C2+C3)...")
    y2_tr = (y_tr >= 2).astype(np.int8)
    y2_va = (y_va >= 2).astype(np.int8)
    bst2 = _lgb_binary_train(X_tr, y2_tr, X_va, y2_va, cat_cols, cfg)

    print("  [C] Training Stage 3: severe delay (C0+C1+C2 vs C3)...")
    y3_tr = (y_tr >= 3).astype(np.int8)
    y3_va = (y_va >= 3).astype(np.int8)
    bst3 = _lgb_binary_train(X_tr, y3_tr, X_va, y3_va, cat_cols, cfg)

    return ChainedBinaryResult(
        stage1_booster=bst1, stage2_booster=bst2, stage3_booster=bst3,
    )


def predict_chained(result: ChainedBinaryResult, X: pd.DataFrame) -> np.ndarray:
    """Derive class probabilities from cumulative binary predictions.

    Returns (N, 4) probability matrix.
    """
    p_ge1 = result.stage1_booster.predict(X)  # P(class >= 1)
    p_ge2 = result.stage2_booster.predict(X)  # P(class >= 2)
    p_ge3 = result.stage3_booster.predict(X)  # P(class >= 3)

    # Enforce monotonicity: P(>=1) >= P(>=2) >= P(>=3)
    p_ge2 = np.minimum(p_ge2, p_ge1)
    p_ge3 = np.minimum(p_ge3, p_ge2)

    # Derive class probabilities
    p_c0 = 1.0 - p_ge1
    p_c1 = p_ge1 - p_ge2
    p_c2 = p_ge2 - p_ge3
    p_c3 = p_ge3

    proba = np.column_stack([p_c0, p_c1, p_c2, p_c3])
    # Clip negatives from numerical precision and renormalize
    proba = np.clip(proba, 0.0, None)
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = np.where(row_sums > 0, proba / row_sums, 0.25)
    return proba

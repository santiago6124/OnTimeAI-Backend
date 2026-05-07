"""Ensemble methods for flight delay prediction.

v2 — Stacking Ensemble:
    Level-0: LightGBM + XGBoost + ExtraTrees (diverse base learners)
    Level-1: Logistic Regression meta-learner on out-of-fold predictions
    Uses temporal K-fold to respect chronological ordering.

v3 — Weighted Soft Voting (Blending):
    Models: LightGBM + XGBoost + HistGradientBoosting
    Blend weights optimized on validation set via scipy.optimize (maximize AUC).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from ontimeai.config import TrainConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lgb_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
    cfg: TrainConfig,
) -> lgb.Booster:
    """Train a single LightGBM booster (reuses existing logic)."""
    sw = compute_sample_weight("balanced", y_train) if cfg.balance_classes else None
    ds_train = lgb.Dataset(X_train, label=y_train, weight=sw,
                           categorical_feature=cat_cols, free_raw_data=False)
    ds_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols,
                         reference=ds_train, free_raw_data=False)
    return lgb.train(
        params=cfg.resolved_lgb_params(),
        train_set=ds_train,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[ds_val], valid_names=["val"],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                   lgb.log_evaluation(period=0)],
    )


def _xgb_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cfg: TrainConfig,
) -> xgb.Booster:
    """Train a single XGBoost booster."""
    sw = compute_sample_weight("balanced", y_train) if cfg.balance_classes else None
    X_tr_num = X_train.copy()
    X_va_num = X_val.copy()
    for c in X_tr_num.select_dtypes("category").columns:
        X_tr_num[c] = X_tr_num[c].cat.codes.replace(-1, np.nan).astype("float32")
        X_va_num[c] = X_va_num[c].cat.codes.replace(-1, np.nan).astype("float32")

    dtrain = xgb.DMatrix(X_tr_num, label=y_train, weight=sw, enable_categorical=False)
    dval = xgb.DMatrix(X_va_num, label=y_val, enable_categorical=False)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "min_child_weight": 200,
        "seed": cfg.random_state,
        "verbosity": 0,
    }
    booster = xgb.train(
        params, dtrain,
        num_boost_round=cfg.num_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=cfg.early_stopping_rounds,
        verbose_eval=False,
    )
    return booster


def _cats_to_codes(X: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to numeric codes for sklearn models."""
    out = X.copy()
    for c in out.select_dtypes("category").columns:
        out[c] = out[c].cat.codes.replace(-1, np.nan).astype("float32")
    return out


# ---------------------------------------------------------------------------
# v2 — Stacking Ensemble
# ---------------------------------------------------------------------------

@dataclass
class StackingResult:
    meta_model: LogisticRegression
    lgb_booster: lgb.Booster
    xgb_booster: xgb.Booster
    et_model: ExtraTreesClassifier
    proba_val: np.ndarray
    proba_test: np.ndarray
    base_probas_val: dict[str, np.ndarray]
    base_probas_test: dict[str, np.ndarray]


def train_stacking(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cat_cols: list[str],
    cfg: TrainConfig,
    n_folds: int = 3,
) -> StackingResult:
    """Train a 2-level stacking ensemble with temporal folds.

    Level 0: LightGBM, XGBoost, ExtraTrees
    Level 1: Logistic Regression on out-of-fold probabilities
    """
    n = len(X_train)
    fold_size = n // n_folds
    oof_lgb = np.zeros(n, dtype=np.float64)
    oof_xgb = np.zeros(n, dtype=np.float64)
    oof_et = np.zeros(n, dtype=np.float64)

    print(f"  [Stacking] Generating OOF predictions with {n_folds} temporal folds...")
    for fold_i in range(n_folds):
        start = fold_i * fold_size
        end = (fold_i + 1) * fold_size if fold_i < n_folds - 1 else n
        mask = np.zeros(n, dtype=bool)
        mask[start:end] = True

        Xf_train, yf_train = X_train.iloc[~mask], y_train[~mask]
        Xf_oof, yf_oof = X_train.iloc[mask], y_train[mask]

        # LightGBM fold
        bst_lgb = _lgb_train(Xf_train, yf_train, Xf_oof, yf_oof, cat_cols, cfg)
        oof_lgb[mask] = bst_lgb.predict(Xf_oof)

        # XGBoost fold
        bst_xgb = _xgb_train(Xf_train, yf_train, Xf_oof, yf_oof, cfg)
        Xf_oof_num = _cats_to_codes(Xf_oof)
        oof_xgb[mask] = bst_xgb.predict(xgb.DMatrix(Xf_oof_num))

        # ExtraTrees fold
        Xf_train_num = _cats_to_codes(Xf_train).fillna(-999)
        Xf_oof_num2 = _cats_to_codes(Xf_oof).fillna(-999)
        et = ExtraTreesClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=50,
            n_jobs=-1, random_state=cfg.random_state, class_weight="balanced",
        )
        et.fit(Xf_train_num, yf_train)
        oof_et[mask] = et.predict_proba(Xf_oof_num2)[:, 1]

        print(f"    Fold {fold_i+1}/{n_folds} done.")

    # Train final base models on full train set
    print("  [Stacking] Training final base models on full train set...")
    final_lgb = _lgb_train(X_train, y_train, X_val, y_val, cat_cols, cfg)
    final_xgb = _xgb_train(X_train, y_train, X_val, y_val, cfg)

    X_train_num = _cats_to_codes(X_train).fillna(-999)
    final_et = ExtraTreesClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=50,
        n_jobs=-1, random_state=cfg.random_state, class_weight="balanced",
    )
    final_et.fit(X_train_num, y_train)

    # Meta-learner on OOF predictions
    print("  [Stacking] Fitting meta-learner...")
    meta_X = np.column_stack([oof_lgb, oof_xgb, oof_et])
    meta_model = LogisticRegression(max_iter=1000, random_state=cfg.random_state)
    meta_model.fit(meta_X, y_train)

    # Predict val and test
    X_val_num = _cats_to_codes(X_val).fillna(-999)
    X_test_num = _cats_to_codes(X_test).fillna(-999)

    base_val = {
        "lgb": final_lgb.predict(X_val),
        "xgb": final_xgb.predict(xgb.DMatrix(X_val_num)),
        "et": final_et.predict_proba(X_val_num)[:, 1],
    }
    base_test = {
        "lgb": final_lgb.predict(X_test),
        "xgb": final_xgb.predict(xgb.DMatrix(X_test_num)),
        "et": final_et.predict_proba(X_test_num)[:, 1],
    }

    meta_val_X = np.column_stack([base_val["lgb"], base_val["xgb"], base_val["et"]])
    meta_test_X = np.column_stack([base_test["lgb"], base_test["xgb"], base_test["et"]])

    proba_val = meta_model.predict_proba(meta_val_X)[:, 1]
    proba_test = meta_model.predict_proba(meta_test_X)[:, 1]

    return StackingResult(
        meta_model=meta_model,
        lgb_booster=final_lgb,
        xgb_booster=final_xgb,
        et_model=final_et,
        proba_val=proba_val,
        proba_test=proba_test,
        base_probas_val=base_val,
        base_probas_test=base_test,
    )


# ---------------------------------------------------------------------------
# v3 — Weighted Soft Voting (Blending)
# ---------------------------------------------------------------------------

@dataclass
class BlendingResult:
    lgb_booster: lgb.Booster
    xgb_booster: xgb.Booster
    hgb_model: HistGradientBoostingClassifier
    weights: np.ndarray
    proba_val: np.ndarray
    proba_test: np.ndarray
    base_probas_val: dict[str, np.ndarray]
    base_probas_test: dict[str, np.ndarray]


def _optimize_weights(probas: list[np.ndarray], y_true: np.ndarray) -> np.ndarray:
    """Find weights that maximize ROC AUC on validation set."""
    n_models = len(probas)

    def neg_auc(w: np.ndarray) -> float:
        w_norm = w / w.sum()
        blended = sum(w_norm[i] * probas[i] for i in range(n_models))
        return -roc_auc_score(y_true, blended)

    # Start with equal weights
    w0 = np.ones(n_models) / n_models
    bounds = [(0.01, 1.0)] * n_models
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

    result = minimize(neg_auc, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x / result.x.sum()


def train_blending(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cat_cols: list[str],
    cfg: TrainConfig,
) -> BlendingResult:
    """Train a weighted blending ensemble.

    Models: LightGBM, XGBoost, HistGradientBoosting
    Weights: optimized on validation AUC via SLSQP.
    """
    print("  [Blending] Training LightGBM...")
    bst_lgb = _lgb_train(X_train, y_train, X_val, y_val, cat_cols, cfg)

    print("  [Blending] Training XGBoost...")
    bst_xgb = _xgb_train(X_train, y_train, X_val, y_val, cfg)

    print("  [Blending] Training HistGradientBoosting...")
    X_train_num = _cats_to_codes(X_train).fillna(-999)
    X_val_num = _cats_to_codes(X_val).fillna(-999)
    X_test_num = _cats_to_codes(X_test).fillna(-999)

    hgb = HistGradientBoostingClassifier(
        max_iter=cfg.num_boost_round,
        learning_rate=0.05,
        max_depth=8,
        max_leaf_nodes=63,
        min_samples_leaf=200,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=cfg.early_stopping_rounds,
        random_state=cfg.random_state,
        class_weight="balanced" if cfg.balance_classes else None,
    )
    hgb.fit(X_train_num, y_train)

    # Base predictions
    base_val = {
        "lgb": bst_lgb.predict(X_val),
        "xgb": bst_xgb.predict(xgb.DMatrix(X_val_num)),
        "hgb": hgb.predict_proba(X_val_num)[:, 1],
    }
    base_test = {
        "lgb": bst_lgb.predict(X_test),
        "xgb": bst_xgb.predict(xgb.DMatrix(X_test_num)),
        "hgb": hgb.predict_proba(X_test_num)[:, 1],
    }

    # Optimize weights on val
    print("  [Blending] Optimizing blend weights on validation AUC...")
    weights = _optimize_weights(
        [base_val["lgb"], base_val["xgb"], base_val["hgb"]], y_val,
    )
    print(f"    Optimal weights: LGB={weights[0]:.3f}, XGB={weights[1]:.3f}, HGB={weights[2]:.3f}")

    proba_val = sum(weights[i] * p for i, p in enumerate(base_val.values()))
    proba_test = sum(weights[i] * p for i, p in enumerate(base_test.values()))

    return BlendingResult(
        lgb_booster=bst_lgb,
        xgb_booster=bst_xgb,
        hgb_model=hgb,
        weights=weights,
        proba_val=proba_val,
        proba_test=proba_test,
        base_probas_val=base_val,
        base_probas_test=base_test,
    )

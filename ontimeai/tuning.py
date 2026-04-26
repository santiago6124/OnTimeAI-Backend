"""Optuna-based hyperparameter tuning for LightGBM.

Optimizes val ROC AUC over num_leaves, learning_rate, min_data_in_leaf,
feature_fraction, bagging_fraction, reg_alpha, reg_lambda. Uses the same
prepare_dataset + temporal split as the production pipeline.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from ontimeai.config import TARGET_COL, TrainConfig
from ontimeai.features import build_feature_matrix
from ontimeai.model import _sample_weights
from ontimeai.pipeline import prepare_dataset
from ontimeai.split import temporal_split


def _objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
    base_cfg: TrainConfig,
) -> float:
    params: dict[str, Any] = dict(base_cfg.lgb_params)
    params["objective"] = "binary"
    params["metric"] = ["auc"]
    params["verbose"] = -1
    params["seed"] = base_cfg.random_state

    params["num_leaves"] = trial.suggest_int("num_leaves", 31, 511, log=True)
    params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
    params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 50, 2000, log=True)
    params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.5, 1.0)
    params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.5, 1.0)
    params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 10)
    params["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
    params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)

    sw_train = _sample_weights(y_train, base_cfg.balance_classes)

    train_set = lgb.Dataset(
        X_train, label=y_train, weight=sw_train, categorical_feature=cat_cols, free_raw_data=False
    )
    val_set = lgb.Dataset(
        X_val, label=y_val, categorical_feature=cat_cols, reference=train_set, free_raw_data=False
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=base_cfg.num_boost_round,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=callbacks,
    )
    proba_val = booster.predict(X_val, num_iteration=booster.best_iteration or None)
    return float(roc_auc_score(y_val, proba_val))


def tune_hyperparams(
    df_raw: pd.DataFrame,
    cfg: TrainConfig,
    n_trials: int = 30,
    study_name: str = "ontimeai_lgb",
    sampler_seed: int = 42,
) -> dict[str, Any]:
    df_ready, _ = prepare_dataset(df_raw, cfg)
    tr, va, _ = temporal_split(df_ready, train_frac=cfg.train_frac, val_frac=cfg.val_frac)
    X_full, cat_cols, _ = build_feature_matrix(df_ready)
    y_full = df_ready[TARGET_COL].to_numpy()
    X_train, X_val = X_full.iloc[tr], X_full.iloc[va]
    y_train, y_val = y_full[tr], y_full[va]

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        lambda t: _objective(t, X_train, y_train, X_val, y_val, cat_cols, cfg),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": n_trials,
        "study_name": study_name,
    }
    return best


def save_best_params(best: dict[str, Any], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(best, f, indent=2)


def build_config_with_best(base_cfg: TrainConfig, best_params: dict[str, Any]) -> TrainConfig:
    new_cfg = deepcopy(base_cfg)
    merged = dict(new_cfg.lgb_params)
    merged.update(best_params)
    new_cfg.lgb_params = merged
    return new_cfg

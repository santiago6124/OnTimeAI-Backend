"""Tests for SHAP explainability wrappers."""
from __future__ import annotations

import pandas as pd

from ontimeai.config import TrainConfig
from ontimeai.explainability import (
    compute_shap_values,
    explain_instance,
    global_feature_importance,
)
from ontimeai.features import build_feature_matrix
from ontimeai.model import train_booster
from ontimeai.pipeline import prepare_dataset
from ontimeai.split import temporal_split


def _quick_booster(larger_master: pd.DataFrame):
    cfg = TrainConfig(target="binary", num_boost_round=20, early_stopping_rounds=5)
    df_ready, _ = prepare_dataset(larger_master, cfg)
    tr, va, te = temporal_split(df_ready)
    X, cat_cols, _ = build_feature_matrix(df_ready)
    y = df_ready["TARGET"].to_numpy()
    booster = train_booster(X.iloc[tr], y[tr], X.iloc[va], y[va], cat_cols, cfg)
    return booster, X.iloc[te].head(20)


def test_shap_global_importance(larger_master: pd.DataFrame) -> None:
    booster, X_sample = _quick_booster(larger_master)
    sv = compute_shap_values(booster, X_sample)
    imp = global_feature_importance(sv, list(X_sample.columns))
    assert len(imp) == len(X_sample.columns)
    assert (imp >= 0).all()


def test_explain_single_instance(larger_master: pd.DataFrame) -> None:
    booster, X_sample = _quick_booster(larger_master)
    sv = compute_shap_values(booster, X_sample)
    exp = explain_instance(sv, list(X_sample.columns), row_idx=0, top_n=5)
    assert len(exp) == 5
    assert set(exp.columns) == {"feature", "contribution"}

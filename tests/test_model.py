"""Tests for LightGBM trainer, threshold tuning, artifact persistence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ontimeai.config import TrainConfig
from ontimeai.features import build_feature_matrix
from ontimeai.model import (
    load_artifact,
    predict_label,
    predict_proba,
    save_artifact,
    train_booster,
    tune_threshold,
)
from ontimeai.pipeline import prepare_dataset
from ontimeai.split import temporal_split


@pytest.fixture
def trained_binary(larger_master: pd.DataFrame, tmp_path: Path) -> dict:
    cfg = TrainConfig(
        target="binary",
        num_boost_round=30,
        early_stopping_rounds=5,
        artifacts_dir=tmp_path / "art",
    )
    df_ready, _ = prepare_dataset(larger_master, cfg)
    tr, va, te = temporal_split(df_ready, train_frac=cfg.train_frac, val_frac=cfg.val_frac)
    X, cat_cols, cat_mapping = build_feature_matrix(df_ready)
    y = df_ready["TARGET"].to_numpy()
    booster = train_booster(X.iloc[tr], y[tr], X.iloc[va], y[va], cat_cols, cfg)
    return {
        "cfg": cfg,
        "booster": booster,
        "X": X,
        "y": y,
        "tr": tr, "va": va, "te": te,
        "cat_cols": cat_cols,
        "cat_mapping": cat_mapping,
    }


def test_predict_proba_shape_binary(trained_binary: dict) -> None:
    X_te = trained_binary["X"].iloc[trained_binary["te"]]
    proba = predict_proba(trained_binary["booster"], X_te)
    assert proba.shape == (len(X_te),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_tune_threshold_in_range(trained_binary: dict) -> None:
    X_va = trained_binary["X"].iloc[trained_binary["va"]]
    y_va = trained_binary["y"][trained_binary["va"]]
    proba = predict_proba(trained_binary["booster"], X_va)
    thr = tune_threshold(proba, y_va, metric="f1")
    assert 0.05 <= thr <= 0.95


def test_save_and_load_artifact_round_trip(trained_binary: dict, tmp_path: Path) -> None:
    out = save_artifact(
        trained_binary["booster"],
        threshold=0.5,
        feature_cols=list(trained_binary["X"].columns),
        cat_cols=trained_binary["cat_cols"],
        cat_mapping=trained_binary["cat_mapping"],
        target="binary",
        metadata={"smoke": True},
        out_dir=tmp_path / "art2",
    )
    assert (out / "model.lgb").exists()
    assert (out / "meta.joblib").exists()
    assert (out / "metrics.json").exists()

    loaded = load_artifact(out)
    assert loaded["target"] == "binary"
    assert loaded["threshold"] == 0.5
    assert loaded["feature_cols"] == list(trained_binary["X"].columns)

    X_te = trained_binary["X"].iloc[trained_binary["te"]]
    proba_orig = predict_proba(trained_binary["booster"], X_te)
    proba_loaded = predict_proba(loaded["booster"], X_te)
    assert np.allclose(proba_orig, proba_loaded)


def test_predict_label_binary() -> None:
    proba = np.array([0.2, 0.6, 0.8])
    assert (predict_label(proba, 0.5, "binary") == np.array([0, 1, 1])).all()


def test_predict_label_multiclass() -> None:
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])
    assert (predict_label(proba, 0.5, "multiclass") == np.array([0, 1, 2])).all()


def test_tune_threshold_rejects_2d() -> None:
    with pytest.raises(ValueError):
        tune_threshold(np.array([[0.1, 0.9], [0.5, 0.5]]), np.array([0, 1]))

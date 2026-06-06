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
    quantile_threshold,
    save_artifact,
    select_threshold,
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


def test_quantile_threshold_matches_target_pos_rate() -> None:
    rng = np.random.default_rng(0)
    proba = rng.uniform(0.0, 1.0, size=10_000)
    thr = quantile_threshold(proba, target_pos_rate=0.26)
    pos_rate = (proba >= thr).mean()
    assert abs(pos_rate - 0.26) < 0.01


def test_quantile_threshold_robust_to_shift() -> None:
    rng = np.random.default_rng(1)
    # Train-time-like distribution centered at 0.5
    train_like = np.clip(rng.normal(0.5, 0.15, size=5_000), 0, 1)
    # Live-like distribution shifted up to 0.72 (the v3_full live observation)
    live_like = np.clip(rng.normal(0.72, 0.13, size=200), 0, 1)
    target = 0.26

    # Static threshold tuned on train collapses on live (predicts ~all positives).
    static_thr = float(np.quantile(train_like, 1 - target))
    static_pos_rate = (live_like >= static_thr).mean()

    # Quantile threshold computed on the live batch keeps pos rate near target.
    dyn_thr = quantile_threshold(live_like, target)
    dyn_pos_rate = (live_like >= dyn_thr).mean()

    # Static threshold over-predicts on the shifted distribution; dynamic stays on target.
    assert static_pos_rate > target + 0.4, (
        f"static threshold should over-predict on shifted dist (got {static_pos_rate:.2f})"
    )
    assert abs(dyn_pos_rate - target) < 0.05
    assert abs(dyn_pos_rate - target) < abs(static_pos_rate - target)


def test_quantile_threshold_handles_nan_and_small_input() -> None:
    proba = np.array([0.1, np.nan, 0.4, np.inf, 0.9])
    thr = quantile_threshold(proba, target_pos_rate=0.5)
    assert np.isfinite(thr)

    # All-NaN falls back to 0.5
    all_nan = np.array([np.nan, np.nan])
    assert quantile_threshold(all_nan, 0.5) == 0.5


def test_quantile_threshold_validates_inputs() -> None:
    with pytest.raises(ValueError):
        quantile_threshold(np.array([[0.1, 0.9]]), 0.5)
    with pytest.raises(ValueError):
        quantile_threshold(np.array([0.1, 0.5]), 0.0)
    with pytest.raises(ValueError):
        quantile_threshold(np.array([0.1, 0.5]), 1.0)


# ── select_threshold: decision-rule precedence (abs > quantile > artifact) ──

def test_select_threshold_abs_overrides_quantile() -> None:
    proba = np.linspace(0.0, 1.0, 100)
    thr, strat = select_threshold(
        proba, target_pos_rate=0.22, artifact_threshold=0.32, abs_threshold=0.5,
    )
    assert thr == 0.5
    assert strat == "abs@0.50"


def test_select_threshold_falls_back_to_quantile() -> None:
    # abs_threshold == 0 (default) → keep existing quantile behavior (no regression).
    proba = np.linspace(0.0, 1.0, 100)
    thr, strat = select_threshold(
        proba, target_pos_rate=0.22, artifact_threshold=0.32,
    )
    assert strat == "quantile@0.22"
    # ~22% of the batch flagged at this threshold.
    assert abs((proba >= thr).mean() - 0.22) <= 0.03


def test_select_threshold_artifact_when_batch_too_small() -> None:
    proba = np.array([0.1, 0.2, 0.3])  # < 5 finite values
    thr, strat = select_threshold(
        proba, target_pos_rate=0.22, artifact_threshold=0.32,
    )
    assert thr == 0.32
    assert strat == "artifact"


def test_select_threshold_abs_adapts_positive_rate_to_conditions() -> None:
    # The core fix: a fixed cutoff flags MORE on storm-like batches and FEWER on
    # calm batches, instead of forcing a constant ~22% like the quantile rule.
    calm = np.concatenate([np.full(90, 0.05), np.full(10, 0.6)])   # ~10% truly high
    storm = np.concatenate([np.full(40, 0.05), np.full(60, 0.6)])  # ~60% truly high
    t_calm, _ = select_threshold(calm, target_pos_rate=0.22, artifact_threshold=0.32, abs_threshold=0.5)
    t_storm, _ = select_threshold(storm, target_pos_rate=0.22, artifact_threshold=0.32, abs_threshold=0.5)
    calm_rate = (calm >= t_calm).mean()
    storm_rate = (storm >= t_storm).mean()
    assert calm_rate < 0.22 < storm_rate  # adapts; quantile@0.22 would pin both at 0.22

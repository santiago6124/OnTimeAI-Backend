"""End-to-end pipeline tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ontimeai.config import LEAKY_COLS, TrainConfig
from ontimeai.pipeline import PipelineResult, prepare_dataset, run_training


def test_prepare_dataset_has_no_leakage(larger_master: pd.DataFrame) -> None:
    cfg = TrainConfig(target="binary")
    df_ready, y = prepare_dataset(larger_master, cfg)
    for c in LEAKY_COLS:
        assert c not in df_ready.columns, f"leakage: {c}"
    assert "ARR_DELAY" not in df_ready.columns
    assert "TARGET" in df_ready.columns
    assert len(df_ready) == len(y)
    # Engineered features present
    for c in ("dep_hour_sin", "dep_hour_cos", "congestion_orig_window", "congestion_dest_window"):
        assert c in df_ready.columns


def test_run_training_binary_end_to_end(larger_master: pd.DataFrame, tmp_path: Path) -> None:
    cfg = TrainConfig(
        target="binary",
        num_boost_round=30,
        early_stopping_rounds=5,
        artifacts_dir=tmp_path / "art",
    )
    result: PipelineResult = run_training(cfg, df_raw=larger_master)
    assert 0.05 <= result.threshold <= 0.95
    assert 0.0 <= result.metrics["test"]["accuracy"] <= 1.0
    assert 0.0 <= result.metrics["test"]["roc_auc"] <= 1.0 or result.metrics["test"]["roc_auc"] != result.metrics["test"]["roc_auc"]  # nan allowed
    assert result.artifact_dir is not None
    assert (result.artifact_dir / "model.lgb").exists()
    assert (result.artifact_dir / "meta.joblib").exists()
    assert (result.artifact_dir / "metrics.json").exists()


def test_run_training_multiclass_end_to_end(larger_master: pd.DataFrame, tmp_path: Path) -> None:
    cfg = TrainConfig(
        target="multiclass",
        num_boost_round=30,
        early_stopping_rounds=5,
        artifacts_dir=tmp_path / "art_mc",
    )
    result: PipelineResult = run_training(cfg, df_raw=larger_master)
    assert "f1_macro" in result.metrics["test"]
    assert 0.0 <= result.metrics["test"]["accuracy"] <= 1.0
    assert result.artifact_dir is not None


def test_cascade_stub_raises() -> None:
    from ontimeai.cascade import predict_cascade
    import pytest

    with pytest.raises(NotImplementedError):
        predict_cascade()

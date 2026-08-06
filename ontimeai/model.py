"""LightGBM trainer, threshold tuner, and artifact serialization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_curve
from sklearn.utils.class_weight import compute_sample_weight

from ontimeai.config import TrainConfig


def _sample_weights(y: np.ndarray, balance: bool) -> np.ndarray | None:
    if not balance:
        return None
    return compute_sample_weight("balanced", y)


def train_booster(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
    cfg: TrainConfig,
) -> lgb.Booster:
    sw_train = _sample_weights(y_train, cfg.balance_classes)
    sw_val = _sample_weights(y_val, cfg.balance_classes)

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        weight=sw_train,
        categorical_feature=cat_cols,
        free_raw_data=True,
    )
    val_set = lgb.Dataset(
        X_val,
        label=y_val,
        weight=sw_val,
        categorical_feature=cat_cols,
        reference=train_set,
        free_raw_data=True,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    booster = lgb.train(
        params=cfg.resolved_lgb_params(),
        train_set=train_set,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[val_set],
        valid_names=["val"],
        callbacks=callbacks,
    )
    return booster


def tune_threshold(
    proba: np.ndarray,
    y_true: np.ndarray,
    metric: str = "f1",
) -> float:
    if proba.ndim != 1:
        raise ValueError("tune_threshold expects 1-D probability array (binary)")
    if metric == "f1":
        thresholds = np.linspace(0.05, 0.95, 91)
        scores = np.array(
            [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in thresholds]
        )
        return float(thresholds[int(np.argmax(scores))])
    if metric == "youden":
        fpr, tpr, ths = roc_curve(y_true, proba)
        j = tpr - fpr
        return float(ths[int(np.argmax(j))])
    raise ValueError(f"Unknown metric: {metric}")


def quantile_threshold(proba: np.ndarray, target_pos_rate: float) -> float:
    """Pick a binary threshold so the predicted positive rate matches a target.

    Robust to monotone shifts in the raw probability distribution (the typical
    failure mode in production cold-start). Requires no ground-truth labels —
    only the proba distribution of the batch being scored.
    """
    if proba.ndim != 1:
        raise ValueError("quantile_threshold expects 1-D probability array")
    if not 0.0 < target_pos_rate < 1.0:
        raise ValueError("target_pos_rate must be in (0, 1)")
    finite = proba[np.isfinite(proba)]
    if finite.size == 0:
        return 0.5
    return float(np.quantile(finite, 1.0 - target_pos_rate))


def select_threshold(
    target_proba: np.ndarray,
    *,
    target_pos_rate: float,
    artifact_threshold: float,
    abs_threshold: float = 0.0,
) -> tuple[float, str]:
    """Choose the binary decision threshold for a scoring batch.

    Precedence (first match wins):
      1. ``abs_threshold > 0``  → fixed absolute probability cutoff (``"abs@T"``).
         Adapts the predicted-positive rate to live conditions: it flags more on
         storm days and fewer on calm days, instead of forcing a constant rate.
      2. ``target_pos_rate`` in (0, 1) and batch has >= 5 finite probabilities
         → per-batch quantile so ~``target_pos_rate`` is flagged (``"quantile@X"``).
         Robust to monotone proba shift but mis-fires when the live base rate
         diverges from ``target_pos_rate``.
      3. otherwise → the artifact's static training threshold (``"artifact"``).

    Returns ``(threshold, strategy_label)``.
    """
    finite = np.asarray(target_proba)[np.isfinite(target_proba)] if target_proba.size else target_proba
    if abs_threshold > 0:
        return float(abs_threshold), f"abs@{abs_threshold:.2f}"
    if 0.0 < target_pos_rate < 1.0 and finite.size >= 5:
        return quantile_threshold(target_proba, target_pos_rate), f"quantile@{target_pos_rate:.2f}"
    return float(artifact_threshold), "artifact"


def predict_proba(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return booster.predict(X, num_iteration=booster.best_iteration or None)


def predict_label(proba: np.ndarray, threshold: float, target: str) -> np.ndarray:
    if target == "binary":
        if proba.ndim != 1:
            raise ValueError("Binary predict expects 1-D probabilities")
        return (proba >= threshold).astype(np.int8)
    if target == "multiclass":
        if proba.ndim != 2:
            raise ValueError("Multiclass predict expects 2-D probabilities")
        return np.argmax(proba, axis=1).astype(np.int8)
    raise ValueError(f"Unknown target: {target}")


def save_artifact(
    booster: lgb.Booster,
    *,
    threshold: float,
    feature_cols: list[str],
    cat_cols: list[str],
    cat_mapping: dict[str, list],
    target: str,
    metadata: dict[str, Any],
    out_dir: Path,
    calibrator: Any | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.lgb"))
    joblib.dump(
        {
            "threshold": threshold,
            "feature_cols": feature_cols,
            "cat_cols": cat_cols,
            "cat_mapping": cat_mapping,
            "target": target,
            "calibrator": calibrator,
        },
        out_dir / "meta.joblib",
    )
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return out_dir


def load_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    # `params={'num_threads': 1}` forces single-threaded model parse → avoids a
    # SIGSEGV (with "Model format error, expect a tree here" cascade) we saw in
    # Cloud Run when lightgbm 4.6 + libgomp1 parsed the booster concurrently.
    booster = lgb.Booster(
        model_file=str(artifact_dir / "model.lgb"),
        params={"num_threads": 1},
    )
    meta = joblib.load(artifact_dir / "meta.joblib")
    meta["booster"] = booster
    meta.setdefault("calibrator", None)
    return meta

"""Tests for binary and multiclass evaluation metrics."""
from __future__ import annotations

import numpy as np

from ontimeai.evaluation import binary_metrics, confusion_df, multiclass_metrics


def test_binary_metrics_perfect_prediction() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.98, 0.99])
    pred = (p >= 0.5).astype(int)
    m = binary_metrics(y, pred, p)
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["f1"] == 1.0
    assert m["support_pos"] == 2
    assert m["support_neg"] == 2


def test_binary_metrics_random_values() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    pred = (p >= 0.5).astype(int)
    m = binary_metrics(y, pred, p)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert m["brier"] >= 0.0


def test_multiclass_metrics_perfect() -> None:
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    proba = np.eye(4)[y]
    pred = y.copy()
    m = multiclass_metrics(y, pred, proba, labels=[0, 1, 2, 3])
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0


def test_confusion_df_shape_and_values() -> None:
    y = np.array([0, 1, 1, 0])
    pred = np.array([0, 1, 0, 0])
    cm = confusion_df(y, pred, labels=[0, 1])
    assert cm.shape == (2, 2)
    assert cm.loc["true_0", "pred_0"] == 2
    assert cm.loc["true_1", "pred_0"] == 1
    assert cm.loc["true_1", "pred_1"] == 1
    assert cm.loc["true_0", "pred_1"] == 0

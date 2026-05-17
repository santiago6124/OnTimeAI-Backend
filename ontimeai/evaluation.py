"""Classification metrics for binary and multiclass delay targets."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-9, 1 - 1e-9))),
        "support_pos": int(np.sum(y_true == 1)),
        "support_neg": int(np.sum(y_true == 0)),
    }


def multiclass_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, labels: list[int]
) -> dict[str, Any]:
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    macro_auc = float("nan")
    try:
        y_true_1h = np.eye(len(labels))[y_true.astype(int)]
        macro_auc = float(roc_auc_score(y_true_1h, y_proba, average="macro", multi_class="ovr"))
    except Exception:
        pass
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": {
            str(c): float(f1_score(y_true, y_pred, labels=[c], average="macro", zero_division=0))
            for c in labels
        },
        "roc_auc_macro_ovr": macro_auc,
        "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-9, 1.0), labels=labels)),
        "per_class_report": report,
    }


def confusion_df(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None) -> pd.DataFrame:
    if labels is None:
        labels = sorted(list({*y_true.tolist(), *y_pred.tolist()}))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )

"""Probability calibration for LightGBM binary outputs.

Fits an isotonic or sigmoid (Platt) calibrator on validation probabilities so
that raw model scores can be converted into well-calibrated probabilities. This
improves downstream threshold decisions and Brier score without changing ROC AUC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class Calibrator:
    method: Literal["isotonic", "sigmoid"]
    estimator: Any

    def transform(self, proba: np.ndarray) -> np.ndarray:
        p = np.asarray(proba, dtype=float)
        if self.method == "isotonic":
            return self.estimator.predict(np.clip(p, 1e-9, 1 - 1e-9))
        # sigmoid / Platt
        logits = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9))
        return self.estimator.predict_proba(logits.reshape(-1, 1))[:, 1]


@dataclass
class StackedCalibrator:
    """Two-stage calibrator: applies cal1 first, then cal2.

    Used when refitting calibration on live data without discarding the
    original calibration trained on the historical validation set.
    """
    cal1: Any
    cal2: Any

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return self.cal2.transform(self.cal1.transform(proba))


def fit_calibrator(
    proba_val: np.ndarray,
    y_val: np.ndarray,
    method: Literal["isotonic", "sigmoid"] = "isotonic",
) -> Calibrator:
    p = np.asarray(proba_val, dtype=float)
    y = np.asarray(y_val, dtype=int)
    if p.ndim != 1:
        raise ValueError("Calibration expects 1-D binary probabilities")

    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(np.clip(p, 1e-9, 1 - 1e-9), y)
        return Calibrator(method="isotonic", estimator=iso)
    if method == "sigmoid":
        logits = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9))
        lr = LogisticRegression()
        lr.fit(logits.reshape(-1, 1), y)
        return Calibrator(method="sigmoid", estimator=lr)
    raise ValueError(f"Unknown method: {method}")

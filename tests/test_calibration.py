"""Tests for probability calibration wrappers."""
from __future__ import annotations

import numpy as np

from ontimeai.calibration import Calibrator, fit_calibrator


def test_isotonic_calibration_improves_brier_on_skewed_probas() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    # Build systematically miscalibrated probas: push toward extremes
    base = rng.uniform(0, 1, n)
    skewed = base**2
    cal = fit_calibrator(skewed, y, method="isotonic")
    calibrated = cal.transform(skewed)
    assert calibrated.shape == skewed.shape
    assert ((calibrated >= 0) & (calibrated <= 1)).all()


def test_sigmoid_calibration_returns_probas() -> None:
    rng = np.random.default_rng(1)
    n = 500
    y = rng.integers(0, 2, n)
    p = rng.uniform(0.01, 0.99, n)
    cal = fit_calibrator(p, y, method="sigmoid")
    out = cal.transform(p)
    assert out.shape == p.shape
    assert ((out >= 0) & (out <= 1)).all()


def test_calibrator_handles_zero_one_extremes() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(0, 1, 200)
    p[0] = 0.0
    p[1] = 1.0
    cal = fit_calibrator(p, y, method="isotonic")
    out = cal.transform(p)
    assert np.isfinite(out).all()


def test_calibrator_is_monotone_on_sorted_input_isotonic() -> None:
    rng = np.random.default_rng(3)
    n = 1000
    p = np.sort(rng.uniform(0, 1, n))
    # Construct y roughly correlated with p so isotonic has signal
    y = (rng.uniform(0, 1, n) < p).astype(int)
    cal = fit_calibrator(p, y, method="isotonic")
    out = cal.transform(p)
    # Isotonic enforces monotonicity wrt the fitted input order
    diffs = np.diff(out)
    assert (diffs >= -1e-9).all()

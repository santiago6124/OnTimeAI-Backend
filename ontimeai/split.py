"""Chronological train/val/test split — never random for time series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    date_col: str = "FL_DATE",
    intraday_col: str = "CRS_DEP_MIN",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(
            f"Invalid fractions: train={train_frac}, val={val_frac} (train+val must be <1)"
        )

    sort_cols = [c for c in (date_col, intraday_col) if c in df.columns]
    if not sort_cols:
        raise ValueError(f"Missing sort columns {date_col}/{intraday_col}")

    order = df.sort_values(sort_cols, kind="stable").index.to_numpy()
    n = len(order)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return train_idx, val_idx, test_idx

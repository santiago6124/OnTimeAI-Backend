"""Tests for data loading and leakage-safe filtering."""
from __future__ import annotations

import pandas as pd

from ontimeai.config import ARR_DELAY_COL, FILTER_COLS, LEAKY_COLS
from ontimeai.data import drop_leaky_target_columns, filter_valid_flights


def test_filter_removes_cancelled_diverted_and_null_target(tiny_master: pd.DataFrame) -> None:
    filtered = filter_valid_flights(tiny_master)
    assert filtered["CANCELLED"].sum() == 0
    assert filtered["DIVERTED"].sum() == 0
    assert filtered[ARR_DELAY_COL].notna().all()
    assert len(filtered) <= len(tiny_master)


def test_filter_is_idempotent(tiny_master: pd.DataFrame) -> None:
    once = filter_valid_flights(tiny_master)
    twice = filter_valid_flights(once)
    assert len(once) == len(twice)


def test_drop_leaky_removes_all_leaky_cols(tiny_master: pd.DataFrame) -> None:
    df = filter_valid_flights(tiny_master)
    dropped = drop_leaky_target_columns(df)
    forbidden = {*LEAKY_COLS, ARR_DELAY_COL, *FILTER_COLS}
    assert forbidden.isdisjoint(dropped.columns), f"leaked: {forbidden & set(dropped.columns)}"


def test_drop_leaky_preserves_schedule_cols(tiny_master: pd.DataFrame) -> None:
    df = filter_valid_flights(tiny_master)
    dropped = drop_leaky_target_columns(df)
    for col in ("FL_DATE", "CRS_DEP_MIN", "ORIGIN", "DEST", "OP_CARRIER", "DISTANCE"):
        assert col in dropped.columns

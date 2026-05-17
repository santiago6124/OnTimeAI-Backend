"""Tests for chronological train/val/test split."""
from __future__ import annotations

import pandas as pd
import pytest

from ontimeai.split import temporal_split


def test_split_sizes_sum_to_n(tiny_master: pd.DataFrame) -> None:
    tr, va, te = temporal_split(tiny_master, train_frac=0.6, val_frac=0.2)
    assert len(tr) + len(va) + len(te) == len(tiny_master)


def test_split_no_index_overlap(tiny_master: pd.DataFrame) -> None:
    tr, va, te = temporal_split(tiny_master, train_frac=0.6, val_frac=0.2)
    assert set(tr).isdisjoint(va)
    assert set(va).isdisjoint(te)
    assert set(tr).isdisjoint(te)


def test_split_is_chronological(tiny_master: pd.DataFrame) -> None:
    tr, va, te = temporal_split(tiny_master, train_frac=0.6, val_frac=0.2)
    sort_key = tiny_master["FL_DATE"] + pd.to_timedelta(tiny_master["CRS_DEP_MIN"], unit="m")
    max_tr = sort_key.loc[tr].max()
    min_va = sort_key.loc[va].min()
    max_va = sort_key.loc[va].max()
    min_te = sort_key.loc[te].min()
    assert max_tr <= min_va
    assert max_va <= min_te


def test_split_raises_on_bad_fractions(tiny_master: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        temporal_split(tiny_master, train_frac=0.8, val_frac=0.3)
    with pytest.raises(ValueError):
        temporal_split(tiny_master, train_frac=0.0, val_frac=0.2)

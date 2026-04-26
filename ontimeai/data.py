"""Load the BTS + IEM master dataset and apply leakage-safe filters."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ontimeai.config import ARR_DELAY_COL, DATA_PATH, FILTER_COLS, LEAKY_COLS


def load_master(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["FL_DATE"], low_memory=False)


def filter_valid_flights(df: pd.DataFrame) -> pd.DataFrame:
    cancelled = df.get("CANCELLED", pd.Series(0, index=df.index)).fillna(0).astype(int) == 1
    diverted = df.get("DIVERTED", pd.Series(0, index=df.index)).fillna(0).astype(int) == 1
    has_target = df[ARR_DELAY_COL].notna()
    return df.loc[~cancelled & ~diverted & has_target].reset_index(drop=True)


def drop_leaky_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in (*LEAKY_COLS, ARR_DELAY_COL, *FILTER_COLS) if c in df.columns]
    return df.drop(columns=cols)

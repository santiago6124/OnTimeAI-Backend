"""Load the BTS + IEM master dataset and apply leakage-safe filters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ontimeai.config import ARR_DELAY_COL, CATEGORICAL_COLS, DATA_PATH, FILTER_COLS, LEAKY_COLS


# Columns guaranteed not to have NaN (calendar fields BTS always populates).
INT16_COLS: tuple[str, ...] = (
    "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK",
)

# All other numeric columns → float32. NaN is handled natively by float;
# precision (~7 sig digits) is more than enough for delays, weather, schedule.
# Using float32 instead of pandas nullable Int* avoids LightGBM compatibility issues.
FLOAT32_COLS: tuple[str, ...] = (
    "YEAR", "OP_CARRIER_FL_NUM",
    "CRS_DEP_TIME", "CRS_ARR_TIME", "DEP_TIME", "ARR_TIME",
    "CRS_DEP_MIN",
    "DEP_DELAY", "ARR_DELAY", "DEP_DELAY_NEW", "ARR_DELAY_NEW",
    "CANCELLED", "DIVERTED",
    "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME", "DISTANCE",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    "ORIG_WX_TMPC", "ORIG_WX_DWPC", "ORIG_WX_RELH", "ORIG_WX_DRCT",
    "ORIG_WX_SKNT", "ORIG_WX_ALTI", "ORIG_WX_P01M", "ORIG_WX_VSBY", "ORIG_WX_GUST",
    "ORIG_WX_PRECIP_FLAG", "ORIG_WX_LOW_VIS_FLAG", "ORIG_WX_STRONG_WIND_FLAG",
    "ORIG_WX_MATCH_GAP_MIN",
    "DEST_WX_TMPC", "DEST_WX_DWPC", "DEST_WX_RELH", "DEST_WX_DRCT",
    "DEST_WX_SKNT", "DEST_WX_ALTI", "DEST_WX_P01M", "DEST_WX_VSBY", "DEST_WX_GUST",
    "DEST_WX_PRECIP_FLAG", "DEST_WX_LOW_VIS_FLAG", "DEST_WX_STRONG_WIND_FLAG",
    "DEST_WX_MATCH_GAP_MIN",
)


def optimize_dtypes(df: pd.DataFrame, *, verbose: bool = True, in_place: bool = True) -> pd.DataFrame:
    """Downcast columns to memory-efficient dtypes without losing precision.

    Reduces a default-dtype DataFrame from ~19 GB to ~7 GB on the FULL US
    master (27.5M rows × 65 cols), making it tractable on 16 GB Macs.

    - Strings with low cardinality → category (carrier, tail, origin, dest)
    - Calendar fields (month, dow, day) → int16
    - Everything else numeric → float32 (handles NaN natively, LightGBM-friendly)

    Uses plain numpy dtypes (not pandas nullable Int*) to avoid any downstream
    incompatibility with LightGBM, scikit-learn, etc.

    By default mutates the input DataFrame in place to keep memory peak low
    (one column conversion at a time, no full copy). Set `in_place=False`
    if you need the original preserved.
    """
    out = df if in_place else df.copy()
    mem_before = out.memory_usage(deep=True).sum() / 1e9

    # Categoricals (carrier, tail, origin, dest, flow, par_airport, wx codes)
    for col in CATEGORICAL_COLS:
        if col in out.columns and out[col].dtype != "category":
            out[col] = out[col].astype("category")

    # Calendar fields — never NaN, fit easily in int16
    for col in INT16_COLS:
        if col in out.columns:
            try:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int16)
            except Exception:
                pass

    # All other numeric — float32
    for col in FLOAT32_COLS:
        if col in out.columns:
            try:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
            except Exception:
                pass

    mem_after = out.memory_usage(deep=True).sum() / 1e9
    if verbose:
        print(f"  optimize_dtypes: {mem_before:.2f} GB → {mem_after:.2f} GB "
              f"({(1 - mem_after/mem_before)*100:.0f}% reduction)")
    return out


def load_master(path: Path | str = DATA_PATH, *, optimize: bool = True) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        if "FL_DATE" in df.columns:
            df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    else:
        df = pd.read_csv(path, parse_dates=["FL_DATE"], low_memory=False)

    if optimize:
        df = optimize_dtypes(df)
    return df


def filter_valid_flights(df: pd.DataFrame) -> pd.DataFrame:
    cancelled = df.get("CANCELLED", pd.Series(0, index=df.index)).fillna(0).astype(int) == 1
    diverted = df.get("DIVERTED", pd.Series(0, index=df.index)).fillna(0).astype(int) == 1
    has_target = df[ARR_DELAY_COL].notna()
    return df.loc[~cancelled & ~diverted & has_target].reset_index(drop=True)


def drop_leaky_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in (*LEAKY_COLS, ARR_DELAY_COL, *FILTER_COLS) if c in df.columns]
    return df.drop(columns=cols)

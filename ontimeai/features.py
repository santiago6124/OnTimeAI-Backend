"""Feature engineering: temporal cyclical, congestion windows, weather interactions, target."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ontimeai.config import (
    ARR_DELAY_COL,
    CATEGORICAL_COLS,
    DROP_COLS,
    MULTICLASS_BINS,
    MULTICLASS_LABELS,
    TARGET_COL,
)


CYCLICAL_COLS: tuple[str, ...] = (
    "dep_hour_sin",
    "dep_hour_cos",
    "dep_dow_sin",
    "dep_dow_cos",
    "dep_month_sin",
    "dep_month_cos",
)

CONGESTION_COLS: tuple[str, ...] = (
    "congestion_orig_window",
    "congestion_dest_window",
)

INTERACTION_COLS: tuple[str, ...] = (
    "wx_both_precip",
    "wx_both_low_vis",
    "wx_both_strong_wind",
)


def build_target(df: pd.DataFrame, target: str, threshold_min: float) -> pd.Series:
    arr = pd.to_numeric(df[ARR_DELAY_COL], errors="coerce")
    if target == "binary":
        return (arr > threshold_min).astype(np.int8).rename(TARGET_COL)
    if target == "multiclass":
        cat = pd.cut(
            arr,
            bins=list(MULTICLASS_BINS),
            labels=list(MULTICLASS_LABELS),
            right=True,
            include_lowest=True,
        )
        codes = cat.cat.codes.astype(np.int16)
        codes = codes.where(codes >= 0, other=-1)
        return codes.rename(TARGET_COL)
    raise ValueError(f"Unknown target: {target}")


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mins = pd.to_numeric(out["CRS_DEP_MIN"], errors="coerce").astype(float)
    out["dep_hour_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    out["dep_hour_cos"] = np.cos(2 * np.pi * mins / 1440.0)
    dow = pd.to_numeric(out["DAY_OF_WEEK"], errors="coerce").astype(float)
    out["dep_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dep_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    month = pd.to_numeric(out["MONTH"], errors="coerce").astype(float)
    out["dep_month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["dep_month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return out


def _window_counts_per_group(mins: np.ndarray, window: float) -> np.ndarray:
    order = np.argsort(mins, kind="stable")
    sorted_mins = mins[order]
    left = np.searchsorted(sorted_mins, sorted_mins - window, side="left")
    right = np.searchsorted(sorted_mins, sorted_mins + window, side="right")
    counts_sorted = (right - left).astype(np.int32)
    counts = np.empty_like(counts_sorted)
    counts[order] = counts_sorted
    return counts


def add_congestion_features(df: pd.DataFrame, window_min: int = 30) -> pd.DataFrame:
    original_index = df.index
    out = df.reset_index(drop=True).copy()
    date_key = pd.to_datetime(out["FL_DATE"]).dt.strftime("%Y-%m-%d").astype(str)
    mins_all = pd.to_numeric(out["CRS_DEP_MIN"], errors="coerce").fillna(-1.0).to_numpy(dtype=float)

    for airport_col, out_col in (
        ("ORIGIN", "congestion_orig_window"),
        ("DEST", "congestion_dest_window"),
    ):
        grp_series = pd.Series((date_key + "|" + out[airport_col].astype(str)).to_numpy())
        counts = np.zeros(len(out), dtype=np.int32)
        for pos in grp_series.groupby(grp_series).indices.values():
            pos = np.asarray(pos, dtype=int)
            counts[pos] = _window_counts_per_group(mins_all[pos], float(window_min))
        out[out_col] = counts

    out.index = original_index
    return out


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) != 0)
    return s.astype("string").str.lower().isin(["true", "1"]).fillna(False)


WEATHER_FLAGS: tuple[str, ...] = ("PRECIP_FLAG", "LOW_VIS_FLAG", "STRONG_WIND_FLAG")


def normalize_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for prefix in ("ORIG", "DEST"):
        for flag in WEATHER_FLAGS:
            col = f"{prefix}_WX_{flag}"
            if col in out.columns:
                out[col] = _to_bool(out[col]).astype(np.int8)
    return out


def add_weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapping = (
        ("PRECIP_FLAG", "wx_both_precip"),
        ("LOW_VIS_FLAG", "wx_both_low_vis"),
        ("STRONG_WIND_FLAG", "wx_both_strong_wind"),
    )
    for flag, out_col in mapping:
        orig = f"ORIG_WX_{flag}"
        dest = f"DEST_WX_{flag}"
        if orig in out.columns and dest in out.columns:
            out[out_col] = (_to_bool(out[orig]) & _to_bool(out[dest])).astype(np.int8)
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    categorical_cols: tuple[str, ...] = CATEGORICAL_COLS,
    drop_cols: tuple[str, ...] = DROP_COLS,
) -> tuple[pd.DataFrame, list[str], dict[str, list]]:
    to_drop = [c for c in (*drop_cols, target_col) if c in df.columns]
    X = df.drop(columns=to_drop).copy()

    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(np.int8)

    present_cats = [c for c in categorical_cols if c in X.columns]
    cat_mapping: dict[str, list] = {}
    for c in present_cats:
        X[c] = X[c].astype("category")
        cat_mapping[c] = list(X[c].cat.categories)
    return X, present_cats, cat_mapping


def apply_categorical_mapping(X: pd.DataFrame, cat_mapping: dict[str, list]) -> pd.DataFrame:
    out = X.copy()
    for col, cats in cat_mapping.items():
        if col in out.columns:
            out[col] = pd.Categorical(out[col], categories=cats)
    return out

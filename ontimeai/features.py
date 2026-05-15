"""Feature engineering: temporal cyclical, congestion windows, weather interactions, target."""
from __future__ import annotations

import datetime

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

DELAY_THRESHOLD_MIN = 15.0

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

HOLIDAY_COLS: tuple[str, ...] = (
    "is_us_holiday",
    "days_to_nearest_holiday",
    "is_thanksgiving_window",
    "is_summer_peak",
)

ABSORB_COLS: tuple[str, ...] = ("absorb_score_origin",)


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
    out = df
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
    out = df
    date_key = pd.to_datetime(out["FL_DATE"]).dt.strftime("%Y-%m-%d").astype(str)
    mins_all = pd.to_numeric(out["CRS_DEP_MIN"], errors="coerce").fillna(-1.0).to_numpy(dtype=float)

    for airport_col, out_col in (
        ("ORIGIN", "congestion_orig_window"),
        ("DEST", "congestion_dest_window"),
    ):
        # grp_series is built from a fresh numpy array so groupby.indices returns
        # 0-based positions that align with mins_all and counts directly.
        grp_series = pd.Series((date_key + "|" + out[airport_col].astype(str)).to_numpy())
        counts = np.zeros(len(out), dtype=np.int32)
        for pos in grp_series.groupby(grp_series).indices.values():
            pos = np.asarray(pos, dtype=int)
            counts[pos] = _window_counts_per_group(mins_all[pos], float(window_min))
        out[out_col] = counts

    return out


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) != 0)
    return s.astype("string").str.lower().isin(["true", "1"]).fillna(False)


WEATHER_FLAGS: tuple[str, ...] = ("PRECIP_FLAG", "LOW_VIS_FLAG", "STRONG_WIND_FLAG")


def normalize_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    for prefix in ("ORIG", "DEST"):
        for flag in WEATHER_FLAGS:
            col = f"{prefix}_WX_{flag}"
            if col in out.columns:
                out[col] = _to_bool(out[col]).astype(np.int8)
    return out


def add_weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df
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


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add US holiday proximity and seasonal peak features (vectorized).

    Requires FL_DATE column. Emits:
      - is_us_holiday            (0/1 binary)
      - days_to_nearest_holiday  (signed float, clipped ±30)
      - is_thanksgiving_window   (Wed before → Sun after Thanksgiving)
      - is_summer_peak           (Jul-4 week + Labor Day weekend)
    """
    out = df
    fl_dates = pd.to_datetime(out.get("FL_DATE"), errors="coerce")
    has_date = fl_dates.notna()

    # --- is_us_holiday ---
    try:
        import holidays as hols_pkg
        years = set(fl_dates.dt.year.dropna().astype(int).tolist()) | {2022, 2023, 2024, 2025, 2026}
        us_hols = hols_pkg.US(years=sorted(years))
        hol_dates_sorted = sorted(us_hols.keys())
    except ImportError:
        # Fallback: major US federal holidays (hard-coded for training years 2022-2026)
        hol_dates_sorted = [
            datetime.date(y, m, d)
            for y in range(2022, 2027)
            for m, d in [
                (1, 1),   # New Year's Day
                (7, 4),   # Independence Day
                (11, 11), # Veterans Day
                (12, 25), # Christmas
            ]
        ]

    hol_set = set(hol_dates_sorted)
    fl_date_only = fl_dates.dt.date
    out["is_us_holiday"] = fl_date_only.apply(
        lambda d: int(d in hol_set) if d is not None and d is not pd.NaT else 0
    ).astype(np.int8)

    # --- days_to_nearest_holiday (searchsorted, vectorized) ---
    hol_arr = np.array([pd.Timestamp(d).value for d in hol_dates_sorted], dtype=np.int64)
    dates_ns = fl_dates.astype("int64").to_numpy()
    ns_per_day = 86_400_000_000_000

    idx = np.searchsorted(hol_arr, dates_ns, side="left")
    idx_r = np.clip(idx, 0, len(hol_arr) - 1)
    idx_l = np.clip(idx - 1, 0, len(hol_arr) - 1)
    diff_r = (hol_arr[idx_r] - dates_ns) / ns_per_day
    diff_l = (hol_arr[idx_l] - dates_ns) / ns_per_day
    nearest = np.where(np.abs(diff_r) <= np.abs(diff_l), diff_r, diff_l)
    nearest = np.clip(nearest, -30.0, 30.0)
    nearest[~has_date.to_numpy()] = np.nan
    out["days_to_nearest_holiday"] = nearest.astype(np.float32)

    # --- is_thanksgiving_window (4th Thu Nov ± a few days) ---
    year_arr = fl_dates.dt.year.to_numpy(dtype=float)
    month_arr = fl_dates.dt.month.to_numpy(dtype=float)
    day_arr = fl_dates.dt.day.to_numpy(dtype=float)

    # Nov 1 day-of-week per year (0=Mon, 3=Thu)
    unique_years = [y for y in np.unique(year_arr[~np.isnan(year_arr)]).astype(int) if 1900 < y < 2100]
    nov1_dow_map = {y: datetime.date(y, 11, 1).weekday() for y in unique_years}
    sep1_dow_map = {y: datetime.date(y, 9, 1).weekday() for y in unique_years}

    nov1_dow = np.array([nov1_dow_map.get(int(y), 0) if not np.isnan(y) else 0 for y in year_arr])
    first_thu = 1 + (3 - nov1_dow) % 7  # day-of-month of first Thursday in Nov
    tgiving_dom = first_thu + 21         # 4th Thursday
    is_tgiving = (
        has_date.to_numpy()
        & (month_arr == 11)
        & (day_arr >= tgiving_dom - 1)
        & (day_arr <= tgiving_dom + 3)
    )
    out["is_thanksgiving_window"] = is_tgiving.astype(np.int8)

    # --- is_summer_peak (Jul 4 week + Labor Day weekend) ---
    is_jul4 = has_date.to_numpy() & (
        ((month_arr == 6) & (day_arr >= 30)) | ((month_arr == 7) & (day_arr <= 7))
    )
    sep1_dow = np.array([sep1_dow_map.get(int(y), 0) if not np.isnan(y) else 0 for y in year_arr])
    labor_day_dom = 1 + (7 - sep1_dow) % 7   # first Monday of September
    is_labor_day = (
        has_date.to_numpy()
        & (month_arr == 9)
        & (day_arr >= labor_day_dom - 3)
        & (day_arr <= labor_day_dom)
    )
    out["is_summer_peak"] = (is_jul4 | is_labor_day).astype(np.int8)

    return out


def add_absorb_score(
    df: pd.DataFrame,
    window_hours: float = 2.0,
    out_col: str = "absorb_score_origin",
) -> pd.DataFrame:
    """Airport resilience: fraction of inbound flights that arrived on-time at ORIGIN.

    For each flight i with ORIGIN=X:
      - Looks at all flights j where DEST[j] == X
      - Whose actual arrival (EVENT_DEST_UTC[j] + ARR_DELAY[j]) falls in
        (EVENT_ORIGIN_UTC[i] - window, EVENT_ORIGIN_UTC[i])
      - Returns on_time_count / total_count for that window

    A high absorb_score means the origin airport is functioning well (not receiving
    many late inbounds → lower upstream propagation risk). Follows Zhou et al. 2025
    (arXiv:2512.08197) delay-absorption methodology.
    """
    required = {"DEST", "ORIGIN", "EVENT_ORIGIN_UTC", "EVENT_DEST_UTC", ARR_DELAY_COL}
    if not required.issubset(df.columns):
        return df

    out = df
    n = len(out)

    arr_delay_raw = pd.to_numeric(out[ARR_DELAY_COL], errors="coerce").to_numpy()
    event_dest = pd.to_datetime(out["EVENT_DEST_UTC"], errors="coerce").astype("datetime64[ns]").to_numpy()
    event_orig = pd.to_datetime(out["EVENT_ORIGIN_UTC"], errors="coerce").astype("datetime64[ns]").to_numpy()

    arr_td = (
        pd.to_timedelta(pd.Series(arr_delay_raw), unit="m", errors="coerce")
        .fillna(pd.Timedelta(0))
        .to_numpy()
        .astype("timedelta64[ns]")
    )
    visible_at = event_dest + arr_td          # when flight j actually landed at DEST[j]
    is_on_time = (arr_delay_raw <= DELAY_THRESHOLD_MIN).astype(np.int32)
    has_valid = ~pd.isna(pd.to_datetime(event_dest, errors="coerce")) & ~np.isnan(arr_delay_raw)

    window_td = np.timedelta64(int(window_hours * 3_600_000_000_000), "ns")
    nat = np.datetime64("NaT", "ns")

    dest_vals = out["DEST"].astype("string").fillna("__nan__").to_numpy()
    origin_vals = out["ORIGIN"].astype("string").fillna("__nan__").to_numpy()

    absorb = np.full(n, np.nan, dtype=np.float64)

    for airport in np.unique(dest_vals):
        if airport == "__nan__":
            continue

        # Flights ARRIVING at this airport (supply side)
        arr_mask = (dest_vals == airport) & has_valid
        arr_pos = np.where(arr_mask)[0]
        if len(arr_pos) == 0:
            continue

        arr_vis = visible_at[arr_pos]
        arr_ot = is_on_time[arr_pos]
        order = np.argsort(arr_vis, kind="stable")
        arr_vis_s = arr_vis[order]
        arr_ot_s = arr_ot[order]
        cum = np.concatenate(([0], np.cumsum(arr_ot_s)))

        # Flights DEPARTING from this airport (query side)
        dep_pos = np.where(origin_vals == airport)[0]
        if len(dep_pos) == 0:
            continue

        dep_times = event_orig[dep_pos]
        valid_q = dep_times != nat

        safe_dep = np.where(valid_q, dep_times, np.datetime64("2000-01-01T00:00:00", "ns"))
        safe_start = safe_dep - window_td

        idx_right = np.searchsorted(arr_vis_s, safe_dep, side="left")
        idx_left = np.searchsorted(arr_vis_s, safe_start, side="right")

        cnt = idx_right - idx_left
        on_time_cnt = cum[idx_right] - cum[idx_left]

        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(cnt > 0, on_time_cnt / np.maximum(cnt, 1), np.nan)
        rate[~valid_q] = np.nan

        absorb[dep_pos] = rate

    out[out_col] = absorb
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    categorical_cols: tuple[str, ...] = CATEGORICAL_COLS,
    drop_cols: tuple[str, ...] = DROP_COLS,
) -> tuple[pd.DataFrame, list[str], dict[str, list]]:
    to_drop = [c for c in (*drop_cols, target_col) if c in df.columns]
    X = df.drop(columns=to_drop)  # drop() already returns a new DataFrame; .copy() was redundant

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

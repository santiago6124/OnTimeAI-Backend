"""Leakage-safe lineage features for flight delay prediction.

Three feature families, all computed from ARR_DELAY of *prior* observable events
(verified via actual arrival time < current scheduled departure):

1. Per-tail intra-day lineage:
   - prev_arr_delay_tail           — minutes of delay of the previous same-day flight
                                      of the same aircraft, observable at CRS_DEP_UTC
   - prev_turnaround_tail_min      — scheduled turnaround between arrivals/departures
   - tail_flights_today_prior      — count of same-tail flights earlier that day

2. Per-carrier daily lag:
   - carrier_delay_rate_yday       — share of carrier's flights with ARR_DELAY > 15
                                      on the previous calendar day

3. Per-origin daily lag:
   - origin_delay_rate_yday        — same share but grouped by ORIGIN airport
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ontimeai.config import ARR_DELAY_COL


DELAY_THRESHOLD_MIN = 15.0


def add_tail_lineage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute prev_arr_delay_tail / prev_turnaround / tail_flights_today_prior.

    Memory-efficient implementation: works entirely with numpy index arrays
    instead of sorting the full DataFrame. Peak overhead is ~1-2 GB (index
    arrays + per-group output buffers) vs ~9 GB for a full DataFrame sort.
    """
    required = {"TAIL_NUM", "EVENT_ORIGIN_UTC", "EVENT_DEST_UTC", ARR_DELAY_COL}
    if not required.issubset(df.columns):
        return df

    n = len(df)

    # Extract only the needed columns as numpy arrays — cheap, no DataFrame copy.
    tail_arr  = df["TAIL_NUM"].astype("string").fillna("__nan__").to_numpy()
    t_orig    = pd.to_datetime(df["EVENT_ORIGIN_UTC"], errors="coerce").astype("datetime64[ns]").to_numpy()
    t_dest    = pd.to_datetime(df["EVENT_DEST_UTC"],   errors="coerce").astype("datetime64[ns]").to_numpy()
    delay_arr = pd.to_numeric(df[ARR_DELAY_COL], errors="coerce").to_numpy(dtype="float32")

    # Sort by (tail_code, departure time) — argsort only, no DataFrame copy.
    tail_codes  = pd.Categorical(tail_arr).codes.astype(np.int32)
    sort_order  = np.lexsort([t_orig, tail_codes])   # primary: tail, secondary: time
    inv_order   = np.empty(n, dtype=np.int32)
    inv_order[sort_order] = np.arange(n, dtype=np.int32)

    # Sorted arrays (one copy per column, ~220 MB each — much less than full DF sort)
    tc_s   = tail_codes[sort_order]
    orig_s = t_orig[sort_order]
    dest_s = t_dest[sort_order]
    dly_s  = delay_arr[sort_order]

    # Group boundaries by tail code
    unique_tc    = np.unique(tc_s)
    boundaries   = np.searchsorted(tc_s, np.append(unique_tc, unique_tc[-1] + 1))

    # Output buffers in sorted order
    prev_delay_s     = np.full(n, np.nan, dtype="float32")
    prev_dest_s      = np.full(n, np.datetime64("NaT", "ns"))
    prev_orig_s      = np.full(n, np.datetime64("NaT", "ns"))
    today_count_s    = np.zeros(n, dtype="int16")

    NaT = np.datetime64("NaT", "ns")

    for i, g in enumerate(unique_tc):
        b, e = int(boundaries[i]), int(boundaries[i + 1])
        if e - b < 2:
            continue

        d = dly_s[b:e]
        o = orig_s[b:e]
        de = dest_s[b:e]
        m = e - b

        # For each position i, find the last settled (non-NaN delay) position < i.
        # settled_pos[j] = the position within the group of the j-th settled flight.
        settled_pos = np.where(np.isfinite(d))[0]  # positions where delay is known
        if len(settled_pos) > 0:
            # For each position i (1..m-1), how many settled flights are before i?
            # = searchsorted(settled_pos, i) gives count of settled_pos < i
            query_idx = np.arange(1, m)
            cnt_before = np.searchsorted(settled_pos, query_idx, side="left")
            has_prior  = cnt_before > 0
            prior_pos  = settled_pos[np.maximum(cnt_before - 1, 0)]

            out_slice = slice(b + 1, e)
            prev_delay_s[out_slice] = np.where(has_prior, d[prior_pos], np.nan)
            prev_dest_s[out_slice]  = np.where(has_prior, de[prior_pos], NaT)
            prev_orig_s[out_slice]  = np.where(has_prior, o[prior_pos],  NaT)

        # tail_flights_today_prior: cumcount per UTC date within this group.
        # datetime64[D] is backed by int64 — must view as int64, not int32.
        utc_days = o.astype("datetime64[D]").view(np.int64)
        for day_val in np.unique(utc_days):
            mask = utc_days == day_val
            positions_in_day = np.where(mask)[0]
            today_count_s[b + positions_in_day] = np.arange(len(positions_in_day), dtype="int16")

    # Validity checks (done in sorted-array space)
    delay_td    = (prev_delay_s * 60 * 1e9).astype("int64")  # minutes → ns
    prev_arr_ns = prev_dest_s.view("int64") + delay_td
    nat_val     = np.datetime64("NaT", "ns").view("int64")
    not_nat     = prev_dest_s.view("int64") != nat_val

    observable  = not_nat & (prev_arr_ns <= orig_s.view("int64"))
    delta_ns    = orig_s.view("int64") - prev_orig_s.view("int64")
    delta_h     = delta_ns / 3_600_000_000_000.0
    within_24h  = (delta_h > 0) & (delta_h < 24)
    valid       = observable & within_24h

    turnaround_s = (orig_s.view("int64") - prev_dest_s.view("int64")) / 60_000_000_000.0

    prev_delay_out    = np.where(valid, prev_delay_s,  np.nan)
    prev_turn_out     = np.where(valid, turnaround_s,  np.nan)

    # Map back to original DataFrame order and write directly (no copy)
    df["prev_arr_delay_tail"]      = prev_delay_out[inv_order].astype("float32")
    df["prev_turnaround_tail_min"] = prev_turn_out[inv_order].astype("float32")
    df["tail_flights_today_prior"] = today_count_s[inv_order].astype(np.int16)
    return df


def _daily_rate_lag(
    df: pd.DataFrame, key_col: str, out_col: str, threshold: float = DELAY_THRESHOLD_MIN
) -> pd.DataFrame:
    if key_col not in df.columns or "FL_DATE" not in df.columns or ARR_DELAY_COL not in df.columns:
        return df
    out = df  # in-place: only adds one new column, never modifies existing ones

    date_day = pd.to_datetime(out["FL_DATE"], errors="coerce").dt.floor("D")
    key_str = out[key_col].astype("string")
    arr = pd.to_numeric(out[ARR_DELAY_COL], errors="coerce")

    agg_in = pd.DataFrame({
        "_key": key_str,
        "_date": date_day,
        "_delayed": (arr > threshold).astype(float),
        "_valid": arr.notna().astype(float),
    })
    daily = (
        agg_in.dropna(subset=["_date", "_key"])
        .groupby(["_key", "_date"], sort=False)[["_delayed", "_valid"]]
        .sum()
        .reset_index()
    )
    daily["rate"] = daily["_delayed"] / daily["_valid"].where(daily["_valid"] > 0, np.nan)
    daily["_lookup_date"] = daily["_date"] + pd.Timedelta(days=1)
    lookup = daily[["_key", "_lookup_date", "rate"]].rename(
        columns={"_key": key_col, "_lookup_date": "_date_ts", "rate": out_col}
    )

    merge_frame = pd.DataFrame({key_col: key_str, "_date_ts": date_day, "_pos": np.arange(len(out))})
    merged = merge_frame.merge(lookup, on=[key_col, "_date_ts"], how="left")
    merged = merged.sort_values("_pos").reset_index(drop=True)
    out[out_col] = merged[out_col].to_numpy()
    return out


def add_carrier_day_lag(df: pd.DataFrame) -> pd.DataFrame:
    return _daily_rate_lag(df, "OP_CARRIER", "carrier_delay_rate_yday")


def add_origin_day_lag(df: pd.DataFrame) -> pd.DataFrame:
    return _daily_rate_lag(df, "ORIGIN", "origin_delay_rate_yday")


def add_group_rolling_rate(
    df: pd.DataFrame,
    group_col: str,
    window_hours: float,
    out_col: str,
    threshold: float = DELAY_THRESHOLD_MIN,
) -> pd.DataFrame:
    """Share of same-group flights whose *visibility* time (actual arrival) falls in
    (current CRS_DEP - window, current CRS_DEP). Leakage-safe: only observes flights
    that already landed before the current flight departs.
    """
    required = {group_col, "EVENT_ORIGIN_UTC", "EVENT_DEST_UTC", ARR_DELAY_COL}
    if not required.issubset(df.columns):
        return df

    out = df  # in-place: only adds one new column, never modifies existing ones
    n = len(out)

    arr_delay = pd.to_numeric(out[ARR_DELAY_COL], errors="coerce").to_numpy()
    event_dest = pd.to_datetime(out["EVENT_DEST_UTC"], errors="coerce").astype("datetime64[ns]").to_numpy()
    event_orig = pd.to_datetime(out["EVENT_ORIGIN_UTC"], errors="coerce").astype("datetime64[ns]").to_numpy()

    arr_delay_td = pd.to_timedelta(pd.Series(arr_delay), unit="m").to_numpy().astype("timedelta64[ns]")
    visible_at = event_dest + arr_delay_td
    is_delayed = (arr_delay > threshold).astype(np.int32)

    window_td = np.timedelta64(int(window_hours * 3_600_000_000_000), "ns")
    group_vals = out[group_col].astype("string").fillna("__nan__").to_numpy()

    rates = np.full(n, np.nan, dtype=np.float64)

    unique_groups, inverse = np.unique(group_vals, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inv = inverse[order]
    boundaries = np.searchsorted(sorted_inv, np.arange(len(unique_groups) + 1))

    for g in range(len(unique_groups)):
        grp_pos = order[boundaries[g] : boundaries[g + 1]]
        if len(grp_pos) == 0:
            continue

        grp_obs = visible_at[grp_pos]
        grp_query = event_orig[grp_pos]
        grp_delay = is_delayed[grp_pos]

        valid_obs = ~pd.isna(grp_obs)
        if not valid_obs.any():
            continue

        obs_t = grp_obs[valid_obs]
        obs_d = grp_delay[valid_obs].astype(np.int32)

        obs_order = np.argsort(obs_t, kind="stable")
        obs_t_sorted = obs_t[obs_order]
        obs_d_sorted = obs_d[obs_order]
        cum = np.concatenate(([0], np.cumsum(obs_d_sorted)))

        valid_q = ~pd.isna(grp_query)
        safe_q = np.where(valid_q, grp_query, np.datetime64("2000-01-01", "ns"))
        window_start = safe_q - window_td

        idx_left = np.searchsorted(obs_t_sorted, window_start, side="right")
        idx_right = np.searchsorted(obs_t_sorted, safe_q, side="left")

        cnt = idx_right - idx_left
        dly = cum[idx_right] - cum[idx_left]

        with np.errstate(divide="ignore", invalid="ignore"):
            group_rate = np.where(cnt > 0, dly / np.maximum(cnt, 1), np.nan)
        group_rate[~valid_q] = np.nan

        rates[grp_pos] = group_rate

    out[out_col] = rates
    return out


CARRIER_ROLLING_WINDOWS: tuple[tuple[float, str], ...] = (
    (24.0, "carrier_delay_rate_24h"),
    (168.0, "carrier_delay_rate_7d"),
)

ORIGIN_ROLLING_WINDOWS: tuple[tuple[float, str], ...] = (
    (1.0, "origin_delay_rate_1h"),
    (6.0, "origin_delay_rate_6h"),
    (24.0, "origin_delay_rate_24h"),
)

# Destination-side congestion: delay rate of flights arriving at DEST airport.
# Captures whether the destination is currently absorbing many delayed inbounds.
DEST_ROLLING_WINDOWS: tuple[tuple[float, str], ...] = (
    (1.0, "dest_delay_rate_1h"),
    (6.0, "dest_delay_rate_6h"),
    (24.0, "dest_delay_rate_24h"),
)


def add_carrier_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    for hours, col in CARRIER_ROLLING_WINDOWS:
        out = add_group_rolling_rate(out, "OP_CARRIER", hours, col)
    return out


def add_origin_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    for hours, col in ORIGIN_ROLLING_WINDOWS:
        out = add_group_rolling_rate(out, "ORIGIN", hours, col)
    return out


def add_dest_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Delay rate of flights arriving AT the destination airport in rolling windows.

    Groups by DEST and uses each flight's actual arrival time as the visibility
    timestamp (same leakage-safe logic as origin rolling). Captures destination-side
    congestion that often propagates to the current flight's arrival delay.
    """
    out = df
    for hours, col in DEST_ROLLING_WINDOWS:
        out = add_group_rolling_rate(out, "DEST", hours, col)
    return out


LINEAGE_FEATURE_COLS: tuple[str, ...] = (
    "prev_arr_delay_tail",
    "prev_turnaround_tail_min",
    "tail_flights_today_prior",
    "carrier_delay_rate_yday",
    "origin_delay_rate_yday",
    *(name for _, name in CARRIER_ROLLING_WINDOWS),
    *(name for _, name in ORIGIN_ROLLING_WINDOWS),
    *(name for _, name in DEST_ROLLING_WINDOWS),
)

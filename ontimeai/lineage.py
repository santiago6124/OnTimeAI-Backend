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

    Groups by TAIL_NUM only (across days) and uses a 24h UTC window to identify
    the prior leg. The previous implementation grouped by (TAIL_NUM, FL_DATE)
    where FL_DATE is local-tz, which silently dropped overnight rotations
    (target's local date != prior's local date when crossing midnight at any
    timezone). With ATL EDT vs e.g. LAX PDT, even within-tz rotations could
    miss when scheduled_off straddled midnight.

    The 24h window keeps the original anti-leakage intent (don't pull stale
    7-day-old "priors") while correctly capturing every truly-recent rotation.
    """
    required = {"TAIL_NUM", "EVENT_ORIGIN_UTC", "EVENT_DEST_UTC", ARR_DELAY_COL}
    if not required.issubset(df.columns):
        return df

    out = df.copy()
    out["_orig_pos"] = np.arange(len(out))

    tail_str = out["TAIL_NUM"].astype("string").fillna("__nan__")
    cur_event_origin = pd.to_datetime(out["EVENT_ORIGIN_UTC"], errors="coerce")
    event_dest = pd.to_datetime(out["EVENT_DEST_UTC"], errors="coerce")
    arr_delay = pd.to_numeric(out[ARR_DELAY_COL], errors="coerce")

    # Use UTC date for the "today" group, not local FL_DATE — this is part of the bug-fix.
    utc_date = cur_event_origin.dt.strftime("%Y-%m-%d").fillna("")
    out["_tail_only_key"] = tail_str
    out["_tail_utcdate_key"] = (tail_str + "|" + utc_date).astype("string")

    out = out.sort_values(["_tail_only_key", "EVENT_ORIGIN_UTC"], kind="stable").reset_index(drop=True)

    # Re-extract after sort
    cur_event_origin = pd.to_datetime(out["EVENT_ORIGIN_UTC"], errors="coerce")
    event_dest = pd.to_datetime(out["EVENT_DEST_UTC"], errors="coerce")
    arr_delay = pd.to_numeric(out[ARR_DELAY_COL], errors="coerce")
    has_arr = arr_delay.notna()

    # ── KEY FIX: ffill then shift(1) within each tail group ───────────────────
    # Inference frames mix history rows (ARR_DELAY known) with target rows (ARR_DELAY=NaN).
    # A naive shift(1) would propagate the NaN of an immediately-prior target instead of
    # finding the most recent SETTLED prior. We fix this by ffill-ing the valid-only
    # triplet (arr_delay, event_dest, event_origin) within each tail before shifting.
    out["_v_delay"] = arr_delay.where(has_arr)
    out["_v_dest"] = event_dest.where(has_arr)
    out["_v_orig"] = cur_event_origin.where(has_arr)

    grp_tail = out.groupby("_tail_only_key", sort=False)
    # Use transform with ffill+shift inside the lambda so both ops stay within each
    # group (a bare .ffill().shift(1) on a SeriesGroupBy returns a flat Series whose
    # subsequent .shift(1) crosses group boundaries, contaminating different tails).
    def _ffill_shift(s: pd.Series) -> pd.Series:
        return s.ffill().shift(1)

    prev_arr_delay = grp_tail["_v_delay"].transform(_ffill_shift)
    prev_event_dest = pd.to_datetime(
        grp_tail["_v_dest"].transform(_ffill_shift), errors="coerce",
    )
    prev_event_origin = pd.to_datetime(
        grp_tail["_v_orig"].transform(_ffill_shift), errors="coerce",
    )

    prev_actual_arr = prev_event_dest + pd.to_timedelta(prev_arr_delay, unit="m")
    observable = (prev_actual_arr <= cur_event_origin).fillna(False)
    # Anti-stale-prior guard: prev leg must be within 24h.
    delta_hours = (cur_event_origin - prev_event_origin).dt.total_seconds() / 3600.0
    within_24h = (delta_hours > 0) & (delta_hours < 24)
    valid = observable & within_24h.fillna(False)

    scheduled_turnaround_min = (cur_event_origin - prev_event_dest).dt.total_seconds() / 60.0

    out["prev_arr_delay_tail"] = np.where(valid, prev_arr_delay, np.nan)
    out["prev_turnaround_tail_min"] = np.where(valid, scheduled_turnaround_min, np.nan)

    # ── tail_flights_today_prior: count per (TAIL, UTC-date) ──────────────────
    out["tail_flights_today_prior"] = (
        out.groupby("_tail_utcdate_key", sort=False).cumcount().astype(np.int16)
    )

    out = out.sort_values("_orig_pos").reset_index(drop=True)
    return out.drop(columns=[
        "_orig_pos", "_tail_only_key", "_tail_utcdate_key",
        "_v_delay", "_v_dest", "_v_orig",
    ])


def _daily_rate_lag(
    df: pd.DataFrame, key_col: str, out_col: str, threshold: float = DELAY_THRESHOLD_MIN
) -> pd.DataFrame:
    if key_col not in df.columns or "FL_DATE" not in df.columns or ARR_DELAY_COL not in df.columns:
        return df
    out = df.copy()

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

    out = df.copy()
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

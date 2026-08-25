"""Cold-deck imputation for lineage / rolling features.

Built once from the historical BTS master. Used at inference when the live
history buffer + chain-walk both fail to provide a value (e.g. first flight
of the day for a tail with no inbound, carrier never seen in last 24h).

Hierarchy of fallbacks, from most to least specific:

  prev_arr_delay_tail (minutes):
    1. mean ARR_DELAY by (carrier, origin, dest, hour-of-day)
    2. mean ARR_DELAY by (carrier, origin, hour-of-day)
    3. mean ARR_DELAY by (carrier, hour-of-day)
    4. global mean ARR_DELAY

  carrier_delay_rate_{yday,24h,7d}:
    1. late-rate (>15min) by (carrier, day-of-week)
    2. late-rate by (carrier)
    3. global late-rate

  origin_delay_rate_{yday,1h,6h,24h}:
    1. late-rate by (origin, hour-of-day)
    2. late-rate by (origin)
    3. global late-rate

  tail_flights_today_prior  → 0 (assume target is first leg of the day)
  prev_turnaround_tail_min  → 60 minutes (typical)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ontimeai.config import ARR_DELAY_COL


DELAY_THRESHOLD_MIN = 15.0
MIN_GROUP_COUNT = 5  # below this, ignore the bucket as too noisy


def _agg(df: pd.DataFrame, group_cols: list[str], value_col: str,
         min_count: int = MIN_GROUP_COUNT) -> pd.Series:
    g = df.groupby(group_cols, observed=True)[value_col].agg(["mean", "count"])
    keep = g[g["count"] >= min_count]
    result = keep["mean"].astype(float)
    # Convert CategoricalIndex levels to plain object so MultiIndex.reindex works.
    if isinstance(result.index, pd.MultiIndex):
        result.index = result.index.set_levels(
            [lvl.astype(object) for lvl in result.index.levels]
        )
    elif hasattr(result.index, "categories"):
        result.index = result.index.astype(object)
    return result


def build_lookups(master_df: pd.DataFrame) -> dict[str, Any]:
    """Compute cold-deck lookups from the historical BTS master.

    Expects columns: ARR_DELAY, OP_CARRIER, ORIGIN, DEST, CRS_DEP_TIME,
    DAY_OF_WEEK, CANCELLED. Filters out cancelled flights and rows with
    missing ARR_DELAY before aggregating.
    """
    df = master_df.copy()
    df = df.loc[df[ARR_DELAY_COL].notna()]
    if "CANCELLED" in df.columns:
        df = df.loc[pd.to_numeric(df["CANCELLED"], errors="coerce").fillna(0).eq(0)]

    df["_hour"] = (
        pd.to_numeric(df.get("CRS_DEP_TIME"), errors="coerce").fillna(0) // 100
    ).astype(int)
    df["_late"] = (df[ARR_DELAY_COL] > DELAY_THRESHOLD_MIN).astype(float)

    return {
        "carrier_route_hour_mean_delay": _agg(
            df, ["OP_CARRIER", "ORIGIN", "DEST", "_hour"], ARR_DELAY_COL
        ),
        "carrier_origin_hour_mean_delay": _agg(
            df, ["OP_CARRIER", "ORIGIN", "_hour"], ARR_DELAY_COL
        ),
        "carrier_hour_mean_delay": _agg(df, ["OP_CARRIER", "_hour"], ARR_DELAY_COL),
        "carrier_dow_late_rate": _agg(df, ["OP_CARRIER", "DAY_OF_WEEK"], "_late"),
        "origin_hour_late_rate": _agg(df, ["ORIGIN", "_hour"], "_late"),
        "dest_hour_late_rate": _agg(df, ["DEST", "_hour"], "_late"),
        "carrier_late_rate": _agg(df, ["OP_CARRIER"], "_late"),
        "origin_late_rate": _agg(df, ["ORIGIN"], "_late"),
        "dest_late_rate": _agg(df, ["DEST"], "_late"),
        "global_mean_delay": float(df[ARR_DELAY_COL].mean()),
        "global_late_rate": float(df["_late"].mean()),
        "global_on_time_rate": float(1.0 - df["_late"].mean()),
        "_meta": {
            "n_rows": int(len(df)),
            "min_group_count": MIN_GROUP_COUNT,
        },
    }


def _series_or_str(out: pd.DataFrame, col: str) -> pd.Series:
    if col in out.columns:
        s = out[col].astype("string").fillna("")
    else:
        s = pd.Series([""] * len(out), index=out.index, dtype="string")
    return s


def _series_or_int(out: pd.DataFrame, col: str, default: int = 0) -> pd.Series:
    if col in out.columns:
        s = pd.to_numeric(out[col], errors="coerce").fillna(default).astype(int)
    else:
        s = pd.Series([default] * len(out), index=out.index, dtype=int)
    return s


def _decat(lookup: pd.Series) -> pd.Series:
    """Convert any CategoricalIndex levels to plain object so reindex works."""
    if isinstance(lookup.index, pd.MultiIndex):
        new_levels = [
            lvl.astype(object) if hasattr(lvl, "categories") else lvl
            for lvl in lookup.index.levels
        ]
        lookup = lookup.copy()
        lookup.index = lookup.index.set_levels(new_levels)
    elif hasattr(lookup.index, "categories"):
        lookup = lookup.copy()
        lookup.index = lookup.index.astype(object)
    return lookup


def _hierarchical_lookup(
    layers: list[tuple[list[pd.Series], pd.Series | None]],
    target_index: pd.Index,
    global_value: float,
) -> pd.Series:
    """Compute imputed values by walking layers from most-specific (last) to
    least-specific (first). Each layer is (key_arrays, lookup_series).

    Returns a Series indexed like target_index with no NaNs (falls back to
    global_value where every layer misses).
    """
    imputed = pd.Series(global_value, index=target_index, dtype=float)
    for key_arrays, lookup in layers:
        if lookup is None or lookup.empty:
            continue
        lookup = _decat(lookup)
        idx = pd.MultiIndex.from_arrays(key_arrays)
        layer_vals = lookup.reindex(idx)
        layer_vals.index = target_index
        mask = layer_vals.notna()
        imputed = imputed.where(~mask, layer_vals)
    return imputed


def apply_lineage_fallback(df: pd.DataFrame, lookups: dict[str, Any]) -> pd.DataFrame:
    """Fill NaN values in lineage / rolling features using cold-deck lookups.

    Non-destructive: leaves non-NaN values untouched. Tolerates missing
    feature columns and partial lookup tables.
    """
    out = df.copy()
    n = len(out)
    if n == 0:
        return out

    if "CRS_DEP_MIN" in out.columns:
        hour = (
            pd.to_numeric(out["CRS_DEP_MIN"], errors="coerce").fillna(0) // 60
        ).astype(int)
    elif "CRS_DEP_TIME" in out.columns:
        hour = (
            pd.to_numeric(out["CRS_DEP_TIME"], errors="coerce").fillna(0) // 100
        ).astype(int)
    else:
        hour = pd.Series([0] * n, index=out.index, dtype=int)

    carrier = _series_or_str(out, "OP_CARRIER")
    origin = _series_or_str(out, "ORIGIN")
    dest = _series_or_str(out, "DEST")
    dow = _series_or_int(out, "DAY_OF_WEEK")

    g_mean = float(lookups.get("global_mean_delay", 0.0))
    g_late = float(lookups.get("global_late_rate", 0.2))

    # ---- prev_arr_delay_tail: hierarchical mean ARR_DELAY ----
    if "prev_arr_delay_tail" in out.columns:
        layers = [
            ([carrier, hour], lookups.get("carrier_hour_mean_delay")),
            ([carrier, origin, hour], lookups.get("carrier_origin_hour_mean_delay")),
            ([carrier, origin, dest, hour], lookups.get("carrier_route_hour_mean_delay")),
        ]
        imputed = _hierarchical_lookup(layers, out.index, g_mean)
        out["prev_arr_delay_tail"] = out["prev_arr_delay_tail"].fillna(imputed)

    # ---- defaults for non-rate lineage features ----
    if "tail_flights_today_prior" in out.columns:
        out["tail_flights_today_prior"] = out["tail_flights_today_prior"].fillna(0)

    # ---- prev_turnaround_tail_min: cascade (Layer 3) ----
    # When the per-tail chain-walk failed (stale cache, unseen tail, NULL tail),
    # fall back through carrier+route+hour -> carrier+route -> carrier+hour ->
    # carrier -> hour -> global -> 60.0 default. The lookups come from the live
    # `flights` table via build_live_turnaround_lookups (loaded at runtime), so
    # the model gets a more informed prior than the constant 60.0.
    if "prev_turnaround_tail_min" in out.columns:
        layers = [
            ([carrier], lookups.get("turnaround_carrier_mean")),
            ([hour], lookups.get("turnaround_hour_mean")),
            ([carrier, hour], lookups.get("turnaround_carrier_hour_mean")),
            ([carrier, origin, dest], lookups.get("turnaround_carrier_route_mean")),
            ([carrier, origin, dest, hour],
             lookups.get("turnaround_carrier_route_hour_mean")),
        ]
        g_turnaround = float(lookups.get("global_turnaround_mean", 60.0))
        imputed = _hierarchical_lookup(layers, out.index, g_turnaround)
        out["prev_turnaround_tail_min"] = out["prev_turnaround_tail_min"].fillna(imputed)

    # ---- carrier rate features ----
    carrier_rate_cols = [
        "carrier_delay_rate_yday",
        "carrier_delay_rate_24h",
        "carrier_delay_rate_7d",
    ]
    if any(c in out.columns for c in carrier_rate_cols):
        layers = [
            ([carrier], lookups.get("carrier_late_rate")),
            ([carrier, dow], lookups.get("carrier_dow_late_rate")),
        ]
        imputed = _hierarchical_lookup(layers, out.index, g_late)
        for c in carrier_rate_cols:
            if c in out.columns:
                out[c] = out[c].fillna(imputed)

    # ---- origin rate features ----
    origin_rate_cols = [
        "origin_delay_rate_yday",
        "origin_delay_rate_1h",
        "origin_delay_rate_6h",
        "origin_delay_rate_24h",
    ]
    if any(c in out.columns for c in origin_rate_cols):
        layers = [
            ([origin], lookups.get("origin_late_rate")),
            ([origin, hour], lookups.get("origin_hour_late_rate")),
        ]
        imputed = _hierarchical_lookup(layers, out.index, g_late)
        for c in origin_rate_cols:
            if c in out.columns:
                out[c] = out[c].fillna(imputed)

    # ---- dest rate features ----
    dest_rate_cols = [
        "dest_delay_rate_1h",
        "dest_delay_rate_6h",
        "dest_delay_rate_24h",
    ]
    if any(c in out.columns for c in dest_rate_cols):
        layers = [
            ([dest], lookups.get("dest_late_rate")),
            ([dest, hour], lookups.get("dest_hour_late_rate")),
        ]
        imputed_dest = _hierarchical_lookup(layers, out.index, g_late)
        for c in dest_rate_cols:
            if c in out.columns:
                out[c] = out[c].fillna(imputed_dest)

    # ---- absorb_score_origin: 1 - origin inbound late rate ----
    if "absorb_score_origin" in out.columns:
        g_on_time = float(lookups.get("global_on_time_rate", 1.0 - g_late))
        layers = [
            ([origin], lookups.get("origin_late_rate")),
            ([origin, hour], lookups.get("origin_hour_late_rate")),
        ]
        imputed_late = _hierarchical_lookup(layers, out.index, g_late)
        imputed_absorb = 1.0 - imputed_late
        out["absorb_score_origin"] = out["absorb_score_origin"].fillna(imputed_absorb)

    # ADS-B prev-leg features: always 100% NaN in prod (no live ADS-B tracking).
    # Fill with neutral defaults so the model receives a value in its learned range
    # rather than going to the NaN branch (trained on very few/no examples).
    for _c in ("PREV_BLOCK_DELTA_MIN", "PREV_HOLDING_MIN", "PREV_ROUTE_DEVIATION_PCT"):
        if _c in out.columns:
            out[_c] = out[_c].fillna(0.0)

    if "PREV_ADSB_AVAILABLE" in out.columns:
        out["PREV_ADSB_AVAILABLE"] = out["PREV_ADSB_AVAILABLE"].fillna(0)

    # PREV block times: use CRS_ELAPSED_TIME as a same-aircraft-type proxy
    if "CRS_ELAPSED_TIME" in out.columns:
        for _c in ("PREV_ACTUAL_BLOCK_MIN", "PREV_SCHED_BLOCK_MIN"):
            if _c in out.columns:
                out[_c] = out[_c].fillna(out["CRS_ELAPSED_TIME"])

    return out


def build_live_turnaround_lookups(conn, days: int = 14,
                                  min_count: int = MIN_GROUP_COUNT) -> dict[str, Any]:
    """Compute turnaround-minutes lookups from the live `flights` table.

    Joins each flight to its inbound leg via `inbound_fa_flight_id` and
    computes (scheduled_out - scheduled_in_inbound) as the turnaround. Returns
    a dict shaped like the keys consumed by apply_lineage_fallback for
    `prev_turnaround_tail_min`.

    Builds five cascading aggregations from most-specific to least-specific:
      - (op_carrier, origin, dest, hour)
      - (op_carrier, origin, dest)
      - (op_carrier, hour)
      - (op_carrier,)
      - (hour,)
    plus a global mean.

    Skipped buckets (< min_count) fall through to the next layer at lookup time.
    """
    sql = f"""
        SELECT f.op_carrier, f.origin, f.dest,
               CAST(substr(f.scheduled_out_utc, 12, 2) AS INTEGER) AS hour,
               (julianday(f.scheduled_out_utc) - julianday(p.scheduled_in_utc))
                 * 24.0 * 60.0 AS turn_min
        FROM flights f
        JOIN flights p ON p.fa_flight_id = f.inbound_fa_flight_id
        WHERE f.scheduled_out_utc IS NOT NULL
          AND p.scheduled_in_utc IS NOT NULL
          AND f.op_carrier IS NOT NULL
          AND f.origin IS NOT NULL
          AND f.dest IS NOT NULL
          AND date(f.scheduled_out_utc) >= date('now', '-{int(days)} days')
    """
    df = pd.read_sql_query(sql, conn)
    # Clip wildly implausible values (chain-walk timestamp glitches)
    df = df[(df["turn_min"] >= 10.0) & (df["turn_min"] <= 720.0)]
    if df.empty:
        return {"global_turnaround_mean": 60.0,
                "_meta": {"n_rows": 0, "source": "live_flights"}}

    out: dict[str, Any] = {}
    g = df.groupby(["op_carrier", "origin", "dest", "hour"], observed=True)["turn_min"]
    grp = g.agg(["mean", "count"])
    out["turnaround_carrier_route_hour_mean"] = (
        grp[grp["count"] >= min_count]["mean"].astype(float)
    )
    g = df.groupby(["op_carrier", "origin", "dest"], observed=True)["turn_min"]
    grp = g.agg(["mean", "count"])
    out["turnaround_carrier_route_mean"] = (
        grp[grp["count"] >= min_count]["mean"].astype(float)
    )
    g = df.groupby(["op_carrier", "hour"], observed=True)["turn_min"]
    grp = g.agg(["mean", "count"])
    out["turnaround_carrier_hour_mean"] = (
        grp[grp["count"] >= min_count]["mean"].astype(float)
    )
    g = df.groupby(["op_carrier"], observed=True)["turn_min"]
    grp = g.agg(["mean", "count"])
    out["turnaround_carrier_mean"] = (
        grp[grp["count"] >= min_count]["mean"].astype(float)
    )
    g = df.groupby(["hour"], observed=True)["turn_min"]
    grp = g.agg(["mean", "count"])
    out["turnaround_hour_mean"] = (
        grp[grp["count"] >= min_count]["mean"].astype(float)
    )
    out["global_turnaround_mean"] = float(df["turn_min"].mean())
    out["_meta"] = {"n_rows": int(len(df)), "source": "live_flights",
                    "days": int(days)}
    return out


def save_lookups(lookups: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lookups, path)
    return path


def load_lookups(path: Path) -> dict[str, Any]:
    return joblib.load(Path(path))

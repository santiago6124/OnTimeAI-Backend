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
    if "prev_turnaround_tail_min" in out.columns:
        out["prev_turnaround_tail_min"] = out["prev_turnaround_tail_min"].fillna(60.0)

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

    return out


def save_lookups(lookups: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lookups, path)
    return path


def load_lookups(path: Path) -> dict[str, Any]:
    return joblib.load(Path(path))

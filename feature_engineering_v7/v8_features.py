"""v8 feature engineering additions.

Three new feature families:

1. TAIL_DELAY_DECAY
   prev_arr_delay_tail decayed by turnaround slack.
   If prev_delay=60 and turnaround=120 min → factor exp(-2) ≈ 0.14.
   If prev_delay=60 and turnaround=5 min  → factor exp(-0.08) ≈ 0.92.

2. ORIGIN_PAGERANK / DEST_PAGERANK
   PageRank centrality of each airport in the BTS directed route graph.
   High-centrality hubs (ATL, ORD, DFW) propagate delays more and accumulate
   more upstream pressure — a static but structurally meaningful signal.

3. NAS_RATE_2H
   Rolling 2-hour fraction of flights at origin that had NAS_DELAY > 0.
   NAS delays (ATC, GDP, weather programs) are systemic: one active EDCT
   affects an entire bank of departures. This complements origin_delay_rate_1h
   (which captures overall delay pressure) with a cause-specific signal.

Usage (offline, in build_v8_dataset.py):
    from feature_engineering_v7.v8_features import (
        add_tail_delay_decay, build_pagerank_lookup, add_pagerank_features,
        add_nas_rolling_rate,
    )

Usage (live, in predict.py / live.py):
    from feature_engineering_v7.v8_features import (
        add_tail_delay_decay, add_pagerank_features,
    )
    # NAS_RATE_2H at inference defaults to 0 — GDP_FLAG covers the live signal.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

_PAGERANK_CACHE: dict | None = None
_PAGERANK_DEFAULT = 0.0


# ---------------------------------------------------------------------------
# 1. TAIL_DELAY_DECAY
# ---------------------------------------------------------------------------

def add_tail_delay_decay(df: pd.DataFrame) -> pd.DataFrame:
    """Add TAIL_DELAY_DECAY = prev_arr_delay_tail * exp(-turnaround_min / 60).

    Must be called AFTER add_tail_lineage_features() has populated
    prev_arr_delay_tail and prev_turnaround_tail_min.
    """
    if "prev_arr_delay_tail" not in df.columns:
        df["TAIL_DELAY_DECAY"] = np.nan
        return df

    delay = df["prev_arr_delay_tail"].fillna(0.0).to_numpy(dtype=np.float64)
    slack = df.get("prev_turnaround_tail_min", pd.Series(60.0, index=df.index))
    slack = slack.clip(lower=0).fillna(60.0).to_numpy(dtype=np.float64)

    decay = delay * np.exp(-slack / 60.0)
    # Where delay was NaN, result should also be NaN (no info)
    mask_nan = df["prev_arr_delay_tail"].isna().to_numpy()
    decay[mask_nan] = np.nan

    df = df.copy()
    df["TAIL_DELAY_DECAY"] = decay
    return df


# ---------------------------------------------------------------------------
# 2. AIRPORT_PAGERANK
# ---------------------------------------------------------------------------

def build_pagerank_lookup(
    df: pd.DataFrame,
    out_path: Path | str | None = None,
    damping: float = 0.85,
    max_iter: int = 100,
) -> dict[str, float]:
    """Compute PageRank on the directed ORIGIN→DEST route graph.

    Weight of each edge = number of flights on that route (raw frequency).
    Returns {airport_iata: pagerank_score}.

    If out_path is given, saves the result as JSON for reuse.
    """
    import networkx as nx

    counts = (
        df.groupby(["ORIGIN", "DEST"])
        .size()
        .reset_index(name="weight")
    )
    G = nx.DiGraph()
    for _, row in counts.iterrows():
        G.add_edge(row["ORIGIN"], row["DEST"], weight=float(row["weight"]))

    pr = nx.pagerank(G, alpha=damping, weight="weight", max_iter=max_iter)

    if out_path is not None:
        Path(out_path).write_text(json.dumps(pr, indent=2))

    return pr


def load_pagerank_lookup(path: Path | str) -> dict[str, float]:
    return json.loads(Path(path).read_text())


def add_pagerank_features(
    df: pd.DataFrame,
    pr_lookup: dict[str, float] | None = None,
    lookup_path: Path | str | None = None,
) -> pd.DataFrame:
    """Join ORIGIN_PAGERANK and DEST_PAGERANK from a precomputed lookup.

    Pass either pr_lookup (dict) or lookup_path (JSON file). If neither is
    available, fills with 0.
    """
    global _PAGERANK_CACHE

    if pr_lookup is None:
        if lookup_path is not None and Path(lookup_path).exists():
            if _PAGERANK_CACHE is None:
                _PAGERANK_CACHE = load_pagerank_lookup(lookup_path)
            pr_lookup = _PAGERANK_CACHE
        else:
            df["ORIGIN_PAGERANK"] = _PAGERANK_DEFAULT
            df["DEST_PAGERANK"] = _PAGERANK_DEFAULT
            return df

    df = df.copy()
    df["ORIGIN_PAGERANK"] = df["ORIGIN"].map(pr_lookup).fillna(_PAGERANK_DEFAULT).astype("float64")
    df["DEST_PAGERANK"]   = df["DEST"].map(pr_lookup).fillna(_PAGERANK_DEFAULT).astype("float64")
    return df


# ---------------------------------------------------------------------------
# 3. NAS_RATE_2H  (offline only — uses BTS NAS_DELAY)
# ---------------------------------------------------------------------------

def add_nas_rolling_rate(
    df: pd.DataFrame,
    window_hours: float = 2.0,
    out_col: str = "nas_rate_2h",
) -> pd.DataFrame:
    """Rolling fraction of flights at ORIGIN with NAS_DELAY > 0.

    Uses the same leakage-safe binary-search approach as lineage.add_group_rolling_rate:
    a flight's NAS result is only "visible" once it has landed (EVENT_DEST_UTC + delay).

    Requires columns: NAS_DELAY, EVENT_ORIGIN_UTC, EVENT_DEST_UTC, ARR_DELAY, ORIGIN.
    Falls back to NaN if columns are missing.
    """
    required = {"NAS_DELAY", "EVENT_ORIGIN_UTC", "EVENT_DEST_UTC", "ARR_DELAY", "ORIGIN"}
    if not required.issubset(df.columns):
        df[out_col] = np.nan
        return df

    out = df.copy()
    n = len(out)

    arr_delay = pd.to_numeric(out["ARR_DELAY"], errors="coerce").fillna(0).to_numpy()
    event_dest = pd.to_datetime(out["EVENT_DEST_UTC"], errors="coerce").astype("datetime64[ns]").to_numpy()
    event_orig = pd.to_datetime(out["EVENT_ORIGIN_UTC"], errors="coerce").astype("datetime64[ns]").to_numpy()
    nas_delay = pd.to_numeric(out["NAS_DELAY"], errors="coerce").fillna(0).to_numpy()

    arr_delay_td = pd.to_timedelta(pd.Series(arr_delay), unit="m").to_numpy().astype("timedelta64[ns]")
    visible_at = event_dest + arr_delay_td
    is_nas = (nas_delay > 0).astype(np.int32)

    window_td = np.timedelta64(int(window_hours * 3_600_000_000_000), "ns")
    group_vals = out["ORIGIN"].astype("string").fillna("__nan__").to_numpy()

    rates = np.full(n, np.nan, dtype=np.float64)

    unique_groups, inverse = np.unique(group_vals, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inv = inverse[order]
    boundaries = np.searchsorted(sorted_inv, np.arange(len(unique_groups) + 1))

    for g in range(len(unique_groups)):
        grp_pos = order[boundaries[g]: boundaries[g + 1]]
        if len(grp_pos) == 0:
            continue

        grp_obs = visible_at[grp_pos]
        grp_query = event_orig[grp_pos]
        grp_nas = is_nas[grp_pos]

        valid_obs = ~pd.isna(grp_obs)
        if not valid_obs.any():
            continue

        obs_t = grp_obs[valid_obs]
        obs_n = grp_nas[valid_obs].astype(np.int32)

        obs_order = np.argsort(obs_t, kind="stable")
        obs_t_sorted = obs_t[obs_order]
        obs_n_sorted = obs_n[obs_order]
        cum = np.concatenate(([0], np.cumsum(obs_n_sorted)))

        valid_q = ~pd.isna(grp_query)
        safe_q = np.where(valid_q, grp_query, np.datetime64("2000-01-01", "ns"))
        window_start = safe_q - window_td

        idx_left = np.searchsorted(obs_t_sorted, window_start, side="right")
        idx_right = np.searchsorted(obs_t_sorted, safe_q, side="left")

        cnt = idx_right - idx_left
        nas_cnt = cum[idx_right] - cum[idx_left]

        with np.errstate(divide="ignore", invalid="ignore"):
            group_rate = np.where(cnt > 0, nas_cnt / np.maximum(cnt, 1), np.nan)
        group_rate[~valid_q] = np.nan

        rates[grp_pos] = group_rate

    out[out_col] = rates
    return out

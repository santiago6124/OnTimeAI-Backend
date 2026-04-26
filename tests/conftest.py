"""Shared pytest fixtures with synthetic master datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


MASTER_COLUMNS = [
    "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "FL_DATE",
    "OP_CARRIER", "TAIL_NUM", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST",
    "CRS_DEP_TIME", "CRS_ARR_TIME", "DEP_TIME", "ARR_TIME",
    "DEP_DELAY", "ARR_DELAY", "DEP_DELAY_NEW", "ARR_DELAY_NEW",
    "CANCELLED", "DIVERTED",
    "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME", "DISTANCE",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    "FLOW_ATL", "PAR_AIRPORT", "CRS_DEP_MIN",
    "DEP_LOCAL_DT", "EVENT_ORIGIN_UTC", "EVENT_DEST_UTC",
    "ORIG_WX_VALID_UTC", "ORIG_WX_TMPC", "ORIG_WX_DWPC", "ORIG_WX_RELH",
    "ORIG_WX_DRCT", "ORIG_WX_SKNT", "ORIG_WX_ALTI", "ORIG_WX_P01M",
    "ORIG_WX_VSBY", "ORIG_WX_GUST", "ORIG_WX_CODES",
    "ORIG_WX_PRECIP_FLAG", "ORIG_WX_LOW_VIS_FLAG", "ORIG_WX_STRONG_WIND_FLAG",
    "ORIG_WX_MATCH_GAP_MIN",
    "DEST_WX_VALID_UTC", "DEST_WX_TMPC", "DEST_WX_DWPC", "DEST_WX_RELH",
    "DEST_WX_DRCT", "DEST_WX_SKNT", "DEST_WX_ALTI", "DEST_WX_P01M",
    "DEST_WX_VSBY", "DEST_WX_GUST", "DEST_WX_CODES",
    "DEST_WX_PRECIP_FLAG", "DEST_WX_LOW_VIS_FLAG", "DEST_WX_STRONG_WIND_FLAG",
    "DEST_WX_MATCH_GAP_MIN",
]


def _wx_block(rng: np.random.Generator, prefix: str, n: int) -> dict:
    return {
        f"{prefix}_WX_VALID_UTC": pd.NaT,
        f"{prefix}_WX_TMPC": rng.normal(15, 10, n),
        f"{prefix}_WX_DWPC": rng.normal(10, 8, n),
        f"{prefix}_WX_RELH": rng.uniform(20, 100, n),
        f"{prefix}_WX_DRCT": rng.uniform(0, 360, n),
        f"{prefix}_WX_SKNT": rng.uniform(0, 30, n),
        f"{prefix}_WX_ALTI": rng.uniform(29.5, 30.5, n),
        f"{prefix}_WX_P01M": np.where(rng.random(n) < 0.1, rng.uniform(0.1, 5, n), 0.0),
        f"{prefix}_WX_VSBY": rng.uniform(0.5, 10, n),
        f"{prefix}_WX_GUST": np.where(rng.random(n) < 0.1, rng.uniform(20, 50, n), np.nan),
        f"{prefix}_WX_CODES": rng.choice(np.array(["M", "BR", "RA", "-RA", "SN"]), n),
        f"{prefix}_WX_PRECIP_FLAG": rng.random(n) < 0.1,
        f"{prefix}_WX_LOW_VIS_FLAG": rng.random(n) < 0.05,
        f"{prefix}_WX_STRONG_WIND_FLAG": rng.random(n) < 0.05,
        f"{prefix}_WX_MATCH_GAP_MIN": rng.uniform(0, 30, n),
    }


def make_synthetic_master(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    end = pd.Timestamp("2025-12-20")
    delta_days = (end - start).days
    day_offsets = np.sort(rng.integers(0, delta_days + 1, size=n))
    fl_dates = pd.to_datetime(start + pd.to_timedelta(day_offsets, unit="D"))

    airports = np.array(["ATL", "MCO", "LGA", "MIA", "ORD", "DFW", "LAX"])
    carriers = np.array(["DL", "AA", "WN", "UA", "B6"])
    tails = np.array([f"N{i:04d}" for i in range(80)])

    origins = rng.choice(airports, size=n)
    dests = rng.choice(airports, size=n)
    half = n // 2
    origins[:half] = "ATL"
    dests[half:] = "ATL"
    same_mask = origins == dests
    dests[same_mask] = np.where(origins[same_mask] == "MCO", "LGA", "MCO")

    crs_dep_min = rng.integers(0, 1440, size=n).astype(float)
    crs_elapsed = rng.integers(60, 300, size=n).astype(float)
    distance = (crs_elapsed * rng.uniform(4.0, 8.0, size=n)).astype(float)

    # Inject mild signal so tests don't degenerate: late flights + low visibility → more delay
    noise = rng.normal(0, 15, size=n)
    late_push = (crs_dep_min > 900) * rng.uniform(0, 10, n)
    arr_delay = noise + late_push
    arr_delay[rng.random(n) < 0.05] = np.nan

    cancelled = (rng.random(n) < 0.03).astype(int)
    diverted = (rng.random(n) < 0.01).astype(int)

    df = pd.DataFrame({
        "YEAR": 2025,
        "MONTH": fl_dates.month,
        "DAY_OF_MONTH": fl_dates.day,
        "DAY_OF_WEEK": fl_dates.dayofweek + 1,
        "FL_DATE": fl_dates,
        "OP_CARRIER": rng.choice(carriers, n),
        "TAIL_NUM": rng.choice(tails, n),
        "OP_CARRIER_FL_NUM": rng.integers(100, 9999, n),
        "ORIGIN": origins,
        "DEST": dests,
        "CRS_DEP_TIME": ((crs_dep_min // 60) * 100 + crs_dep_min % 60).astype(int),
        "CRS_ARR_TIME": 1200,
        "DEP_TIME": np.nan,
        "ARR_TIME": np.nan,
        "DEP_DELAY": rng.normal(0, 15, n),
        "ARR_DELAY": arr_delay,
        "DEP_DELAY_NEW": np.clip(rng.normal(0, 15, n), 0, None),
        "ARR_DELAY_NEW": np.clip(arr_delay, 0, None),
        "CANCELLED": cancelled,
        "DIVERTED": diverted,
        "CRS_ELAPSED_TIME": crs_elapsed,
        "ACTUAL_ELAPSED_TIME": crs_elapsed + rng.normal(0, 10, n),
        "AIR_TIME": crs_elapsed - 20,
        "DISTANCE": distance,
        "CARRIER_DELAY": np.nan,
        "WEATHER_DELAY": np.nan,
        "NAS_DELAY": np.nan,
        "SECURITY_DELAY": np.nan,
        "LATE_AIRCRAFT_DELAY": np.nan,
        "FLOW_ATL": np.where(origins == "ATL", "DEP_FROM_ATL", "ARR_TO_ATL"),
        "PAR_AIRPORT": np.where(origins == "ATL", dests, origins),
        "CRS_DEP_MIN": crs_dep_min,
        "DEP_LOCAL_DT": pd.NaT,
        "EVENT_ORIGIN_UTC": pd.NaT,
        "EVENT_DEST_UTC": pd.NaT,
        **_wx_block(rng, "ORIG", n),
        **_wx_block(rng, "DEST", n),
    })
    missing = set(MASTER_COLUMNS) - set(df.columns)
    assert not missing, f"synthetic master missing columns: {missing}"
    return df[MASTER_COLUMNS]


@pytest.fixture
def tiny_master() -> pd.DataFrame:
    return make_synthetic_master(n=400, seed=42)


@pytest.fixture
def larger_master() -> pd.DataFrame:
    return make_synthetic_master(n=2000, seed=7)

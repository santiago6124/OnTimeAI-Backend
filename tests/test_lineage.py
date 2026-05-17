"""Tests for leakage-safe lineage features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ontimeai.lineage import (
    add_carrier_day_lag,
    add_carrier_rolling_features,
    add_group_rolling_rate,
    add_origin_day_lag,
    add_origin_rolling_features,
    add_tail_lineage_features,
)


def _mini_tail_df() -> pd.DataFrame:
    return pd.DataFrame({
        "TAIL_NUM": ["N1", "N1", "N1", "N2", "N1"],
        "FL_DATE": pd.to_datetime(["2024-05-01", "2024-05-01", "2024-05-01", "2024-05-01", "2024-05-02"]),
        "EVENT_ORIGIN_UTC": pd.to_datetime([
            "2024-05-01 06:00", "2024-05-01 10:00", "2024-05-01 15:00",
            "2024-05-01 08:00", "2024-05-02 07:00",
        ]),
        "EVENT_DEST_UTC": pd.to_datetime([
            "2024-05-01 08:00", "2024-05-01 12:00", "2024-05-01 17:00",
            "2024-05-01 10:00", "2024-05-02 09:00",
        ]),
        "ARR_DELAY": [10.0, -5.0, 30.0, 45.0, 0.0],
    })


def test_tail_lineage_first_flight_has_nans() -> None:
    df = _mini_tail_df()
    out = add_tail_lineage_features(df)
    # Row 0 is N1's very first flight in the frame → no prior → NaN
    assert pd.isna(out.loc[0, "prev_arr_delay_tail"])
    # Row 3 is N2's only flight → no prior → NaN
    assert pd.isna(out.loc[3, "prev_arr_delay_tail"])
    # Row 4 is N1 on day 2 at 07:00 UTC. Prior is N1's row 2 (day 1, 15:00 UTC dep,
    # 17:00 sched arr + 30min delay = actual 17:30). Gap = 16h < 24h → lineage VALID.
    # (Previously this returned NaN because of the (TAIL, FL_DATE-local) grouping
    # bug; the fix now correctly captures cross-midnight rotations.)
    assert out.loc[4, "prev_arr_delay_tail"] == 30.0


def test_tail_lineage_uses_prior_actual_arrival() -> None:
    df = _mini_tail_df()
    out = add_tail_lineage_features(df)
    # Row 1 (N1, 10:00 dep) — prior is row 0 (N1, sched_arr 08:00, delay 10min → actual 08:10)
    # 08:10 <= 10:00 so observable; prev_arr_delay_tail = 10.0
    assert out.loc[1, "prev_arr_delay_tail"] == 10.0
    # Turnaround = 10:00 - 08:00 = 120 min
    assert out.loc[1, "prev_turnaround_tail_min"] == 120.0


def test_tail_lineage_skips_unobservable_prior() -> None:
    df = _mini_tail_df().copy()
    # Force the prior of row 1 to be actually late enough to still be airborne
    df.loc[0, "ARR_DELAY"] = 300.0  # 5-hour delay -> prev actual arr = 13:00 > current 10:00
    out = add_tail_lineage_features(df)
    assert pd.isna(out.loc[1, "prev_arr_delay_tail"]), "should not leak if prior not landed"


def test_tail_flights_today_prior_counts_correctly() -> None:
    df = _mini_tail_df()
    out = add_tail_lineage_features(df)
    # N1 day1: positions 0,1,2 -> counts 0,1,2
    assert out.loc[0, "tail_flights_today_prior"] == 0
    assert out.loc[1, "tail_flights_today_prior"] == 1
    assert out.loc[2, "tail_flights_today_prior"] == 2
    # N2 day1: position 3 -> count 0
    assert out.loc[3, "tail_flights_today_prior"] == 0
    # N1 day2: position 4 -> count 0 (new day)
    assert out.loc[4, "tail_flights_today_prior"] == 0


def test_tail_lineage_different_tails_dont_contaminate() -> None:
    df = pd.DataFrame({
        "TAIL_NUM": ["N1", "N2"],
        "FL_DATE": pd.to_datetime(["2024-05-01", "2024-05-01"]),
        "EVENT_ORIGIN_UTC": pd.to_datetime(["2024-05-01 06:00", "2024-05-01 10:00"]),
        "EVENT_DEST_UTC": pd.to_datetime(["2024-05-01 08:00", "2024-05-01 12:00"]),
        "ARR_DELAY": [60.0, 0.0],
    })
    out = add_tail_lineage_features(df)
    assert pd.isna(out.loc[1, "prev_arr_delay_tail"]), "N2 row must not pull from N1"


def test_carrier_day_lag_looks_at_previous_day() -> None:
    df = pd.DataFrame({
        "OP_CARRIER": ["DL", "DL", "DL", "AA"],
        "FL_DATE": pd.to_datetime(["2024-05-01", "2024-05-01", "2024-05-02", "2024-05-02"]),
        "ARR_DELAY": [20.0, 10.0, 5.0, 100.0],  # DL day1: 1/2 delayed (>15)
    })
    out = add_carrier_day_lag(df)
    # DL on day 2 → previous day rate = 1/2 = 0.5
    assert out.loc[2, "carrier_delay_rate_yday"] == 0.5
    # AA on day 2 → no AA on day 1 → NaN
    assert pd.isna(out.loc[3, "carrier_delay_rate_yday"])
    # DL on day 1 → no DL on day 0 → NaN
    assert pd.isna(out.loc[0, "carrier_delay_rate_yday"])
    assert pd.isna(out.loc[1, "carrier_delay_rate_yday"])


def test_origin_day_lag_computes_per_airport() -> None:
    df = pd.DataFrame({
        "ORIGIN": ["ATL", "ATL", "MCO", "ATL"],
        "FL_DATE": pd.to_datetime(["2024-05-01", "2024-05-01", "2024-05-01", "2024-05-02"]),
        "ARR_DELAY": [30.0, 0.0, 20.0, 10.0],  # ATL day1: 1/2 delayed
    })
    out = add_origin_day_lag(df)
    # ATL day 2 → rate = 1/2
    assert out.loc[3, "origin_delay_rate_yday"] == 0.5
    # ATL day 1 → no prior → NaN
    assert pd.isna(out.loc[0, "origin_delay_rate_yday"])


def test_tail_lineage_preserves_row_order() -> None:
    df = _mini_tail_df()
    out = add_tail_lineage_features(df)
    assert (out["TAIL_NUM"].values == df["TAIL_NUM"].values).all()
    assert (out["ARR_DELAY"].values == df["ARR_DELAY"].values).all()


def _rolling_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "OP_CARRIER": ["DL", "DL", "DL", "DL", "AA"],
        "ORIGIN": ["ATL", "ATL", "ATL", "ATL", "ATL"],
        "EVENT_ORIGIN_UTC": pd.to_datetime([
            "2024-05-01 06:00", "2024-05-01 12:00", "2024-05-01 14:00",
            "2024-05-02 10:00", "2024-05-01 13:00",
        ]),
        "EVENT_DEST_UTC": pd.to_datetime([
            "2024-05-01 08:00", "2024-05-01 14:00", "2024-05-01 16:00",
            "2024-05-02 12:00", "2024-05-01 15:00",
        ]),
        "ARR_DELAY": [30.0, 10.0, 50.0, 5.0, 20.0],
    })


def test_rolling_rate_24h_windows_observable_prior_events() -> None:
    df = _rolling_fixture()
    out = add_group_rolling_rate(df, "OP_CARRIER", 24.0, "carrier_delay_rate_24h")
    # Row 0 (DL @ 06:00) — no prior DL obs → NaN
    assert pd.isna(out.loc[0, "carrier_delay_rate_24h"])
    # Row 1 (DL @ 12:00) — prior DL obs: row 0 arr 08:30 (delayed=1). Rate 1/1 = 1.0
    assert out.loc[1, "carrier_delay_rate_24h"] == 1.0
    # Row 2 (DL @ 14:00) — prior DL obs: row 0 (08:30, 1), row 1 (14:10, 0). Row 1's visible_at
    # is 14:10 > 14:00 so NOT observable yet. Rate is 1/1 = 1.0
    assert out.loc[2, "carrier_delay_rate_24h"] == 1.0
    # Row 4 (AA @ 13:00) — no AA obs → NaN (different carrier, no leakage)
    assert pd.isna(out.loc[4, "carrier_delay_rate_24h"])


def test_rolling_rate_excludes_events_outside_window() -> None:
    df = _rolling_fixture()
    out = add_group_rolling_rate(df, "OP_CARRIER", 1.0, "rate_1h")
    # Row 1 (DL @ 12:00) — only obs is row 0 at 08:30 (30+ min ago). For 1h window:
    # window is (11:00, 12:00). Row 0's visibility 08:30 is outside. → NaN (no obs in window)
    assert pd.isna(out.loc[1, "rate_1h"])


def test_carrier_rolling_adds_all_windows() -> None:
    df = _rolling_fixture()
    out = add_carrier_rolling_features(df)
    for col in ("carrier_delay_rate_24h", "carrier_delay_rate_7d"):
        assert col in out.columns


def test_origin_rolling_adds_all_windows() -> None:
    df = _rolling_fixture()
    out = add_origin_rolling_features(df)
    for col in ("origin_delay_rate_1h", "origin_delay_rate_6h", "origin_delay_rate_24h"):
        assert col in out.columns
    # Row 3 (DL @ 10:00 on 2024-05-02) — prior ATL obs: rows 0,1,2 (all on 2024-05-01),
    # and row 4 (AA, but same ORIGIN ATL). For 24h window ending at 2024-05-02 10:00,
    # window is (05-01 10:00, 05-02 10:00). Visibility times: row0 08:30 OUT, row1 14:10 IN,
    # row2 16:50 IN, row4 15:20 IN → 3 events. Delayed: row1 no (10<=15), row2 yes (50>15),
    # row4 yes (20>15). Rate = 2/3
    assert abs(out.loc[3, "origin_delay_rate_24h"] - 2 / 3) < 1e-9


def test_rolling_rate_preserves_row_order() -> None:
    df = _rolling_fixture()
    out = add_group_rolling_rate(df, "OP_CARRIER", 24.0, "rate")
    # Non-feature columns untouched
    assert (out["OP_CARRIER"].values == df["OP_CARRIER"].values).all()
    assert (out["ARR_DELAY"].values == df["ARR_DELAY"].values).all()

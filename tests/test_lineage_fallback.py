"""Tests for the cold-deck lineage fallback (build + apply + persistence)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ontimeai.lineage_fallback import (
    apply_lineage_fallback,
    build_lookups,
    load_lookups,
    save_lookups,
)


# ----------------------- build_lookups ------------------------------------


def test_build_lookups_smoke(larger_master: pd.DataFrame) -> None:
    lookups = build_lookups(larger_master)

    assert {
        "carrier_route_hour_mean_delay",
        "carrier_origin_hour_mean_delay",
        "carrier_hour_mean_delay",
        "carrier_dow_late_rate",
        "origin_hour_late_rate",
        "carrier_late_rate",
        "origin_late_rate",
        "global_mean_delay",
        "global_late_rate",
        "_meta",
    }.issubset(lookups.keys())

    assert np.isfinite(lookups["global_mean_delay"])
    assert 0.0 <= lookups["global_late_rate"] <= 1.0
    assert lookups["_meta"]["n_rows"] > 0


def test_build_lookups_excludes_cancelled_and_nan(tiny_master: pd.DataFrame) -> None:
    n_orig = len(tiny_master)
    lookups = build_lookups(tiny_master)
    n_used = lookups["_meta"]["n_rows"]
    n_cancelled = (tiny_master["CANCELLED"] == 1).sum()
    n_nan_arr = tiny_master["ARR_DELAY"].isna().sum()
    assert n_used <= n_orig - n_cancelled
    assert n_used <= n_orig - n_nan_arr


# ----------------------- apply_lineage_fallback ---------------------------


def _frame_with(carrier: list[str], origin: list[str], dest: list[str],
                hour: list[int], dow: list[int],
                **extra: list) -> pd.DataFrame:
    n = len(carrier)
    df = pd.DataFrame({
        "OP_CARRIER": carrier,
        "ORIGIN": origin,
        "DEST": dest,
        "DAY_OF_WEEK": dow,
        "CRS_DEP_MIN": [h * 60 for h in hour],
    })
    for k, v in extra.items():
        df[k] = v
    return df


def test_fallback_imputes_prev_arr_delay_from_specific_layer() -> None:
    lookups = {
        "carrier_route_hour_mean_delay": pd.Series(
            {("DL", "ATL", "MCO", 18): 25.0}, name="mean"
        ),
        "carrier_origin_hour_mean_delay": pd.Series(
            {("DL", "ATL", 18): 12.0}, name="mean"
        ),
        "carrier_hour_mean_delay": pd.Series({("DL", 18): 8.0}, name="mean"),
        "global_mean_delay": 5.0,
        "global_late_rate": 0.25,
    }
    # Set names for MultiIndex
    lookups["carrier_route_hour_mean_delay"].index = pd.MultiIndex.from_tuples(
        lookups["carrier_route_hour_mean_delay"].index
    )
    lookups["carrier_origin_hour_mean_delay"].index = pd.MultiIndex.from_tuples(
        lookups["carrier_origin_hour_mean_delay"].index
    )
    lookups["carrier_hour_mean_delay"].index = pd.MultiIndex.from_tuples(
        lookups["carrier_hour_mean_delay"].index
    )

    df = _frame_with(
        carrier=["DL"],
        origin=["ATL"],
        dest=["MCO"],
        hour=[18],
        dow=[3],
        prev_arr_delay_tail=[np.nan],
    )
    out = apply_lineage_fallback(df, lookups)
    # Most-specific layer (carrier+origin+dest+hour) should win
    assert out.loc[0, "prev_arr_delay_tail"] == pytest.approx(25.0)


def test_fallback_walks_to_less_specific_layer_when_specific_missing() -> None:
    lookups = {
        "carrier_route_hour_mean_delay": pd.Series([], dtype=float),
        "carrier_origin_hour_mean_delay": pd.Series(
            {("DL", "ATL", 18): 12.0}, name="mean"
        ),
        "carrier_hour_mean_delay": pd.Series([], dtype=float),
        "global_mean_delay": 5.0,
        "global_late_rate": 0.25,
    }
    lookups["carrier_origin_hour_mean_delay"].index = pd.MultiIndex.from_tuples(
        lookups["carrier_origin_hour_mean_delay"].index
    )

    df = _frame_with(
        carrier=["DL"],
        origin=["ATL"],
        dest=["MCO"],
        hour=[18],
        dow=[3],
        prev_arr_delay_tail=[np.nan],
    )
    out = apply_lineage_fallback(df, lookups)
    # carrier_origin_hour layer is the only one with data
    assert out.loc[0, "prev_arr_delay_tail"] == pytest.approx(12.0)


def test_fallback_uses_global_when_no_layer_matches() -> None:
    lookups = {
        "carrier_route_hour_mean_delay": pd.Series([], dtype=float),
        "carrier_origin_hour_mean_delay": pd.Series([], dtype=float),
        "carrier_hour_mean_delay": pd.Series([], dtype=float),
        "global_mean_delay": 5.0,
        "global_late_rate": 0.25,
    }
    df = _frame_with(
        carrier=["XX"], origin=["YYY"], dest=["ZZZ"], hour=[3], dow=[1],
        prev_arr_delay_tail=[np.nan],
    )
    out = apply_lineage_fallback(df, lookups)
    assert out.loc[0, "prev_arr_delay_tail"] == pytest.approx(5.0)


def test_fallback_does_not_overwrite_present_values() -> None:
    lookups = {
        "carrier_route_hour_mean_delay": pd.Series([], dtype=float),
        "carrier_origin_hour_mean_delay": pd.Series([], dtype=float),
        "carrier_hour_mean_delay": pd.Series([], dtype=float),
        "global_mean_delay": 99.0,
        "global_late_rate": 0.8,
    }
    df = _frame_with(
        carrier=["DL", "DL"], origin=["ATL", "ATL"], dest=["MCO", "MCO"],
        hour=[18, 18], dow=[3, 3],
        prev_arr_delay_tail=[42.0, np.nan],
        carrier_delay_rate_24h=[0.10, np.nan],
    )
    out = apply_lineage_fallback(df, lookups)
    # Row 0 was already present → unchanged
    assert out.loc[0, "prev_arr_delay_tail"] == pytest.approx(42.0)
    assert out.loc[0, "carrier_delay_rate_24h"] == pytest.approx(0.10)
    # Row 1 was NaN → filled with global
    assert out.loc[1, "prev_arr_delay_tail"] == pytest.approx(99.0)
    assert out.loc[1, "carrier_delay_rate_24h"] == pytest.approx(0.8)


def test_fallback_handles_missing_columns_gracefully() -> None:
    lookups = {"global_mean_delay": 0.0, "global_late_rate": 0.0}
    # No lineage columns present at all → returns df unchanged
    df = pd.DataFrame({"OP_CARRIER": ["DL"], "ORIGIN": ["ATL"], "DEST": ["MCO"]})
    out = apply_lineage_fallback(df, lookups)
    assert out.equals(df)


def test_fallback_sets_default_for_count_and_turnaround() -> None:
    lookups = {"global_mean_delay": 0.0, "global_late_rate": 0.0}
    df = _frame_with(
        carrier=["DL"], origin=["ATL"], dest=["MCO"], hour=[12], dow=[2],
        tail_flights_today_prior=[np.nan],
        prev_turnaround_tail_min=[np.nan],
    )
    out = apply_lineage_fallback(df, lookups)
    assert out.loc[0, "tail_flights_today_prior"] == 0
    assert out.loc[0, "prev_turnaround_tail_min"] == pytest.approx(60.0)


# ----------------------- save / load round-trip --------------------------


def test_save_load_round_trip(larger_master: pd.DataFrame, tmp_path: Path) -> None:
    lookups = build_lookups(larger_master)
    out_path = tmp_path / "fb.joblib"
    saved = save_lookups(lookups, out_path)
    assert saved.exists()

    reloaded = load_lookups(saved)
    assert reloaded["global_mean_delay"] == pytest.approx(lookups["global_mean_delay"])
    assert reloaded["global_late_rate"] == pytest.approx(lookups["global_late_rate"])
    pd.testing.assert_series_equal(
        reloaded["carrier_late_rate"].sort_index(),
        lookups["carrier_late_rate"].sort_index(),
    )


# ----------------------- end-to-end: fallback rescues cold-start --------


def test_apply_fallback_brings_cold_start_features_to_finite_values(
    larger_master: pd.DataFrame,
) -> None:
    lookups = build_lookups(larger_master)
    df = pd.DataFrame({
        "OP_CARRIER": ["DL", "AA"],
        "ORIGIN": ["ATL", "ATL"],
        "DEST": ["MCO", "LAX"],
        "DAY_OF_WEEK": [3, 5],
        "CRS_DEP_MIN": [600, 1080],
        "prev_arr_delay_tail": [np.nan, np.nan],
        "tail_flights_today_prior": [np.nan, np.nan],
        "prev_turnaround_tail_min": [np.nan, np.nan],
        "carrier_delay_rate_24h": [np.nan, np.nan],
        "carrier_delay_rate_7d": [np.nan, np.nan],
        "carrier_delay_rate_yday": [np.nan, np.nan],
        "origin_delay_rate_1h": [np.nan, np.nan],
        "origin_delay_rate_6h": [np.nan, np.nan],
        "origin_delay_rate_24h": [np.nan, np.nan],
        "origin_delay_rate_yday": [np.nan, np.nan],
    })
    out = apply_lineage_fallback(df, lookups)
    fallback_cols = [c for c in df.columns if c not in ("OP_CARRIER", "ORIGIN", "DEST", "DAY_OF_WEEK", "CRS_DEP_MIN")]
    for col in fallback_cols:
        assert out[col].notna().all(), f"{col} still has NaN after fallback"
        assert np.isfinite(out[col].to_numpy()).all(), f"{col} non-finite"

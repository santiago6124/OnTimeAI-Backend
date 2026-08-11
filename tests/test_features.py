"""Tests for feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ontimeai.features import (
    _window_counts_per_group,
    add_congestion_features,
    add_cyclical_features,
    add_weather_interactions,
    apply_categorical_mapping,
    build_feature_matrix,
    build_target,
    normalize_weather_flags,
)


def test_cyclical_values_on_unit_circle(tiny_master: pd.DataFrame) -> None:
    out = add_cyclical_features(tiny_master)
    for sin_col, cos_col in [
        ("dep_hour_sin", "dep_hour_cos"),
        ("dep_dow_sin", "dep_dow_cos"),
        ("dep_month_sin", "dep_month_cos"),
    ]:
        radius2 = out[sin_col] ** 2 + out[cos_col] ** 2
        assert np.allclose(radius2.dropna(), 1.0, atol=1e-9)


def test_cyclical_handles_nan() -> None:
    df = pd.DataFrame({
        "CRS_DEP_MIN": [0.0, 360.0, np.nan],
        "DAY_OF_WEEK": [1, 2, 3],
        "MONTH": [1, 2, 3],
    })
    out = add_cyclical_features(df)
    assert pd.isna(out["dep_hour_sin"].iloc[2])
    assert np.isclose(out["dep_hour_sin"].iloc[0], 0.0, atol=1e-9)
    assert np.isclose(out["dep_hour_cos"].iloc[0], 1.0, atol=1e-9)


def test_window_counts_small() -> None:
    mins = np.array([0.0, 10.0, 25.0, 50.0, 55.0])
    counts = _window_counts_per_group(mins, window=30.0)
    expected = np.array([3, 3, 5, 3, 3])
    assert (counts == expected).all(), f"got {counts.tolist()}, want {expected.tolist()}"


def test_window_counts_preserve_order() -> None:
    # Unsorted input must produce counts aligned to the original positions.
    mins = np.array([50.0, 0.0, 25.0, 55.0, 10.0])
    counts = _window_counts_per_group(mins, window=30.0)
    # sorted equivalent would give [3, 3, 5, 3, 3] for sorted [0,10,25,50,55];
    # original positions map: 50->3, 0->3, 25->5, 55->3, 10->3
    expected = np.array([3, 3, 5, 3, 3])
    assert (counts == expected).all()


def test_congestion_features_added(tiny_master: pd.DataFrame) -> None:
    out = add_congestion_features(tiny_master)
    assert "congestion_orig_window" in out.columns
    assert "congestion_dest_window" in out.columns
    assert (out["congestion_orig_window"] >= 1).all()
    assert (out["congestion_dest_window"] >= 1).all()


def test_congestion_no_cross_day_bleed() -> None:
    df = pd.DataFrame({
        "FL_DATE": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]),
        "CRS_DEP_MIN": [600.0, 610.0, 600.0],
        "ORIGIN": ["ATL", "ATL", "ATL"],
        "DEST": ["MCO", "MCO", "MCO"],
    })
    out = add_congestion_features(df, window_min=30)
    # Rows 0 and 1 are same date+origin within 30min => 2; row 2 is alone on 01-02 => 1
    assert out["congestion_orig_window"].tolist() == [2, 2, 1]


def test_weather_interactions(tiny_master: pd.DataFrame) -> None:
    out = add_weather_interactions(tiny_master)
    for c in ("wx_both_precip", "wx_both_low_vis", "wx_both_strong_wind"):
        assert c in out.columns
        assert out[c].isin([0, 1]).all()


def test_build_target_binary(tiny_master: pd.DataFrame) -> None:
    y = build_target(tiny_master, "binary", 15.0)
    assert y.dtype == np.int8
    arr = pd.to_numeric(tiny_master["ARR_DELAY"], errors="coerce")
    mask = (arr > 15).fillna(False)
    assert (y.loc[mask] == 1).all()
    assert (y.loc[arr <= 15] == 0).all()


def test_build_target_multiclass_edges() -> None:
    df = pd.DataFrame({"ARR_DELAY": [0, 15, 15.01, 30, 30.01, 60, 60.01, 120]})
    y = build_target(df, "multiclass", 15.0)
    assert list(y) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_build_feature_matrix_casts_categoricals(tiny_master: pd.DataFrame) -> None:
    from ontimeai.data import drop_leaky_target_columns, filter_valid_flights

    df = filter_valid_flights(tiny_master)
    df = drop_leaky_target_columns(df)
    X, cat_cols, mapping = build_feature_matrix(df)
    for c in cat_cols:
        assert str(X[c].dtype) == "category"
    # all cat mappings recorded
    assert set(mapping.keys()) == set(cat_cols)
    # no TARGET or schedule-replaced cols
    for forbidden in ("TARGET", "CRS_DEP_TIME", "FL_DATE"):
        assert forbidden not in X.columns


def test_normalize_weather_flags_accepts_mixed_types() -> None:
    df = pd.DataFrame({
        "ORIG_WX_PRECIP_FLAG": ["True", "False", "true", np.nan],
        "DEST_WX_PRECIP_FLAG": [True, False, True, False],
        "ORIG_WX_LOW_VIS_FLAG": [1, 0, 1, 0],
        "DEST_WX_LOW_VIS_FLAG": ["False", "True", "False", "True"],
        "ORIG_WX_STRONG_WIND_FLAG": [False, False, True, True],
        "DEST_WX_STRONG_WIND_FLAG": [0, 1, 0, 1],
    })
    out = normalize_weather_flags(df)
    for c in df.columns:
        assert out[c].dtype == np.int8
        assert out[c].isin([0, 1]).all()
    assert out["ORIG_WX_PRECIP_FLAG"].tolist() == [1, 0, 1, 0]
    assert out["DEST_WX_LOW_VIS_FLAG"].tolist() == [0, 1, 0, 1]


def test_apply_categorical_mapping_round_trips() -> None:
    mapping = {"col": ["a", "b", "c"]}
    X = pd.DataFrame({"col": ["a", "b", "a", "z"]})
    out = apply_categorical_mapping(X, mapping)
    assert str(out["col"].dtype) == "category"
    assert list(out["col"].cat.categories) == ["a", "b", "c"]
    # unknown values become NaN
    assert pd.isna(out["col"].iloc[3])


def test_inference_can_preserve_unknown_category_for_challenger_capture() -> None:
    from predict import prepare_inference_frame

    source = pd.DataFrame(
        {
            "OP_CARRIER": ["ZZ"],
            "FL_DATE": ["2026-08-11"],
            "CRS_DEP_MIN": [720],
            "ORIGIN": ["ATL"],
            "DEST": ["MCO"],
            "DAY_OF_WEEK": [2],
            "MONTH": [8],
        }
    )
    mapping = {"OP_CARRIER": ["AA", "DL"]}

    raw = prepare_inference_frame(
        source,
        ["OP_CARRIER"],
        mapping,
        apply_category_mapping=False,
    )
    champion = prepare_inference_frame(source, ["OP_CARRIER"], mapping)

    assert raw.loc[0, "OP_CARRIER"] == "ZZ"
    assert pd.isna(champion.loc[0, "OP_CARRIER"])

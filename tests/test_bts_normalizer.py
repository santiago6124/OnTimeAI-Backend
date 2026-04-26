"""Tests for BTS chunk normalization (PREZIP friendly vs legacy SQL-style)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# construir_dataset_maestro_multi lives at project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from construir_dataset_maestro_multi import KEEP_COLS, parse_fl_date, standardize_bts_chunk


def test_standardize_prezip_renames_and_keeps_schema() -> None:
    df = pd.DataFrame({
        "Year": [2024, 2024],
        "Month": [1, 1],
        "DayofMonth": [8, 8],
        "DayOfWeek": [1, 1],
        "FlightDate": ["2024-01-08", "2024-01-08"],
        "Reporting_Airline": ["DL", "AA"],
        "Tail_Number": ["N1234", "N5678"],
        "Flight_Number_Reporting_Airline": [100, 200],
        "Origin": ["atl ", "MCO"],
        "Dest": [" MCO", "ATL"],
        "CRSDepTime": [600, 1500],
        "CRSArrTime": [735, 1730],
        "DepTime": [np.nan, 1510],
        "ArrTime": [np.nan, 1740],
        "DepDelay": [np.nan, 10.0],
        "ArrDelay": [np.nan, 10.0],
        "DepDelayMinutes": [np.nan, 10.0],
        "ArrDelayMinutes": [np.nan, 10.0],
        "Cancelled": [1.0, 0.0],
        "Diverted": [0.0, 0.0],
        "CRSElapsedTime": [95.0, 150.0],
        "ActualElapsedTime": [np.nan, 150.0],
        "AirTime": [np.nan, 130.0],
        "Distance": [406, 404],
        "CarrierDelay": [np.nan, np.nan],
        "WeatherDelay": [np.nan, np.nan],
        "NASDelay": [np.nan, np.nan],
        "SecurityDelay": [np.nan, np.nan],
        "LateAircraftDelay": [np.nan, np.nan],
    })

    out = standardize_bts_chunk(df)

    assert list(out.columns) == list(KEEP_COLS)
    assert out["ORIGIN"].tolist() == ["ATL", "MCO"]
    assert out["DEST"].tolist() == ["MCO", "ATL"]
    assert out["OP_CARRIER"].tolist() == ["DL", "AA"]
    assert out["TAIL_NUM"].tolist() == ["N1234", "N5678"]
    assert out["FL_DATE"].dt.year.iloc[0] == 2024
    assert out["CRS_DEP_TIME"].dtype.kind in "if"


def test_standardize_legacy_sqlstyle() -> None:
    df = pd.DataFrame({
        "YEAR": [2025],
        "MONTH": [1],
        "DAY_OF_MONTH": [1],
        "DAY_OF_WEEK": [3],
        "FL_DATE": ["01/01/2025 12:00:00 AM"],
        "OP_CARRIER": ["DL"],
        "TAIL_NUM": ["N555NW"],
        "OP_CARRIER_FL_NUM": [1545],
        "ORIGIN": ["TPA"],
        "DEST": ["ATL"],
        "CRS_DEP_TIME": [600],
        "CRS_ARR_TIME": [732],
        "DEP_TIME": [555.0],
        "ARR_TIME": [722.0],
        "DEP_DELAY": [0.0],
        "ARR_DELAY": [0.0],
        "DEP_DELAY_NEW": [0.0],
        "ARR_DELAY_NEW": [0.0],
        "CANCELLED": [0.0],
        "DIVERTED": [0.0],
        "CRS_ELAPSED_TIME": [92.0],
        "ACTUAL_ELAPSED_TIME": [87.0],
        "AIR_TIME": [66.0],
        "DISTANCE": [406.0],
        "CARRIER_DELAY": [np.nan],
        "WEATHER_DELAY": [np.nan],
        "NAS_DELAY": [np.nan],
        "SECURITY_DELAY": [np.nan],
        "LATE_AIRCRAFT_DELAY": [np.nan],
    })

    out = standardize_bts_chunk(df)
    assert list(out.columns) == list(KEEP_COLS)
    assert out["FL_DATE"].iloc[0] == pd.Timestamp("2025-01-01")
    assert out["ORIGIN"].iloc[0] == "TPA"


def test_parse_fl_date_iso() -> None:
    s = pd.Series(["2024-01-08", "2024-02-15", "2024-12-31"])
    out = parse_fl_date(s)
    assert out.iloc[0] == pd.Timestamp("2024-01-08")
    assert out.iloc[-1] == pd.Timestamp("2024-12-31")


def test_parse_fl_date_legacy() -> None:
    s = pd.Series(["01/08/2024 12:00:00 AM", "02/15/2024 12:00:00 AM"])
    out = parse_fl_date(s)
    assert out.iloc[0] == pd.Timestamp("2024-01-08")
    assert out.iloc[1] == pd.Timestamp("2024-02-15")


def test_parse_fl_date_nan_tolerant() -> None:
    s = pd.Series(["2024-01-08", "not-a-date", None])
    out = parse_fl_date(s)
    assert out.iloc[0] == pd.Timestamp("2024-01-08")
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])

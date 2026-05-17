"""Builds the BTS + IEM master dataset for arbitrary year ranges.

Handles both BTS schemas:
- PREZIP (friendly names): FlightDate, Reporting_Airline, Origin, Dest, CRSDepTime, ...
- Legacy SQL-style: FL_DATE, OP_CARRIER, ORIGIN, DEST, CRS_DEP_TIME, ...

And both date formats: ISO (YYYY-MM-DD) and MM/DD/YYYY HH:MM:SS AM/PM.

Usage:
    python3 construir_dataset_maestro_multi.py                  # default: 2022 2023 2024
    python3 construir_dataset_maestro_multi.py 2022 2023 2024
    python3 construir_dataset_maestro_multi.py 2024 2025        # only these years
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
BTS_DIR = PROJECT_ROOT / "data_raw" / "bts"
IEM_DIR = PROJECT_ROOT / "data_raw" / "iem"

AIRPORTS = {
    "ATL", "LGA", "MCO", "FLL", "MIA", "DFW", "DCA", "EWR",
    "TPA", "ORD", "PHL", "DEN", "LAX", "BWI", "LAS", "BOS",
}
TZ_BY_AIRPORT = {
    "ATL": "America/New_York", "LGA": "America/New_York",
    "MCO": "America/New_York", "FLL": "America/New_York",
    "MIA": "America/New_York", "DFW": "America/Chicago",
    "DCA": "America/New_York", "EWR": "America/New_York",
    "TPA": "America/New_York", "ORD": "America/Chicago",
    "PHL": "America/New_York", "DEN": "America/Denver",
    "LAX": "America/Los_Angeles", "BWI": "America/New_York",
    "LAS": "America/Los_Angeles", "BOS": "America/New_York",
}
WX_NUM_COLS = ["tmpc", "dwpc", "relh", "drct", "sknt", "alti", "p01m", "vsby", "gust"]
MONTH_PREFIXES = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                  "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

PREZIP_RENAME = {
    "Year": "YEAR",
    "Month": "MONTH",
    "DayofMonth": "DAY_OF_MONTH",
    "DayOfWeek": "DAY_OF_WEEK",
    "FlightDate": "FL_DATE",
    "Reporting_Airline": "OP_CARRIER",
    "Tail_Number": "TAIL_NUM",
    "Flight_Number_Reporting_Airline": "OP_CARRIER_FL_NUM",
    "Origin": "ORIGIN",
    "Dest": "DEST",
    "CRSDepTime": "CRS_DEP_TIME",
    "DepTime": "DEP_TIME",
    "DepDelay": "DEP_DELAY",
    "DepDelayMinutes": "DEP_DELAY_NEW",
    "CRSArrTime": "CRS_ARR_TIME",
    "ArrTime": "ARR_TIME",
    "ArrDelay": "ARR_DELAY",
    "ArrDelayMinutes": "ARR_DELAY_NEW",
    "Cancelled": "CANCELLED",
    "Diverted": "DIVERTED",
    "CRSElapsedTime": "CRS_ELAPSED_TIME",
    "ActualElapsedTime": "ACTUAL_ELAPSED_TIME",
    "AirTime": "AIR_TIME",
    "Distance": "DISTANCE",
    "CarrierDelay": "CARRIER_DELAY",
    "WeatherDelay": "WEATHER_DELAY",
    "NASDelay": "NAS_DELAY",
    "SecurityDelay": "SECURITY_DELAY",
    "LateAircraftDelay": "LATE_AIRCRAFT_DELAY",
}

KEEP_COLS = [
    "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "FL_DATE",
    "OP_CARRIER", "TAIL_NUM", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST",
    "CRS_DEP_TIME", "CRS_ARR_TIME", "DEP_TIME", "ARR_TIME",
    "DEP_DELAY", "ARR_DELAY", "DEP_DELAY_NEW", "ARR_DELAY_NEW",
    "CANCELLED", "DIVERTED",
    "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME", "DISTANCE",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]

BTS_FILE_RE = re.compile(r"^(" + "|".join(MONTH_PREFIXES) + r")_(\d{4})_.*\.csv$", re.I)
IEM_FILE_RE = re.compile(r"clima_iem_asos_(\d{4})_utc\.csv$", re.I)


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    hours = np.floor(values / 100)
    minutes = values % 100
    invalid = (values < 0) | (hours > 23) | (minutes >= 60)
    out = (hours * 60 + minutes).astype("float64")
    out[invalid] = np.nan
    return out


def local_series_to_utc_naive(local_naive: pd.Series, tz_name: str) -> pd.Series:
    tz = ZoneInfo(tz_name)

    def _convert(dt):
        if pd.isna(dt):
            return pd.NaT
        try:
            return dt.replace(tzinfo=tz).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        except Exception:
            return pd.NaT

    return local_naive.apply(_convert)


def parse_fl_date(series: pd.Series) -> pd.Series:
    iso = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    if iso.notna().mean() > 0.5:
        return iso
    legacy = pd.to_datetime(series, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    if legacy.notna().mean() > 0.5:
        return legacy
    return pd.to_datetime(series, errors="coerce")


def standardize_bts_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_applicable = {k: v for k, v in PREZIP_RENAME.items() if k in df.columns}
    if rename_applicable:
        df = df.rename(columns=rename_applicable)
    df.columns = [c.strip().upper() for c in df.columns]

    if "OP_UNIQUE_CARRIER" in df.columns and "OP_CARRIER" not in df.columns:
        df["OP_CARRIER"] = df["OP_UNIQUE_CARRIER"]
    if "DEP_DELAY" not in df.columns and "DEP_DELAY_NEW" in df.columns:
        df["DEP_DELAY"] = df["DEP_DELAY_NEW"]
    if "ARR_DELAY" not in df.columns and "ARR_DELAY_NEW" in df.columns:
        df["ARR_DELAY"] = df["ARR_DELAY_NEW"]

    df["ORIGIN"] = df["ORIGIN"].astype("string").str.strip().str.upper()
    df["DEST"] = df["DEST"].astype("string").str.strip().str.upper()
    df["FL_DATE"] = parse_fl_date(df["FL_DATE"])

    num_cols = [
        "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "OP_CARRIER_FL_NUM",
        "CRS_DEP_TIME", "CRS_ARR_TIME", "DEP_TIME", "ARR_TIME",
        "DEP_DELAY", "ARR_DELAY", "DEP_DELAY_NEW", "ARR_DELAY_NEW",
        "CANCELLED", "DIVERTED",
        "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME", "DISTANCE",
        "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[KEEP_COLS]


def discover_bts_csvs(years: set[int]) -> list[Path]:
    paths: list[Path] = []
    for base in (BTS_DIR, PROJECT_ROOT):
        if not base.is_dir():
            continue
        for p in sorted(base.glob("*.csv")):
            m = BTS_FILE_RE.match(p.name)
            if not m:
                continue
            year = int(m.group(2))
            if year not in years:
                continue
            try:
                cols = pd.read_csv(p, nrows=0).columns
            except Exception:
                continue
            cols_upper = {c.strip().upper() for c in cols}
            if "FL_DATE" in cols_upper or "FLIGHTDATE" in cols_upper:
                paths.append(p)
    seen: set[str] = set()
    unique: list[Path] = []
    for p in paths:
        key = p.name.upper()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def discover_iem_csvs(years: set[int]) -> list[Path]:
    paths: list[Path] = []
    for base in (IEM_DIR, PROJECT_ROOT):
        if not base.is_dir():
            continue
        for p in sorted(base.glob("clima_iem_asos_*_utc.csv")):
            m = IEM_FILE_RE.search(p.name)
            if not m:
                continue
            year = int(m.group(1))
            if year in years:
                paths.append(p)
    return paths


def load_all_weather(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        print(f"  weather <- {p}")
        wx = pd.read_csv(p, low_memory=False)
        wx.columns = [c.strip().lower() for c in wx.columns]
        wx["station"] = wx["station"].astype("string").str.strip().str.upper()
        wx["valid"] = (
            pd.to_datetime(wx["valid"], errors="coerce", utc=True)
            .dt.tz_convert(None)
            .astype("datetime64[ns]")
        )
        for col in WX_NUM_COLS:
            wx[col] = pd.to_numeric(wx[col].replace({"M": np.nan, "T": 0.001}), errors="coerce")
        wx["wx_precip_flag"] = wx["p01m"].fillna(0) > 0
        wx["wx_low_vis_flag"] = wx["vsby"] < 3
        wx["wx_strong_wind_flag"] = (wx["sknt"] >= 20) | (wx["gust"] >= 30)
        frames.append(wx[[
            "station", "valid", "tmpc", "dwpc", "relh", "drct", "sknt", "alti",
            "p01m", "vsby", "gust", "wxcodes",
            "wx_precip_flag", "wx_low_vis_flag", "wx_strong_wind_flag",
        ]])
    weather = pd.concat(frames, ignore_index=True)
    weather = weather.drop_duplicates(subset=["station", "valid"])
    weather = weather.sort_values(["station", "valid"]).reset_index(drop=True)
    return weather


def load_bts_flights(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        print(f"  flights <- {path.name}", flush=True)
        for chunk in pd.read_csv(path, chunksize=250_000, low_memory=False):
            std = standardize_bts_chunk(chunk)
            mask_atl = std["ORIGIN"].eq("ATL") | std["DEST"].eq("ATL")
            mask_known = std["ORIGIN"].isin(AIRPORTS) & std["DEST"].isin(AIRPORTS)
            sub = std.loc[mask_atl & mask_known].copy()
            if not sub.empty:
                frames.append(sub)
    if not frames:
        raise ValueError("No flights remaining after ATL + known-airports filter")
    flights = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    flights["FLOW_ATL"] = np.where(flights["ORIGIN"].eq("ATL"), "DEP_FROM_ATL", "ARR_TO_ATL")
    flights["PAR_AIRPORT"] = np.where(flights["ORIGIN"].eq("ATL"), flights["DEST"], flights["ORIGIN"])
    return flights


def add_utc_event_times(flights: pd.DataFrame) -> pd.DataFrame:
    out = flights.copy()
    out["CRS_DEP_MIN"] = hhmm_to_minutes(out["CRS_DEP_TIME"])
    out["DEP_LOCAL_DT"] = out["FL_DATE"] + pd.to_timedelta(out["CRS_DEP_MIN"], unit="m")
    out["EVENT_ORIGIN_UTC"] = pd.NaT
    for airport, tz_name in TZ_BY_AIRPORT.items():
        mask = out["ORIGIN"].eq(airport)
        if mask.any():
            out.loc[mask, "EVENT_ORIGIN_UTC"] = local_series_to_utc_naive(
                out.loc[mask, "DEP_LOCAL_DT"], tz_name
            )
    out["EVENT_ORIGIN_UTC"] = pd.to_datetime(out["EVENT_ORIGIN_UTC"], errors="coerce").astype("datetime64[ns]")
    elapsed = pd.to_numeric(out["CRS_ELAPSED_TIME"], errors="coerce")
    out["EVENT_DEST_UTC"] = (out["EVENT_ORIGIN_UTC"] + pd.to_timedelta(elapsed, unit="m")).astype("datetime64[ns]")
    return out


def merge_weather(flights: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    def asof_by_station(left, wx_df, station_col, event_col, prefix):
        out_parts = []
        for station in sorted(left[station_col].dropna().astype(str).unique()):
            lsub = left[left[station_col].eq(station)].copy()
            wsub = wx_df[wx_df["station"].eq(station)].copy()
            if lsub.empty:
                continue
            lsub = lsub.sort_values(event_col)
            wsub = wsub.sort_values("valid")
            lsub[event_col] = lsub[event_col].astype("datetime64[ns]")
            wsub["valid"] = wsub["valid"].astype("datetime64[ns]")
            merged = pd.merge_asof(
                lsub, wsub,
                left_on=event_col, right_on="valid",
                direction="nearest", tolerance=pd.Timedelta("90min"),
            )
            out_parts.append(merged)
        if not out_parts:
            raise ValueError(f"No matches for prefix={prefix}")
        merged_all = pd.concat(out_parts, ignore_index=True)
        gap = (merged_all[event_col] - merged_all["valid"]).abs().dt.total_seconds() / 60
        merged_all = merged_all.rename(columns={
            "valid": f"{prefix}_WX_VALID_UTC",
            "tmpc": f"{prefix}_WX_TMPC", "dwpc": f"{prefix}_WX_DWPC",
            "relh": f"{prefix}_WX_RELH", "drct": f"{prefix}_WX_DRCT",
            "sknt": f"{prefix}_WX_SKNT", "alti": f"{prefix}_WX_ALTI",
            "p01m": f"{prefix}_WX_P01M", "vsby": f"{prefix}_WX_VSBY",
            "gust": f"{prefix}_WX_GUST", "wxcodes": f"{prefix}_WX_CODES",
            "wx_precip_flag": f"{prefix}_WX_PRECIP_FLAG",
            "wx_low_vis_flag": f"{prefix}_WX_LOW_VIS_FLAG",
            "wx_strong_wind_flag": f"{prefix}_WX_STRONG_WIND_FLAG",
        })
        merged_all[f"{prefix}_WX_MATCH_GAP_MIN"] = gap
        return merged_all.drop(columns=["station"])

    origin_merge = asof_by_station(flights, wx, "ORIGIN", "EVENT_ORIGIN_UTC", "ORIG")
    return asof_by_station(origin_merge, wx, "DEST", "EVENT_DEST_UTC", "DEST")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    years = sorted(set(int(a) for a in argv)) if argv else [2022, 2023, 2024]
    years_set = set(years)
    print(f"Target years: {years}")

    bts_paths = discover_bts_csvs(years_set)
    iem_paths = discover_iem_csvs(years_set)

    if not bts_paths:
        raise FileNotFoundError(
            f"No BTS monthly CSVs found for {years} in {BTS_DIR} or {PROJECT_ROOT}. "
            f"Run descarga_bts.py first."
        )
    if not iem_paths:
        raise FileNotFoundError(
            f"No IEM CSVs found for {years} in {IEM_DIR} or {PROJECT_ROOT}. "
            f"Run descarga_data_iem_multi.py first."
        )

    print(f"\nBTS inputs ({len(bts_paths)}):")
    for p in bts_paths:
        print(f"  {p}")
    print(f"\nIEM inputs ({len(iem_paths)}):")
    for p in iem_paths:
        print(f"  {p}")

    print("\nLoading weather...")
    wx = load_all_weather(iem_paths)
    print(f"  {len(wx):,} METAR observations across {wx['station'].nunique()} stations")

    print("\nLoading flights...")
    flights = load_bts_flights(bts_paths)
    print(f"  {len(flights):,} ATL-related flights after filter")

    flights = add_utc_event_times(flights)

    print("\nMerging weather by asof (90min tolerance)...")
    master = merge_weather(flights, wx)
    print(f"  {len(master):,} rows × {len(master.columns)} columns")

    label = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])
    out_path = PROJECT_ROOT / f"dataset_maestro_ATL_{label}_BTS_IEM_ORIG_DEST.csv"
    master.to_csv(out_path, index=False)
    print(f"\n  -> {out_path}")
    print(f"  ORIG wx coverage: {master['ORIG_WX_VALID_UTC'].notna().mean()*100:.2f}%")
    print(f"  DEST wx coverage: {master['DEST_WX_VALID_UTC'].notna().mean()*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

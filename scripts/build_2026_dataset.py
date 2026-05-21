"""Build the BTS+IEM master dataset for Jan-Feb 2026.

Optimized version that:
  1. Reads the newly uploaded JAN_2026 and FEB_2026 BTS CSVs
  2. Downloads IEM weather ONLY for airports that appear in the data
  3. Merges weather using the same asof-join logic as the training pipeline
  4. Outputs a parquet ready for ground truth evaluation with v9

Usage:
    python3 scripts/build_2026_dataset.py
    python3 scripts/build_2026_dataset.py --skip-weather    # use without weather
    python3 scripts/build_2026_dataset.py --atl-only        # only ATL-touching flights
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BTS_DIR = PROJECT_ROOT / "data_raw" / "bts"
IEM_DIR = PROJECT_ROOT / "data_raw" / "iem"
UNIVERSE_PATH = PROJECT_ROOT / "airports_universe.csv"

# IEM config
IEM_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
WX_VARS = ["tmpc", "dwpc", "relh", "drct", "sknt", "alti", "p01m", "vsby", "gust", "wxcodes"]


def load_universe() -> tuple[set[str], dict[str, str]]:
    df = pd.read_csv(UNIVERSE_PATH)
    df["iata"] = df["iata"].astype(str).str.strip().str.upper()
    airports = set(df["iata"].tolist())
    tz_by_iata = dict(zip(df["iata"], df["timezone"]))
    return airports, tz_by_iata


def standardize_bts(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the new BTS 2026 format to match the training schema."""
    out = df.copy()

    # Rename columns to match the expected schema
    renames = {
        "OP_UNIQUE_CARRIER": "OP_CARRIER",
        "Flight_Number_Reporting_Airline": "OP_CARRIER_FL_NUM",
        "Reporting_Airline": "OP_CARRIER",
    }
    for old, new in renames.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})

    # Ensure column names are uppercase
    out.columns = [c.strip().upper() for c in out.columns]

    # If OP_CARRIER still missing, check for OP_UNIQUE_CARRIER
    if "OP_CARRIER" not in out.columns and "OP_UNIQUE_CARRIER" in out.columns:
        out["OP_CARRIER"] = out["OP_UNIQUE_CARRIER"]

    # Parse FL_DATE
    out["FL_DATE"] = pd.to_datetime(out["FL_DATE"], errors="coerce")

    # Standardize ORIGIN/DEST
    out["ORIGIN"] = out["ORIGIN"].astype("string").str.strip().str.upper()
    out["DEST"] = out["DEST"].astype("string").str.strip().str.upper()

    # Ensure numeric columns
    num_cols = [
        "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK",
        "CRS_DEP_TIME", "CRS_ARR_TIME", "DEP_TIME", "ARR_TIME",
        "DEP_DELAY", "ARR_DELAY",
        "CANCELLED", "DIVERTED",
        "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME", "DISTANCE",
        "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    ]
    for col in num_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Add missing columns the pipeline expects
    if "DEP_DELAY_NEW" not in out.columns:
        out["DEP_DELAY_NEW"] = out["DEP_DELAY"].clip(lower=0)
    if "ARR_DELAY_NEW" not in out.columns:
        out["ARR_DELAY_NEW"] = out["ARR_DELAY"].clip(lower=0)
    if "OP_CARRIER_FL_NUM" not in out.columns:
        out["OP_CARRIER_FL_NUM"] = np.nan

    return out


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    hours = np.floor(values / 100)
    minutes = values % 100
    invalid = (values < 0) | (hours > 23) | (minutes >= 60)
    result = (hours * 60 + minutes).astype("float64")
    result[invalid] = np.nan
    return result


def add_utc_event_times(flights: pd.DataFrame, tz_by_iata: dict[str, str]) -> pd.DataFrame:
    from zoneinfo import ZoneInfo

    out = flights.copy()
    out["CRS_DEP_MIN"] = hhmm_to_minutes(out["CRS_DEP_TIME"])
    out["DEP_LOCAL_DT"] = out["FL_DATE"] + pd.to_timedelta(out["CRS_DEP_MIN"], unit="m")
    out["EVENT_ORIGIN_UTC"] = pd.NaT

    print(f"  converting local→UTC for {out['ORIGIN'].nunique()} origins...")
    for airport, sub_idx in out.groupby("ORIGIN").indices.items():
        tz_name = tz_by_iata.get(airport)
        if not tz_name:
            continue
        tz = ZoneInfo(tz_name)

        def _to_utc(dt):
            if pd.isna(dt):
                return pd.NaT
            try:
                return dt.replace(tzinfo=tz).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
            except Exception:
                return pd.NaT

        out.iloc[sub_idx, out.columns.get_loc("EVENT_ORIGIN_UTC")] = (
            out.iloc[sub_idx]["DEP_LOCAL_DT"].apply(_to_utc).to_numpy()
        )

    out["EVENT_ORIGIN_UTC"] = pd.to_datetime(out["EVENT_ORIGIN_UTC"], errors="coerce").astype("datetime64[ns]")
    elapsed = pd.to_numeric(out["CRS_ELAPSED_TIME"], errors="coerce")
    out["EVENT_DEST_UTC"] = (
        out["EVENT_ORIGIN_UTC"] + pd.to_timedelta(elapsed, unit="m")
    ).astype("datetime64[ns]")
    return out


def add_flow_columns(flights: pd.DataFrame) -> pd.DataFrame:
    """Add FLOW_ATL and PAR_AIRPORT columns for schema compatibility."""
    flights["FLOW_ATL"] = np.where(
        flights["ORIGIN"].eq("ATL"), "DEP_FROM_ATL",
        np.where(flights["DEST"].eq("ATL"), "ARR_TO_ATL", "NON_ATL"),
    )
    flights["PAR_AIRPORT"] = np.where(
        flights["ORIGIN"].eq("ATL"), flights["DEST"],
        np.where(flights["DEST"].eq("ATL"), flights["ORIGIN"], ""),
    )
    return flights


def download_iem_for_airports(
    airports: set[str],
    tz_by_iata: dict[str, str],
    year: int = 2026,
    month_start: int = 1,
    month_end: int = 2,
) -> pd.DataFrame:
    """Download IEM weather for specific airports and month range."""
    import requests

    # Read IEM network mapping from universe
    uni = pd.read_csv(UNIVERSE_PATH)
    uni["iata"] = uni["iata"].astype(str).str.strip().str.upper()
    network_by_airport = dict(zip(uni["iata"], uni["iem_network"]))

    frames = []
    valid_airports = [a for a in sorted(airports) if a in network_by_airport and pd.notna(network_by_airport[a])]
    print(f"  downloading IEM for {len(valid_airports)} airports (Jan-Feb {year})...")

    for i, station in enumerate(valid_airports):
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(valid_airports)}...")

        network = network_by_airport[station]
        params = [
            ("network", network),
            ("station", station),
            ("year1", str(year)), ("month1", str(month_start)), ("day1", "1"),
            ("year2", str(year)), ("month2", str(month_end)), ("day2", "28"),
            ("tz", "Etc/UTC"), ("format", "onlycomma"),
            ("latlon", "no"), ("elev", "no"),
            ("missing", "M"), ("trace", "T"), ("direct", "no"),
            ("report_type", "3"), ("report_type", "4"),
        ] + [("data", v) for v in WX_VARS]

        try:
            r = requests.get(IEM_BASE, params=params, timeout=30)
            if r.status_code == 503:
                time.sleep(5)
                r = requests.get(IEM_BASE, params=params, timeout=30)
            r.raise_for_status()
            text = r.text
            lines = text.splitlines()
            hdr_idx = next((j for j, ln in enumerate(lines) if ln.lower().startswith("station,valid")), None)
            if hdr_idx is None:
                continue
            df = pd.read_csv(StringIO("\n".join(lines[hdr_idx:])))
            df.columns = [c.strip().lower() for c in df.columns]
            df["station"] = df["station"].astype("string").str.strip().str.upper()
            df["valid"] = pd.to_datetime(df["valid"], errors="coerce", utc=True).dt.tz_convert(None).astype("datetime64[ns]")
            for c in ["tmpc", "dwpc", "relh", "drct", "sknt", "alti", "p01m", "vsby", "gust"]:
                df[c] = pd.to_numeric(df[c].replace({"M": np.nan, "T": 0.001}), errors="coerce")
            df["wx_precip_flag"] = df["p01m"].fillna(0) > 0
            df["wx_low_vis_flag"] = df["vsby"] < 3
            df["wx_strong_wind_flag"] = (df["sknt"] >= 20) | (df["gust"] >= 30)
            keep = [
                "station", "valid", "tmpc", "dwpc", "relh", "drct", "sknt", "alti",
                "p01m", "vsby", "gust", "wxcodes",
                "wx_precip_flag", "wx_low_vis_flag", "wx_strong_wind_flag",
            ]
            frames.append(df[keep])
        except Exception as e:
            pass  # silently skip failed stations
        time.sleep(0.5)

    if not frames:
        return pd.DataFrame()
    wx = pd.concat(frames, ignore_index=True).drop_duplicates(["station", "valid"])
    wx = wx.sort_values(["station", "valid"]).reset_index(drop=True)
    print(f"  downloaded {len(wx):,} weather obs from {wx['station'].nunique()} stations")
    return wx


def merge_weather(flights: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    """Merge weather data using asof join (same logic as training pipeline)."""

    def asof_by_station(left, wx_df, station_col, event_col, prefix):
        nat_mask = left[event_col].isna() | left[station_col].isna()
        invalid = left[nat_mask].copy()
        valid_left = left[~nat_mask]

        out_parts = []
        unique_stations = sorted(valid_left[station_col].astype(str).unique())
        print(f"    merging {prefix} weather for {len(unique_stations)} stations...")
        for i, station in enumerate(unique_stations, 1):
            if i % 50 == 0:
                print(f"      {i}/{len(unique_stations)}")
            lsub = valid_left[valid_left[station_col].eq(station)].copy()
            wsub = wx_df[wx_df["station"].eq(station)].copy()
            if lsub.empty:
                continue
            if wsub.empty:
                out_parts.append(lsub)
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

        if not invalid.empty:
            out_parts.append(invalid)
        if not out_parts:
            return left

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
        return merged_all.drop(columns=["station"], errors="ignore")

    print("  ORIG weather merge...")
    origin_merge = asof_by_station(flights, wx, "ORIGIN", "EVENT_ORIGIN_UTC", "ORIG")
    print("  DEST weather merge...")
    return asof_by_station(origin_merge, wx, "DEST", "EVENT_DEST_UTC", "DEST")


def main() -> int:
    p = argparse.ArgumentParser(description="Build Jan-Feb 2026 BTS+IEM master dataset")
    p.add_argument("--skip-weather", action="store_true", help="Build without weather")
    p.add_argument("--atl-only", action="store_true", help="Only ATL-touching flights")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "dataset_maestro_2026_JAN_FEB.parquet")
    args = p.parse_args()

    # Load airports universe
    airports, tz_by_iata = load_universe()
    print(f"Universe: {len(airports)} airports")

    # Load BTS data
    bts_files = sorted(BTS_DIR.glob("*2026*raw.csv"))
    if not bts_files:
        print("ERROR: No BTS 2026 files found in data_raw/bts/")
        return 1

    frames = []
    for f in bts_files:
        print(f"\nLoading {f.name}...")
        df = pd.read_csv(f, low_memory=False)
        df = standardize_bts(df)
        # Filter to known airports
        mask = df["ORIGIN"].isin(airports) & df["DEST"].isin(airports)
        df = df[mask].copy()
        print(f"  {len(df):,} flights (both endpoints in universe)")
        frames.append(df)

    flights = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    print(f"\nTotal: {len(flights):,} flights")

    # Add flow columns
    flights = add_flow_columns(flights)
    is_atl = flights["ORIGIN"].eq("ATL") | flights["DEST"].eq("ATL")
    print(f"  ATL-touching: {is_atl.sum():,} ({is_atl.mean()*100:.1f}%)")

    if args.atl_only:
        flights = flights[is_atl].reset_index(drop=True)
        print(f"  filtered to ATL-only: {len(flights):,}")

    # Add UTC event times
    print("\nAdding UTC event timestamps...")
    flights = add_utc_event_times(flights, tz_by_iata)

    # Weather
    if not args.skip_weather:
        print("\nDownloading IEM weather for Jan-Feb 2026...")
        active_airports = set(flights["ORIGIN"].unique()) | set(flights["DEST"].unique())
        wx = download_iem_for_airports(active_airports, tz_by_iata, 2026, 1, 2)

        if not wx.empty:
            # Save weather for reuse
            IEM_DIR.mkdir(parents=True, exist_ok=True)
            wx_path = IEM_DIR / "clima_iem_asos_2026_jan_feb_utc.csv"
            wx.to_csv(wx_path, index=False)
            print(f"  saved weather to {wx_path.name}")

            print("\nMerging weather...")
            flights = merge_weather(flights, wx)
            orig_cov = flights.get("ORIG_WX_VALID_UTC", pd.Series(dtype="datetime64[ns]")).notna().mean()
            dest_cov = flights.get("DEST_WX_VALID_UTC", pd.Series(dtype="datetime64[ns]")).notna().mean()
            print(f"  ORIG wx coverage: {orig_cov:.1%}")
            print(f"  DEST wx coverage: {dest_cov:.1%}")
            del wx
            gc.collect()
        else:
            print("  no weather data downloaded — proceeding without")

    # Save
    print(f"\nSaving to {args.out.name}...")
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(flights, preserve_index=False)
    pq.write_table(table, args.out, compression="zstd")
    size_mb = args.out.stat().st_size / 1e6
    print(f"  → {args.out.name} ({size_mb:.1f} MB, {len(flights):,} rows × {len(flights.columns)} cols)")

    # Quick stats
    valid = flights[flights["ARR_DELAY"].notna() & (flights["CANCELLED"].fillna(0) == 0)]
    delay_rate = (valid["ARR_DELAY"] > 15).mean()
    print(f"\n  Valid flights: {len(valid):,}")
    print(f"  Delay rate (>15min): {delay_rate:.1%}")
    print(f"  Carriers: {flights['OP_CARRIER'].nunique()}")
    print(f"  Airports: {flights['ORIGIN'].nunique()} origins, {flights['DEST'].nunique()} dests")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

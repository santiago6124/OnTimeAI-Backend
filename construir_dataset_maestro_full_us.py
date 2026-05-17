"""Builds the BTS + IEM master dataset for ALL US airports (no ATL filter).

Adapted from `construir_dataset_maestro_multi.py`. Differences:

- AIRPORTS / TZ_BY_AIRPORT come from `airports_universe.csv` (built by
  `build_airports_universe.py`) instead of being hardcoded to 16 hubs.
- The `mask_atl` filter is removed — flights between any two US airports
  are kept, not just those touching ATL.
- `FLOW_ATL` and `PAR_AIRPORT` columns are kept (set to NaN-equivalent for
  flights that don't touch ATL) so the schema stays compatible with the
  existing model artifacts and the live pipeline.
- Output filename: `dataset_maestro_FULL_US_{year-range}_BTS_IEM.csv`.

Same columns as the existing master (65 cols) — drop-in replacement for
training the next model version.

Usage:
    python3 construir_dataset_maestro_full_us.py                    # default 2022-2025
    python3 construir_dataset_maestro_full_us.py 2022 2023 2024 2025
    python3 construir_dataset_maestro_full_us.py --out-format parquet 2022 2025
"""
from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
BTS_DIR = PROJECT_ROOT / "data_raw" / "bts"
IEM_DIR = PROJECT_ROOT / "data_raw" / "iem"
DEFAULT_UNIVERSE = PROJECT_ROOT / "airports_universe.csv"

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


def filter_paths_by_year(paths: list[Path], year: int, regex: re.Pattern) -> list[Path]:
    out = []
    for p in paths:
        m = regex.match(p.name) if hasattr(regex, "match") else regex.search(p.name)
        if not m:
            continue
        # Year is the LAST captured group in both BTS_FILE_RE and IEM_FILE_RE
        try:
            y = int(m.groups()[-1])
        except (ValueError, IndexError):
            continue
        if y == year:
            out.append(p)
    return out


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


def load_universe(path: Path) -> tuple[set[str], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Universe file not found at {path}. "
            f"Run `python3 build_airports_universe.py` first."
        )
    df = pd.read_csv(path)
    required = {"iata", "timezone"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Universe CSV missing columns: {missing}")
    df["iata"] = df["iata"].astype(str).str.strip().str.upper()
    airports = set(df["iata"].tolist())
    tz_by_iata = dict(zip(df["iata"], df["timezone"]))
    return airports, tz_by_iata


def load_all_weather(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        print(f"  weather <- {p.name} ", end="", flush=True)
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
        keep = [
            "station", "valid", "tmpc", "dwpc", "relh", "drct", "sknt", "alti",
            "p01m", "vsby", "gust", "wxcodes",
            "wx_precip_flag", "wx_low_vis_flag", "wx_strong_wind_flag",
        ]
        frames.append(wx[keep])
        print(f"({len(wx):,} rows)")
    weather = pd.concat(frames, ignore_index=True)
    weather = weather.drop_duplicates(subset=["station", "valid"])
    weather = weather.sort_values(["station", "valid"]).reset_index(drop=True)
    return weather


def load_bts_flights(paths: list[Path], airports: set[str]) -> pd.DataFrame:
    frames = []
    n_total = 0
    n_kept = 0
    for path in paths:
        print(f"  flights <- {path.name}", flush=True)
        for chunk in pd.read_csv(path, chunksize=250_000, low_memory=False):
            std = standardize_bts_chunk(chunk)
            n_total += len(std)
            # Keep flights where both endpoints are in the universe (no ATL filter)
            mask_known = std["ORIGIN"].isin(airports) & std["DEST"].isin(airports)
            sub = std.loc[mask_known].copy()
            n_kept += len(sub)
            if not sub.empty:
                frames.append(sub)
    if not frames:
        raise ValueError("No flights remaining after universe filter")
    flights = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    print(f"  kept {n_kept:,} of {n_total:,} flights "
          f"({n_kept/n_total*100:.1f}% — both endpoints in universe)")

    # Schema-compatible columns. FLOW_ATL and PAR_AIRPORT only meaningful for
    # ATL-touching flights; for all others we set them but they'll be ignored
    # by the model (or treated as a "non-ATL" category).
    is_atl = flights["ORIGIN"].eq("ATL") | flights["DEST"].eq("ATL")
    flights["FLOW_ATL"] = np.where(
        flights["ORIGIN"].eq("ATL"), "DEP_FROM_ATL",
        np.where(flights["DEST"].eq("ATL"), "ARR_TO_ATL", "NON_ATL"),
    )
    flights["PAR_AIRPORT"] = np.where(
        flights["ORIGIN"].eq("ATL"), flights["DEST"],
        np.where(flights["DEST"].eq("ATL"), flights["ORIGIN"], ""),
    )
    print(f"  ATL-touching: {is_atl.sum():,} ({is_atl.mean()*100:.1f}%); non-ATL: {(~is_atl).sum():,}")
    return flights


def add_utc_event_times(flights: pd.DataFrame, tz_by_iata: dict[str, str]) -> pd.DataFrame:
    out = flights.copy()
    out["CRS_DEP_MIN"] = hhmm_to_minutes(out["CRS_DEP_TIME"])
    out["DEP_LOCAL_DT"] = out["FL_DATE"] + pd.to_timedelta(out["CRS_DEP_MIN"], unit="m")
    out["EVENT_ORIGIN_UTC"] = pd.NaT

    # Group by origin so we apply each timezone once.
    print(f"  converting local→UTC for {out['ORIGIN'].nunique()} unique origins...")
    for airport, sub_idx in out.groupby("ORIGIN").indices.items():
        tz_name = tz_by_iata.get(airport)
        if not tz_name:
            continue  # leave as NaT, will be filtered downstream
        out.iloc[sub_idx, out.columns.get_loc("EVENT_ORIGIN_UTC")] = (
            local_series_to_utc_naive(out.iloc[sub_idx]["DEP_LOCAL_DT"], tz_name).to_numpy()
        )

    out["EVENT_ORIGIN_UTC"] = pd.to_datetime(out["EVENT_ORIGIN_UTC"], errors="coerce").astype("datetime64[ns]")
    elapsed = pd.to_numeric(out["CRS_ELAPSED_TIME"], errors="coerce")
    out["EVENT_DEST_UTC"] = (
        out["EVENT_ORIGIN_UTC"] + pd.to_timedelta(elapsed, unit="m")
    ).astype("datetime64[ns]")
    return out


def merge_weather(flights: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    def asof_by_station(left, wx_df, station_col, event_col, prefix):
        # `pd.merge_asof` requires the join key to be non-null. Some BTS rows
        # have invalid CRS_DEP_TIME (cancelled flights with corrupt schedules),
        # which produces NaT EVENT_ORIGIN_UTC / EVENT_DEST_UTC. Split those out
        # and re-attach with NaN weather columns at the end — they still carry
        # ARR_DELAY so they're useful for training.
        nat_mask = left[event_col].isna() | left[station_col].isna()
        invalid = left[nat_mask].copy()
        valid_left = left[~nat_mask]

        n_invalid = len(invalid)
        if n_invalid:
            print(f"    skipping {n_invalid:,} rows with NaT/NaN keys "
                  f"(will get NaN weather)")

        out_parts: list[pd.DataFrame] = []
        unique_stations = sorted(valid_left[station_col].astype(str).unique())
        print(f"    merging {prefix} weather for {len(unique_stations)} stations...")
        for i, station in enumerate(unique_stations, 1):
            if i % 25 == 0:
                print(f"      {i}/{len(unique_stations)}", flush=True)
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

        # Re-attach the NaT/NaN rows so we don't lose them.
        if not invalid.empty:
            out_parts.append(invalid)

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
        return merged_all.drop(columns=["station"], errors="ignore")

    print("  ORIG weather merge...")
    origin_merge = asof_by_station(flights, wx, "ORIGIN", "EVENT_ORIGIN_UTC", "ORIG")
    print("  DEST weather merge...")
    return asof_by_station(origin_merge, wx, "DEST", "EVENT_DEST_UTC", "DEST")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("years", nargs="*", type=int,
                   help="Years (default: 2022 2023 2024 2025)")
    p.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE,
                   help=f"Airports universe CSV (default: {DEFAULT_UNIVERSE.name})")
    p.add_argument("--out-format", choices=("csv", "parquet"), default="csv",
                   help="Output file format (parquet recommended for large datasets)")
    args = p.parse_args(argv)

    years = sorted(set(args.years)) if args.years else [2022, 2023, 2024, 2025]
    years_set = set(years)
    print(f"Target years: {years}")

    airports, tz_by_iata = load_universe(args.universe)
    print(f"Universe: {len(airports)} airports, {len(set(tz_by_iata.values()))} timezones")

    bts_paths = discover_bts_csvs(years_set)
    iem_paths = discover_iem_csvs(years_set)
    if not bts_paths:
        raise FileNotFoundError(
            f"No BTS CSVs for {years} in {BTS_DIR}. "
            f"Run `python3 descarga_bts.py {' '.join(map(str, years))}` first."
        )
    if not iem_paths:
        raise FileNotFoundError(
            f"No IEM CSVs for {years} in {IEM_DIR}. "
            f"Run `python3 descarga_data_iem_full.py {' '.join(map(str, years))}` first."
        )

    print(f"\nBTS inputs ({len(bts_paths)}):")
    for p in bts_paths:
        print(f"  {p.name}")
    print(f"\nIEM inputs ({len(iem_paths)}):")
    for p in iem_paths:
        print(f"  {p.name}")

    label = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])

    if args.out_format == "parquet":
        out_path = PROJECT_ROOT / f"dataset_maestro_FULL_US_{label}_BTS_IEM.parquet"
        return _build_streaming_parquet(
            years, bts_paths, iem_paths, airports, tz_by_iata, out_path,
        )

    # CSV fallback: also process by year and append, but no streaming write available.
    out_path = PROJECT_ROOT / f"dataset_maestro_FULL_US_{label}_BTS_IEM.csv"
    return _build_streaming_csv(
        years, bts_paths, iem_paths, airports, tz_by_iata, out_path,
    )


def _build_streaming_parquet(
    years: list[int],
    bts_paths: list[Path],
    iem_paths: list[Path],
    airports: set[str],
    tz_by_iata: dict[str, str],
    out_path: Path,
) -> int:
    """Process year-by-year and stream to a single parquet with ParquetWriter.

    Avoids holding all years in RAM at once (~12 GB) — peak per year is ~5-6 GB.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: pyarrow is required for streaming parquet output.")
        print("Install: pip install pyarrow")
        return 1

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    cov_orig: list[float] = []
    cov_dest: list[float] = []
    atl_total = 0

    try:
        for year in years:
            print(f"\n========== YEAR {year} ==========")
            iem_year = filter_paths_by_year(iem_paths, year, IEM_FILE_RE)
            bts_year = filter_paths_by_year(bts_paths, year, BTS_FILE_RE)

            if not iem_year:
                print(f"  [skip] no IEM file for {year}")
                continue
            if not bts_year:
                print(f"  [skip] no BTS files for {year}")
                continue

            print(f"  Loading weather for {year}...")
            wx = load_all_weather(iem_year)
            print(f"    {len(wx):,} obs / {wx['station'].nunique()} stations")

            print(f"  Loading flights for {year}...")
            flights = load_bts_flights(bts_year, airports)

            print(f"  Adding UTC event timestamps...")
            flights = add_utc_event_times(flights, tz_by_iata)

            print(f"  Merging weather...")
            master_year = merge_weather(flights, wx)
            n = len(master_year)
            print(f"  master {year}: {n:,} rows × {len(master_year.columns)} cols")

            cov_orig.append(master_year["ORIG_WX_VALID_UTC"].notna().mean())
            cov_dest.append(master_year["DEST_WX_VALID_UTC"].notna().mean())
            atl_total += int(((master_year["ORIGIN"] == "ATL") |
                              (master_year["DEST"] == "ATL")).sum())
            total_rows += n

            # Drop intermediate intermediates BEFORE pa.Table conversion to free RAM
            del wx, flights
            gc.collect()

            table = pa.Table.from_pandas(master_year, preserve_index=False)
            del master_year
            gc.collect()

            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
            else:
                # Schema unification: cast to first writer's schema to avoid mismatches
                # caused by per-year category differences (e.g. carrier set).
                try:
                    table = table.cast(writer.schema)
                except Exception:
                    print(f"  [warn] schema cast for {year} failed; writing as-is")
            writer.write_table(table)
            print(f"  wrote {year} chunk to {out_path.name}")
            del table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        print("\nERROR: no data written")
        return 1

    print(f"\n  -> {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  Total rows: {total_rows:,}")
    print(f"  ORIG wx coverage (avg across years): {sum(cov_orig)/len(cov_orig)*100:.2f}%")
    print(f"  DEST wx coverage (avg across years): {sum(cov_dest)/len(cov_dest)*100:.2f}%")
    print(f"  ATL-touching flights: {atl_total:,}")
    return 0


def _build_streaming_csv(
    years: list[int],
    bts_paths: list[Path],
    iem_paths: list[Path],
    airports: set[str],
    tz_by_iata: dict[str, str],
    out_path: Path,
) -> int:
    """Same year-by-year approach for CSV output: append mode, header only on first."""
    total_rows = 0
    out_path.unlink(missing_ok=True)
    first_year = True

    for year in years:
        print(f"\n========== YEAR {year} ==========")
        iem_year = filter_paths_by_year(iem_paths, year, IEM_FILE_RE)
        bts_year = filter_paths_by_year(bts_paths, year, BTS_FILE_RE)
        if not iem_year or not bts_year:
            print(f"  [skip] missing inputs for {year}")
            continue
        wx = load_all_weather(iem_year)
        flights = load_bts_flights(bts_year, airports)
        flights = add_utc_event_times(flights, tz_by_iata)
        master_year = merge_weather(flights, wx)
        n = len(master_year)
        master_year.to_csv(out_path, mode="a", header=first_year, index=False)
        total_rows += n
        first_year = False
        print(f"  appended {year}: {n:,} rows")
        del wx, flights, master_year
        gc.collect()

    print(f"\n  -> {out_path}  total {total_rows:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

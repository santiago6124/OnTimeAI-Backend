"""Downloads ASOS/METAR data from IEM for ALL airports in airports_universe.csv.

Same output layout as descarga_data_iem_multi.py: one CSV per year in
`data_raw/iem/clima_iem_asos_{year}_utc.csv`. Stations come from the universe
CSV (built by `build_airports_universe.py`) instead of the 16 hardcoded hubs.

Usage:
    python3 descarga_data_iem_full.py                      # default: 2022-2025
    python3 descarga_data_iem_full.py 2022 2023 2024 2025
    python3 descarga_data_iem_full.py --universe airports_universe.csv 2022 2025
    python3 descarga_data_iem_full.py --skip-existing-stations 2022 2025
"""
from __future__ import annotations

import argparse
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "data_raw" / "iem"
DEFAULT_UNIVERSE = PROJECT_ROOT / "airports_universe.csv"

BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
DATA_VARS = ["tmpc", "dwpc", "relh", "drct", "sknt", "alti", "p01m", "vsby", "gust", "wxcodes"]
SLEEP_BETWEEN_REQUESTS = 3.0   # courtesy pause for IEM (~370 stations × 4y → ~25 min/year)


def parse_iem_csv(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("station,valid"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("IEM response did not contain a CSV header line")
    df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
    df.columns = df.columns.str.strip()
    return df


def download_year(year: int, stations: list[dict], out_dir: Path,
                  skip_existing_stations: bool) -> Path:
    out_path = out_dir / f"clima_iem_asos_{year}_utc.csv"
    already_have: set[str] = set()
    if out_path.exists() and skip_existing_stations:
        try:
            existing = pd.read_csv(out_path, usecols=["station"])
            already_have = set(existing["station"].astype(str).str.upper().unique())
            print(f"  [resume] {len(already_have)} stations already in {out_path.name}")
        except Exception:
            already_have = set()

    frames: list[pd.DataFrame] = []
    if out_path.exists() and skip_existing_stations:
        # we'll append, so seed with what's there
        try:
            existing = pd.read_csv(out_path, low_memory=False)
            frames.append(existing)
        except Exception:
            pass

    n_done = 0
    n_total = len(stations)
    for idx, cfg in enumerate(stations, start=1):
        station = cfg["iata"]
        network = cfg["iem_network"]
        if station in already_have:
            continue

        print(f"  [{idx}/{n_total}] {station} ({network}) {year}", flush=True)
        params = [
            ("network", network),
            ("station", station),
            ("year1", str(year)), ("month1", "1"), ("day1", "1"),
            ("year2", str(year + 1)), ("month2", "1"), ("day2", "1"),
            ("tz", "Etc/UTC"),
            ("format", "onlycomma"),
            ("latlon", "no"), ("elev", "no"),
            ("missing", "M"), ("trace", "T"), ("direct", "no"),
            ("report_type", "3"), ("report_type", "4"),
        ] + [("data", v) for v in DATA_VARS]

        try:
            r = requests.get(BASE_URL, params=params, timeout=180)
            r.raise_for_status()
            df = parse_iem_csv(r.text)
            if df.empty or len(df.columns) <= 2:
                print(f"           [empty]")
            else:
                df["source_station"] = station
                df["source_network"] = network
                frames.append(df)
                n_done += 1
                print(f"           {len(df)} rows")
        except Exception as e:
            print(f"           [FAIL] {e}")
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        # Periodic checkpoint write so we don't lose all progress on crash
        if idx % 20 == 0 and frames:
            partial = pd.concat(frames, ignore_index=True)
            partial.to_csv(out_path, index=False)
            print(f"           [checkpoint] saved {len(partial):,} rows so far")

    if not frames:
        raise RuntimeError(f"No data downloaded for {year}")

    df_year = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["station", "valid"], keep="last"
    )
    df_year.to_csv(out_path, index=False)
    print(f"  -> {out_path} ({len(df_year):,} rows, {df_year['station'].nunique()} stations)")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("years", nargs="*", type=int,
                   help="Years to download (default: 2022 2023 2024 2025)")
    p.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE,
                   help=f"Airports universe CSV (default: {DEFAULT_UNIVERSE.name})")
    p.add_argument("--skip-existing-stations", action="store_true",
                   help="If output file exists, skip stations already present (resume mode)")
    args = p.parse_args(argv)

    years = sorted(set(args.years)) if args.years else [2022, 2023, 2024, 2025]

    if not args.universe.exists():
        print(f"ERROR: airports universe not found: {args.universe}")
        print("Run `python3 build_airports_universe.py` first.")
        return 1

    universe = pd.read_csv(args.universe)
    n_total = len(universe)
    if "iem_validated" in universe.columns:
        universe = universe[universe["iem_validated"].astype(bool)].copy()
        print(f"Filtered to {len(universe)} IEM-validated airports (of {n_total} in universe)")
    else:
        print(f"WARNING: universe CSV has no `iem_validated` column. "
              f"Will attempt all {n_total} airports — expect ~10-15% empty responses. "
              f"Re-run `build_airports_universe.py` to add validation.")
    stations = universe.to_dict("records")
    print(f"Years: {years}")
    print(f"Output dir: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for y in years:
        print(f"\n=== {y} ===")
        download_year(y, stations, OUT_DIR, args.skip_existing_stations)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

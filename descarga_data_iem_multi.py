"""Downloads ASOS/METAR data from Iowa Environmental Mesonet (IEM) for multiple years.

Reuses the station list from the single-year script and produces one CSV per year:
    data_raw/iem/clima_iem_asos_{year}_utc.csv

Usage:
    python3 descarga_data_iem_multi.py                  # default: 2022 2023 2024
    python3 descarga_data_iem_multi.py 2022 2023 2024   # explicit years
"""
from __future__ import annotations

import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path(__file__).resolve().parent / "data_raw" / "iem"

AIRPORTS_CONFIG = [
    {"network": "GA_ASOS", "station": "ATL"},
    {"network": "NY_ASOS", "station": "LGA"},
    {"network": "FL_ASOS", "station": "MCO"},
    {"network": "FL_ASOS", "station": "FLL"},
    {"network": "FL_ASOS", "station": "MIA"},
    {"network": "TX_ASOS", "station": "DFW"},
    {"network": "DC_ASOS", "station": "DCA"},
    {"network": "NJ_ASOS", "station": "EWR"},
    {"network": "FL_ASOS", "station": "TPA"},
    {"network": "IL_ASOS", "station": "ORD"},
    {"network": "PA_ASOS", "station": "PHL"},
    {"network": "CO_ASOS", "station": "DEN"},
    {"network": "CA_ASOS", "station": "LAX"},
    {"network": "MD_ASOS", "station": "BWI"},
    {"network": "NV_ASOS", "station": "LAS"},
    {"network": "MA_ASOS", "station": "BOS"},
]

BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
DATA_VARS = ["tmpc", "dwpc", "relh", "drct", "sknt", "alti", "p01m", "vsby", "gust", "wxcodes"]


def parse_iem_csv(response_text: str) -> pd.DataFrame:
    lines = response_text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("station,valid"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("IEM response did not contain a CSV header line")
    csv_data = StringIO("\n".join(lines[header_idx:]))
    df = pd.read_csv(csv_data)
    df.columns = df.columns.str.strip()
    return df


def download_year(year: int, out_dir: Path) -> Path:
    out_path = out_dir / f"clima_iem_asos_{year}_utc.csv"
    if out_path.exists():
        print(f"  [skip] {out_path.name} exists")
        return out_path

    frames = []
    for cfg in AIRPORTS_CONFIG:
        station = cfg["station"]
        network = cfg["network"]
        print(f"  [get ] {station} ({network}) {year}", flush=True)
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
            df["source_station"] = station
            df["source_network"] = network
            frames.append(df)
            print(f"         {len(df)} rows")
        except Exception as e:
            print(f"         [FAIL] {e}")
        time.sleep(2)

    if not frames:
        raise RuntimeError(f"No data downloaded for {year}")

    df_year = pd.concat(frames, ignore_index=True)
    df_year.to_csv(out_path, index=False)
    print(f"  -> {out_path} ({len(df_year)} rows)")
    return out_path


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    years = [int(a) for a in argv] if argv else [2022, 2023, 2024]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading IEM ASOS for years: {years}")
    print(f"Output dir: {OUT_DIR}")
    for y in years:
        print(f"\n=== {y} ===")
        download_year(y, OUT_DIR)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

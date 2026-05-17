"""Downloads monthly BTS On-Time Performance CSVs from the PREZIP endpoint.

Usage:
    python3 descarga_bts.py                  # default: 2022 2023 2024
    python3 descarga_bts.py 2022 2023 2024   # explicit years
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import requests

BASE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
MONTH_NAMES = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
OUT_DIR = Path(__file__).resolve().parent / "data_raw" / "bts"


def download_one(year: int, month: int, keep_zip: bool = False) -> Path:
    zip_path = OUT_DIR / f"bts_{year}_{month:02d}.zip"
    csv_target = OUT_DIR / f"{MONTH_NAMES[month - 1]}_{year}_raw.csv"

    if csv_target.exists():
        print(f"  [skip] {csv_target.name} exists")
        return csv_target

    if not zip_path.exists():
        url = BASE_URL.format(year=year, month=month)
        print(f"  [get ] {url}")
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        size_mb = zip_path.stat().st_size / 1e6
        print(f"         zip {size_mb:.1f} MB")

    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if not names:
            raise RuntimeError(f"No CSV in {zip_path.name}")
        with z.open(names[0]) as src, open(csv_target, "wb") as dst:
            while True:
                buf = src.read(1024 * 1024)
                if not buf:
                    break
                dst.write(buf)
    csv_mb = csv_target.stat().st_size / 1e6
    print(f"         csv {csv_mb:.1f} MB -> {csv_target.name}")

    if not keep_zip:
        zip_path.unlink(missing_ok=True)

    return csv_target


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    years = [int(a) for a in argv] if argv else [2022, 2023, 2024]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading BTS On-Time Performance for years: {years}")
    print(f"Output dir: {OUT_DIR}")

    for year in years:
        print(f"\n=== {year} ===")
        for month in range(1, 13):
            try:
                download_one(year, month)
            except Exception as e:
                print(f"  [FAIL] {year}-{month:02d}: {e}")
            time.sleep(1)

    csvs = sorted(OUT_DIR.glob("*_*_raw.csv"))
    total_mb = sum(p.stat().st_size for p in csvs) / 1e6
    print(f"\nDone. {len(csvs)} CSVs, {total_mb:.0f} MB total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

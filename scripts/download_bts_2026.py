"""Download BTS On-Time Performance data for 2025-2026 and build an incremental dataset.

Downloads the latest available BTS months (up to Feb 2026 as of May 2026) plus
corresponding IEM weather data, then builds a supplementary master dataset that
can be used to:
  1. Evaluate the v9 model on recent ground truth (no AeroAPI needed)
  2. Extend the training corpus for v10

Usage:
    python3 scripts/download_bts_2026.py                  # default: 2025 Jan-Jun + 2026 Jan-Feb
    python3 scripts/download_bts_2026.py --years 2026     # only 2026
    python3 scripts/download_bts_2026.py --months 1 2     # only Jan-Feb
    python3 scripts/download_bts_2026.py --skip-weather    # skip IEM download
    python3 scripts/download_bts_2026.py --skip-build      # download only, don't build master
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from descarga_bts import download_one, OUT_DIR as BTS_DIR
from descarga_data_iem_full import main as download_iem_main


def main() -> int:
    p = argparse.ArgumentParser(description="Download BTS 2025-2026 + IEM weather")
    p.add_argument("--years", nargs="*", type=int, default=[2025, 2026])
    p.add_argument("--months", nargs="*", type=int, default=None,
                   help="Specific months (1-12). Default: all available")
    p.add_argument("--skip-weather", action="store_true")
    p.add_argument("--skip-build", action="store_true")
    args = p.parse_args()

    BTS_DIR.mkdir(parents=True, exist_ok=True)

    # BTS data has ~3 month lag. As of May 2026, Feb 2026 is the latest.
    # For 2025, all 12 months should be available.
    max_months = {2025: 12, 2026: 2}

    total_downloaded = 0
    for year in args.years:
        max_m = max_months.get(year, 12)
        months = args.months or list(range(1, max_m + 1))
        months = [m for m in months if m <= max_m]

        print(f"\n=== BTS {year} (months {months}) ===")
        for month in months:
            try:
                csv_path = download_one(year, month)
                total_downloaded += 1
                print(f"  ✅ {csv_path.name}")
            except Exception as e:
                print(f"  ❌ {year}-{month:02d}: {e}")
            time.sleep(2)

    print(f"\nDownloaded {total_downloaded} BTS CSVs to {BTS_DIR}")

    # Download IEM weather for the same years
    if not args.skip_weather:
        print("\n=== IEM Weather ===")
        for year in args.years:
            print(f"\nDownloading IEM weather for {year}...")
            try:
                download_iem_main([str(year)])
            except Exception as e:
                print(f"  ❌ IEM {year}: {e}")

    # Build supplementary master
    if not args.skip_build and total_downloaded > 0:
        print("\n=== Building supplementary master ===")
        try:
            from construir_dataset_maestro_full_us import main as build_main
            year_args = [str(y) for y in args.years]
            build_main(["--out-format", "parquet"] + year_args)
        except Exception as e:
            print(f"  ❌ Build failed: {e}")
            print("  You can manually run:")
            print(f"    python3 construir_dataset_maestro_full_us.py --out-format parquet {' '.join(str(y) for y in args.years)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

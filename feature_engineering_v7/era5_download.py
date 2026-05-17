"""Descarga medias mensuales ERA5 de viento a 250 hPa para US.

Requiere:
  1. pip install cdsapi netCDF4
  2. Cuenta gratuita en https://cds.climate.copernicus.eu/
  3. Archivo ~/.cdsapirc con tu UID y API key:
       url: https://cds.climate.copernicus.eu/api
       key: <tu-uid>:<tu-api-key>

Descarga:
  - Variables: u-component of wind (u), v-component of wind (v)
  - Nivel de presión: 250 hPa  (altitud de crucero típica ~FL340-FL380)
  - Resolución: 0.25° × 0.25° (≈ 27 km sobre EEUU)
  - Bounding box: 20°N–55°N, 130°W–60°W  (contiguous US + Canada sur)
  - Período: 2022-01 a 2025-12  →  48 archivos, ~8 MB c/u ≈ 380 MB total
  - Tipo: monthly means (reanalysis-era5-pressure-levels-monthly-means)

Uso:
    python3 feature_engineering_v7/era5_download.py
    python3 feature_engineering_v7/era5_download.py --years 2022 2023
    python3 feature_engineering_v7/era5_download.py --out data_raw/era5_wind

Cada archivo se guarda como era5_wind_<YYYY>_<MM>.nc y se omite si ya existe.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

ERA5_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "data_raw" / "era5_wind"
YEARS_DEFAULT = list(range(2022, 2026))
MONTHS_ALL = [f"{m:02d}" for m in range(1, 13)]

# Bounding box: [norte, oeste, sur, este] en convención CDS
BBOX_US = [55, -130, 20, -60]
PRESSURE_LEVEL = "250"
VARIABLES = ["u_component_of_wind", "v_component_of_wind"]


def download_month(year: int, month: str, out_dir: Path) -> Path:
    """Descarga el archivo ERA5 mensual para un año/mes dado. Retorna la ruta."""
    import cdsapi
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"era5_wind_{year}_{month}.nc"
    if out_path.exists():
        print(f"  ya existe: {out_path.name}")
        return out_path

    c = cdsapi.Client(quiet=True)
    print(f"  descargando ERA5 {year}-{month}...", end=" ", flush=True)
    t0 = time.time()
    c.retrieve(
        "reanalysis-era5-pressure-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": VARIABLES,
            "pressure_level": PRESSURE_LEVEL,
            "year": str(year),
            "month": month,
            "time": "00:00",  # monthly mean no depende de la hora
            "area": BBOX_US,
            "format": "netcdf",
        },
        str(out_path),
    )
    print(f"OK ({time.time()-t0:.0f}s, {out_path.stat().st_size/1e6:.1f} MB)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int, default=YEARS_DEFAULT)
    p.add_argument("--months", nargs="+", default=MONTHS_ALL)
    p.add_argument("--out", default=str(ERA5_DIR_DEFAULT))
    args = p.parse_args()

    out_dir = Path(args.out)
    total = len(args.years) * len(args.months)
    done = 0
    errors = []

    print(f"Descargando ERA5 250hPa wind — {total} archivos → {out_dir}")
    print(f"Bounding box: {BBOX_US}  |  variables: {', '.join(VARIABLES)}")
    print()

    for year in sorted(args.years):
        for month in sorted(args.months):
            try:
                download_month(year, month, out_dir)
                done += 1
            except Exception as e:
                print(f"  ERROR {year}-{month}: {e}")
                errors.append(f"{year}-{month}: {e}")

    print(f"\nListo: {done}/{total} archivos descargados en {out_dir}")
    if errors:
        print(f"Errores ({len(errors)}):")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()

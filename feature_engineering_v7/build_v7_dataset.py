"""Construye el dataset maestro con todas las features nuevas para v7.

Parte de: dataset_maestro_FULL_US_2022-2025_BTS_IEM.parquet  (69 cols, 27.5M vuelos)
Agrega:
  Fase 1 — Bearing (disponible ahora):
    BEARING_DEG

  Fase 2 — Viento ERA5 250hPa (requiere era5_download.py ejecutado):
    ERA5_U_KT, ERA5_V_KT, ERA5_HEADWIND_KT, ERA5_TAILWIND_FLAG, ERA5_CROSSWIND_KT

  Fase 3 — ADS-B vuelo anterior (requiere --setup de adsb_lineage.py):
    PREV_ACTUAL_BLOCK_MIN, PREV_BLOCK_DELTA_MIN, PREV_HOLDING_MIN,
    PREV_ROUTE_DEVIATION_PCT, PREV_ADSB_AVAILABLE

Salida: dataset_maestro_FULL_US_2022-2025_v7.parquet

Uso:
    # Solo bearing (sin ERA5 ni ADS-B):
    python3 feature_engineering_v7/build_v7_dataset.py --skip-era5 --skip-adsb

    # Con ERA5 (después de era5_download.py):
    python3 feature_engineering_v7/build_v7_dataset.py --skip-adsb

    # Completo (después de era5_download.py y adsb_lineage.py --setup):
    python3 feature_engineering_v7/build_v7_dataset.py

    # Test rápido en muestra:
    python3 feature_engineering_v7/build_v7_dataset.py --sample 500000 --skip-adsb
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engineering_v7.airport_lookup import great_circle_bearing
from feature_engineering_v7.enroute_wind import add_enroute_wind_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PARQUET = PROJECT_ROOT / "dataset_maestro_FULL_US_2022-2025_BTS_IEM.parquet"
ERA5_DIR = PROJECT_ROOT / "data_raw" / "era5_wind"


def _add_bearing(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega BEARING_DEG calculado por par único ORIGIN-DEST."""
    pairs = df[["ORIGIN", "DEST"]].drop_duplicates().copy()
    pairs["BEARING_DEG"] = pairs.apply(
        lambda r: great_circle_bearing(r["ORIGIN"], r["DEST"]), axis=1
    )
    return df.merge(pairs, on=["ORIGIN", "DEST"], how="left")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(SRC_PARQUET))
    p.add_argument("--out", default=str(PROJECT_ROOT / "dataset_maestro_FULL_US_2022-2025_v7.parquet"))
    p.add_argument("--era5-dir", default=str(ERA5_DIR))
    p.add_argument("--sample", type=int, default=0,
                   help="Si > 0, procesa solo N vuelos (para test rápido)")
    p.add_argument("--skip-era5", action="store_true",
                   help="Omitir features ERA5 (si aún no descargaste los datos)")
    p.add_argument("--skip-adsb", action="store_true",
                   help="Omitir features ADS-B (si aún no corriste adsb_lineage --setup)")
    p.add_argument("--chunk-size", type=int, default=500_000,
                   help="Vuelos por chunk en memoria (default 500k)")
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    print(f"Dataset fuente : {src}")
    print(f"Dataset destino: {out}")
    print(f"ERA5            : {'SKIP' if args.skip_era5 else args.era5_dir}")
    print(f"ADS-B           : {'SKIP' if args.skip_adsb else 'habilitado'}")
    print()

    # ---- cargar ----
    t0 = time.time()
    print("Cargando dataset fuente...")
    df_full = pq.read_table(src).to_pandas()
    print(f"  {len(df_full):,} vuelos, {len(df_full.columns)} columnas en {time.time()-t0:.0f}s")

    if args.sample:
        df_full = df_full.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"  Muestra reducida a {len(df_full):,} vuelos")

    # ---- Fase 1: Bearing ----
    print("\n[Fase 1] Calculando BEARING_DEG...")
    t1 = time.time()
    df_full = _add_bearing(df_full)
    coverage = df_full["BEARING_DEG"].notna().mean()
    print(f"  Cobertura: {coverage:.1%}  ({time.time()-t1:.0f}s)")

    # ---- Fase 2: ERA5 viento ----
    if not args.skip_era5:
        era5_dir = Path(args.era5_dir)
        if not era5_dir.exists() or not list(era5_dir.glob("era5_wind_*.nc")):
            print(f"\n[Fase 2] SKIP — no se encontraron archivos ERA5 en {era5_dir}")
            print("  Ejecutá: python3 feature_engineering_v7/era5_download.py")
        else:
            print(f"\n[Fase 2] Computando features ERA5 viento 250hPa...")
            t2 = time.time()
            from feature_engineering_v7.era5_wind_features import add_era5_wind_features
            df_full = add_era5_wind_features(df_full, era5_dir=era5_dir)
            cov = df_full["ERA5_HEADWIND_KT"].notna().mean()
            print(f"  Cobertura ERA5_HEADWIND_KT: {cov:.1%}  ({time.time()-t2:.0f}s)")
    else:
        print("\n[Fase 2] ERA5 — skipped (--skip-era5)")
        for c in ["ERA5_U_KT", "ERA5_V_KT", "ERA5_HEADWIND_KT",
                  "ERA5_TAILWIND_FLAG", "ERA5_CROSSWIND_KT"]:
            df_full[c] = np.nan
        df_full["ERA5_TAILWIND_FLAG"] = 0

    # ---- Fase 3: ADS-B ----
    if not args.skip_adsb:
        print("\n[Fase 3] ADS-B features del vuelo anterior...")
        try:
            from feature_engineering_v7.adsb_lineage import (
                build_tail_icao24_map, add_adsb_lineage_features,
                FAA_CSV_PATH,
            )
            if not FAA_CSV_PATH.exists():
                print(f"  FAA registry no encontrado. Ejecutá: python3 feature_engineering_v7/adsb_lineage.py --setup")
            else:
                tail_map = build_tail_icao24_map()
                df_full = add_adsb_lineage_features(df_full, tail_map)
                hit = df_full.get("PREV_ADSB_AVAILABLE", pd.Series(0)).sum()
                print(f"  ADS-B hit rate: {hit}/{len(df_full)} ({100*hit/max(1,len(df_full)):.1f}%)")
        except Exception as e:
            print(f"  WARN ADS-B: {e}")
    else:
        print("\n[Fase 3] ADS-B — skipped (--skip-adsb)")
        for c in ["PREV_ACTUAL_BLOCK_MIN", "PREV_SCHED_BLOCK_MIN", "PREV_BLOCK_DELTA_MIN",
                  "PREV_HOLDING_MIN", "PREV_ROUTE_DEVIATION_PCT"]:
            df_full[c] = np.nan
        df_full["PREV_ADSB_AVAILABLE"] = 0

    # ---- guardar ----
    print(f"\nGuardando → {out}")
    print(f"  {len(df_full):,} vuelos, {len(df_full.columns)} columnas")
    t3 = time.time()
    df_full.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    print(f"  OK en {time.time()-t3:.0f}s  ({out.stat().st_size/1e9:.2f} GB)")

    print(f"\n=== Resumen de columnas nuevas ===")
    new_cols = ["BEARING_DEG", "ERA5_U_KT", "ERA5_V_KT", "ERA5_HEADWIND_KT",
                "ERA5_TAILWIND_FLAG", "ERA5_CROSSWIND_KT",
                "PREV_BLOCK_DELTA_MIN", "PREV_HOLDING_MIN",
                "PREV_ROUTE_DEVIATION_PCT", "PREV_ADSB_AVAILABLE"]
    for col in new_cols:
        if col in df_full.columns:
            nn = df_full[col].notna().sum()
            print(f"  {col:<30}: {nn:>8,} non-null ({100*nn/len(df_full):.1f}%)")

    total = time.time() - t0
    print(f"\nTiempo total: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()

"""Computa features de viento real en crucero (ERA5 250hPa) por vuelo.

Requiere que era5_download.py ya haya descargado los archivos .nc en
data_raw/era5_wind/.

Columnas nuevas:
  BEARING_DEG         : heading gran círculo ORIGIN→DEST [°]
  ERA5_U_KT           : viento zonal (E-O) en midpoint de ruta [kt]
  ERA5_V_KT           : viento meridional (N-S) en midpoint de ruta [kt]
  ERA5_HEADWIND_KT    : componente headwind sobre la ruta [kt]
  ERA5_TAILWIND_FLAG  : 1 si headwind < -15 kt (cola significativa)
  ERA5_CROSSWIND_KT   : componente transversal [kt] (referencia)

Las features se calculan interpolando el campo ERA5 mensual al midpoint
de cada ruta (lat, lon) y proyectando el vector viento sobre el bearing.
Un mes es suficiente para capturar el ciclo estacional del jet stream.

Uso como módulo:
    from feature_engineering_v7.era5_wind_features import add_era5_wind_features
    df = add_era5_wind_features(df, era5_dir="data_raw/era5_wind")

Uso standalone (verifica correlaciones):
    python3 feature_engineering_v7/era5_wind_features.py --sample 200000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engineering_v7.airport_lookup import great_circle_bearing, airport_coords

ERA5_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "data_raw" / "era5_wind"

# 1 m/s = 1.94384 kt
MPS_TO_KT = 1.94384


class Era5WindGrid:
    """Carga y mantiene en memoria los campos ERA5 de un mes dado.

    El campo tiene dimensiones (lat, lon). Usa RegularGridInterpolator
    de scipy para bilinear interpolation al punto de consulta.
    """

    def __init__(self, nc_path: Path) -> None:
        ds = nc.Dataset(str(nc_path))
        # ERA5 monthly: dims = (time=1, pressure_level=1, lat, lon)
        self.lat = ds.variables["latitude"][:].data.astype(float)
        self.lon = ds.variables["longitude"][:].data.astype(float)
        # Tomar primer (y único) time step
        u_raw = ds.variables["u"][0, 0, :, :].data.astype(float)
        v_raw = ds.variables["v"][0, 0, :, :].data.astype(float)
        ds.close()

        # Asegurar latitudes ascendentes (ERA5 puede venir de 90→-90)
        if self.lat[0] > self.lat[-1]:
            self.lat = self.lat[::-1]
            u_raw = u_raw[::-1, :]
            v_raw = v_raw[::-1, :]

        # Longitudes: ERA5 usa 0→360, convertir a -180→180
        if self.lon.max() > 180:
            shift = self.lon > 180
            self.lon[shift] -= 360
            # Re-ordenar para que lon sea ascendente
            order = np.argsort(self.lon)
            self.lon = self.lon[order]
            u_raw = u_raw[:, order]
            v_raw = v_raw[:, order]

        self._u_interp = RegularGridInterpolator(
            (self.lat, self.lon), u_raw, method="linear", bounds_error=False,
            fill_value=None,
        )
        self._v_interp = RegularGridInterpolator(
            (self.lat, self.lon), v_raw, method="linear", bounds_error=False,
            fill_value=None,
        )

    def wind_at(self, lat: float, lon: float) -> tuple[float, float]:
        """Retorna (u_kt, v_kt) interpolados al punto (lat, lon)."""
        pts = np.array([[lat, lon]])
        u = float(self._u_interp(pts)[0]) * MPS_TO_KT
        v = float(self._v_interp(pts)[0]) * MPS_TO_KT
        return u, v


def _midpoint(origin: str, dest: str) -> tuple[float, float] | None:
    """Retorna (lat, lon) del punto medio de la ruta de gran círculo."""
    co = airport_coords(origin)
    cd = airport_coords(dest)
    if co is None or cd is None:
        return None
    lat1, lon1 = np.radians(co[0]), np.radians(co[1])
    lat2, lon2 = np.radians(cd[0]), np.radians(cd[1])
    Bx = np.cos(lat2) * np.cos(lon2 - lon1)
    By = np.cos(lat2) * np.sin(lon2 - lon1)
    lat_m = np.arctan2(
        np.sin(lat1) + np.sin(lat2),
        np.sqrt((np.cos(lat1) + Bx) ** 2 + By ** 2),
    )
    lon_m = lon1 + np.arctan2(By, np.cos(lat1) + Bx)
    return float(np.degrees(lat_m)), float(np.degrees(lon_m))


def _load_grids(era5_dir: Path) -> dict[tuple[int, int], Era5WindGrid]:
    """Carga todos los archivos .nc en era5_dir, indexados por (año, mes)."""
    grids: dict[tuple[int, int], Era5WindGrid] = {}
    for p in sorted(era5_dir.glob("era5_wind_*.nc")):
        parts = p.stem.split("_")   # era5_wind_2022_01
        if len(parts) >= 4:
            year, month = int(parts[2]), int(parts[3])
            try:
                grids[(year, month)] = Era5WindGrid(p)
            except Exception as e:
                print(f"  WARN: no se pudo cargar {p.name}: {e}")
    return grids


def add_era5_wind_features(
    df: pd.DataFrame,
    era5_dir: str | Path = ERA5_DIR_DEFAULT,
) -> pd.DataFrame:
    """Agrega columnas ERA5 de viento en crucero al DataFrame de vuelos.

    Espera columnas: ORIGIN, DEST, YEAR, MONTH
    Retorna el DataFrame con columnas nuevas agregadas.
    """
    era5_dir = Path(era5_dir)
    if not era5_dir.exists():
        raise FileNotFoundError(
            f"Directorio ERA5 no encontrado: {era5_dir}\n"
            "Ejecutá primero: python3 feature_engineering_v7/era5_download.py"
        )

    print(f"Cargando grids ERA5 desde {era5_dir}...")
    grids = _load_grids(era5_dir)
    if not grids:
        raise RuntimeError(f"No se encontraron archivos era5_wind_*.nc en {era5_dir}")
    print(f"  {len(grids)} archivos mensuales cargados")

    # Pre-computar midpoints y bearings por par único ORIGIN-DEST
    # Usamos _MIDPOINT y _ERA5_BEARING para no colisionar con BEARING_DEG
    # que puede ya existir en df (agregado por build_v7_dataset Fase 1)
    pairs = df[["ORIGIN", "DEST"]].drop_duplicates().copy()
    pairs["_MIDPOINT"] = pairs.apply(
        lambda r: _midpoint(r["ORIGIN"], r["DEST"]), axis=1
    )
    pairs["_ERA5_BEARING"] = pairs.apply(
        lambda r: great_circle_bearing(r["ORIGIN"], r["DEST"]), axis=1
    )

    # Computar viento por par × mes
    # Agrupamos por (ORIGIN, DEST, YEAR, MONTH) para minimizar interpolaciones
    df = df.merge(pairs, on=["ORIGIN", "DEST"], how="left")

    n = len(df)
    era5_u = np.full(n, np.nan)
    era5_v = np.full(n, np.nan)

    groups = df.groupby(["ORIGIN", "DEST", "YEAR", "MONTH"]).indices

    for (origin, dest, year, month), idx in groups.items():
        midpt = df.at[idx[0], "_MIDPOINT"]
        bearing = df.at[idx[0], "_ERA5_BEARING"]
        if midpt is None or bearing is None:
            continue

        # Buscar grid del mes; si no hay, buscar el mes más cercano disponible
        grid = grids.get((int(year), int(month)))
        if grid is None:
            # Fallback: año disponible más cercano con el mismo mes
            candidates = [(abs(y - int(year)), g) for (y, m), g in grids.items() if m == int(month)]
            if candidates:
                grid = min(candidates, key=lambda x: x[0])[1]
        if grid is None:
            continue

        u_kt, v_kt = grid.wind_at(midpt[0], midpt[1])
        era5_u[list(idx)] = u_kt
        era5_v[list(idx)] = v_kt

    df = df.drop(columns=["_MIDPOINT"])

    df["ERA5_U_KT"] = era5_u
    df["ERA5_V_KT"] = era5_v

    bearing_rad = np.radians(df["_ERA5_BEARING"].values.astype(float))
    flight_e = np.sin(bearing_rad)
    flight_n = np.cos(bearing_rad)
    tailwind = era5_u * flight_e + era5_v * flight_n
    df["ERA5_HEADWIND_KT"] = -tailwind
    df["ERA5_CROSSWIND_KT"] = (era5_u * flight_n - era5_v * flight_e)
    df["ERA5_TAILWIND_FLAG"] = (df["ERA5_HEADWIND_KT"] < -15).astype("int8")

    # Promover _ERA5_BEARING a BEARING_DEG solo si no existe ya en df
    if "BEARING_DEG" not in df.columns:
        df = df.rename(columns={"_ERA5_BEARING": "BEARING_DEG"})
    else:
        df = df.drop(columns=["_ERA5_BEARING"])

    return df


# ------------------------------------------------------------------ #
# Standalone: verifica distribución y correlación con ARR_DELAY        #
# ------------------------------------------------------------------ #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet",
                   default=str(Path(__file__).resolve().parents[1] /
                               "dataset_maestro_FULL_US_2022-2025_BTS_IEM.parquet"))
    p.add_argument("--era5-dir", default=str(ERA5_DIR_DEFAULT))
    p.add_argument("--sample", type=int, default=200_000)
    args = p.parse_args()

    import pyarrow.parquet as pq

    needed = ["ORIGIN", "DEST", "YEAR", "MONTH", "ARR_DELAY",
              "DISTANCE", "CRS_ELAPSED_TIME"]
    df = pq.read_table(args.parquet, columns=needed).to_pandas()
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=42)

    print(f"Muestra: {len(df):,} vuelos")
    df = add_era5_wind_features(df, era5_dir=args.era5_dir)

    print(f"\n=== Cobertura ===")
    for col in ["BEARING_DEG", "ERA5_U_KT", "ERA5_V_KT",
                "ERA5_HEADWIND_KT", "ERA5_TAILWIND_FLAG"]:
        nn = df[col].notna().sum() if col in df else 0
        print(f"  {col:<25}: {nn:>8,} ({100*nn/len(df):.1f}%)")

    print(f"\n=== Distribución ERA5_HEADWIND_KT ===")
    hw = df["ERA5_HEADWIND_KT"].dropna()
    print(f"  mean={hw.mean():.1f} kt  std={hw.std():.1f} kt")
    print(f"  p5={hw.quantile(.05):.1f}  p25={hw.quantile(.25):.1f}  "
          f"median={hw.median():.1f}  p75={hw.quantile(.75):.1f}  p95={hw.quantile(.95):.1f}")
    print(f"  tailwind flag: {df['ERA5_TAILWIND_FLAG'].mean():.3f}")

    print(f"\n=== Correlación con ARR_DELAY ===")
    mask = df["ARR_DELAY"].notna() & df["ERA5_HEADWIND_KT"].notna()
    corr = df.loc[mask, ["ARR_DELAY", "ERA5_HEADWIND_KT",
                          "ERA5_U_KT", "ERA5_V_KT"]].corr()
    print(corr["ARR_DELAY"].to_string())


if __name__ == "__main__":
    main()

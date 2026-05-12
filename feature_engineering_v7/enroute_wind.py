"""Features de viento en ruta para el dataset maestro.

Fase 1 — Proxy METAR (sin datos externos):
  Usa el viento superficial de origen y destino para estimar el componente
  de viento de proa (headwind) sobre la ruta de vuelo.

  Columnas nuevas:
    BEARING_DEG           : heading de gran círculo ORIGIN → DEST (0-360°)
    ORIG_HEADWIND_KT      : componente headwind del viento en ORIGIN [kt]
    DEST_HEADWIND_KT      : componente headwind del viento en DEST [kt]
    ENROUTE_HEADWIND_KT   : promedio de ORIG y DEST (proxy en ruta)
    ENROUTE_TAILWIND_FLAG : 1 si proxy < -10 kt (viento de cola significativo)

  Convención: positivo = headwind (frena el avión), negativo = tailwind (ayuda).

Fase 2 — ERA5 real (ver era5_download.py):
  Una vez descargados los datos ERA5 de viento a 250hPa, reemplaza el proxy
  por ENROUTE_WIND_U_KT y ENROUTE_WIND_V_KT interpolados al midpoint de ruta.

Uso:
    from feature_engineering_v7.enroute_wind import add_enroute_wind_features
    df = add_enroute_wind_features(df)

O como script standalone:
    python3 feature_engineering_v7/enroute_wind.py --sample 100000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engineering_v7.airport_lookup import great_circle_bearing


def _headwind_kt(bearing_deg: float, wind_from_dir: float, wind_speed_kt: float) -> float:
    """Componente de headwind [kt] sobre la ruta de vuelo.

    bearing_deg   : heading del vuelo (0=N, 90=E, 180=S, 270=W)
    wind_from_dir : dirección meteorológica del viento (de donde VIENE) [°]
    wind_speed_kt : velocidad del viento [kt]

    Fórmula: headwind = speed × cos(bearing − wind_from)
      +  = headwind (viento de frente, aumenta block time)
      −  = tailwind (viento de cola, reduce block time)

    Verificación rápida:
      vuelo E (bear=90°) + viento del E (dir=90°) → cos(0°)=1 → headwind ✓
      vuelo E (bear=90°) + viento del O (dir=270°) → cos(-180°)=-1 → tailwind ✓
    """
    angle = np.radians(bearing_deg - wind_from_dir)
    return float(wind_speed_kt * np.cos(angle))


def add_enroute_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas de viento en ruta al DataFrame de vuelos.

    Espera las columnas:
      ORIGIN, DEST,
      ORIG_WX_DRCT, ORIG_WX_SKNT,
      DEST_WX_DRCT, DEST_WX_SKNT

    Retorna el DataFrame con columnas nuevas agregadas.
    """
    out = df.copy()

    # ---- Bearing ----
    # Calculamos por pares únicos ORIGIN-DEST para no hacer N millones de cálculos
    pairs = out[["ORIGIN", "DEST"]].drop_duplicates()
    pairs["BEARING_DEG"] = pairs.apply(
        lambda r: great_circle_bearing(r["ORIGIN"], r["DEST"]),
        axis=1,
    )
    out = out.merge(pairs, on=["ORIGIN", "DEST"], how="left")

    # ---- Headwind en origen ----
    has_orig = (
        out["BEARING_DEG"].notna()
        & out["ORIG_WX_DRCT"].notna()
        & out["ORIG_WX_SKNT"].notna()
        & (out["ORIG_WX_SKNT"] > 0)
    )
    out["ORIG_HEADWIND_KT"] = np.where(
        has_orig,
        (out["ORIG_WX_SKNT"] * np.cos(
            np.radians(out["BEARING_DEG"] - out["ORIG_WX_DRCT"])
        )).where(has_orig),
        np.nan,
    )

    # ---- Headwind en destino ----
    has_dest = (
        out["BEARING_DEG"].notna()
        & out["DEST_WX_DRCT"].notna()
        & out["DEST_WX_SKNT"].notna()
        & (out["DEST_WX_SKNT"] > 0)
    )
    out["DEST_HEADWIND_KT"] = np.where(
        has_dest,
        (out["DEST_WX_SKNT"] * np.cos(
            np.radians(out["BEARING_DEG"] - out["DEST_WX_DRCT"])
        )).where(has_dest),
        np.nan,
    )

    # ---- Proxy en ruta: promedio ponderado (origen tiene mayor peso) ----
    both = out["ORIG_HEADWIND_KT"].notna() & out["DEST_HEADWIND_KT"].notna()
    orig_only = out["ORIG_HEADWIND_KT"].notna() & out["DEST_HEADWIND_KT"].isna()
    dest_only = out["ORIG_HEADWIND_KT"].isna() & out["DEST_HEADWIND_KT"].notna()

    out["ENROUTE_HEADWIND_KT"] = np.nan
    out.loc[both, "ENROUTE_HEADWIND_KT"] = (
        0.6 * out.loc[both, "ORIG_HEADWIND_KT"]
        + 0.4 * out.loc[both, "DEST_HEADWIND_KT"]
    )
    out.loc[orig_only, "ENROUTE_HEADWIND_KT"] = out.loc[orig_only, "ORIG_HEADWIND_KT"]
    out.loc[dest_only, "ENROUTE_HEADWIND_KT"] = out.loc[dest_only, "DEST_HEADWIND_KT"]

    # ---- Flag de tailwind significativo ----
    out["ENROUTE_TAILWIND_FLAG"] = (
        out["ENROUTE_HEADWIND_KT"].fillna(0) < -10
    ).astype("int8")

    return out


# ------------------------------------------------------------------ #
# Standalone: verifica distribución de las features sobre una muestra #
# ------------------------------------------------------------------ #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet",
                   default=str(Path(__file__).resolve().parents[1] /
                               "dataset_maestro_FULL_US_2022-2025_BTS_IEM.parquet"))
    p.add_argument("--sample", type=int, default=200_000,
                   help="Vuelos a muestrear para verificación rápida")
    args = p.parse_args()

    import pyarrow.parquet as pq

    needed = ["ORIGIN", "DEST", "ARR_DELAY",
              "ORIG_WX_DRCT", "ORIG_WX_SKNT", "DEST_WX_DRCT", "DEST_WX_SKNT"]
    df = pq.read_table(args.parquet, columns=needed).to_pandas()
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=42)

    print(f"Muestra: {len(df):,} vuelos")
    df = add_enroute_wind_features(df)

    print(f"\n=== Cobertura de features ===")
    for col in ["BEARING_DEG", "ORIG_HEADWIND_KT", "DEST_HEADWIND_KT",
                "ENROUTE_HEADWIND_KT", "ENROUTE_TAILWIND_FLAG"]:
        non_null = df[col].notna().sum()
        print(f"  {col:<28}: {non_null:>8,} ({100*non_null/len(df):.1f}% non-null)")

    print(f"\n=== Distribución ENROUTE_HEADWIND_KT ===")
    hw = df["ENROUTE_HEADWIND_KT"].dropna()
    print(f"  mean={hw.mean():.1f} kt   std={hw.std():.1f} kt")
    print(f"  p5={hw.quantile(.05):.1f}  p25={hw.quantile(.25):.1f}  "
          f"median={hw.median():.1f}  p75={hw.quantile(.75):.1f}  p95={hw.quantile(.95):.1f}")
    print(f"  tailwind flag rate: {df['ENROUTE_TAILWIND_FLAG'].mean():.3f}")

    print(f"\n=== Correlación con ARR_DELAY ===")
    mask = df["ARR_DELAY"].notna() & df["ENROUTE_HEADWIND_KT"].notna()
    corr = df.loc[mask, ["ARR_DELAY", "ENROUTE_HEADWIND_KT",
                          "ORIG_HEADWIND_KT", "DEST_HEADWIND_KT"]].corr()
    print(corr["ARR_DELAY"].to_string())


if __name__ == "__main__":
    main()

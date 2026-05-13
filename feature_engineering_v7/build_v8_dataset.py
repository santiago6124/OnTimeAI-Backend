"""Construye el dataset maestro con todas las features nuevas para v8.

Parte de: dataset_maestro_FULL_US_2022-2025_v7.parquet  (77 cols, 27.5M vuelos)
Agrega:
  Fase 4 — v8 features (este script):
    NAS_RATE_2H     — fracción de vuelos con NAS_DELAY > 0 en origin en las 2h previas
    ORIGIN_PAGERANK — centralidad PageRank del aeropuerto origen en la red de rutas
    DEST_PAGERANK   — centralidad PageRank del aeropuerto destino

Nota: TAIL_DELAY_DECAY se calcula en tiempo de inferencia en prepare_inference_frame
(depende de lineage features que se computan on-the-fly, no en el parquet).

Salida: dataset_maestro_FULL_US_2022-2025_v8.parquet

Uso:
    python3 feature_engineering_v7/build_v8_dataset.py
    python3 feature_engineering_v7/build_v8_dataset.py --sample 500000   # test
    python3 feature_engineering_v7/build_v8_dataset.py --skip-nas        # sin NAS (más rápido)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_engineering_v7.v8_features import (
    add_nas_rolling_rate,
    add_pagerank_features,
    build_pagerank_lookup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PARQUET  = PROJECT_ROOT / "dataset_maestro_FULL_US_2022-2025_v7.parquet"
PAGERANK_JSON = PROJECT_ROOT / "artifacts" / "airport_pagerank.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src",  default=str(SRC_PARQUET))
    p.add_argument("--out",  default=str(PROJECT_ROOT / "dataset_maestro_FULL_US_2022-2025_v8.parquet"))
    p.add_argument("--sample", type=int, default=0,
                   help="Si > 0, procesa solo N vuelos (para test rápido)")
    p.add_argument("--skip-nas", action="store_true",
                   help="Omitir NAS_RATE_2H (más rápido, menos features)")
    p.add_argument("--pagerank-json", default=str(PAGERANK_JSON),
                   help="Ruta al JSON de PageRank (se genera si no existe)")
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    print(f"Dataset fuente : {src}")
    print(f"Dataset destino: {out}")
    print()

    # ---- cargar ----
    t0 = time.time()
    print("Cargando dataset fuente...")
    df = pq.read_table(src).to_pandas()
    print(f"  {len(df):,} vuelos, {len(df.columns)} columnas  ({time.time()-t0:.0f}s)")

    if args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"  Muestra reducida a {len(df):,} vuelos")

    # ---- Fase 4a: PageRank ----
    pr_path = Path(args.pagerank_json)
    if not pr_path.exists():
        print(f"\n[Fase 4a] Computando PageRank ({df['ORIGIN'].nunique()} aeropuertos)...")
        t1 = time.time()
        pr_lookup = build_pagerank_lookup(df, out_path=pr_path)
        print(f"  Guardado en {pr_path}  ({time.time()-t1:.1f}s)")
    else:
        from feature_engineering_v7.v8_features import load_pagerank_lookup
        pr_lookup = load_pagerank_lookup(pr_path)
        print(f"\n[Fase 4a] PageRank cargado desde {pr_path}  ({len(pr_lookup)} aeropuertos)")

    df = add_pagerank_features(df, pr_lookup=pr_lookup)
    print(f"  ORIGIN_PAGERANK: mean={df['ORIGIN_PAGERANK'].mean():.6f}  "
          f"max={df['ORIGIN_PAGERANK'].max():.6f}")

    # ---- Fase 4b: NAS_RATE_2H ----
    if not args.skip_nas:
        if "NAS_DELAY" not in df.columns:
            print("\n[Fase 4b] NAS_DELAY no encontrado — skip")
            df["nas_rate_2h"] = np.nan
        else:
            print("\n[Fase 4b] Computando NAS_RATE_2H (rolling 2h por origin)...")
            t2 = time.time()
            df = add_nas_rolling_rate(df, window_hours=2.0, out_col="nas_rate_2h")
            cov = df["nas_rate_2h"].notna().mean()
            nas_mean = df["nas_rate_2h"].mean()
            print(f"  Cobertura: {cov:.1%}  mean={nas_mean:.4f}  ({time.time()-t2:.0f}s)")
    else:
        print("\n[Fase 4b] NAS_RATE_2H — skipped (--skip-nas)")
        df["nas_rate_2h"] = np.nan

    # ---- guardar ----
    print(f"\nGuardando → {out}")
    print(f"  {len(df):,} vuelos, {len(df.columns)} columnas")
    t3 = time.time()
    df.to_parquet(out, index=False, engine="pyarrow", compression="snappy")
    print(f"  OK en {time.time()-t3:.0f}s  ({out.stat().st_size/1e9:.2f} GB)")

    print("\n=== Columnas nuevas (v8) ===")
    for col in ["ORIGIN_PAGERANK", "DEST_PAGERANK", "nas_rate_2h"]:
        if col in df.columns:
            nn = df[col].notna().sum()
            print(f"  {col:<20}: {nn:>8,} non-null ({100*nn/len(df):.1f}%)")

    print(f"\nTiempo total: {time.time()-t0:.0f}s  ({(time.time()-t0)/60:.1f} min)")
    print(f"\nPróximos pasos:")
    print(f"  1. Verificar airport_pagerank.json en artifacts/")
    print(f"  2. Retrain: python3 train.py --data {out}")
    print(f"  3. Agregar TAIL_DELAY_DECAY, ORIGIN_PAGERANK, DEST_PAGERANK, nas_rate_2h a feature_cols en train.py")


if __name__ == "__main__":
    main()

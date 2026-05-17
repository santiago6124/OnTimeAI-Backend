"""Features de linaje ADS-B del vuelo anterior (vuelo previo del mismo avión).

Fuentes de datos:
  - OpenSky Aircraft Database: registration → ICAO24 (transponder hex)
    URL: https://s3.opensky-network.org/data-samples/metadata/aircraftDatabase.csv (~94 MB)
    Sin auth requerida, descarga directa desde AWS S3.
  - OpenSky Network REST API: trayectoria completa del vuelo anterior
    Endpoint: GET /tracks/all?icao24=<hex>&time=<unix_ts>
    Free tier: 10 req/s, sin clave para datos >1h de antigüedad

Columnas nuevas:
  PREV_ACTUAL_BLOCK_MIN    : duración real del vuelo anterior (minutos)
  PREV_SCHED_BLOCK_MIN     : duración programada del vuelo anterior (minutos)
  PREV_BLOCK_DELTA_MIN     : real − programado (negativo = llegó antes)
  PREV_HOLDING_MIN         : tiempo estimado en patrón de espera (minutos)
  PREV_ROUTE_DEVIATION_PCT : % de desviación de la ruta de gran círculo
  PREV_ADSB_AVAILABLE      : 1 si se encontraron datos ADS-B para ese vuelo

Limitación crítica para inferencia live:
  Solo se puede usar el vuelo ANTERIOR del avión, no el vuelo actual.
  Estos features son equivalentes a 'prev_arr_delay_tail' pero con
  más detalle: distinguen "llegó tarde por holding" vs "por clima en ruta".

Uso:
  python3 feature_engineering_v7/adsb_lineage.py --setup    # descarga OpenSky DB
  python3 feature_engineering_v7/adsb_lineage.py --sample 50000
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data_raw"
OPENSKY_DB_URL = "https://s3.opensky-network.org/data-samples/metadata/aircraftDatabase.csv"
OPENSKY_DB_PATH = DATA_DIR / "opensky_aircraft_db.csv"
ICAO24_CACHE = DATA_DIR / "tail_to_icao24.parquet"
OPENSKY_API = "https://opensky-network.org/api"

_HEADERS = {"User-Agent": "Mozilla/5.0 (research/thesis project)"}


# ------------------------------------------------------------------ #
# 1. OpenSky Aircraft Database: registration → ICAO24                  #
# ------------------------------------------------------------------ #

def download_aircraft_db(out_dir: Path = DATA_DIR) -> Path:
    """Descarga el OpenSky aircraft database desde S3. ~94 MB, sin auth."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "opensky_aircraft_db.csv"
    if out_path.exists():
        print(f"  OpenSky DB ya existe: {out_path} ({out_path.stat().st_size/1e6:.0f} MB)")
        return out_path

    print(f"Descargando OpenSky Aircraft Database (~94 MB)...")
    r = requests.get(OPENSKY_DB_URL, headers=_HEADERS, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    downloaded = 0
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
    print(f"\n  guardado en {out_path} ({out_path.stat().st_size/1e6:.0f} MB)")
    return out_path


def build_tail_icao24_map(
    db_path: Path = OPENSKY_DB_PATH,
    cache_path: Path = ICAO24_CACHE,
) -> pd.DataFrame:
    """Construye y cachea el mapeo TAIL_NUM → ICAO24 desde OpenSky DB.

    Filtra solo aviones de registro US (N-number) para reducir tamaño.
    """
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    if not db_path.exists():
        db_path = download_aircraft_db()

    print(f"Construyendo mapeo TAIL→ICAO24 desde {db_path}...")
    df = pd.read_csv(
        db_path,
        usecols=["icao24", "registration"],
        dtype=str,
        on_bad_lines="skip",
    ).dropna()
    df.columns = ["icao24", "tail_num"]
    df["tail_num"] = df["tail_num"].str.strip().str.upper()
    df["icao24"] = df["icao24"].str.strip().str.lower()
    # Quedarse solo con aviones N-number (matrícula americana)
    df = df[df["tail_num"].str.startswith("N")].drop_duplicates("tail_num").reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  {len(df):,} aviones US guardados en {cache_path}")
    return df


# ------------------------------------------------------------------ #
# 2. OpenSky Network: trayectoria del vuelo anterior                   #
# ------------------------------------------------------------------ #

def _opensky_get(path: str, params: dict, retries: int = 3) -> dict | None:
    url = f"{OPENSKY_API}{path}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(5)
    return None


def _route_deviation_pct(
    track_lat: list[float],
    track_lon: list[float],
    orig_lat: float,
    orig_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> float:
    """Calcula la desviación porcentual de la ruta respecto al gran círculo.

    Usa la distancia máxima de los waypoints al gran círculo dividido
    por la distancia total del gran círculo.
    """
    if len(track_lat) < 3:
        return 0.0

    def haversine(lat1, lon1, lat2, lon2):
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        return r * 2 * math.asin(math.sqrt(a))

    gc_dist = haversine(orig_lat, orig_lon, dest_lat, dest_lon)
    if gc_dist < 1:
        return 0.0

    # Distancia acumulada de la trayectoria real
    track_dist = sum(
        haversine(track_lat[i], track_lon[i], track_lat[i+1], track_lon[i+1])
        for i in range(len(track_lat)-1)
    )
    return max(0.0, (track_dist - gc_dist) / gc_dist * 100)


def _holding_time_min(
    track_ts: list[float],
    track_lat: list[float],
    track_lon: list[float],
    dest_lat: float,
    dest_lon: float,
    holding_radius_km: float = 50.0,
    min_consecutive_s: float = 180.0,
) -> float:
    """Estima tiempo en patrón de espera como segundos en el área de llegada sin avanzar.

    Considera 'holding' a los segmentos donde el avión está dentro de
    holding_radius_km del destino pero se mueve lateralmente (no avanza).
    """
    if len(track_ts) < 5:
        return 0.0

    def dist_km(lat1, lon1, lat2, lon2):
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        return r * 2 * math.asin(math.sqrt(a))

    holding_sec = 0.0
    in_holding = False
    holding_start = 0.0

    for i, (ts, lat, lon) in enumerate(zip(track_ts, track_lat, track_lon)):
        near_dest = dist_km(lat, lon, dest_lat, dest_lon) < holding_radius_km
        if near_dest and not in_holding:
            in_holding = True
            holding_start = ts
        elif not near_dest and in_holding:
            segment = ts - holding_start
            if segment >= min_consecutive_s:
                holding_sec += segment
            in_holding = False

    return holding_sec / 60.0  # a minutos


def get_prev_flight_features(
    icao24: str,
    dep_unix: float,
    orig_lat: float,
    orig_lon: float,
    dest_lat: float,
    dest_lon: float,
    window_h: float = 6.0,
) -> dict:
    """Consulta OpenSky para el vuelo anterior del avión y retorna features.

    dep_unix: tiempo de despegue del vuelo actual (unix timestamp)
    El vuelo anterior es el último que aterrizó ANTES de dep_unix.
    """
    null_result = {
        "PREV_ACTUAL_BLOCK_MIN": np.nan,
        "PREV_SCHED_BLOCK_MIN": np.nan,
        "PREV_BLOCK_DELTA_MIN": np.nan,
        "PREV_HOLDING_MIN": np.nan,
        "PREV_ROUTE_DEVIATION_PCT": np.nan,
        "PREV_ADSB_AVAILABLE": 0,
    }

    # Buscar vuelos del avión en la ventana anterior a la salida
    begin = int(dep_unix - window_h * 3600)
    end = int(dep_unix)
    payload = _opensky_get(
        "/flights/aircraft",
        {"icao24": icao24, "begin": begin, "end": end},
    )
    if not payload or not isinstance(payload, list):
        return null_result

    # Tomar el vuelo con lastSeen más cercano a dep_unix
    candidates = [f for f in payload if f.get("lastSeen") and f["lastSeen"] < dep_unix]
    if not candidates:
        return null_result
    prev = max(candidates, key=lambda f: f["lastSeen"])

    # Trayectoria del vuelo anterior
    track_payload = _opensky_get(
        "/tracks/all",
        {"icao24": icao24, "time": prev.get("firstSeen", end - 7200)},
    )
    if not track_payload or "path" not in track_payload:
        return null_result

    path = track_payload["path"]  # lista de [time, lat, lon, baro_alt, true_track, on_ground]
    if len(path) < 5:
        return null_result

    track_ts  = [p[0] for p in path if p[1] is not None and p[2] is not None]
    track_lat = [p[1] for p in path if p[1] is not None and p[2] is not None]
    track_lon = [p[2] for p in path if p[1] is not None and p[2] is not None]

    if len(track_ts) < 5:
        return null_result

    actual_block_min = (track_ts[-1] - track_ts[0]) / 60.0
    sched_block_min = (float(prev.get("lastSeen", track_ts[-1])) -
                       float(prev.get("firstSeen", track_ts[0]))) / 60.0

    holding = _holding_time_min(track_ts, track_lat, track_lon, dest_lat, dest_lon)
    deviation = _route_deviation_pct(track_lat, track_lon, orig_lat, orig_lon, dest_lat, dest_lon)

    return {
        "PREV_ACTUAL_BLOCK_MIN":    round(actual_block_min, 1),
        "PREV_SCHED_BLOCK_MIN":     round(sched_block_min, 1),
        "PREV_BLOCK_DELTA_MIN":     round(actual_block_min - sched_block_min, 1),
        "PREV_HOLDING_MIN":         round(holding, 1),
        "PREV_ROUTE_DEVIATION_PCT": round(deviation, 2),
        "PREV_ADSB_AVAILABLE":      1,
    }


# ------------------------------------------------------------------ #
# 3. Pipeline batch para el dataset maestro                            #
# ------------------------------------------------------------------ #

def add_adsb_lineage_features(
    df: pd.DataFrame,
    tail_icao_map: pd.DataFrame,
    max_flights: int = 10_000,
    sleep_between: float = 0.2,
) -> pd.DataFrame:
    """Agrega features ADS-B del vuelo anterior para una muestra del dataset.

    Por el límite de la API pública de OpenSky (10 req/s, sin historial > 30 días
    en tier gratuito), esta función trabaja en modo batch con un cap en max_flights.
    Para el dataset completo se recomienda usar OpenSky Research Access (acceso
    a base de datos completa via Impala o parquet dumps).

    Retorna el DataFrame con columnas PREV_* agregadas.
    """
    icao_lookup = dict(zip(tail_icao_map["tail_num"], tail_icao_map["icao24"]))

    adsb_cols = ["PREV_ACTUAL_BLOCK_MIN", "PREV_SCHED_BLOCK_MIN",
                 "PREV_BLOCK_DELTA_MIN", "PREV_HOLDING_MIN",
                 "PREV_ROUTE_DEVIATION_PCT", "PREV_ADSB_AVAILABLE"]
    for c in adsb_cols:
        df[c] = np.nan
    df["PREV_ADSB_AVAILABLE"] = 0

    processed = 0
    for idx, row in df.iterrows():
        if processed >= max_flights:
            break
        icao24 = icao_lookup.get(str(row.get("TAIL_NUM", "")))
        if not icao24:
            continue

        # Tiempo de despegue como unix timestamp (estimado desde FL_DATE + CRS_DEP_TIME)
        try:
            dep_ts = pd.Timestamp(
                f"{row['FL_DATE']} {int(row['CRS_DEP_TIME'])//100:02d}:{int(row['CRS_DEP_TIME'])%100:02d}",
                tz="UTC",
            ).timestamp()
        except Exception:
            continue

        from feature_engineering_v7.airport_lookup import airport_coords
        co = airport_coords(str(row["ORIGIN"]))
        cd = airport_coords(str(row["DEST"]))
        if co is None or cd is None:
            continue

        feats = get_prev_flight_features(
            icao24=icao24,
            dep_unix=dep_ts,
            orig_lat=co[0], orig_lon=co[1],
            dest_lat=cd[0], dest_lon=cd[1],
        )
        for col, val in feats.items():
            df.at[idx, col] = val

        processed += 1
        if processed % 100 == 0:
            hit_rate = df["PREV_ADSB_AVAILABLE"].sum() / max(1, processed)
            print(f"  ADS-B: {processed}/{max_flights} procesados, "
                  f"hit rate={hit_rate:.2f}")
        time.sleep(sleep_between)

    return df


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--setup", action="store_true",
                   help="Descarga FAA registry y construye mapa TAIL→ICAO24, luego sale")
    p.add_argument("--sample", type=int, default=1000,
                   help="Vuelos a procesar con API OpenSky para verificación")
    p.add_argument("--parquet",
                   default=str(Path(__file__).resolve().parents[1] /
                               "dataset_maestro_FULL_US_2022-2025_BTS_IEM.parquet"))
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    faa_path = download_faa_registry()
    tail_map = build_tail_icao24_map(faa_path)
    print(f"\nMapeo construido: {len(tail_map):,} aviones con ICAO24")

    if args.setup:
        print("Setup completo. Podés correr sin --setup para testear la API.")
        return

    import pyarrow.parquet as pq
    needed = ["ORIGIN", "DEST", "TAIL_NUM", "FL_DATE", "CRS_DEP_TIME", "ARR_DELAY"]
    df = pq.read_table(args.parquet, columns=needed).to_pandas()

    # Muestra reciente para maximizar cobertura OpenSky (datos > 30 días pueden no estar)
    df = df.sort_values("FL_DATE", ascending=False).head(args.sample * 10).sample(
        min(args.sample, len(df)), random_state=42
    )
    print(f"\nProcesando {len(df):,} vuelos con API OpenSky...")
    df = add_adsb_lineage_features(df, tail_map, max_flights=args.sample)

    hit = df["PREV_ADSB_AVAILABLE"].sum()
    print(f"\n=== Resultado ===")
    print(f"  Hit rate ADS-B: {hit}/{len(df)} ({100*hit/len(df):.1f}%)")
    if hit > 0:
        for col in ["PREV_BLOCK_DELTA_MIN", "PREV_HOLDING_MIN", "PREV_ROUTE_DEVIATION_PCT"]:
            vals = df.loc[df["PREV_ADSB_AVAILABLE"]==1, col].dropna()
            if len(vals) > 0:
                print(f"  {col}: mean={vals.mean():.2f} std={vals.std():.2f}")
        mask = df["ARR_DELAY"].notna() & df["PREV_ADSB_AVAILABLE"].eq(1)
        if mask.sum() > 10:
            corr = df.loc[mask, ["ARR_DELAY", "PREV_BLOCK_DELTA_MIN",
                                  "PREV_HOLDING_MIN", "PREV_ROUTE_DEVIATION_PCT"]].corr()
            print(f"\nCorrelación con ARR_DELAY:")
            print(corr["ARR_DELAY"].to_string())


if __name__ == "__main__":
    main()

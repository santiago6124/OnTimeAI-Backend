"""Lookup de lat/lon y bearing entre aeropuertos para features de ruta.

Carga desde airports_universe.csv una sola vez al importar.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import pandas as pd

_CSV = Path(__file__).resolve().parents[1] / "airports_universe.csv"
_df = pd.read_csv(_CSV)[["iata", "iem_lat", "iem_lon"]].dropna()
_LAT: dict[str, float] = dict(zip(_df["iata"], _df["iem_lat"]))
_LON: dict[str, float] = dict(zip(_df["iata"], _df["iem_lon"]))

# Secondary lookup: airportsdata covers 11,700+ IATA airports globally.
# Fills gaps for international destinations and small regional airports
# not present in airports_universe.csv (built from BTS US-only dataset).
try:
    import airportsdata as _airportsdata_pkg
    _adb = _airportsdata_pkg.load("IATA")
    for _code, _ap in _adb.items():
        if _code not in _LAT and _ap.get("lat") is not None and _ap.get("lon") is not None:
            _LAT[_code] = float(_ap["lat"])
            _LON[_code] = float(_ap["lon"])
    del _adb, _airportsdata_pkg, _code, _ap
except (ImportError, Exception):
    pass


def airport_coords(iata: str) -> tuple[float, float] | None:
    """Retorna (lat, lon) o None si el aeropuerto no está en el universo."""
    lat = _LAT.get(iata)
    lon = _LON.get(iata)
    return (lat, lon) if lat is not None else None


@lru_cache(maxsize=8192)
def great_circle_bearing(origin: str, dest: str) -> float | None:
    """Bearing de gran círculo en grados (0-360) de origin a dest.

    Retorna None si algún aeropuerto no tiene coordenadas.
    Convención: 0=Norte, 90=Este, 180=Sur, 270=Oeste.
    """
    co = airport_coords(origin)
    cd = airport_coords(dest)
    if co is None or cd is None:
        return None
    lat1, lon1 = math.radians(co[0]), math.radians(co[1])
    lat2, lon2 = math.radians(cd[0]), math.radians(cd[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360


@lru_cache(maxsize=8192)
def great_circle_distance_nm(origin: str, dest: str) -> float | None:
    """Distancia de gran círculo en millas náuticas."""
    co = airport_coords(origin)
    cd = airport_coords(dest)
    if co is None or cd is None:
        return None
    lat1, lon1 = math.radians(co[0]), math.radians(co[1])
    lat2, lon2 = math.radians(cd[0]), math.radians(cd[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * math.asin(math.sqrt(a)) * 3440.065  # radio tierra en nm

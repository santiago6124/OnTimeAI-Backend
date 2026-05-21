"""OpenSky Network REST API client for ADS-B flight tracking.

Provides free, no-auth access to:
  - Live aircraft positions (states/all)
  - Flight tracking by tail number (registration)
  - On-ground detection for estimating actual arrival times
  - Departure/arrival detection for lineage enrichment

Academic users can register at opensky-network.org for unlimited access.
Anonymous users: 10 requests per 10 seconds.

Usage as module:
    from ontimeai.opensky import OpenSkyClient
    client = OpenSkyClient()
    states = client.get_states_at_airport("KATL", radius_km=50)
    flights = client.get_flights_by_aircraft("N12345", begin, end)
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

OPENSKY_BASE = "https://opensky-network.org/api"


class OpenSkyClient:
    """Minimal OpenSky Network REST API client.

    Supports both anonymous and authenticated access. Authentication
    increases rate limits and enables access to historical data.
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
    ):
        self.username = username or os.environ.get("OPENSKY_USER")
        self.password = password or os.environ.get("OPENSKY_PASS")
        self._session = requests.Session()
        if self.username and self.password:
            self._session.auth = (self.username, self.password)
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        """Enforce minimum interval between requests (anonymous: 1s, auth: 0.5s)."""
        min_interval = 0.5 if self.username else 1.0
        elapsed = time.time() - self._last_request
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request = time.time()

    def _get(self, path: str, params: dict | None = None) -> dict | list | None:
        """Make a GET request with rate limiting and error handling."""
        self._rate_limit()
        url = f"{OPENSKY_BASE}{path}"
        try:
            r = self._session.get(url, params=params or {}, timeout=30)
            if r.status_code == 429:
                print(f"  [OpenSky] rate-limited, waiting 10s...")
                time.sleep(10)
                r = self._session.get(url, params=params or {}, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"  [OpenSky] request failed: {e}")
            return None

    def get_all_states(
        self,
        *,
        icao24: str | list[str] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """Get current state vectors for all (or filtered) aircraft.

        Args:
            icao24: Filter by ICAO 24-bit address(es)
            bbox: (lat_min, lat_max, lon_min, lon_max) bounding box

        Returns list of state dicts with keys:
            icao24, callsign, origin_country, time_position, last_contact,
            longitude, latitude, baro_altitude, on_ground, velocity,
            true_track, vertical_rate, sensors, geo_altitude, squawk, spi,
            position_source, category
        """
        params = {}
        if icao24:
            if isinstance(icao24, list):
                params["icao24"] = "&icao24=".join(icao24)
            else:
                params["icao24"] = icao24
        if bbox:
            params["lamin"] = bbox[0]
            params["lamax"] = bbox[1]
            params["lomin"] = bbox[2]
            params["lomax"] = bbox[3]

        data = self._get("/states/all", params)
        if not data or "states" not in data:
            return []

        keys = [
            "icao24", "callsign", "origin_country", "time_position",
            "last_contact", "longitude", "latitude", "baro_altitude",
            "on_ground", "velocity", "true_track", "vertical_rate",
            "sensors", "geo_altitude", "squawk", "spi",
            "position_source", "category",
        ]
        return [dict(zip(keys, s)) for s in data["states"]]

    def get_states_near_airport(
        self,
        lat: float,
        lon: float,
        radius_km: float = 50.0,
    ) -> list[dict]:
        """Get aircraft states near an airport (bounding box approximation).

        Common airports:
            ATL: (33.6407, -84.4277)
        """
        # Approximate bounding box from center + radius
        # 1 degree lat ≈ 111 km
        dlat = radius_km / 111.0
        dlon = radius_km / (111.0 * abs(max(0.01, __import__("math").cos(__import__("math").radians(lat)))))
        bbox = (lat - dlat, lat + dlat, lon - dlon, lon + dlon)
        return self.get_all_states(bbox=bbox)

    def get_flights_by_aircraft(
        self,
        icao24: str,
        begin: datetime,
        end: datetime,
    ) -> list[dict]:
        """Get flights for a specific aircraft (by ICAO 24-bit address) in a time range.

        Requires authentication for ranges > 2 hours.
        Max range: 30 days.

        Returns list of flight dicts with:
            icao24, firstSeen, estDepartureAirport, lastSeen,
            estArrivalAirport, callsign, ...
        """
        params = {
            "icao24": icao24.lower(),
            "begin": int(begin.timestamp()),
            "end": int(end.timestamp()),
        }
        data = self._get("/flights/aircraft", params)
        return data if isinstance(data, list) else []

    def get_arrivals_by_airport(
        self,
        icao_airport: str,
        begin: datetime,
        end: datetime,
    ) -> list[dict]:
        """Get arrivals at an airport in a time range (max 7 days).

        Returns list of flight dicts with estimated departure/arrival airports
        and timestamps.
        """
        params = {
            "airport": icao_airport,
            "begin": int(begin.timestamp()),
            "end": int(end.timestamp()),
        }
        data = self._get("/flights/arrival", params)
        return data if isinstance(data, list) else []

    def get_departures_by_airport(
        self,
        icao_airport: str,
        begin: datetime,
        end: datetime,
    ) -> list[dict]:
        """Get departures from an airport in a time range (max 7 days)."""
        params = {
            "airport": icao_airport,
            "begin": int(begin.timestamp()),
            "end": int(end.timestamp()),
        }
        data = self._get("/flights/departure", params)
        return data if isinstance(data, list) else []


# --- ATL-specific helpers ---

ATL_LAT = 33.6407
ATL_LON = -84.4277


def get_atl_traffic(client: OpenSkyClient | None = None) -> list[dict]:
    """Get current aircraft near ATL (50km radius)."""
    if client is None:
        client = OpenSkyClient()
    return client.get_states_near_airport(ATL_LAT, ATL_LON, radius_km=50)


def detect_arrivals(
    states: list[dict],
    airport_lat: float = ATL_LAT,
    airport_lon: float = ATL_LON,
    ground_radius_km: float = 5.0,
) -> list[dict]:
    """From a list of state vectors, detect aircraft that are on_ground near the airport.

    These are recently-arrived aircraft whose actual landing time can be
    estimated from their last_contact timestamp.
    """
    import math

    arrived = []
    for s in states:
        if not s.get("on_ground"):
            continue
        lat, lon = s.get("latitude"), s.get("longitude")
        if lat is None or lon is None:
            continue

        # Haversine distance approximation
        dlat = math.radians(lat - airport_lat)
        dlon = math.radians(lon - airport_lon)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(airport_lat)) * \
            math.cos(math.radians(lat)) * math.sin(dlon / 2) ** 2
        dist_km = 6371 * 2 * math.asin(math.sqrt(a))

        if dist_km <= ground_radius_km:
            arrived.append({
                "icao24": s["icao24"],
                "callsign": (s.get("callsign") or "").strip(),
                "last_contact": s.get("last_contact"),
                "latitude": lat,
                "longitude": lon,
                "velocity": s.get("velocity"),
                "dist_km": round(dist_km, 2),
            })
    return arrived

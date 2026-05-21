"""AirLabs free-tier API client for flight data.

Supplements AeroAPI with free flight schedule and delay data.
Free tier: 50 results per request, unspecified monthly limit.

Provides:
  - Flight schedules (up to 10h ahead)
  - Flight delay information
  - Actual arrival/departure times (for actuals settlement)

Usage:
    from ontimeai.airlabs import AirLabsClient
    client = AirLabsClient()
    flights = client.get_arrivals("ATL")
    flights = client.get_departures("ATL")
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

AIRLABS_BASE = "https://airlabs.co/api/v9"

# Load API key from env or .env file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def _load_airlabs_key() -> str | None:
    key = os.environ.get("AIRLABS_API_KEY")
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("AIRLABS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


class AirLabsClient:
    """AirLabs free-tier API client.

    Requires AIRLABS_API_KEY in environment or .env file.
    Free tier limits: 50 items per request.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or _load_airlabs_key()
        if not self.api_key:
            raise RuntimeError(
                "AIRLABS_API_KEY not found. Register at https://airlabs.co/ "
                "and set AIRLABS_API_KEY in your .env file."
            )
        self._session = requests.Session()
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request = time.time()

    def _get(self, endpoint: str, params: dict | None = None) -> list[dict]:
        self._rate_limit()
        url = f"{AIRLABS_BASE}/{endpoint}"
        p = {"api_key": self.api_key}
        if params:
            p.update(params)
        try:
            r = self._session.get(url, params=p, timeout=30)
            if r.status_code == 429:
                print("  [AirLabs] rate-limited, waiting 60s...")
                time.sleep(60)
                r = self._session.get(url, params=p, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                print(f"  [AirLabs] API error: {data['error']}")
                return []
            return data.get("response", [])
        except Exception as e:
            print(f"  [AirLabs] request failed: {e}")
            return []

    def get_schedules(
        self,
        iata_airport: str,
        *,
        dep: bool = True,
    ) -> list[dict]:
        """Get flight schedules for an airport (up to 10h ahead).

        Args:
            iata_airport: IATA code (e.g. "ATL")
            dep: True for departures, False for arrivals

        Returns list of schedule dicts with:
            airline_iata, flight_iata, dep_iata, arr_iata,
            dep_time_utc, arr_time_utc, status, delayed, ...
        """
        params = {
            "dep_iata" if dep else "arr_iata": iata_airport.upper(),
        }
        return self._get("schedules", params)

    def get_delays(
        self,
        iata_airport: str | None = None,
    ) -> list[dict]:
        """Get current flight delays, optionally filtered by airport.

        Returns list of delay dicts with:
            airline_iata, flight_iata, dep_iata, arr_iata,
            dep_delayed, arr_delayed, dep_time, arr_time, ...
        """
        params = {}
        if iata_airport:
            params["arr_iata"] = iata_airport.upper()
        return self._get("delays", params)

    def get_flights_live(
        self,
        iata_airport: str | None = None,
    ) -> list[dict]:
        """Get live flight data (in-air and recently landed).

        Free tier returns up to 50 results.
        """
        params = {}
        if iata_airport:
            params["arr_iata"] = iata_airport.upper()
        return self._get("flights", params)

    def extract_actuals(self, flights: list[dict]) -> list[dict]:
        """Extract actual arrival data from AirLabs flight records.

        Maps to the actuals schema used by live_data.db.
        Only returns flights that have actual arrival information.
        """
        actuals = []
        for f in flights:
            arr_actual = f.get("arr_actual_utc") or f.get("arr_actual")
            if not arr_actual:
                continue

            dep_actual = f.get("dep_actual_utc") or f.get("dep_actual")
            arr_delayed = f.get("arr_delayed")
            dep_delayed = f.get("dep_delayed")

            actuals.append({
                "flight_iata": f.get("flight_iata", ""),
                "airline_iata": f.get("airline_iata", ""),
                "dep_iata": f.get("dep_iata", ""),
                "arr_iata": f.get("arr_iata", ""),
                "dep_actual": dep_actual,
                "arr_actual": arr_actual,
                "arr_delay_min": int(arr_delayed) if arr_delayed is not None else None,
                "dep_delay_min": int(dep_delayed) if dep_delayed is not None else None,
                "status": f.get("status", ""),
            })
        return actuals

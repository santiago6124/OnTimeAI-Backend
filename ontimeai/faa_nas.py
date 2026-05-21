"""FAA NAS (National Airspace System) Status client.

Parses the official FAA airport status XML/JSON feed to get:
  - Ground Delay Programs (GDP)
  - Ground Stops
  - Arrival/Departure delays
  - Airport closures

This replaces the existing gdp_scraper.py with the official FAA data source.

Source: https://nasstatus.faa.gov/api/airport-status-information

Usage:
    from ontimeai.faa_nas import FaaNasClient
    client = FaaNasClient()
    status = client.get_airport_status("ATL")
    gdp_features = client.get_gdp_features("ATL")
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import requests

FAA_STATUS_URL = "https://nasstatus.faa.gov/api/airport-status-information"


class FaaNasClient:
    """Client for the FAA NAS Status API.

    Provides airport delay status including GDP (Ground Delay Program)
    information, which is a key feature for flight delay prediction.
    """

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "OnTimeAI-Academic/1.0",
        })
        self._last_request = 0.0
        self._cache: dict[str, tuple[float, dict]] = {}
        self._cache_ttl = 300  # 5 minute cache

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)
        self._last_request = time.time()

    def get_nas_status(self) -> dict:
        """Fetch the NAS-wide status XML and parse into structured data.

        Returns dict keyed by airport IATA with delay info:
        {
            "ATL": {"ground_stop": True, "reason": "thunderstorms", ...},
            "BNA": {"gdp": True, "avg_delay_min": 54, "max_delay_min": 106, ...},
            ...
        }
        """
        # Check cache
        cache_key = "__NAS_ALL__"
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return data

        self._rate_limit()
        try:
            r = self._session.get(FAA_STATUS_URL, timeout=15,
                                  headers={"Accept": "application/xml"})
            r.raise_for_status()
            data = _parse_nas_xml(r.content)
            self._cache[cache_key] = (time.time(), data)
            return data
        except Exception as e:
            print(f"  [FAA NAS] NAS status fetch failed: {e}")
            return {}

    def get_gdp_features(self, iata: str) -> dict:
        """Extract GDP-related features for a given airport from NAS XML feed.

        Returns:
            {
                "GDP_FLAG": 0 or 1,
                "GDP_DELAY_MIN": float (average delay in minutes),
                "GROUND_STOP": 0 or 1,
                "ARRIVAL_DELAY": 0 or 1,
                "DEPARTURE_DELAY": 0 or 1,
                "CLOSURE": 0 or 1,
            }
        """
        defaults = {
            "GDP_FLAG": 0,
            "GDP_DELAY_MIN": 0.0,
            "GROUND_STOP": 0,
            "ARRIVAL_DELAY": 0,
            "DEPARTURE_DELAY": 0,
            "CLOSURE": 0,
        }

        nas = self.get_nas_status()
        iata = iata.upper()
        if iata not in nas:
            return defaults

        info = nas[iata]
        result = dict(defaults)
        result["GROUND_STOP"] = 1 if info.get("ground_stop") else 0
        result["GDP_FLAG"] = 1 if info.get("gdp") else 0
        result["ARRIVAL_DELAY"] = 1 if info.get("arrival_delay") else 0
        result["DEPARTURE_DELAY"] = 1 if info.get("departure_delay") else 0
        result["CLOSURE"] = 1 if info.get("closure") else 0

        avg = info.get("avg_delay_min")
        if avg is not None:
            result["GDP_DELAY_MIN"] = avg
        elif info.get("min_delay_min") is not None and info.get("max_delay_min") is not None:
            result["GDP_DELAY_MIN"] = (info["min_delay_min"] + info["max_delay_min"]) / 2.0

        return result


def _parse_delay_string(s: str) -> float | None:
    """Parse FAA delay strings like '1 hour and 15 minutes' into minutes."""
    if not s or s.strip() == "":
        return None
    s = s.lower().strip()
    total = 0.0
    import re

    # Match patterns like "1 hour", "30 minutes", "1 hour and 30 minutes"
    hour_match = re.search(r"(\d+)\s*hour", s)
    min_match = re.search(r"(\d+)\s*min", s)

    if hour_match:
        total += int(hour_match.group(1)) * 60
    if min_match:
        total += int(min_match.group(1))

    # Also try just a plain number (assumed minutes)
    if total == 0:
        try:
            total = float(s)
        except ValueError:
            return None

    return total if total > 0 else None


def _parse_nas_xml(xml_bytes: bytes) -> dict:
    """Parse FAA NAS status XML into a dict keyed by airport IATA.

    XML structure:
      <AIRPORT_STATUS_INFORMATION>
        <Delay_type>
          <Name>Ground Stop Programs</Name>
          <Ground_Stop_List>
            <Program><ARPT>DTW</ARPT><Reason>thunderstorms</Reason><End_Time>...</End_Time></Program>
          </Ground_Stop_List>
        </Delay_type>
        <Delay_type>
          <Name>Ground Delay Programs</Name>
          <Ground_Delay_List>
            <Ground_Delay><ARPT>BNA</ARPT><Reason>other</Reason><Avg>54 minutes</Avg><Max>...</Max></Ground_Delay>
          </Ground_Delay_List>
        </Delay_type>
        <Delay_type>
          <Name>Arrival/Departure Delay Information</Name>
          ...
        </Delay_type>
        <Delay_type>
          <Name>Airport Closures</Name>
          ...
        </Delay_type>
      </AIRPORT_STATUS_INFORMATION>
    """
    import xml.etree.ElementTree as ET

    result: dict = {}

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        print(f"  [FAA NAS] XML parse error: {e}")
        return result

    for delay_type in root.findall(".//Delay_type"):
        name_el = delay_type.find("Name")
        if name_el is None:
            continue
        dtype_name = (name_el.text or "").strip()

        if "Ground Stop" in dtype_name:
            for prog in delay_type.findall(".//Program"):
                arpt = (prog.findtext("ARPT") or "").strip().upper()
                if not arpt:
                    continue
                if arpt not in result:
                    result[arpt] = {}
                result[arpt]["ground_stop"] = True
                result[arpt]["reason"] = (prog.findtext("Reason") or "").strip()
                result[arpt]["end_time"] = (prog.findtext("End_Time") or "").strip()

        elif "Ground Delay" in dtype_name:
            for gd in delay_type.findall(".//Ground_Delay"):
                arpt = (gd.findtext("ARPT") or "").strip().upper()
                if not arpt:
                    continue
                if arpt not in result:
                    result[arpt] = {}
                result[arpt]["gdp"] = True
                result[arpt]["reason"] = (gd.findtext("Reason") or "").strip()
                avg_str = (gd.findtext("Avg") or "").strip()
                max_str = (gd.findtext("Max") or "").strip()
                if avg_str:
                    result[arpt]["avg_delay_min"] = _parse_delay_string(avg_str)
                if max_str:
                    result[arpt]["max_delay_min"] = _parse_delay_string(max_str)

        elif "Arrival" in dtype_name or "Departure" in dtype_name:
            for arpt_el in delay_type.findall(".//*[ARPT]"):
                arpt = (arpt_el.findtext("ARPT") or "").strip().upper()
                if not arpt:
                    continue
                if arpt not in result:
                    result[arpt] = {}
                if "Arrival" in dtype_name:
                    result[arpt]["arrival_delay"] = True
                if "Departure" in dtype_name:
                    result[arpt]["departure_delay"] = True
                reason = (arpt_el.findtext("Reason") or "").strip()
                if reason:
                    result[arpt]["reason"] = reason
                min_str = (arpt_el.findtext("Min") or "").strip()
                max_str = (arpt_el.findtext("Max") or "").strip()
                if min_str:
                    result[arpt]["min_delay_min"] = _parse_delay_string(min_str)
                if max_str:
                    result[arpt]["max_delay_min"] = _parse_delay_string(max_str)

        elif "Closure" in dtype_name:
            for arpt_el in delay_type.findall(".//*[ARPT]"):
                arpt = (arpt_el.findtext("ARPT") or "").strip().upper()
                if not arpt:
                    continue
                if arpt not in result:
                    result[arpt] = {}
                result[arpt]["closure"] = True

    return result

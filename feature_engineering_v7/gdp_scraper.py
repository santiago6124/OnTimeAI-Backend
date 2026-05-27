"""Real-time FAA Ground Delay Program (GDP) / NAS status scraper.

Fetches active Ground Delay Programs, Ground Stops, and Arrival/Departure
Delay Programs from the FAA NAS Status API and exposes them as a binary
flag and estimated delay for use in live flight delay prediction.

At live inference time, a flight departing from an airport under an active
GDP is almost certainly going to be delayed — this is one of the strongest
possible signals. It's absent from most academic models because it requires
real-time scraping.

Usage:
    client = GdpClient()
    status = client.get_airport_status("ATL")   # dict or None
    flag = client.gdp_flag("ATL")               # 1 if under GDP/GS, else 0
    delay = client.gdp_delay_min("ATL")         # avg delay minutes or 0

    # Batch: add GDP_FLAG and GDP_DELAY_MIN to a DataFrame
    df = client.add_gdp_features(df)

The client caches results for CACHE_TTL_SEC (default 300 s = 5 min).
Set CACHE_TTL_SEC = 0 to disable caching (useful for testing).

FAA API:
    https://nasstatus.faa.gov/api/airport-status-information
    Returns JSON array of active programs.
"""
from __future__ import annotations

import time
import warnings
from typing import Any

import pandas as pd

FAA_NAS_URL = "https://nasstatus.faa.gov/api/airport-status-information"
CACHE_TTL_SEC = 300  # 5 min


_PROGRAM_TYPES_GDP = {
    "Ground Delay",
    "Ground Stop",
    "Arrival/Departure Delay Program",
    "Departure Delay",
}

# Substrings (case-insensitive) inside an Airport_Closure NOTAM reason that
# indicate the closure only affects General Aviation / non-scheduled traffic
# and therefore does NOT impact commercial scheduled flights. Without this
# filter, items like "!LAS 04/141 LAS AD AP CLSD TO NON SKED TRANSIENT GA
# ACFT..." were being treated as full 180-minute closures and inflating
# GDP_ORIG for commercial flights (e.g. F91116 LAS->ATL).
_GA_ONLY_CLOSURE_MARKERS = (
    "NON SKED",
    "NON-SKED",
    "GA ACFT",
    "GA TRANSIENT",
    "TRANSIENT GA",
    "EXC HEL",
    "EXCEPT HEL",
)


def _is_ga_only_notam(reason: str) -> bool:
    """True if a closure reason refers only to General Aviation / non-scheduled
    traffic restrictions (does not affect commercial scheduled flights)."""
    if not reason:
        return False
    upper = reason.upper()
    return any(marker in upper for marker in _GA_ONLY_CLOSURE_MARKERS)


def _parse_end_time_utc(end_time: str | None) -> str | None:
    """Parse an FAA NAS End_Time string into an ISO-8601 UTC timestamp.

    The FAA feed uses several formats:
      - "0030"        -> 00:30 UTC today (or tomorrow if already past)
      - "20:30"       -> 20:30 UTC today
      - "midnight"    -> 00:00 UTC tomorrow
      - "indefinite"  -> returns None (no defined end)
      - empty / junk  -> returns None
    """
    if not end_time:
        return None
    import re
    from datetime import datetime, timezone, timedelta
    s = end_time.strip().lower()
    if not s or "indefinite" in s or "tba" in s or "until further" in s:
        return None

    now = datetime.now(timezone.utc)
    hour = minute = None

    if s == "midnight":
        end_dt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return end_dt.isoformat()

    m = re.match(r"^(\d{1,2}):(\d{2})", s)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
    else:
        m = re.match(r"^(\d{4})$", s)
        if m:
            hour, minute = int(s[:2]), int(s[2:])

    if hour is None or not (0 <= hour < 24 and 0 <= minute < 60):
        return None

    end_dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    # If the HHMM time is more than ~30 min in the past, assume it rolled
    # over to tomorrow (a program ending at 03:00 captured at 22:00 is
    # tomorrow morning, not 19 hours ago).
    if (now - end_dt).total_seconds() > 1800:
        end_dt = end_dt + timedelta(days=1)
    return end_dt.isoformat()


def _parse_delay_minutes(val: Any) -> float:
    """Parse delay strings into minutes.

    Handles: '45', '1:30', '1 hour and 30 minutes', '2 hours', '30 minutes'.
    """
    if val is None:
        return 0.0
    s = str(val).strip().lower()

    # "1:30" style
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) * 60 + float(parts[1])
        except (ValueError, IndexError):
            return 0.0

    # "1 hour and 58 minutes" / "2 hours" / "30 minutes" style
    import re
    hours = minutes = 0.0
    m = re.search(r"(\d+)\s*hour", s)
    if m:
        hours = float(m.group(1))
    m = re.search(r"(\d+)\s*minute", s)
    if m:
        minutes = float(m.group(1))
    if hours or minutes:
        return hours * 60 + minutes

    try:
        return float(s)
    except ValueError:
        return 0.0


class GdpClient:
    """Thread-safe FAA GDP status client with time-based caching."""

    def __init__(self, url: str = FAA_NAS_URL, cache_ttl: int = CACHE_TTL_SEC):
        self._url = url
        self._cache_ttl = cache_ttl
        self._cache: dict[str, dict] = {}  # {iata: {type, delay_min, reason}}
        self._last_fetch: float = 0.0

    def _refresh(self) -> None:
        """Fetch and parse FAA NAS status. Silently degrades on error."""
        import ssl
        import urllib.request

        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()

        try:
            req = urllib.request.Request(self._url)
            with urllib.request.urlopen(req, timeout=8, context=ctx) as resp:
                body = resp.read()
        except Exception as exc:
            warnings.warn(f"GdpClient: FAA fetch failed — {exc}. Using stale/empty cache.")
            self._last_fetch = time.time()
            return

        new_cache: dict[str, dict] = {}
        try:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(body)
            for delay_type in root.findall("Delay_type"):
                type_name = (delay_type.findtext("Name") or "").strip()

                # Ground Stop Programs
                for prog in delay_type.findall(".//Ground_Stop_List/Program"):
                    airport = (prog.findtext("ARPT") or "").upper()
                    reason  = prog.findtext("Reason") or ""
                    end_iso = _parse_end_time_utc(prog.findtext("End_Time"))
                    if airport:
                        new_cache[airport] = {"type": "Ground Stop", "delay_min": 90.0,
                                              "max_delay_min": 90.0, "reason": reason,
                                              "end_time_utc": end_iso}

                # Ground Delay Programs
                for prog in delay_type.findall(".//Ground_Delay_List/Ground_Delay"):
                    airport   = (prog.findtext("ARPT") or "").upper()
                    reason    = prog.findtext("Reason") or ""
                    avg_delay = prog.findtext("Avg") or "0"
                    max_delay = prog.findtext("Max") or avg_delay
                    delay_min = _parse_delay_minutes(avg_delay)
                    max_delay_min = _parse_delay_minutes(max_delay)
                    end_iso = _parse_end_time_utc(prog.findtext("End_Time"))
                    if airport:
                        if airport not in new_cache or delay_min > new_cache[airport]["delay_min"]:
                            new_cache[airport] = {"type": "Ground Delay", "delay_min": delay_min,
                                                  "max_delay_min": max(delay_min, max_delay_min),
                                                  "reason": reason, "end_time_utc": end_iso}

                # Arrival/Departure Delay Info
                for prog in delay_type.findall(".//Arrival_Departure_Delay_List/Delay"):
                    airport = (prog.findtext("ARPT") or "").upper()
                    reason  = prog.findtext("Reason") or ""
                    end_iso = _parse_end_time_utc(prog.findtext("End_Time"))
                    # Take the max of Min delay across departure/arrival sub-elements
                    max_delay = 0.0
                    for ad in prog.findall("Arrival_Departure"):
                        max_delay = max(max_delay, _parse_delay_minutes(ad.findtext("Max") or "0"))
                    if airport and max_delay > 0:
                        if airport not in new_cache or max_delay > new_cache[airport]["delay_min"]:
                            new_cache[airport] = {"type": type_name, "delay_min": max_delay,
                                                  "max_delay_min": max_delay, "reason": reason,
                                                  "end_time_utc": end_iso}

                # Airport Closures — treat as GDP_FLAG=1, delay=180 min,
                # but SKIP NOTAMs that only restrict General Aviation /
                # non-scheduled traffic (they don't impact commercial flights).
                for prog in delay_type.findall(".//Airport_Closure_List/Airport"):
                    airport = (prog.findtext("ARPT") or "").upper()
                    reason  = prog.findtext("Reason") or ""
                    if not airport or airport in new_cache:
                        continue
                    if _is_ga_only_notam(reason):
                        continue
                    end_iso = _parse_end_time_utc(prog.findtext("Reopen") or prog.findtext("End_Time"))
                    new_cache[airport] = {"type": "Airport Closure", "delay_min": 180.0,
                                          "max_delay_min": 180.0, "reason": reason,
                                          "end_time_utc": end_iso}

        except Exception as exc:
            warnings.warn(f"GdpClient: XML parse failed — {exc}")
            self._last_fetch = time.time()
            return

        self._cache = new_cache
        self._last_fetch = time.time()

    def _ensure_fresh(self) -> None:
        age = time.time() - self._last_fetch
        if age > self._cache_ttl or not self._cache:
            self._refresh()

    def get_airport_status(self, iata: str) -> dict | None:
        """Return active program info for airport or None if no program."""
        self._ensure_fresh()
        return self._cache.get(iata.upper())

    def gdp_flag(self, iata: str) -> int:
        """1 if airport is under any GDP/GS/AFP, else 0."""
        status = self.get_airport_status(iata)
        if status is None:
            return 0
        return 1 if status["type"] in _PROGRAM_TYPES_GDP or status["delay_min"] > 0 else 0

    def gdp_delay_min(self, iata: str) -> float:
        """Expected GDP delay in minutes (0 if no program)."""
        status = self.get_airport_status(iata)
        return status["delay_min"] if status else 0.0

    def add_gdp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add GDP_FLAG and GDP_DELAY_MIN columns to df (uses ORIGIN column)."""
        self._ensure_fresh()
        df = df.copy()
        origins = df["ORIGIN"].astype(str).str.upper()
        df["GDP_FLAG"] = origins.map(
            lambda a: 1 if a in self._cache and (
                self._cache[a]["type"] in _PROGRAM_TYPES_GDP
                or self._cache[a]["delay_min"] > 0
            ) else 0
        ).astype("int8")
        df["GDP_DELAY_MIN"] = origins.map(
            lambda a: self._cache[a]["delay_min"] if a in self._cache else 0.0
        ).astype("float32")
        return df

    def status_report(self) -> str:
        """Human-readable summary of active programs."""
        self._ensure_fresh()
        if not self._cache:
            return "No active GDP/GS programs."
        lines = [f"Active FAA programs ({len(self._cache)} airports):"]
        for ap, info in sorted(self._cache.items()):
            lines.append(
                f"  {ap:4s}  {info['type']:<35}  avg={info['delay_min']:.0f}min  {info['reason'][:50]}"
            )
        return "\n".join(lines)


# Module-level singleton for use in live inference
_default_client: GdpClient | None = None


def get_client() -> GdpClient:
    global _default_client
    if _default_client is None:
        _default_client = GdpClient()
    return _default_client


if __name__ == "__main__":
    client = GdpClient()
    print(client.status_report())

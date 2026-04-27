"""Pulls live ATL flights from AviationStack and emits a BTS-master-shaped CSV
ready to score with predict.py.

Reads API key from .env (AVIATIONSTACK_API_KEY=...).

Usage:
    python3 descarga_aviationstack.py 2026-04-25 2026-04-26
    python3 descarga_aviationstack.py 2026-04-25 2026-04-26 --out vuelos_live.csv
    python3 descarga_aviationstack.py 2026-04-25 --no-weather   # skip IEM enrichment
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
DISTANCE_LOOKUP = PROJECT_ROOT / "artifacts" / "distance_lookup.csv"

API_BASE = "http://api.aviationstack.com/v1/flights"  # free tier requires http
PAGE_SIZE = 100  # free tier max

AIRPORTS = {
    "ATL", "LGA", "MCO", "FLL", "MIA", "DFW", "DCA", "EWR",
    "TPA", "ORD", "PHL", "DEN", "LAX", "BWI", "LAS", "BOS",
}
TZ_BY_AIRPORT = {
    "ATL": "America/New_York", "LGA": "America/New_York",
    "MCO": "America/New_York", "FLL": "America/New_York",
    "MIA": "America/New_York", "DFW": "America/Chicago",
    "DCA": "America/New_York", "EWR": "America/New_York",
    "TPA": "America/New_York", "ORD": "America/Chicago",
    "PHL": "America/New_York", "DEN": "America/Denver",
    "LAX": "America/Los_Angeles", "BWI": "America/New_York",
    "LAS": "America/Los_Angeles", "BOS": "America/New_York",
}
NETWORK_BY_AIRPORT = {
    "ATL": "GA_ASOS", "LGA": "NY_ASOS", "MCO": "FL_ASOS", "FLL": "FL_ASOS",
    "MIA": "FL_ASOS", "DFW": "TX_ASOS", "DCA": "DC_ASOS", "EWR": "NJ_ASOS",
    "TPA": "FL_ASOS", "ORD": "IL_ASOS", "PHL": "PA_ASOS", "DEN": "CO_ASOS",
    "LAX": "CA_ASOS", "BWI": "MD_ASOS", "LAS": "NV_ASOS", "BOS": "MA_ASOS",
}
WX_VARS = ["tmpc", "dwpc", "relh", "drct", "sknt", "alti", "p01m", "vsby", "gust", "wxcodes"]


def load_api_key() -> str:
    key = os.environ.get("AVIATIONSTACK_API_KEY")
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("AVIATIONSTACK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("AVIATIONSTACK_API_KEY not found in env or .env file")


def fetch_flights_page(api_key: str, dep_iata: str, offset: int) -> dict:
    """Free-tier endpoint: no flight_date filter (paid feature). Returns current+recent."""
    params = {
        "access_key": api_key,
        "dep_iata": dep_iata,
        "limit": PAGE_SIZE,
        "offset": offset,
    }
    r = requests.get(API_BASE, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_flights_paginated(api_key: str, dep_iata: str, max_pages: int) -> list[dict]:
    """Page through all results sorted newest-first. Stops at max_pages or when exhausted."""
    out: list[dict] = []
    offset = 0
    for page in range(max_pages):
        payload = fetch_flights_page(api_key, dep_iata, offset)
        if "error" in payload:
            raise RuntimeError(f"AviationStack error: {payload['error']}")
        data = payload.get("data", [])
        out.extend(data)
        pagination = payload.get("pagination", {})
        total = int(pagination.get("total", 0))
        count = int(pagination.get("count", len(data)))
        offset += count
        print(f"    page {page + 1}/{max_pages} offset={offset - count} got={count} total={total}")
        if offset >= total or count == 0:
            break
        time.sleep(1)
    return out


def parse_iso_local(dt_str: str | None) -> pd.Timestamp:
    if not dt_str:
        return pd.NaT
    try:
        ts = pd.to_datetime(dt_str)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts
    except Exception:
        return pd.NaT


def hhmm_from_local(ts: pd.Timestamp) -> float:
    if pd.isna(ts):
        return np.nan
    return float(ts.hour * 100 + ts.minute)


def map_to_bts_schema(records: list[dict], distance_lookup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in records:
        dep = r.get("departure") or {}
        arr = r.get("arrival") or {}
        airline = r.get("airline") or {}
        flight = r.get("flight") or {}
        aircraft = r.get("aircraft") or {}

        origin = (dep.get("iata") or "").upper()
        dest = (arr.get("iata") or "").upper()
        if not origin or not dest:
            continue

        dep_sched_local = parse_iso_local(dep.get("scheduled"))
        arr_sched_local = parse_iso_local(arr.get("scheduled"))
        elapsed = np.nan
        if pd.notna(dep_sched_local) and pd.notna(arr_sched_local):
            elapsed = (arr_sched_local - dep_sched_local).total_seconds() / 60.0

        status = (r.get("flight_status") or "").lower()
        cancelled = 1 if status == "cancelled" else 0
        diverted = 1 if status == "diverted" else 0
        arr_delay = arr.get("delay")  # minutes, only if landed/active

        rows.append({
            "FL_DATE": dep_sched_local.normalize() if pd.notna(dep_sched_local) else pd.NaT,
            "YEAR": dep_sched_local.year if pd.notna(dep_sched_local) else np.nan,
            "MONTH": dep_sched_local.month if pd.notna(dep_sched_local) else np.nan,
            "DAY_OF_MONTH": dep_sched_local.day if pd.notna(dep_sched_local) else np.nan,
            "DAY_OF_WEEK": (dep_sched_local.weekday() + 1) if pd.notna(dep_sched_local) else np.nan,  # BTS uses 1=Mon..7=Sun
            "OP_CARRIER": (airline.get("iata") or "").upper(),
            "TAIL_NUM": (aircraft.get("registration") or ""),
            "OP_CARRIER_FL_NUM": flight.get("number"),
            "ORIGIN": origin,
            "DEST": dest,
            "CRS_DEP_TIME": hhmm_from_local(dep_sched_local),
            "CRS_ARR_TIME": hhmm_from_local(arr_sched_local),
            "DEP_TIME": hhmm_from_local(parse_iso_local(dep.get("actual"))),
            "ARR_TIME": hhmm_from_local(parse_iso_local(arr.get("actual"))),
            "DEP_DELAY": dep.get("delay"),
            "ARR_DELAY": arr_delay,
            "CRS_ELAPSED_TIME": elapsed,
            "ACTUAL_ELAPSED_TIME": np.nan,
            "AIR_TIME": np.nan,
            "DISTANCE": np.nan,  # filled later
            "CANCELLED": cancelled,
            "DIVERTED": diverted,
            "CARRIER_DELAY": np.nan,
            "WEATHER_DELAY": np.nan,
            "NAS_DELAY": np.nan,
            "SECURITY_DELAY": np.nan,
            "LATE_AIRCRAFT_DELAY": np.nan,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Filter to ATL-related + known airports (same rule as builder)
    mask = (df["ORIGIN"].eq("ATL") | df["DEST"].eq("ATL")) & \
           df["ORIGIN"].isin(AIRPORTS) & df["DEST"].isin(AIRPORTS)
    df = df.loc[mask].copy()

    # FLOW + PAR_AIRPORT
    df["FLOW_ATL"] = np.where(df["ORIGIN"].eq("ATL"), "DEP_FROM_ATL", "ARR_TO_ATL")
    df["PAR_AIRPORT"] = np.where(df["ORIGIN"].eq("ATL"), df["DEST"], df["ORIGIN"])

    # Distance lookup
    df = df.merge(distance_lookup, on=["ORIGIN", "DEST"], how="left", suffixes=("", "_lk"))
    df["DISTANCE"] = df["DISTANCE_lk"].combine_first(df["DISTANCE"])
    df = df.drop(columns=["DISTANCE_lk"])

    return df.reset_index(drop=True)


def add_utc_event_times(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    crs_min = pd.to_numeric(out["CRS_DEP_TIME"], errors="coerce")
    hours = np.floor(crs_min / 100)
    minutes = crs_min % 100
    out["CRS_DEP_MIN"] = hours * 60 + minutes
    out["DEP_LOCAL_DT"] = out["FL_DATE"] + pd.to_timedelta(out["CRS_DEP_MIN"], unit="m")

    out["EVENT_ORIGIN_UTC"] = pd.NaT
    for airport, tz_name in TZ_BY_AIRPORT.items():
        mask = out["ORIGIN"].eq(airport)
        if not mask.any():
            continue
        tz = ZoneInfo(tz_name)

        def _to_utc(dt):
            if pd.isna(dt):
                return pd.NaT
            try:
                return dt.replace(tzinfo=tz).astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
            except Exception:
                return pd.NaT

        out.loc[mask, "EVENT_ORIGIN_UTC"] = out.loc[mask, "DEP_LOCAL_DT"].apply(_to_utc)

    out["EVENT_ORIGIN_UTC"] = pd.to_datetime(out["EVENT_ORIGIN_UTC"], errors="coerce").astype("datetime64[ns]")
    elapsed = pd.to_numeric(out["CRS_ELAPSED_TIME"], errors="coerce")
    out["EVENT_DEST_UTC"] = (out["EVENT_ORIGIN_UTC"] + pd.to_timedelta(elapsed, unit="m")).astype("datetime64[ns]")
    return out


def fetch_iem_for_dates(stations: set[str], date_min_utc: pd.Timestamp, date_max_utc: pd.Timestamp) -> pd.DataFrame:
    """Pull METAR for the set of stations covering [min, max+1day) UTC window."""
    base = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    frames = []
    pad = pd.Timedelta(days=1)
    start = (date_min_utc - pad).floor("D")
    end = (date_max_utc + pad).ceil("D")
    for st in sorted(stations):
        if st not in NETWORK_BY_AIRPORT:
            continue
        params = [
            ("network", NETWORK_BY_AIRPORT[st]),
            ("station", st),
            ("year1", str(start.year)), ("month1", str(start.month)), ("day1", str(start.day)),
            ("year2", str(end.year)), ("month2", str(end.month)), ("day2", str(end.day)),
            ("tz", "Etc/UTC"), ("format", "onlycomma"),
            ("latlon", "no"), ("elev", "no"),
            ("missing", "M"), ("trace", "T"), ("direct", "no"),
            ("report_type", "3"), ("report_type", "4"),
        ] + [("data", v) for v in WX_VARS]
        try:
            r = requests.get(base, params=params, timeout=120)
            r.raise_for_status()
            text = r.text
            lines = text.splitlines()
            hdr_idx = next((i for i, ln in enumerate(lines) if ln.lower().startswith("station,valid")), None)
            if hdr_idx is None:
                continue
            df = pd.read_csv(StringIO("\n".join(lines[hdr_idx:])))
            df.columns = [c.strip().lower() for c in df.columns]
            df["station"] = df["station"].astype("string").str.strip().str.upper()
            df["valid"] = pd.to_datetime(df["valid"], errors="coerce", utc=True).dt.tz_convert(None).astype("datetime64[ns]")
            for c in ["tmpc", "dwpc", "relh", "drct", "sknt", "alti", "p01m", "vsby", "gust"]:
                df[c] = pd.to_numeric(df[c].replace({"M": np.nan, "T": 0.001}), errors="coerce")
            df["wx_precip_flag"] = df["p01m"].fillna(0) > 0
            df["wx_low_vis_flag"] = df["vsby"] < 3
            df["wx_strong_wind_flag"] = (df["sknt"] >= 20) | (df["gust"] >= 30)
            frames.append(df[[
                "station", "valid", "tmpc", "dwpc", "relh", "drct", "sknt", "alti",
                "p01m", "vsby", "gust", "wxcodes",
                "wx_precip_flag", "wx_low_vis_flag", "wx_strong_wind_flag",
            ]])
            print(f"    IEM {st}: {len(df)} obs")
        except Exception as e:
            print(f"    IEM {st}: FAIL ({e})")
        time.sleep(1)
    if not frames:
        return pd.DataFrame()
    wx = pd.concat(frames, ignore_index=True).drop_duplicates(["station", "valid"])
    return wx.sort_values(["station", "valid"]).reset_index(drop=True)


def merge_weather(flights: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    def _asof(left, station_col, event_col, prefix):
        parts = []
        for st in sorted(left[station_col].dropna().unique()):
            lsub = left[left[station_col].eq(st)].copy().sort_values(event_col)
            wsub = wx[wx["station"].eq(st)].copy().sort_values("valid")
            if lsub.empty or wsub.empty:
                continue
            merged = pd.merge_asof(
                lsub, wsub,
                left_on=event_col, right_on="valid",
                direction="nearest", tolerance=pd.Timedelta("90min"),
            )
            parts.append(merged)
        if not parts:
            return left
        merged_all = pd.concat(parts, ignore_index=True)
        gap = (merged_all[event_col] - merged_all["valid"]).abs().dt.total_seconds() / 60
        merged_all = merged_all.rename(columns={
            "valid": f"{prefix}_WX_VALID_UTC",
            "tmpc": f"{prefix}_WX_TMPC", "dwpc": f"{prefix}_WX_DWPC",
            "relh": f"{prefix}_WX_RELH", "drct": f"{prefix}_WX_DRCT",
            "sknt": f"{prefix}_WX_SKNT", "alti": f"{prefix}_WX_ALTI",
            "p01m": f"{prefix}_WX_P01M", "vsby": f"{prefix}_WX_VSBY",
            "gust": f"{prefix}_WX_GUST", "wxcodes": f"{prefix}_WX_CODES",
            "wx_precip_flag": f"{prefix}_WX_PRECIP_FLAG",
            "wx_low_vis_flag": f"{prefix}_WX_LOW_VIS_FLAG",
            "wx_strong_wind_flag": f"{prefix}_WX_STRONG_WIND_FLAG",
        })
        merged_all[f"{prefix}_WX_MATCH_GAP_MIN"] = gap
        return merged_all.drop(columns=["station"])

    origin_merge = _asof(flights, "ORIGIN", "EVENT_ORIGIN_UTC", "ORIG")
    return _asof(origin_merge, "DEST", "EVENT_DEST_UTC", "DEST")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dates", nargs="+", help="One or more YYYY-MM-DD dates")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "vuelos_aviationstack.csv")
    p.add_argument("--no-weather", action="store_true", help="Skip IEM enrichment")
    p.add_argument("--airports", nargs="+", default=["ATL"],
                   help="Departure airports to query (default: ATL)")
    p.add_argument("--max-pages", type=int, default=80,
                   help="Max API pages per airport (free tier = 100 calls/month total)")
    args = p.parse_args(argv)
    target_dates = set(args.dates)

    api_key = load_api_key()
    if not DISTANCE_LOOKUP.exists():
        raise FileNotFoundError(f"{DISTANCE_LOOKUP} not found. Generate it from training data.")
    distance_lookup = pd.read_csv(DISTANCE_LOOKUP, dtype={"ORIGIN": "string", "DEST": "string", "DISTANCE": "float"})

    all_records: list[dict] = []
    for ap in args.airports:
        print(f"Fetch {ap} (paginated, free tier — no date filter, will filter client-side to {sorted(target_dates)})")
        recs = fetch_flights_paginated(api_key, ap, args.max_pages)
        print(f"  {ap}: {len(recs)} flights returned")
        # Client-side date filter
        filt = [r for r in recs if r.get("flight_date") in target_dates]
        print(f"  {ap}: {len(filt)} flights match target dates")
        all_records.extend(filt)

    if not all_records:
        print("No flights matched target dates. Free tier returns ~7,500 flights total (~2-3 days back).")
        return 1

    print(f"\nMapping {len(all_records)} records to BTS schema...")
    df = map_to_bts_schema(all_records, distance_lookup)
    print(f"  {len(df)} after ATL+known-airports filter")
    if df.empty:
        print("Nothing to score.")
        return 1

    df = add_utc_event_times(df)

    if not args.no_weather:
        stations = set(df["ORIGIN"].dropna().unique()) | set(df["DEST"].dropna().unique())
        date_min = df["EVENT_ORIGIN_UTC"].min()
        date_max = df["EVENT_DEST_UTC"].max()
        print(f"\nFetching IEM weather for {len(stations)} stations, {date_min} → {date_max}")
        wx = fetch_iem_for_dates(stations, date_min, date_max)
        if not wx.empty:
            df = merge_weather(df, wx)
            print(f"  ORIG wx coverage: {df['ORIG_WX_VALID_UTC'].notna().mean()*100:.1f}%")
            print(f"  DEST wx coverage: {df['DEST_WX_VALID_UTC'].notna().mean()*100:.1f}%")

    df.to_csv(args.out, index=False)
    print(f"\n-> {args.out} ({len(df)} rows × {len(df.columns)} cols)")
    print(f"   ARR_DELAY ground truth available: {df['ARR_DELAY'].notna().sum()} flights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Real-time inference primitives: AeroAPI client, SQLite history buffer,
and feature builder that joins live scheduled flights with the buffer.

Designed to be invoked from cron-friendly CLIs:
    live_backfill.py   - one-time seed of the actuals table from 2025 master
    live_pull.py       - per-tick: pull schedules + weather, predict, store
    live_metrics.py    - query accuracy/AUC over completed predictions
"""
from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
DB_PATH = PROJECT_ROOT / "live_data.db"
DISTANCE_LOOKUP = PROJECT_ROOT / "artifacts" / "distance_lookup.csv"

AEROAPI_BASE = "https://aeroapi.flightaware.com/aeroapi"

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


# ----------------------------- env / api key ------------------------------

def load_aeroapi_key() -> str:
    key = os.environ.get("AEROAPI_KEY")
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("AEROAPI_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("AEROAPI_KEY not in env or .env file")


# ----------------------------- sqlite schema ------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS flights (
    fa_flight_id TEXT PRIMARY KEY,
    ident_iata TEXT,
    op_carrier TEXT,
    flight_number TEXT,
    tail_num TEXT,
    origin TEXT,
    dest TEXT,
    inbound_fa_flight_id TEXT,
    fl_date TEXT,           -- local date YYYY-MM-DD (origin tz)
    crs_dep_min INTEGER,    -- minutes after midnight local
    scheduled_out_utc TEXT, -- ISO
    scheduled_off_utc TEXT,
    scheduled_on_utc TEXT,
    scheduled_in_utc TEXT,
    crs_elapsed_min REAL,
    distance REAL,
    aircraft_type TEXT,
    cancelled INTEGER,
    diverted INTEGER,
    first_seen_utc TEXT NOT NULL,
    last_updated_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_flights_date_dep ON flights(fl_date, scheduled_off_utc);
CREATE INDEX IF NOT EXISTS idx_flights_tail ON flights(tail_num, scheduled_off_utc);
CREATE INDEX IF NOT EXISTS idx_flights_carrier ON flights(op_carrier, scheduled_off_utc);

CREATE TABLE IF NOT EXISTS predictions (
    fa_flight_id TEXT NOT NULL,
    predicted_at_utc TEXT NOT NULL,
    proba_delay REAL NOT NULL,
    predicted_delay INTEGER NOT NULL,
    PRIMARY KEY (fa_flight_id, predicted_at_utc)
);

CREATE TABLE IF NOT EXISTS actuals (
    fa_flight_id TEXT PRIMARY KEY,
    actual_out_utc TEXT,
    actual_off_utc TEXT,
    actual_on_utc TEXT,
    actual_in_utc TEXT,
    arr_delay_min REAL,        -- arrival_in actual - scheduled_in (minutes)
    departure_delay_min REAL,
    cancelled INTEGER,
    diverted INTEGER,
    settled_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS weather_obs (
    station TEXT NOT NULL,
    valid_utc TEXT NOT NULL,
    tmpc REAL, dwpc REAL, relh REAL, drct REAL, sknt REAL, alti REAL,
    p01m REAL, vsby REAL, gust REAL, wxcodes TEXT,
    wx_precip_flag INTEGER, wx_low_vis_flag INTEGER, wx_strong_wind_flag INTEGER,
    PRIMARY KEY (station, valid_utc)
);
CREATE INDEX IF NOT EXISTS idx_weather_station_time ON weather_obs(station, valid_utc);

CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_utc TEXT NOT NULL,
    finished_utc TEXT,
    flights_pulled INTEGER,
    flights_predicted INTEGER,
    actuals_updated INTEGER,
    weather_obs_added INTEGER,
    notes TEXT
);
"""


def open_db(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# ----------------------------- aeroapi client -----------------------------

def _api_get(path: str, params: dict | None = None, key: str | None = None,
             retries: int = 3, backoff_s: float = 8.0) -> dict:
    key = key or load_aeroapi_key()
    headers = {"x-apikey": key, "Accept": "application/json"}
    url = f"{AEROAPI_BASE}{path}"
    for attempt in range(retries):
        r = requests.get(url, headers=headers, params=params or {}, timeout=60)
        if r.status_code == 429:
            wait = backoff_s * (2 ** attempt)
            print(f"  [429] rate-limited, sleeping {wait:.0f}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"AeroAPI {r.status_code} {url}: {r.text[:300]}")
        return r.json()
    raise RuntimeError(f"AeroAPI exhausted retries on {url}")


def fetch_airport_flights(airport_icao: str, kind: str, start_iso: str, end_iso: str,
                          max_pages: int = 5, key: str | None = None) -> list[dict]:
    """kind in {'scheduled_departures','scheduled_arrivals','departures','arrivals'}.

    AeroAPI returns up to 15 flights per page with cursor pagination.
    """
    assert kind in ("scheduled_departures", "scheduled_arrivals", "departures", "arrivals")
    # Server-side max_pages controls how many internal pages the API aggregates
    # into a single response. Keep it at 1 to minimize per-call cost.
    params = {"start": start_iso, "end": end_iso, "max_pages": "1"}
    out: list[dict] = []
    next_cursor = None
    pages = 0
    while True:
        if next_cursor:
            params["cursor"] = next_cursor
        payload = _api_get(f"/airports/{airport_icao}/flights/{kind}", params, key)
        data = payload.get(kind, [])
        out.extend(data)
        pages += 1
        nxt = payload.get("links", {}).get("next") if payload.get("links") else None
        if not nxt or pages >= max_pages:
            break
        # parse cursor from the next URL
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(nxt).query)
        next_cursor = (q.get("cursor") or [None])[0]
        if not next_cursor:
            break
        time.sleep(2.0)  # spread cursor requests under typical rate limits
    return out


# ----------------------------- weather from IEM ---------------------------

def fetch_iem_obs(stations: set[str], start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    base = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    frames = []
    for st in sorted(stations):
        if st not in NETWORK_BY_AIRPORT:
            continue
        params = [
            ("network", NETWORK_BY_AIRPORT[st]),
            ("station", st),
            ("year1", str(start_utc.year)), ("month1", str(start_utc.month)), ("day1", str(start_utc.day)),
            ("year2", str(end_utc.year)), ("month2", str(end_utc.month)), ("day2", str(end_utc.day)),
            ("tz", "Etc/UTC"), ("format", "onlycomma"),
            ("latlon", "no"), ("elev", "no"),
            ("missing", "M"), ("trace", "T"), ("direct", "no"),
            ("report_type", "3"), ("report_type", "4"),
        ] + [("data", v) for v in WX_VARS]
        try:
            r = requests.get(base, params=params, timeout=120)
            r.raise_for_status()
            lines = r.text.splitlines()
            hdr = next((i for i, ln in enumerate(lines) if ln.lower().startswith("station,valid")), None)
            if hdr is None:
                continue
            df = pd.read_csv(StringIO("\n".join(lines[hdr:])))
            df.columns = [c.strip().lower() for c in df.columns]
            df["station"] = df["station"].astype("string").str.strip().str.upper()
            df["valid"] = pd.to_datetime(df["valid"], errors="coerce", utc=True).dt.tz_convert(None).astype("datetime64[ns]")
            for c in ["tmpc", "dwpc", "relh", "drct", "sknt", "alti", "p01m", "vsby", "gust"]:
                df[c] = pd.to_numeric(df[c].replace({"M": np.nan, "T": 0.001}), errors="coerce")
            df["wx_precip_flag"] = (df["p01m"].fillna(0) > 0).astype(int)
            df["wx_low_vis_flag"] = (df["vsby"] < 3).astype(int)
            df["wx_strong_wind_flag"] = ((df["sknt"] >= 20) | (df["gust"] >= 30)).astype(int)
            frames.append(df[["station", "valid", "tmpc", "dwpc", "relh", "drct", "sknt", "alti",
                              "p01m", "vsby", "gust", "wxcodes",
                              "wx_precip_flag", "wx_low_vis_flag", "wx_strong_wind_flag"]])
        except Exception as e:
            print(f"  IEM {st}: FAIL ({e})")
        time.sleep(1)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(["station", "valid"]).sort_values(["station", "valid"]).reset_index(drop=True)


def upsert_weather(conn: sqlite3.Connection, wx: pd.DataFrame) -> int:
    if wx.empty:
        return 0
    rows = [
        (r.station, r.valid.isoformat(),
         _f(r.tmpc), _f(r.dwpc), _f(r.relh), _f(r.drct), _f(r.sknt), _f(r.alti),
         _f(r.p01m), _f(r.vsby), _f(r.gust), r.wxcodes if pd.notna(r.wxcodes) else None,
         int(r.wx_precip_flag), int(r.wx_low_vis_flag), int(r.wx_strong_wind_flag))
        for r in wx.itertuples(index=False)
    ]
    cur = conn.executemany(
        """INSERT OR REPLACE INTO weather_obs
           (station, valid_utc, tmpc, dwpc, relh, drct, sknt, alti, p01m, vsby, gust, wxcodes,
            wx_precip_flag, wx_low_vis_flag, wx_strong_wind_flag)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return cur.rowcount


def _f(v):
    return float(v) if pd.notna(v) else None


# ----------------------------- aeroapi → schema ---------------------------

def aeroapi_to_flight_row(rec: dict) -> dict | None:
    """Map an AeroAPI flight record to our flights-table schema."""
    origin = (rec.get("origin") or {}).get("code_iata") or ""
    dest = (rec.get("destination") or {}).get("code_iata") or ""
    origin = origin.upper()
    dest = dest.upper()
    if not origin or not dest:
        return None
    if origin not in AIRPORTS or dest not in AIRPORTS:
        return None
    if not (origin == "ATL" or dest == "ATL"):
        return None

    sched_out = rec.get("scheduled_out")
    if not sched_out:
        return None
    sched_out_ts = pd.to_datetime(sched_out, utc=True).tz_convert(None)
    sched_in = pd.to_datetime(rec.get("scheduled_in"), utc=True).tz_convert(None) if rec.get("scheduled_in") else None
    elapsed_min = float((sched_in - sched_out_ts).total_seconds() / 60) if sched_in is not None else None

    # local-time schedule
    tz = ZoneInfo(TZ_BY_AIRPORT.get(origin, "UTC"))
    local_dt = sched_out_ts.tz_localize("UTC").astimezone(tz)
    fl_date = local_dt.strftime("%Y-%m-%d")
    crs_dep_min = local_dt.hour * 60 + local_dt.minute

    return {
        "fa_flight_id": rec["fa_flight_id"],
        "ident_iata": rec.get("ident_iata"),
        "op_carrier": rec.get("operator_iata"),
        "flight_number": str(rec.get("flight_number")) if rec.get("flight_number") else None,
        "tail_num": rec.get("registration"),
        "origin": origin,
        "dest": dest,
        "inbound_fa_flight_id": rec.get("inbound_fa_flight_id"),
        "fl_date": fl_date,
        "crs_dep_min": crs_dep_min,
        "scheduled_out_utc": sched_out_ts.isoformat(),
        "scheduled_off_utc": pd.to_datetime(rec["scheduled_off"], utc=True).tz_convert(None).isoformat() if rec.get("scheduled_off") else None,
        "scheduled_on_utc": pd.to_datetime(rec["scheduled_on"], utc=True).tz_convert(None).isoformat() if rec.get("scheduled_on") else None,
        "scheduled_in_utc": sched_in.isoformat() if sched_in is not None else None,
        "crs_elapsed_min": elapsed_min,
        "distance": rec.get("route_distance"),
        "aircraft_type": rec.get("aircraft_type"),
        "cancelled": 1 if rec.get("cancelled") else 0,
        "diverted": 1 if rec.get("diverted") else 0,
    }


def upsert_flights(conn: sqlite3.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    payload = [
        (r["fa_flight_id"], r["ident_iata"], r["op_carrier"], r["flight_number"],
         r["tail_num"], r["origin"], r["dest"], r["inbound_fa_flight_id"],
         r["fl_date"], r["crs_dep_min"],
         r["scheduled_out_utc"], r["scheduled_off_utc"], r["scheduled_on_utc"], r["scheduled_in_utc"],
         r["crs_elapsed_min"], r["distance"], r["aircraft_type"],
         r["cancelled"], r["diverted"], now, now)
        for r in rows
    ]
    conn.executemany(
        """INSERT INTO flights
           (fa_flight_id, ident_iata, op_carrier, flight_number, tail_num,
            origin, dest, inbound_fa_flight_id, fl_date, crs_dep_min,
            scheduled_out_utc, scheduled_off_utc, scheduled_on_utc, scheduled_in_utc,
            crs_elapsed_min, distance, aircraft_type, cancelled, diverted,
            first_seen_utc, last_updated_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(fa_flight_id) DO UPDATE SET
             tail_num=excluded.tail_num,
             scheduled_out_utc=excluded.scheduled_out_utc,
             scheduled_off_utc=excluded.scheduled_off_utc,
             scheduled_on_utc=excluded.scheduled_on_utc,
             scheduled_in_utc=excluded.scheduled_in_utc,
             crs_elapsed_min=excluded.crs_elapsed_min,
             cancelled=excluded.cancelled,
             diverted=excluded.diverted,
             last_updated_utc=excluded.last_updated_utc""",
        payload,
    )
    conn.commit()
    return len(payload)


def upsert_actuals_from_aeroapi(conn: sqlite3.Connection, recs: list[dict]) -> int:
    """For arrivals/departures records that have actuals, write them to actuals table."""
    if not recs:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for r in recs:
        sched_in = r.get("scheduled_in")
        actual_in = r.get("actual_in")
        arr_delay_min = None
        if sched_in and actual_in:
            arr_delay_min = (pd.to_datetime(actual_in) - pd.to_datetime(sched_in)).total_seconds() / 60
        elif r.get("arrival_delay") is not None:
            arr_delay_min = r["arrival_delay"] / 60.0  # AeroAPI returns seconds
        dep_delay_min = (r["departure_delay"] / 60.0) if r.get("departure_delay") is not None else None

        rows.append((
            r["fa_flight_id"],
            r.get("actual_out"),
            r.get("actual_off"),
            r.get("actual_on"),
            r.get("actual_in"),
            arr_delay_min,
            dep_delay_min,
            1 if r.get("cancelled") else 0,
            1 if r.get("diverted") else 0,
            now,
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO actuals
           (fa_flight_id, actual_out_utc, actual_off_utc, actual_on_utc, actual_in_utc,
            arr_delay_min, departure_delay_min, cancelled, diverted, settled_at_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return len(rows)


# ----------------------------- feature builder ----------------------------

def build_inference_frame(conn: sqlite3.Connection, fa_flight_ids: list[str],
                          history_days: int = 7) -> pd.DataFrame:
    """Build a master-shaped DataFrame for the given new flights + recent history.

    History flights bring known ARR_DELAY so lineage/rolling features can be computed.
    """
    if not fa_flight_ids:
        return pd.DataFrame()

    # Pull target flights from `flights`
    placeholders = ",".join("?" for _ in fa_flight_ids)
    target = pd.read_sql_query(
        f"SELECT * FROM flights WHERE fa_flight_id IN ({placeholders})",
        conn, params=fa_flight_ids,
    )
    if target.empty:
        return pd.DataFrame()

    # Pull history: completed flights (have actuals) within the last N days
    earliest = (datetime.now(timezone.utc) - timedelta(days=history_days)).isoformat()
    history = pd.read_sql_query(
        """SELECT f.*, a.arr_delay_min, a.departure_delay_min, a.cancelled AS act_cancelled,
                  a.diverted AS act_diverted
           FROM flights f
           JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
           WHERE f.scheduled_off_utc >= ?""",
        conn, params=(earliest,),
    )

    df = pd.concat([
        history.assign(_role="history"),
        target.assign(_role="target", arr_delay_min=np.nan),
    ], ignore_index=True)

    # Map to master schema (the columns predict.py / lineage.py expect)
    df["FL_DATE"] = pd.to_datetime(df["fl_date"])
    df["YEAR"] = df["FL_DATE"].dt.year
    df["MONTH"] = df["FL_DATE"].dt.month
    df["DAY_OF_MONTH"] = df["FL_DATE"].dt.day
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.weekday + 1  # 1=Mon
    df["OP_CARRIER"] = df["op_carrier"].astype("string")
    df["TAIL_NUM"] = df["tail_num"].astype("string")
    df["OP_CARRIER_FL_NUM"] = pd.to_numeric(df["flight_number"], errors="coerce")
    df["ORIGIN"] = df["origin"].astype("string")
    df["DEST"] = df["dest"].astype("string")
    df["CRS_DEP_MIN"] = pd.to_numeric(df["crs_dep_min"], errors="coerce")
    df["CRS_DEP_TIME"] = (df["CRS_DEP_MIN"] // 60) * 100 + (df["CRS_DEP_MIN"] % 60)
    df["CRS_ELAPSED_TIME"] = pd.to_numeric(df["crs_elapsed_min"], errors="coerce")
    df["DISTANCE"] = pd.to_numeric(df["distance"], errors="coerce")
    df["CANCELLED"] = df["cancelled"].fillna(0).astype(int)
    df["DIVERTED"] = df["diverted"].fillna(0).astype(int)
    df["ARR_DELAY"] = pd.to_numeric(df["arr_delay_min"], errors="coerce")

    # Event timestamps
    df["EVENT_ORIGIN_UTC"] = pd.to_datetime(df["scheduled_off_utc"], errors="coerce").astype("datetime64[ns]")
    df["EVENT_DEST_UTC"] = pd.to_datetime(df["scheduled_on_utc"], errors="coerce").astype("datetime64[ns]")
    df["DEP_LOCAL_DT"] = df["FL_DATE"] + pd.to_timedelta(df["CRS_DEP_MIN"], unit="m")

    df["FLOW_ATL"] = np.where(df["ORIGIN"].eq("ATL"), "DEP_FROM_ATL", "ARR_TO_ATL")
    df["PAR_AIRPORT"] = np.where(df["ORIGIN"].eq("ATL"), df["DEST"], df["ORIGIN"])

    # Weather merge_asof from weather_obs table
    if df["EVENT_ORIGIN_UTC"].notna().any():
        wx_min = df["EVENT_ORIGIN_UTC"].min() - timedelta(hours=2)
        wx_max = df["EVENT_DEST_UTC"].max() + timedelta(hours=2)
        wx = pd.read_sql_query(
            "SELECT * FROM weather_obs WHERE valid_utc >= ? AND valid_utc <= ?",
            conn, params=(wx_min.isoformat(), wx_max.isoformat()),
        )
        if not wx.empty:
            wx["valid"] = pd.to_datetime(wx["valid_utc"]).astype("datetime64[ns]")
            wx = wx.sort_values(["station", "valid"]).reset_index(drop=True)
            df = _merge_weather_asof(df, wx, "ORIGIN", "EVENT_ORIGIN_UTC", "ORIG")
            df = _merge_weather_asof(df, wx, "DEST", "EVENT_DEST_UTC", "DEST")

    return df


def _merge_weather_asof(left: pd.DataFrame, wx: pd.DataFrame, station_col: str,
                        event_col: str, prefix: str) -> pd.DataFrame:
    parts = []
    for st in sorted(left[station_col].dropna().unique()):
        lsub = left[left[station_col].eq(st)].sort_values(event_col).copy()
        wsub = wx[wx["station"].eq(st)].sort_values("valid").copy()
        if lsub.empty or wsub.empty:
            parts.append(lsub)
            continue
        merged = pd.merge_asof(
            lsub, wsub, left_on=event_col, right_on="valid",
            direction="nearest", tolerance=pd.Timedelta("90min"),
        )
        parts.append(merged)
    if not parts:
        return left
    out = pd.concat(parts, ignore_index=True)
    rename = {
        "valid": f"{prefix}_WX_VALID_UTC",
        "tmpc": f"{prefix}_WX_TMPC", "dwpc": f"{prefix}_WX_DWPC",
        "relh": f"{prefix}_WX_RELH", "drct": f"{prefix}_WX_DRCT",
        "sknt": f"{prefix}_WX_SKNT", "alti": f"{prefix}_WX_ALTI",
        "p01m": f"{prefix}_WX_P01M", "vsby": f"{prefix}_WX_VSBY",
        "gust": f"{prefix}_WX_GUST", "wxcodes": f"{prefix}_WX_CODES",
        "wx_precip_flag": f"{prefix}_WX_PRECIP_FLAG",
        "wx_low_vis_flag": f"{prefix}_WX_LOW_VIS_FLAG",
        "wx_strong_wind_flag": f"{prefix}_WX_STRONG_WIND_FLAG",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    if event_col in out.columns and f"{prefix}_WX_VALID_UTC" in out.columns:
        out[f"{prefix}_WX_MATCH_GAP_MIN"] = (
            (out[event_col] - out[f"{prefix}_WX_VALID_UTC"]).abs().dt.total_seconds() / 60
        )
    return out.drop(columns=["station"], errors="ignore")

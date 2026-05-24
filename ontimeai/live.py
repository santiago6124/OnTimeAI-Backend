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
DB_PATH = Path(os.environ.get("DB_PATH", str(PROJECT_ROOT / "live_data.db")))
DISTANCE_LOOKUP = PROJECT_ROOT / "artifacts" / "distance_lookup.csv"

AEROAPI_BASE = "https://aeroapi.flightaware.com/aeroapi"

# Load airport universe from CSV at import time. Falls back to the legacy
# 16-airport hub-spoke set if the CSV doesn't exist (e.g. fresh checkout
# before running build_airports_universe.py).
_LEGACY_AIRPORTS = {
    "ATL", "LGA", "MCO", "FLL", "MIA", "DFW", "DCA", "EWR",
    "TPA", "ORD", "PHL", "DEN", "LAX", "BWI", "LAS", "BOS",
}
_LEGACY_TZ = {
    "ATL": "America/New_York", "LGA": "America/New_York",
    "MCO": "America/New_York", "FLL": "America/New_York",
    "MIA": "America/New_York", "DFW": "America/Chicago",
    "DCA": "America/New_York", "EWR": "America/New_York",
    "TPA": "America/New_York", "ORD": "America/Chicago",
    "PHL": "America/New_York", "DEN": "America/Denver",
    "LAX": "America/Los_Angeles", "BWI": "America/New_York",
    "LAS": "America/Los_Angeles", "BOS": "America/New_York",
}
_LEGACY_NETWORK = {
    "ATL": "GA_ASOS", "LGA": "NY_ASOS", "MCO": "FL_ASOS", "FLL": "FL_ASOS",
    "MIA": "FL_ASOS", "DFW": "TX_ASOS", "DCA": "DC_ASOS", "EWR": "NJ_ASOS",
    "TPA": "FL_ASOS", "ORD": "IL_ASOS", "PHL": "PA_ASOS", "DEN": "CO_ASOS",
    "LAX": "CA_ASOS", "BWI": "MD_ASOS", "LAS": "NV_ASOS", "BOS": "MA_ASOS",
}


def _load_airports_universe() -> tuple[set[str], dict[str, str], dict[str, str]]:
    """Returns (AIRPORTS set, TZ_BY_AIRPORT, NETWORK_BY_AIRPORT) from CSV.

    Falls back to the 16-hub legacy set if the universe CSV is missing.
    """
    universe_path = PROJECT_ROOT / "airports_universe.csv"
    if not universe_path.exists():
        return _LEGACY_AIRPORTS, dict(_LEGACY_TZ), dict(_LEGACY_NETWORK)
    df = pd.read_csv(universe_path)
    df["iata"] = df["iata"].astype(str).str.strip().str.upper()
    airports = set(df["iata"].tolist())
    tz_by = dict(zip(df["iata"], df["timezone"], strict=False))
    net_by = dict(zip(df["iata"], df["iem_network"], strict=False))
    return airports, tz_by, net_by


AIRPORTS, TZ_BY_AIRPORT, NETWORK_BY_AIRPORT = _load_airports_universe()
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
    stable_id TEXT,
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
-- idx_flights_stable created in _migrate_stable_ids() since the column may
-- not yet exist on legacy databases.

CREATE TABLE IF NOT EXISTS predictions (
    fa_flight_id TEXT NOT NULL,
    stable_id TEXT,
    predicted_at_utc TEXT NOT NULL,
    proba_delay REAL NOT NULL,
    predicted_delay INTEGER NOT NULL,
    threshold_used REAL,
    threshold_strategy TEXT,
    PRIMARY KEY (fa_flight_id, predicted_at_utc)
);
-- idx_predictions_stable created in _migrate_stable_ids().

CREATE TABLE IF NOT EXISTS actuals (
    fa_flight_id TEXT PRIMARY KEY,
    stable_id TEXT,
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
-- idx_actuals_stable created in _migrate_stable_ids().

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


def stable_id(fa_flight_id: str | None) -> str | None:
    """Strip AeroAPI's unstable trailing suffix from `fa_flight_id`.

    AeroAPI's IDs follow `<IDENT>-<UNIX_TS>-<source>-<suffix>`. The suffix
    differs between `/scheduled_*` and `/departures` / `/arrivals` for the
    same physical flight. The first two parts (IDENT + timestamp) are stable
    and uniquely identify a flight instance.

    Example:
        scheduled: AAL1811-1777701683-airline-1446p
        departed:  AAL1811-1777701683-airline-1447p
        stable:    AAL1811-1777701683
    """
    if not fa_flight_id:
        return None
    parts = str(fa_flight_id).split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return fa_flight_id


def open_db(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    _migrate_predictions_threshold(conn)
    _migrate_stable_ids(conn)
    conn.commit()
    return conn


def _migrate_predictions_threshold(conn: sqlite3.Connection) -> None:
    """Add threshold_used / threshold_strategy columns to existing predictions tables."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    if "threshold_used" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN threshold_used REAL")
    if "threshold_strategy" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN threshold_strategy TEXT")


def _migrate_stable_ids(conn: sqlite3.Connection) -> None:
    """Add and backfill `stable_id` columns on flights / predictions / actuals.

    Without this, joins by `fa_flight_id` miss matches because AeroAPI returns
    different suffixes for the same flight from /scheduled_* vs /departures.
    """
    for table in ("flights", "predictions", "actuals"):
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if "stable_id" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN stable_id TEXT")
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_stable ON {table}(stable_id)"
            )
            # Backfill existing rows
            rows = conn.execute(f"SELECT fa_flight_id FROM {table}").fetchall()
            updates = [(stable_id(r[0]), r[0]) for r in rows if r[0]]
            if updates:
                conn.executemany(
                    f"UPDATE {table} SET stable_id = ? WHERE fa_flight_id = ?",
                    updates,
                )


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
            r = requests.get(base, params=params, timeout=10)
            if r.status_code == 429:
                print(f"  IEM {st}: rate-limited, skipping")
                time.sleep(0.3)
                continue
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
        time.sleep(0.3)
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

def aeroapi_to_flight_row(rec: dict, *, strict_airport_filter: bool = True) -> dict | None:
    """Map an AeroAPI flight record to our flights-table schema.

    `strict_airport_filter=True` (default) restricts to ATL-touching flights
    between the 16 known airports — the right policy for schedule pulls.
    Set to False for chain-walked inbound flights, whose origin may be
    outside the tracked network but whose ARR_DELAY still feeds lineage.
    """
    origin = (rec.get("origin") or {}).get("code_iata") or ""
    dest = (rec.get("destination") or {}).get("code_iata") or ""
    origin = origin.upper()
    dest = dest.upper()
    if not origin or not dest:
        return None
    if strict_airport_filter:
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
        (r["fa_flight_id"], stable_id(r["fa_flight_id"]),
         r["ident_iata"], r["op_carrier"], r["flight_number"],
         r["tail_num"], r["origin"], r["dest"], r["inbound_fa_flight_id"],
         r["fl_date"], r["crs_dep_min"],
         r["scheduled_out_utc"], r["scheduled_off_utc"], r["scheduled_on_utc"], r["scheduled_in_utc"],
         r["crs_elapsed_min"], r["distance"], r["aircraft_type"],
         r["cancelled"], r["diverted"], now, now)
        for r in rows
    ]
    conn.executemany(
        """INSERT INTO flights
           (fa_flight_id, stable_id, ident_iata, op_carrier, flight_number, tail_num,
            origin, dest, inbound_fa_flight_id, fl_date, crs_dep_min,
            scheduled_out_utc, scheduled_off_utc, scheduled_on_utc, scheduled_in_utc,
            crs_elapsed_min, distance, aircraft_type, cancelled, diverted,
            first_seen_utc, last_updated_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(fa_flight_id) DO UPDATE SET
             stable_id=excluded.stable_id,
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


def fetch_flight_by_id(fa_flight_id: str, key: str | None = None) -> list[dict]:
    """Fetch a single AeroAPI flight by its `fa_flight_id`.

    Returns the `flights` array from `/flights/{id}` (typically 1 element).
    Raises RuntimeError on non-2xx after retries.
    """
    payload = _api_get(f"/flights/{fa_flight_id}", key=key)
    return payload.get("flights") or []


def chain_walk_inbound(
    conn: sqlite3.Connection,
    target_rows: list[dict],
    *,
    key: str | None = None,
    max_calls: int = 20,
) -> tuple[int, int]:
    """For each target with an `inbound_fa_flight_id` not yet in `actuals`,
    fetch the inbound flight and persist it (flights + actuals).

    This is the high-ROI path that hydrates `prev_arr_delay_tail` without
    waiting for a 24-48h history backfill. Caller is expected to pass a
    dedup'd list of target flight rows (post `aeroapi_to_flight_row`).

    Returns (calls_made, actuals_written). Hard caps at `max_calls` to
    protect the AeroAPI budget.
    """
    inbound_ids: list[str] = []
    seen: set[str] = set()
    for r in target_rows:
        inbound = r.get("inbound_fa_flight_id")
        if not inbound or inbound in seen:
            continue
        seen.add(inbound)
        # Skip if EITHER the literal fa_flight_id OR the stable_id is already
        # settled — covers both pre-stable-id rows and the post-fix layout.
        cur = conn.execute(
            "SELECT actual_in_utc FROM actuals "
            "WHERE fa_flight_id = ? OR stable_id = ? LIMIT 1",
            (inbound, stable_id(inbound)),
        )
        existing = cur.fetchone()
        if existing and existing[0]:
            continue  # already settled
        inbound_ids.append(inbound)
        if len(inbound_ids) >= max_calls:
            break

    calls = 0
    actuals_written = 0
    for inb_id in inbound_ids:
        try:
            flights = fetch_flight_by_id(inb_id, key=key)
            calls += 1
        except RuntimeError as e:
            print(f"  chain-walk: failed for {inb_id}: {e}")
            continue
        for flight in flights:
            row = aeroapi_to_flight_row(flight, strict_airport_filter=False)
            if row:
                upsert_flights(conn, [row])
            if flight.get("actual_in"):
                upsert_actuals_from_aeroapi(conn, [flight])
                actuals_written += 1
        time.sleep(2.0)  # respect AeroAPI rate limits

    return calls, actuals_written


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
            stable_id(r["fa_flight_id"]),
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
           (fa_flight_id, stable_id,
            actual_out_utc, actual_off_utc, actual_on_utc, actual_in_utc,
            arr_delay_min, departure_delay_min, cancelled, diverted, settled_at_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
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

    # Pull history: completed flights (have actuals) within the last N days.
    # JOIN by stable_id (not fa_flight_id) because AeroAPI returns slightly
    # different fa_flight_id suffixes from /scheduled_* vs /departures-/arrivals
    # endpoints — the actuals could be orphaned from their flight rows otherwise.
    earliest = (datetime.now(timezone.utc) - timedelta(days=history_days)).isoformat()
    history = pd.read_sql_query(
        """SELECT f.*, a.arr_delay_min, a.departure_delay_min, a.cancelled AS act_cancelled,
                  a.diverted AS act_diverted
           FROM flights f
           JOIN actuals a ON a.stable_id = f.stable_id
           WHERE f.scheduled_off_utc >= ?
             AND f.stable_id IS NOT NULL""",
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

    # NA-safe: harvester (FR24) puede traer vuelos privados sin ORIGIN/DEST.
    # En pandas 3.x, Series.eq() ya no coerce NA → False, así que np.where(NA, ...)
    # rompe con "TypeError: boolean value of NA is ambiguous".
    _is_atl_origin = df["ORIGIN"].fillna("").eq("ATL")
    df["FLOW_ATL"] = np.where(_is_atl_origin, "DEP_FROM_ATL", "ARR_TO_ATL")
    df["PAR_AIRPORT"] = np.where(_is_atl_origin, df["DEST"], df["ORIGIN"])

    # Weather merge_asof from weather_obs table
    if df["EVENT_ORIGIN_UTC"].notna().any():
        wx_min = df["EVENT_ORIGIN_UTC"].min() - timedelta(hours=2)
        wx_max = df["EVENT_DEST_UTC"].max() + timedelta(hours=2)
        wx = pd.read_sql_query(
            "SELECT * FROM weather_obs WHERE valid_utc >= ? AND valid_utc <= ?",
            conn, params=(wx_min.isoformat(), wx_max.isoformat()),
        )
        if not wx.empty:
            wx["valid"] = pd.to_datetime(wx["valid_utc"], format="ISO8601", utc=True).dt.tz_convert(None).astype("datetime64[ns]")
            wx = wx.sort_values(["station", "valid"]).reset_index(drop=True)
            df = _merge_weather_asof(df, wx, "ORIGIN", "EVENT_ORIGIN_UTC", "ORIG")
            df = _merge_weather_asof(df, wx, "DEST", "EVENT_DEST_UTC", "DEST")

    # v7 features: BEARING_DEG + ERA5 wind at cruise altitude
    df = _add_v7_wind_features(df)

    # v8 features: PageRank + GDP flag (both gracefully degrade if unavailable)
    df = _add_v8_features(df)

    # v9: AIRCRAFT_FAMILY — use aircraft_type from FlightAware (already ICAO typecode)
    df = _add_aircraft_family(df)

    return df


# ERA5 grids loaded once per process (lazy, cached at module level)
_ERA5_GRIDS: dict | None = None


def _add_v7_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega BEARING_DEG y features ERA5 viento 250hPa al inference frame."""
    global _ERA5_GRIDS
    _np = np
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from feature_engineering_v7.airport_lookup import great_circle_bearing, airport_coords
        from feature_engineering_v7.era5_wind_features import Era5WindGrid, _midpoint as era5_midpoint
        import numpy as _np

        # BEARING_DEG por par único ORIGIN-DEST
        pairs = df[["ORIGIN", "DEST"]].drop_duplicates().copy()
        pairs["BEARING_DEG"] = pairs.apply(
            lambda r: great_circle_bearing(r["ORIGIN"], r["DEST"]), axis=1
        )
        df = df.merge(pairs, on=["ORIGIN", "DEST"], how="left")

        # Cargar ERA5 grids una sola vez
        if _ERA5_GRIDS is None:
            era5_dir = PROJECT_ROOT / "data_raw" / "era5_wind"
            if era5_dir.exists():
                grids: dict = {}
                for p in sorted(era5_dir.glob("era5_wind_*.nc")):
                    parts = p.stem.split("_")
                    if len(parts) >= 4:
                        try:
                            grids[(int(parts[2]), int(parts[3]))] = Era5WindGrid(p)
                        except Exception:
                            pass
                _ERA5_GRIDS = grids
            else:
                _ERA5_GRIDS = {}

        if not _ERA5_GRIDS:
            for c in ["ERA5_U_KT", "ERA5_V_KT", "ERA5_HEADWIND_KT", "ERA5_CROSSWIND_KT"]:
                df[c] = _np.nan
            df["ERA5_TAILWIND_FLAG"] = 0
            return df

        n = len(df)
        era5_u = _np.full(n, _np.nan)
        era5_v = _np.full(n, _np.nan)

        for (origin, dest, year, month), idx in df.groupby(["ORIGIN", "DEST", "YEAR", "MONTH"]).indices.items():
            midpt = era5_midpoint(origin, dest)
            bearing_vals = df["BEARING_DEG"].iloc[list(idx)]
            bearing = bearing_vals.iloc[0] if not bearing_vals.isna().all() else None
            if midpt is None or bearing is None:
                continue
            grid = _ERA5_GRIDS.get((int(year), int(month)))
            if grid is None:
                candidates = [(abs(y - int(year)), g) for (y, m), g in _ERA5_GRIDS.items() if m == int(month)]
                if candidates:
                    grid = min(candidates, key=lambda x: x[0])[1]
            if grid is None:
                continue
            u_kt, v_kt = grid.wind_at(midpt[0], midpt[1])
            era5_u[list(idx)] = u_kt
            era5_v[list(idx)] = v_kt

        df["ERA5_U_KT"] = era5_u
        df["ERA5_V_KT"] = era5_v
        bearing_rad = _np.radians(df["BEARING_DEG"].values.astype(float))
        flight_e = _np.sin(bearing_rad)
        flight_n = _np.cos(bearing_rad)
        tailwind = era5_u * flight_e + era5_v * flight_n
        df["ERA5_HEADWIND_KT"] = -tailwind
        df["ERA5_CROSSWIND_KT"] = era5_u * flight_n - era5_v * flight_e
        df["ERA5_TAILWIND_FLAG"] = (df["ERA5_HEADWIND_KT"] < -15).astype("int8")

    except Exception as e:
        # Si falla por cualquier razón, rellenar con NaN para no romper inferencia
        import warnings
        warnings.warn(f"v7 wind features failed: {e}")
        for c in ["BEARING_DEG", "ERA5_U_KT", "ERA5_V_KT", "ERA5_HEADWIND_KT", "ERA5_CROSSWIND_KT"]:
            if c not in df.columns:
                df[c] = float("nan") if "ERA5" in c or "BEARING" in c else 0
        if "ERA5_TAILWIND_FLAG" not in df.columns:
            df["ERA5_TAILWIND_FLAG"] = 0

    return df


_PAGERANK_PATH_LIVE = PROJECT_ROOT / "artifacts" / "airport_pagerank.json"
_GDP_CLIENT = None


_AIRCRAFT_FAMILY_LOOKUP: dict | None = None
_AIRCRAFT_FAMILY_LOOKUP_PATH = PROJECT_ROOT / "artifacts" / "tail_to_aircraft_family.json"


def _add_aircraft_family(df: pd.DataFrame) -> pd.DataFrame:
    """Add AIRCRAFT_FAMILY from aircraft_type (FlightAware ICAO typecode) or TAIL_NUM lookup.

    Priority:
      1. aircraft_type column (already an ICAO typecode from FlightAware)
      2. TAIL_NUM → OpenSky lookup JSON
      3. Carrier-based fallback
      4. "OTHER"
    """
    global _AIRCRAFT_FAMILY_LOOKUP
    import warnings
    try:
        from feature_engineering_v7.aircraft_type import _FAMILY_MAP, _CARRIER_FALLBACK
        import json

        # Option 1: aircraft_type column from flights table (ICAO typecode)
        if "aircraft_type" in df.columns:
            mapped = df["aircraft_type"].str.strip().str.upper().map(_FAMILY_MAP)
        else:
            mapped = pd.Series(pd.NA, index=df.index)

        # Option 2: TAIL_NUM lookup for rows where aircraft_type was missing
        missing = mapped.isna()
        if missing.any() and "TAIL_NUM" in df.columns:
            if _AIRCRAFT_FAMILY_LOOKUP is None and _AIRCRAFT_FAMILY_LOOKUP_PATH.exists():
                _AIRCRAFT_FAMILY_LOOKUP = json.loads(_AIRCRAFT_FAMILY_LOOKUP_PATH.read_text())
            if _AIRCRAFT_FAMILY_LOOKUP:
                mapped[missing] = df.loc[missing, "TAIL_NUM"].map(_AIRCRAFT_FAMILY_LOOKUP)

        # Option 3: carrier fallback
        still_missing = mapped.isna()
        if still_missing.any() and "OP_CARRIER" in df.columns:
            mapped[still_missing] = df.loc[still_missing, "OP_CARRIER"].map(_CARRIER_FALLBACK)

        df["AIRCRAFT_FAMILY"] = mapped.fillna("OTHER").astype("category")
    except Exception as e:
        warnings.warn(f"AIRCRAFT_FAMILY feature failed: {e}")
        df["AIRCRAFT_FAMILY"] = "OTHER"
    return df


def _add_v8_features(df: "pd.DataFrame") -> "pd.DataFrame":
    """Add v8 features: ORIGIN_PAGERANK, DEST_PAGERANK, GDP_FLAG, GDP_DELAY_MIN.

    All features degrade gracefully: if artifact/API is unavailable they
    fill with 0/NaN so inference is never blocked.
    NAS_RATE_2H is not computed live (no real-time NAS breakdown available);
    GDP_FLAG from the FAA API is the live equivalent.
    """
    global _GDP_CLIENT
    import warnings

    try:
        from feature_engineering_v7.v8_features import add_pagerank_features
        df = add_pagerank_features(df, lookup_path=_PAGERANK_PATH_LIVE)
    except Exception as e:
        warnings.warn(f"v8 PageRank features failed: {e}")
        if "ORIGIN_PAGERANK" not in df.columns:
            df["ORIGIN_PAGERANK"] = 0.0
        if "DEST_PAGERANK" not in df.columns:
            df["DEST_PAGERANK"] = 0.0

    try:
        from feature_engineering_v7.gdp_scraper import GdpClient
        if _GDP_CLIENT is None:
            _GDP_CLIENT = GdpClient()
        df = _GDP_CLIENT.add_gdp_features(df)
    except Exception as e:
        warnings.warn(f"v8 GDP features failed: {e}")
        if "GDP_FLAG" not in df.columns:
            df["GDP_FLAG"] = 0
        if "GDP_DELAY_MIN" not in df.columns:
            df["GDP_DELAY_MIN"] = 0.0

    return df


def _merge_weather_asof(left: pd.DataFrame, wx: pd.DataFrame, station_col: str,
                        event_col: str, prefix: str) -> pd.DataFrame:
    parts = []
    for st in sorted(left[station_col].dropna().unique()):
        lsub = left[left[station_col].eq(st)].sort_values(event_col).copy()
        wsub = wx[wx["station"].eq(st)].sort_values("valid").copy()
        # pandas 3.0: merge_asof raises on null keys — split, merge valid rows only
        lsub_null = lsub[lsub[event_col].isna()]
        lsub = lsub[lsub[event_col].notna()]
        if lsub.empty or wsub.empty:
            parts.append(pd.concat([lsub, lsub_null]) if not lsub_null.empty else lsub)
            continue
        merged = pd.merge_asof(
            lsub, wsub, left_on=event_col, right_on="valid",
            direction="nearest", tolerance=pd.Timedelta("90min"),
        )
        parts.append(merged)
        if not lsub_null.empty:
            parts.append(lsub_null)
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

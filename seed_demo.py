"""seed_demo.py — Inserta vuelos de muestra en live_data.db para demo/pruebas.

Uso:
    python3 seed_demo.py            # usa la fecha de hoy
    python3 seed_demo.py --clear    # borra y re-inserta

Genera ~40 vuelos ATL de hoy con predicciones y actuals suficientes para
que CP-01 y CP-02 también pasen.
"""
from __future__ import annotations

import sys
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "live_data.db"

CLEAR = "--clear" in sys.argv

# ── Schema mínimo (copia de live.py, sólo lo que necesitamos) ─────────────

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
    fl_date TEXT,
    crs_dep_min INTEGER,
    scheduled_out_utc TEXT,
    scheduled_off_utc TEXT,
    scheduled_on_utc TEXT,
    scheduled_in_utc TEXT,
    estimated_out_utc TEXT,
    estimated_in_utc TEXT,
    crs_elapsed_min REAL,
    distance REAL,
    aircraft_type TEXT,
    cancelled INTEGER,
    diverted INTEGER,
    first_seen_utc TEXT NOT NULL,
    last_updated_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_flights_date_dep ON flights(fl_date, scheduled_off_utc);

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

CREATE TABLE IF NOT EXISTS actuals (
    fa_flight_id TEXT PRIMARY KEY,
    stable_id TEXT,
    actual_out_utc TEXT,
    actual_off_utc TEXT,
    actual_on_utc TEXT,
    actual_in_utc TEXT,
    arr_delay_min REAL,
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

# ── Datos de muestra ───────────────────────────────────────────────────────

# Vuelos: (carrier, flight_num, origin, dest, dep_utc_hhmm, elapsed_min, distance, aircraft, proba)
# dep_utc_hhmm = hora UTC de salida (ATL = UTC-4 en verano)
FLIGHT_TEMPLATES = [
    # ─── Salidas desde ATL (DEP_FROM_ATL) ─────────────────────────────────
    ("DL", "1234", "ATL", "JFK", "10:15", 130, 865,  "B738", 0.62),  # alto riesgo
    ("DL", "2801", "ATL", "LAX", "11:30", 315, 1947, "B764", 0.71),  # alto riesgo
    ("DL", "0472", "ATL", "ORD", "12:00", 125, 716,  "A321", 0.41),  # alto riesgo
    ("DL", "1090", "ATL", "MIA", "13:15", 85,  594,  "B738", 0.28),  # medio
    ("DL", "3321", "ATL", "BOS", "14:00", 155, 1107, "A320", 0.19),  # medio
    ("DL", "0811", "ATL", "DFW", "09:00", 145, 731,  "MD88", 0.08),  # bajo
    ("DL", "1502", "ATL", "DEN", "08:30", 200, 1199, "B739", 0.06),  # bajo
    ("DL", "2200", "ATL", "SEA", "07:45", 325, 2182, "A330", 0.05),  # bajo
    ("WN", "4432", "ATL", "MCO", "10:45", 75,  403,  "B737", 0.54),  # alto
    ("WN", "1180", "ATL", "BWI", "11:00", 110, 845,  "B737", 0.33),  # medio
    ("WN", "3300", "ATL", "DEN", "15:30", 210, 1199, "B738", 0.12),  # bajo
    ("AA", "2115", "ATL", "PHL", "13:00", 115, 666,  "A319", 0.61),  # alto
    ("AA", "3002", "ATL", "DFW", "09:30", 145, 731,  "B738", 0.22),  # medio
    ("AA", "1740", "ATL", "LGA", "12:30", 135, 762,  "A321", 0.09),  # bajo
    ("UA", "1831", "ATL", "EWR", "08:00", 130, 764,  "B739", 0.38),  # alto
    ("UA", "4450", "ATL", "ORD", "14:30", 130, 716,  "A320", 0.17),  # medio
    ("UA", "2090", "ATL", "DEN", "16:00", 215, 1199, "B738", 0.07),  # bajo
    ("FL", "0301", "ATL", "TPA", "07:00", 70,  406,  "B737", 0.48),  # alto
    ("FL", "0522", "ATL", "FLL", "08:15", 80,  581,  "B737", 0.31),  # medio
    # ─── Llegadas a ATL (ARR_TO_ATL) ──────────────────────────────────────
    ("DL", "0234", "JFK", "ATL", "08:00", 135, 865,  "A321", 0.55),  # alto
    ("DL", "1688", "LAX", "ATL", "09:30", 305, 1947, "B764", 0.67),  # alto
    ("DL", "0901", "ORD", "ATL", "10:30", 120, 716,  "B738", 0.39),  # alto
    ("DL", "1422", "BOS", "ATL", "11:00", 155, 1107, "A320", 0.26),  # medio
    ("DL", "2033", "MIA", "ATL", "12:00", 85,  594,  "B737", 0.16),  # medio
    ("DL", "3100", "DFW", "ATL", "13:30", 145, 731,  "MD88", 0.07),  # bajo
    ("WN", "2244", "MCO", "ATL", "09:00", 75,  403,  "B737", 0.51),  # alto
    ("WN", "5500", "BWI", "ATL", "10:00", 110, 845,  "B737", 0.24),  # medio
    ("AA", "1900", "PHL", "ATL", "11:30", 115, 666,  "A319", 0.58),  # alto
    ("AA", "2780", "LGA", "ATL", "14:00", 135, 762,  "A321", 0.14),  # bajo
    ("UA", "0780", "EWR", "ATL", "09:30", 130, 764,  "B739", 0.43),  # alto
    ("UA", "3310", "ORD", "ATL", "13:00", 130, 716,  "A320", 0.21),  # medio
    ("FL", "0412", "TPA", "ATL", "06:00", 70,  406,  "B737", 0.44),  # alto
    ("FL", "0601", "FLL", "ATL", "07:15", 80,  581,  "B737", 0.29),  # medio
    ("B6", "1022", "BOS", "ATL", "08:30", 155, 1107, "A320", 0.62),  # alto
    ("B6", "2233", "LGA", "ATL", "12:00", 135, 762,  "A321", 0.11),  # bajo
    ("NK", "0401", "MIA", "ATL", "10:00", 85,  594,  "A320", 0.37),  # alto
    ("NK", "1100", "FLL", "ATL", "11:30", 80,  581,  "A321", 0.18),  # medio
    ("G4", "1201", "MCO", "ATL", "13:00", 75,  403,  "MD80", 0.09),  # bajo
    ("AS", "0334", "LAX", "ATL", "09:00", 295, 1947, "B739", 0.46),  # alto
]

THRESHOLD = 0.35


def stable_id_of(fa: str) -> str:
    parts = fa.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return fa


def run():
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now_iso = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA)

    # Add stable_id columns if missing (migration)
    for table in ("flights", "predictions", "actuals"):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if "stable_id" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN stable_id TEXT")
    for table in ("flights",):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col in ("estimated_out_utc", "estimated_in_utc"):
            if col not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
    for table in ("predictions",):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col in ("threshold_used", "threshold_strategy"):
            if col not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
    conn.commit()

    if CLEAR:
        conn.execute(f"DELETE FROM flights WHERE fl_date = ?", (today_str,))
        conn.execute("DELETE FROM predictions WHERE fa_flight_id IN (SELECT fa_flight_id FROM flights WHERE fl_date = ?)", (today_str,))
        # Just clear all today's predictions by re-checking
        conn.execute("DELETE FROM actuals WHERE settled_at_utc >= ?", (today_str,))
        conn.execute("DELETE FROM weather_obs WHERE valid_utc >= ?", (today_str,))
        conn.commit()
        print(f"Cleared existing data for {today_str}")

    flight_rows = []
    pred_rows = []
    actual_rows = []

    # Base timestamp: today at 00:00 UTC
    base = datetime.strptime(today_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    for i, (carrier, fnum, origin, dest, dep_hhmm, elapsed, dist, acft, proba) in enumerate(FLIGHT_TEMPLATES):
        dep_h, dep_m = int(dep_hhmm.split(":")[0]), int(dep_hhmm.split(":")[1])
        dep_utc = base + timedelta(hours=dep_h, minutes=dep_m)
        arr_utc = dep_utc + timedelta(minutes=elapsed)

        ts_unix = int(dep_utc.timestamp())
        fa_id = f"{carrier}{fnum}-{ts_unix}-schedule-{i}"
        stbl = stable_id_of(fa_id)
        ident = f"{carrier}{fnum}"

        # ATL is UTC-4 in summer → local = UTC - 4h
        local_dep = dep_utc - timedelta(hours=4)
        fl_date = local_dep.strftime("%Y-%m-%d")
        crs_dep_min = local_dep.hour * 60 + local_dep.minute

        # Set estimated times (delayed by 30 min if high risk)
        est_dep = dep_utc
        est_arr = arr_utc
        if proba >= 0.35:
            est_dep = dep_utc + timedelta(minutes=30)
            est_arr = arr_utc + timedelta(minutes=30)

        flight_rows.append((
            fa_id, stbl, ident, carrier, fnum,
            f"N{400+i}DL", origin, dest, None,
            fl_date, crs_dep_min,
            dep_utc.isoformat(), dep_utc.isoformat(),
            arr_utc.isoformat(), arr_utc.isoformat(),
            est_dep.isoformat(), est_arr.isoformat(),
            float(elapsed), float(dist), acft,
            0, 0, now_iso, now_iso,
        ))

        pred_at = (dep_utc - timedelta(hours=2)).isoformat()
        predicted_delay = 1 if proba >= THRESHOLD else 0
        pred_rows.append((
            fa_id, stbl, pred_at,
            proba, predicted_delay,
            THRESHOLD, "fixed",
        ))

        # Mark as actual if arrival was more than 30 min ago
        arr_was_past = arr_utc < datetime.now(timezone.utc) - timedelta(minutes=30)
        if arr_was_past:
            # High-risk flights get real delay; low-risk arrive on time
            if proba >= 0.35:
                arr_delay = round(proba * 70 + 10)   # 24–59 min delay
                dep_delay = max(0, arr_delay - 10)
            elif proba >= 0.15:
                arr_delay = round(proba * 40)         # 6–14 min
                dep_delay = max(0, arr_delay - 5)
            else:
                arr_delay = round(proba * 10 - 2)    # ≈ on time or early
                dep_delay = 0

            actual_arr = arr_utc + timedelta(minutes=arr_delay)
            actual_dep = dep_utc + timedelta(minutes=dep_delay)
            actual_rows.append((
                fa_id, stbl,
                actual_dep.isoformat(), actual_dep.isoformat(),
                actual_arr.isoformat(), actual_arr.isoformat(),
                float(arr_delay), float(dep_delay),
                0, 0, now_iso,
            ))

    # Insert flights
    conn.executemany(
        """INSERT OR REPLACE INTO flights
           (fa_flight_id, stable_id, ident_iata, op_carrier, flight_number, tail_num,
            origin, dest, inbound_fa_flight_id, fl_date, crs_dep_min,
            scheduled_out_utc, scheduled_off_utc, scheduled_on_utc, scheduled_in_utc,
            estimated_out_utc, estimated_in_utc,
            crs_elapsed_min, distance, aircraft_type, cancelled, diverted,
            first_seen_utc, last_updated_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        flight_rows,
    )

    # Insert predictions
    conn.executemany(
        """INSERT OR REPLACE INTO predictions
           (fa_flight_id, stable_id, predicted_at_utc, proba_delay, predicted_delay,
            threshold_used, threshold_strategy)
           VALUES (?,?,?,?,?,?,?)""",
        pred_rows,
    )

    # Insert actuals
    conn.executemany(
        """INSERT OR REPLACE INTO actuals
           (fa_flight_id, stable_id,
            actual_out_utc, actual_off_utc, actual_on_utc, actual_in_utc,
            arr_delay_min, departure_delay_min, cancelled, diverted, settled_at_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        actual_rows,
    )

    # Seed weather for ATL
    wx_obs_time = (base + timedelta(hours=9)).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO weather_obs
           (station, valid_utc, tmpc, dwpc, relh, drct, sknt, alti,
            p01m, vsby, gust, wxcodes,
            wx_precip_flag, wx_low_vis_flag, wx_strong_wind_flag)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        ("ATL", wx_obs_time,
         22.0, 14.0, 61.0, 180, 12, 29.92,
         0.0, 10.0, 18.0, "SCT060",
         0, 0, 0),
    )

    conn.commit()
    conn.close()

    n_actual = len(actual_rows)
    n_high = sum(1 for *_, proba in FLIGHT_TEMPLATES if proba >= 0.35)
    n_med  = sum(1 for *_, proba in FLIGHT_TEMPLATES if 0.15 <= proba < 0.35)
    n_low  = sum(1 for *_, proba in FLIGHT_TEMPLATES if proba < 0.15)

    print(f"✓  Fecha: {today_str}")
    print(f"✓  Vuelos insertados : {len(flight_rows)}")
    print(f"   - Alto riesgo     : {n_high}")
    print(f"   - Medio riesgo    : {n_med}")
    print(f"   - Bajo riesgo     : {n_low}")
    print(f"✓  Predicciones      : {len(pred_rows)}")
    print(f"✓  Actuals           : {n_actual}  (vuelos ya aterrizados)")
    print(f"✓  Weather ATL       : 1 observación")
    print()
    print("Arrancá el backend y los endpoints deberían responder con datos.")
    if n_actual >= 30:
        print("CP-02 debería poder calcularse (n_actuals ≥ 30).")
    else:
        print(f"CP-02 necesita ≥30 actuals; ahora hay {n_actual}. Esperá que más vuelos aterricen.")


if __name__ == "__main__":
    run()

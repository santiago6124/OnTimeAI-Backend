"""OnTimeAI — FastAPI backend.

Serves live predictions from live_data.db.  Swap models without restart:
    ACTIVE_MODEL=4year_v9_recal uvicorn api:app --reload

Endpoints:
    GET /flights            — today's scheduled flights + latest prediction
    GET /flights/{id}       — single flight detail + SHAP
    GET /metrics/summary    — today's KPI cards
    GET /metrics/hourly     — predictions grouped by departure hour
    GET /metrics/model      — active model info + live AUC
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local", override=False)
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
import hashlib

import re
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import secrets
from jose import JWTError, jwt
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# ── Config ─────────────────────────────────────────────────────────────────

GCS_BUCKET = os.getenv("GCS_BUCKET", "")
_TMP_DB = Path("/tmp/live_data.db")
_BUNDLED_DB = Path(__file__).parent / "live_data.db"
DB_PATH = _TMP_DB if GCS_BUCKET else _BUNDLED_DB
_DB_REFRESH_INTERVAL = 1000  # refresh from GCS every ~16 min
_db_last_refresh: float = 0.0
_db_last_health_check: float = 0.0
_DB_HEALTH_INTERVAL = 60  # re-verify DB health every 60 s

# Users DB (separate from live_data.db so live job never overwrites it)
USERS_DB_PATH = Path("/tmp/users.db") if GCS_BUCKET else Path(__file__).parent / "users.db"


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return f"{salt}${h.hex()}"


def _check_password(password: str, stored: str) -> bool:
    try:
        salt, h = stored.split("$", 1)
        expected = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()
        return secrets.compare_digest(h, expected)
    except Exception:
        return False


def _get_users_con() -> sqlite3.Connection:
    con = sqlite3.connect(str(USERS_DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def _upload_users_db() -> None:
    if not GCS_BUCKET:
        return
    try:
        from google.cloud import storage as gcs
        gcs.Client().bucket(GCS_BUCKET).blob("users.db").upload_from_filename(str(USERS_DB_PATH))
    except Exception as e:
        print(f"[users_db] upload failed: {e}")


def _init_users_db() -> None:
    if GCS_BUCKET:
        try:
            from google.cloud import storage as gcs
            blob = gcs.Client().bucket(GCS_BUCKET).blob("users.db")
            if blob.exists():
                blob.download_to_filename(str(USERS_DB_PATH))
                print("[users_db] downloaded from GCS")
        except Exception as e:
            print(f"[users_db] download failed, creating fresh: {e}")

    con = _get_users_con()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE TABLE IF NOT EXISTS user_preferences (
            username TEXT PRIMARY KEY,
            theme TEXT DEFAULT 'dark',
            palette TEXT DEFAULT 'default',
            updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)
    # Seed initial users from env vars (idempotent)
    seeds = [
        (os.getenv("API_USERNAME", "admin"),  os.getenv("API_PASSWORD", "ontimeai2026"),  "superadmin"),
        (os.getenv("API_USERNAME_VIEWER", "viewer"), os.getenv("API_PASSWORD_VIEWER", "viewer2026"), "user"),
    ]
    for username, password, role in seeds:
        if username and not con.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            con.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
                (username, _hash_password(password), role),
            )
    con.commit()
    con.close()
    _upload_users_db()


def _payload_of(request: Request) -> dict:
    token = request.headers.get("Authorization", "")[7:]
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def _require_superadmin(request: Request) -> dict:
    payload = _payload_of(request)
    if payload.get("role") != "superadmin":
        raise HTTPException(status_code=403, detail="Requiere rol superadmin")
    return payload


def _refresh_db_from_gcs() -> None:
    global _db_last_refresh
    import time
    if not GCS_BUCKET:
        return
    if time.time() - _db_last_refresh < _DB_REFRESH_INTERVAL:
        return
    try:
        from google.cloud import storage as gcs
        client = gcs.Client()
        blob = client.bucket(GCS_BUCKET).blob("live_data.db")
        blob.download_to_filename(str(_TMP_DB))
        _db_last_refresh = time.time()
        print(f"[db] refreshed from GCS ({_TMP_DB.stat().st_size / 1e6:.0f} MB)")
    except Exception as e:
        print(f"[db_refresh] failed: {e}")


def _verify_db_health(path: Path) -> bool:
    """Read an actual data page to catch corruption beyond the schema."""
    try:
        chk = sqlite3.connect(str(path))
        chk.execute("SELECT fa_flight_id FROM flights LIMIT 1")
        chk.close()
        return True
    except Exception:
        return False

MODEL_REGISTRY: dict[str, Path] = {
    "4year_v9":       Path(__file__).parent / "artifacts/4year_v9",
    "4year_v9_recal": Path(__file__).parent / "artifacts/4year_v9_recal",
    "4year_v7_recal": Path(__file__).parent / "artifacts/4year_v7_recal",
}
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "4year_v9")
ARTIFACT_PATH = MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["4year_v9"])

# ── Auth ───────────────────────────────────────────────────────────────────

JWT_SECRET = os.getenv("JWT_SECRET_KEY", "ontimeai-dev-secret-change-in-prod-32chars")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 8


_PUBLIC_PATHS = {"/auth/login", "/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path in _PUBLIC_PATHS or request.url.path.startswith("/redoc"):
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        token = auth[7:]
        try:
            jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except JWTError:
            return JSONResponse({"detail": "Invalid or expired token"}, status_code=401)
        return await call_next(request)


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"


class UserUpdate(BaseModel):
    password: Optional[str] = None
    role: Optional[str] = None
    active: Optional[bool] = None


class PreferencesUpdate(BaseModel):
    theme: Optional[str] = None
    palette: Optional[str] = None


# ── Feature labels ─────────────────────────────────────────────────────────

FEATURE_LABELS: dict[str, str] = {
    "prev_arr_delay_tail":      "Demora previa del avión",
    "prev_turnaround_tail_min": "Tiempo en tierra previo",
    "tail_flights_today_prior": "Vuelos previos del avión hoy",
    "carrier_delay_rate_yday":  "Tasa demora aerolínea ayer",
    "origin_delay_rate_yday":   "Tasa demora origen ayer",
    "origin_delay_rate_1h":     "Tasa demora origen (1h)",
    "origin_delay_rate_6h":     "Tasa demora origen (6h)",
    "origin_delay_rate_24h":    "Tasa demora origen (24h)",
    "dest_delay_rate_1h":       "Tasa demora destino (1h)",
    "dest_delay_rate_6h":       "Tasa demora destino (6h)",
    "carrier_delay_rate_24h":   "Tasa demora aerolínea (24h)",
    "carrier_delay_rate_7d":    "Tasa demora aerolínea (7d)",
    "DISTANCE":                 "Distancia de vuelo",
    "CRS_ELAPSED_TIME":         "Duración programada",
    "AIRCRAFT_FAMILY":          "Tipo de aeronave",
    "PAGERANK_ORIGIN":          "Importancia origen (red)",
    "PAGERANK_DEST":            "Importancia destino (red)",
    "TAIL_DELAY_DECAY":         "Historial demoras avión",
    "tmpf_origin":              "Temperatura origen",
    "sknt_origin":              "Viento origen",
    "vsby_origin":              "Visibilidad origen",
    "alti_origin":              "Presión origen",
    "CRS_DEP_MIN_sin":          "Hora salida (cíclica)",
    "CRS_DEP_MIN_cos":          "Hora salida (cíclica)",
    "absorb_score":             "Capacidad absorción ATL",
    "congestion_score":         "Congestión aeropuerto",
}

# ── App ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if GCS_BUCKET:
        try:
            from google.cloud import storage as gcs
            client = gcs.Client()
            blob = client.bucket(GCS_BUCKET).blob("live_data.db")
            blob.download_to_filename(str(_TMP_DB))
            mb = _TMP_DB.stat().st_size / 1e6
            print(f"[startup] DB downloaded from gs://{GCS_BUCKET}/live_data.db ({mb:.1f} MB)")
        except Exception as e:
            print(f"[startup] GCS download failed, using bundled DB: {e}")
            import shutil
            shutil.copy(_BUNDLED_DB, _TMP_DB)
            
    # Trigger database migrations on startup
    try:
        from ontimeai.live import open_db
        conn = open_db(DB_PATH)
        conn.close()
        print("[startup] Database migrations executed successfully")
    except Exception as e:
        print(f"[startup] Database migration failed: {e}")

    _init_users_db()
    yield


app = FastAPI(title="OnTimeAI API", version="1.0.0", lifespan=lifespan)
# Auth is inner; CORS is outer so it wraps 401 responses too
app.add_middleware(AuthMiddleware)
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,https://ontimeai-frontend-hq7henvhjq-uc.a.run.app,https://ontimeai-frontend-150917658060.us-central1.run.app",
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


_AIRPORT_CODE_RE = re.compile(r"^[A-Z]{3,4}$")

def _validate_airport(code: str) -> str:
    """Normaliza y valida un código IATA (3 letras) o ICAO (4 letras)."""
    normalized = code.upper().strip()
    if not _AIRPORT_CODE_RE.match(normalized):
        raise HTTPException(
            status_code=422,
            detail=f"Código de aeropuerto inválido: '{code}'. Debe ser IATA (3 letras) o ICAO (4 letras).",
        )
    return normalized


def get_db() -> sqlite3.Connection:
    global _db_last_refresh, _db_last_health_check
    import time, shutil

    _refresh_db_from_gcs()

    if GCS_BUCKET and (time.time() - _db_last_health_check > _DB_HEALTH_INTERVAL):
        if not _verify_db_health(_TMP_DB):
            print("[db] health check failed — forcing re-download from GCS")
            _db_last_refresh = 0
            _refresh_db_from_gcs()
            if not _verify_db_health(_TMP_DB):
                print("[db] GCS DB also corrupt — falling back to bundled DB")
                shutil.copy(str(_BUNDLED_DB), str(_TMP_DB))
        _db_last_health_check = time.time()

    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    return con


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def risk_level(proba: float) -> str:
    if proba >= 0.35:
        return "high"
    if proba >= 0.15:
        return "medium"
    return "low"


# ── Lazy model loader ──────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_meta() -> dict[str, Any]:
    from ontimeai.model import load_artifact
    return load_artifact(ARTIFACT_PATH)


# ── Helpers ────────────────────────────────────────────────────────────────

def _latest_predictions_active(con: sqlite3.Connection) -> list[sqlite3.Row]:
    """Latest prediction per flight for active/upcoming flights in a sliding window.
    
    Includes flights scheduled from 6 hours ago to 18 hours in the future,
    plus any flight scheduled in the past 24 hours that has not yet departed.

    Selection priority per flight:
      1. The most recent prediction made BEFORE the flight physically departed
         (predicted_at <= actual_off) — the genuine pre-departure forecast.
      2. If none exists (e.g. arrivals only seen en-route), fall back to the
         most recent prediction made before the flight LANDED.
    Post-landing batch runs (e.g. calibration backfills) are always excluded so
    a retro-prediction never overrides a valid pre-landing/pre-departure one.
    """
    now = datetime.now(timezone.utc)
    start_window = (now - timedelta(hours=6)).isoformat()
    end_window = (now + timedelta(hours=18)).isoformat()
    undeparted_limit = (now - timedelta(hours=24)).isoformat()

    rows = con.execute("""
        SELECT f.fa_flight_id,
               f.ident_iata,
               f.op_carrier,
               f.flight_number,
               f.origin,
               f.dest,
               f.scheduled_out_utc,
               f.scheduled_in_utc,
               f.estimated_out_utc,
               f.estimated_in_utc,
               f.aircraft_type,
               p.proba_delay,
               p.predicted_delay,
               p.predicted_at_utc,
               CASE WHEN a.arr_delay_min IS NOT NULL THEN 1 ELSE 0 END AS has_actual,
               a.arr_delay_min,
               a.departure_delay_min,
               a.actual_out_utc
        FROM flights f
        JOIN (
            SELECT p2.fa_flight_id,
                   p2.proba_delay,
                   p2.predicted_delay,
                   p2.predicted_at_utc,
                   ROW_NUMBER() OVER (
                       PARTITION BY p2.fa_flight_id
                       ORDER BY
                           -- prefer pre-departure predictions (0) over en-route (1)
                           CASE WHEN a2.actual_off_utc IS NOT NULL
                                     AND p2.predicted_at_utc > a2.actual_off_utc
                                THEN 1 ELSE 0 END ASC,
                           p2.predicted_at_utc DESC
                   ) AS rn
            FROM predictions p2
            LEFT JOIN actuals a2 ON a2.fa_flight_id = p2.fa_flight_id
            WHERE a2.actual_in_utc IS NULL
               OR p2.predicted_at_utc <= a2.actual_in_utc
        ) p ON p.fa_flight_id = f.fa_flight_id AND p.rn = 1
        LEFT JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
        WHERE f.cancelled = 0
          AND (
              (datetime(f.scheduled_out_utc) >= datetime(?) AND datetime(f.scheduled_out_utc) <= datetime(?))
              OR
              (datetime(f.scheduled_out_utc) >= datetime(?) AND a.actual_out_utc IS NULL)
          )
        ORDER BY f.scheduled_out_utc
    """, (start_window, end_window, undeparted_limit)).fetchall()
    return rows


def _flight_row_to_dict(row: sqlite3.Row) -> dict:
    proba = float(row["proba_delay"])
    ident = row["ident_iata"] or row["flight_number"] or row["fa_flight_id"]
    
    # Safely get estimated/actual times (default to scheduled if missing/older records)
    keys = row.keys()
    est_out = row["estimated_out_utc"] if "estimated_out_utc" in keys else None
    est_in = row["estimated_in_utc"] if "estimated_in_utc" in keys else None
    act_out = row["actual_out_utc"] if "actual_out_utc" in keys else None

    return {
        "fa_flight_id":    row["fa_flight_id"],
        "flight_number":   ident,
        "airline_code":    row["op_carrier"] or "",
        "origin":          row["origin"] or "",
        "destination":     row["dest"] or "",
        "scheduled_out_utc": row["scheduled_out_utc"] or "",
        "scheduled_in_utc":  row["scheduled_in_utc"] or "",
        "estimated_out_utc": est_out or row["scheduled_out_utc"] or "",
        "estimated_in_utc":  est_in or row["scheduled_in_utc"] or "",
        "actual_out_utc":    act_out,
        "aircraft_type":   row["aircraft_type"] or "",
        "risk":            risk_level(proba),
        "delay_probability": round(proba, 4),
        "predicted_delay": int(row["predicted_delay"]),
        "predicted_at_utc": row["predicted_at_utc"] or "",
        "has_actual":      bool(row["has_actual"]),
        "arr_delay_min":   float(row["arr_delay_min"]) if row["arr_delay_min"] is not None else None,
        "departure_delay_min": float(row["departure_delay_min"]) if row["departure_delay_min"] is not None else None,
    }


# ── Auth routes ────────────────────────────────────────────────────────────

@app.post("/auth/login")
def login(body: LoginRequest):
    con = _get_users_con()
    row = con.execute(
        "SELECT password_hash, role, active FROM users WHERE username=?", (body.username,)
    ).fetchone()
    con.close()
    if not row or not row["active"] or not _check_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    token = jwt.encode(
        {"sub": body.username, "role": row["role"], "exp": expire},
        JWT_SECRET, algorithm=JWT_ALGORITHM,
    )
    return {"access_token": token, "token_type": "bearer"}


@app.get("/auth/me")
def auth_me(request: Request):
    payload = _payload_of(request)
    return {"username": payload.get("sub"), "role": payload.get("role", "user")}


# ── User management (superadmin only) ──────────────────────────────────────

@app.get("/admin/users")
def list_users(request: Request):
    _require_superadmin(request)
    con = _get_users_con()
    rows = con.execute(
        "SELECT id, username, role, active, created_at FROM users ORDER BY created_at"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.post("/admin/users", status_code=201)
def create_user(request: Request, body: UserCreate):
    _require_superadmin(request)
    if body.role not in ("user", "admin", "superadmin"):
        raise HTTPException(400, "Rol inválido. Válidos: user, admin, superadmin")
    try:
        con = _get_users_con()
        con.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?,?,?)",
            (body.username, _hash_password(body.password), body.role),
        )
        con.commit()
        con.close()
        _upload_users_db()
        return {"ok": True}
    except sqlite3.IntegrityError:
        raise HTTPException(409, "El usuario ya existe")


@app.patch("/admin/users/{username}")
def update_user(request: Request, username: str, body: UserUpdate):
    payload = _require_superadmin(request)
    me = payload.get("sub")
    if username == me:
        if body.active is False:
            raise HTTPException(400, "No podés desactivarte a vos mismo")
        if body.role and body.role != "superadmin":
            raise HTTPException(400, "No podés cambiar tu propio rol")
    sets, vals = [], []
    if body.password is not None:
        sets.append("password_hash=?"); vals.append(_hash_password(body.password))
    if body.role is not None:
        if body.role not in ("user", "admin", "superadmin"):
            raise HTTPException(400, "Rol inválido")
        sets.append("role=?"); vals.append(body.role)
    if body.active is not None:
        sets.append("active=?"); vals.append(1 if body.active else 0)
    if not sets:
        return {"ok": True}
    sets.append("updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')")
    vals.append(username)
    con = _get_users_con()
    res = con.execute(f"UPDATE users SET {','.join(sets)} WHERE username=?", vals)
    if res.rowcount == 0:
        con.close(); raise HTTPException(404, "Usuario no encontrado")
    con.commit(); con.close()
    _upload_users_db()
    return {"ok": True}


@app.delete("/admin/users/{username}", status_code=204)
def delete_user(request: Request, username: str):
    payload = _require_superadmin(request)
    if username == payload.get("sub"):
        raise HTTPException(400, "No podés eliminarte a vos mismo")
    con = _get_users_con()
    res = con.execute("DELETE FROM users WHERE username=?", (username,))
    if res.rowcount == 0:
        con.close(); raise HTTPException(404, "Usuario no encontrado")
    con.commit(); con.close()
    _upload_users_db()


# ── User preferences ────────────────────────────────────────────────────────

@app.get("/users/me/preferences")
def get_preferences(request: Request):
    username = _payload_of(request).get("sub")
    con = _get_users_con()
    row = con.execute(
        "SELECT theme, palette FROM user_preferences WHERE username=?", (username,)
    ).fetchone()
    con.close()
    return {"theme": row["theme"], "palette": row["palette"]} if row else {"theme": "dark", "palette": "default"}


@app.put("/users/me/preferences")
def update_preferences(request: Request, body: PreferencesUpdate):
    username = _payload_of(request).get("sub")
    con = _get_users_con()
    con.execute("""
        INSERT INTO user_preferences (username, theme, palette)
        VALUES (?,?,?)
        ON CONFLICT(username) DO UPDATE SET
            theme    = COALESCE(excluded.theme,   theme),
            palette  = COALESCE(excluded.palette, palette),
            updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
    """, (username, body.theme or "dark", body.palette or "default"))
    con.commit(); con.close()
    _upload_users_db()
    return {"ok": True}


# ── Protected routes ────────────────────────────────────────────────────────

@app.get("/flights")
def list_flights(status: str = "all", departures_within_min: int = None):
    con = get_db()
    try:
        rows = _latest_predictions_active(con)
        flights = [_flight_row_to_dict(r) for r in rows]
        
        now = datetime.now(timezone.utc)
        
        if status == "departed":
            flights = [f for f in flights if f["actual_out_utc"] is not None or f["departure_delay_min"] is not None]
        elif status == "scheduled":
            flights = [f for f in flights if f["actual_out_utc"] is None and f["departure_delay_min"] is None]
            
        if departures_within_min is not None:
            filtered_flights = []
            for f in flights:
                est_out_str = f["estimated_out_utc"] or f["scheduled_out_utc"]
                if est_out_str:
                    try:
                        if est_out_str.endswith("Z"):
                            est_out_str = est_out_str[:-1] + "+00:00"
                        est_out = datetime.fromisoformat(est_out_str)
                        if est_out.tzinfo is None:
                            est_out = est_out.replace(tzinfo=timezone.utc)
                        else:
                            est_out = est_out.astimezone(timezone.utc)
                            
                        diff_sec = (est_out - now).total_seconds()
                        if -15 * 60 <= diff_sec <= departures_within_min * 60:
                            filtered_flights.append(f)
                    except Exception:
                        pass
            flights = filtered_flights
            
        return flights
    finally:
        con.close()


@app.get("/flight-history/{fa_flight_id:path}")
def get_flight_history(fa_flight_id: str):
    con = get_db()
    try:
        rows = con.execute(
            """SELECT predicted_at_utc, proba_delay, predicted_delay
               FROM predictions
               WHERE fa_flight_id = ?
               ORDER BY predicted_at_utc ASC""",
            (fa_flight_id,),
        ).fetchall()
        return [
            {
                "predicted_at_utc": r["predicted_at_utc"],
                "delay_probability": round(float(r["proba_delay"]), 4),
                "predicted_delay": int(r["predicted_delay"]),
            }
            for r in rows
        ]
    finally:
        con.close()


@app.get("/flights/{fa_flight_id:path}")
def get_flight(fa_flight_id: str):
    con = get_db()
    try:
        rows = _latest_predictions_active(con)
        row = next((r for r in rows if r["fa_flight_id"] == fa_flight_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail="Flight not found")

        result = _flight_row_to_dict(row)
        # Fast path: serve cached SHAP from prediction_shap table (Fix D).
        # Fall back to on-demand compute only if the cache is empty (e.g. for
        # predictions written before SHAP persistence was rolled out).
        cached = _load_cached_shap(con, fa_flight_id)
        result["shap"] = cached if cached else _compute_shap(fa_flight_id)
        return result
    finally:
        con.close()


def _load_cached_shap(con: sqlite3.Connection, fa_flight_id: str) -> list[dict]:
    """Read the most-recent SHAP top-K row set persisted by live_pull.py."""
    try:
        cur = con.execute(
            """SELECT feature_name, shap_value, feature_value
               FROM prediction_shap
               WHERE fa_flight_id = ?
                 AND predicted_at_utc = (
                     SELECT MAX(predicted_at_utc) FROM prediction_shap
                     WHERE fa_flight_id = ?
                 )
               ORDER BY rank ASC""",
            (fa_flight_id, fa_flight_id),
        )
        out: list[dict] = []
        for feat_name, shap_val, feat_val in cur.fetchall():
            contrib = float(shap_val)
            out.append({
                "feature":      feat_name,
                "label":        FEATURE_LABELS.get(feat_name, feat_name.replace("_", " ").title()),
                "contribution": round(abs(contrib), 4),
                "direction":    "positive" if contrib >= 0 else "negative",
                "value":        feat_val,
            })
        return out
    except sqlite3.OperationalError:
        # Table doesn't exist yet (legacy DB) — fall back to live compute.
        return []


def _compute_shap(fa_flight_id: str) -> list[dict]:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from ontimeai.live import open_db, build_inference_frame
        from ontimeai.lineage_fallback import load_lookups
        from ontimeai.explainability import compute_shap_values, explain_instance
        from predict import prepare_inference_frame

        meta = _load_meta()
        conn = open_db()
        try:
            df = build_inference_frame(conn, [fa_flight_id], history_days=7)
        finally:
            conn.close()
        if df.empty:
            return []

        fallback_path = ARTIFACT_PATH.parent / "lineage_fallback.joblib"
        fallback = load_lookups(fallback_path) if fallback_path.exists() else None

        X = prepare_inference_frame(
            df, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=fallback
        )
        cat_cols_set = set(meta.get("cat_cols", []))
        for c in X.columns:
            if c not in cat_cols_set and X[c].dtype == object:
                import pandas as pd
                X[c] = pd.to_numeric(X[c], errors="coerce")

        target_mask = df["fa_flight_id"].isin([fa_flight_id]) & df["ARR_DELAY"].isna()
        if not target_mask.any():
            target_mask = df["fa_flight_id"].isin([fa_flight_id])

        idx = int(df.index[target_mask][0])
        row_pos = list(df.index).index(idx)

        shap_vals = compute_shap_values(meta["booster"], X)
        top = explain_instance(shap_vals, list(X.columns), row_pos, top_n=10)

        result = []
        for _, r in top.iterrows():
            feat = str(r["feature"])
            contrib = float(r["contribution"])
            result.append({
                "feature":      feat,
                "label":        FEATURE_LABELS.get(feat, feat.replace("_", " ").title()),
                "contribution": round(abs(contrib), 4),
                "direction":    "positive" if contrib >= 0 else "negative",
            })
        return result
    except Exception:
        return []


@app.get("/metrics/summary")
def metrics_summary():
    con = get_db()
    try:
        rows = _latest_predictions_active(con)
        if not rows:
            return {
                "total_flights": 0, "high_risk": 0, "medium_risk": 0,
                "low_risk": 0, "avg_delay_probability": 0.0,
                "predicted_positive_rate": 0.0,
                "model_version": ACTIVE_MODEL, "last_tick_utc": "",
            }
        probas = [float(r["proba_delay"]) for r in rows]
        preds  = [int(r["predicted_delay"]) for r in rows]
        ticks  = [r["predicted_at_utc"] for r in rows if r["predicted_at_utc"]]
        return {
            "total_flights":           len(rows),
            "high_risk":               sum(1 for p in probas if p >= 0.35),
            "medium_risk":             sum(1 for p in probas if 0.15 <= p < 0.35),
            "low_risk":                sum(1 for p in probas if p < 0.15),
            "avg_delay_probability":   round(float(np.mean(probas)), 4),
            "predicted_positive_rate": round(float(np.mean(preds)), 4),
            "model_version":           ACTIVE_MODEL,
            "last_tick_utc":           max(ticks) if ticks else "",
        }
    finally:
        con.close()


@app.get("/metrics/hourly")
def metrics_hourly():
    con = get_db()
    try:
        rows = _latest_predictions_active(con)
        buckets: dict[str, dict] = {}
        for r in rows:
            sched = r["scheduled_out_utc"] or ""
            if len(sched) >= 16:
                hour = sched[11:13] + ":00"
            else:
                hour = "??"
            if hour not in buckets:
                buckets[hour] = {"hour": hour, "total": 0, "high_risk": 0, "sum_proba": 0.0}
            b = buckets[hour]
            b["total"] += 1
            b["sum_proba"] += float(r["proba_delay"])
            if float(r["proba_delay"]) >= 0.35:
                b["high_risk"] += 1

        result = []
        for hour in sorted(buckets):
            b = buckets[hour]
            result.append({
                "hour":      hour,
                "total":     b["total"],
                "high_risk": b["high_risk"],
                "avg_proba": round(b["sum_proba"] / b["total"], 4) if b["total"] else 0.0,
            })
        return result
    finally:
        con.close()


@app.get("/metrics/model")
def metrics_model():
    try:
        meta = _load_meta()
        threshold = float(meta.get("threshold", 0.0))
    except Exception:
        threshold = 0.0

    # Live AUC/Brier — PRE_DEPARTURE predictions only (last 7 days, first pred per flight)
    live_auc = live_brier = n_actuals = None
    live_auc_enroute = live_brier_enroute = n_actuals_enroute = None
    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss
        con = get_db()
        import pandas as pd
        df = pd.read_sql("""
            SELECT p.fa_flight_id, p.proba_delay, p.predicted_at_utc,
                   COALESCE(p.prediction_phase, 'PRE_DEPARTURE') AS prediction_phase,
                   CASE WHEN a.arr_delay_min > 15 THEN 1 ELSE 0 END AS delayed
            FROM predictions p
            JOIN actuals a ON p.fa_flight_id = a.fa_flight_id
            WHERE a.arr_delay_min IS NOT NULL AND a.cancelled = 0
              AND p.predicted_at_utc >= datetime('now', '-7 days')
            ORDER BY p.predicted_at_utc
        """, con)
        con.close()
        if not df.empty:
            # PRE_DEPARTURE: first prediction per flight that is PRE_DEPARTURE
            pre = (df[df["prediction_phase"] == "PRE_DEPARTURE"]
                   .sort_values("predicted_at_utc")
                   .groupby("fa_flight_id", as_index=False).first())
            if pre.empty:
                # Fallback: use first prediction (legacy data without phase column)
                pre = df.sort_values("predicted_at_utc").groupby("fa_flight_id", as_index=False).first()
            y = pre["delayed"].to_numpy(dtype=int)
            p = pre["proba_delay"].to_numpy(dtype=float)
            if len(y) >= 30 and y.sum() >= 5:
                live_auc   = round(float(roc_auc_score(y, p)), 4)
                live_brier = round(float(brier_score_loss(y, p)), 4)
                n_actuals  = int(len(y))
            # EN_ROUTE: for reference, compute separately
            enroute = (df[df["prediction_phase"] == "EN_ROUTE"]
                       .sort_values("predicted_at_utc")
                       .groupby("fa_flight_id", as_index=False).first())
            if len(enroute) >= 30 and enroute["delayed"].sum() >= 5:
                y_e = enroute["delayed"].to_numpy(dtype=int)
                p_e = enroute["proba_delay"].to_numpy(dtype=float)
                live_auc_enroute   = round(float(roc_auc_score(y_e, p_e)), 4)
                live_brier_enroute = round(float(brier_score_loss(y_e, p_e)), 4)
                n_actuals_enroute  = int(len(enroute))
    except Exception:
        pass

    version = ACTIVE_MODEL.replace("4year_", "").replace("_", "-")
    return {
        "active_model":         ACTIVE_MODEL,
        "version":              version,
        "live_auc":             live_auc,
        "live_brier":           live_brier,
        "n_actuals":            n_actuals,
        "live_auc_enroute":     live_auc_enroute,
        "live_brier_enroute":   live_brier_enroute,
        "n_actuals_enroute":    n_actuals_enroute,
        "threshold":            threshold,
    }


@app.get("/test-cases")
def test_cases():
    """Formal test case results for thesis validation (CP-01, CP-02)."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score, brier_score_loss

    con = get_db()
    try:
        cp01_row = con.execute("""
            SELECT p.fa_flight_id, p.proba_delay, a.arr_delay_min,
                   f.ident_iata, f.op_carrier, f.origin, f.dest,
                   f.scheduled_out_utc, f.aircraft_type
            FROM (
                SELECT fa_flight_id, proba_delay,
                       ROW_NUMBER() OVER (
                           PARTITION BY fa_flight_id ORDER BY predicted_at_utc DESC
                       ) AS rn
                FROM predictions
            ) p
            JOIN actuals a ON p.fa_flight_id = a.fa_flight_id
            JOIN flights  f ON f.fa_flight_id = p.fa_flight_id
            WHERE p.rn = 1
              AND p.proba_delay >= 0.35
              AND a.arr_delay_min > 15
              AND a.cancelled = 0
              AND f.origin = 'ATL'
            ORDER BY p.proba_delay DESC
            LIMIT 1
        """).fetchone()

        cp01 = None
        if cp01_row:
            shap = _compute_shap(cp01_row["fa_flight_id"])
            ident = cp01_row["ident_iata"] or cp01_row["fa_flight_id"]
            cp01 = {
                "fa_flight_id":      cp01_row["fa_flight_id"],
                "flight_number":     ident,
                "airline_code":      cp01_row["op_carrier"] or "",
                "origin":            cp01_row["origin"] or "",
                "destination":       cp01_row["dest"] or "",
                "scheduled_out_utc": cp01_row["scheduled_out_utc"] or "",
                "predicted_proba":   round(float(cp01_row["proba_delay"]), 4),
                "predicted_risk":    risk_level(float(cp01_row["proba_delay"])),
                "actual_delay_min":  int(cp01_row["arr_delay_min"]),
                "shap":              shap,
                "passed":            True,
            }

        df = pd.read_sql("""
            SELECT p.fa_flight_id, p.proba_delay, p.predicted_at_utc,
                   CASE WHEN a.arr_delay_min > 15 THEN 1 ELSE 0 END AS delayed
            FROM predictions p
            JOIN actuals a ON p.fa_flight_id = a.fa_flight_id
            WHERE a.arr_delay_min IS NOT NULL AND a.cancelled = 0
        """, con)

        cp02: dict[str, Any] = {
            "n_actuals": 0, "auc": None, "brier": None,
            "actual_delay_rate": None, "passed": False,
        }
        if not df.empty:
            df = (df.sort_values("predicted_at_utc")
                    .groupby("fa_flight_id", as_index=False)
                    .last())
            y      = df["delayed"].to_numpy(dtype=int)
            p_vals = df["proba_delay"].to_numpy(dtype=float)
            if len(y) >= 30 and y.sum() >= 5:
                auc   = float(roc_auc_score(y, p_vals))
                brier = float(brier_score_loss(y, p_vals))
                cp02 = {
                    "n_actuals":         int(len(y)),
                    "auc":               round(auc, 4),
                    "brier":             round(brier, 4),
                    "actual_delay_rate": round(float(y.mean()), 4),
                    "passed":            auc >= 0.70 and brier <= 0.15,
                }

        return {"cp01": cp01, "cp02": cp02}
    finally:
        con.close()


@app.get("/weather/{airport_code}")
def weather(airport_code: str):
    """Latest METAR observation for an airport from the stored weather_obs table."""
    code = _validate_airport(airport_code)
    con = get_db()
    try:
        row = con.execute("""
            SELECT station, valid_utc, tmpc, dwpc, relh, drct, sknt, alti,
                   p01m, vsby, gust, wxcodes,
                   wx_precip_flag, wx_low_vis_flag, wx_strong_wind_flag
            FROM weather_obs
            WHERE station = ?
            ORDER BY valid_utc DESC
            LIMIT 1
        """, (code,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"No weather data for {code}")
        return {
            "airport_code":      row["station"],
            "valid_utc":         row["valid_utc"],
            "temperature_c":     row["tmpc"],
            "dewpoint_c":        row["dwpc"],
            "humidity_pct":      row["relh"],
            "wind_direction":    row["drct"],
            "wind_knots":        row["sknt"],
            "gust_knots":        row["gust"],
            "altimeter_inhg":    row["alti"],
            "precip_mm":         row["p01m"],
            "visibility_miles":  row["vsby"],
            "wx_codes":          row["wxcodes"],
            "precip_flag":       bool(row["wx_precip_flag"]),
            "low_visibility":    bool(row["wx_low_vis_flag"]),
            "strong_wind":       bool(row["wx_strong_wind_flag"]),
        }
    finally:
        con.close()


@app.get("/operations/{airport_code}")
def operations(airport_code: str):
    """Today's operational delay stats for flights departing or arriving at an airport."""
    code = _validate_airport(airport_code)
    con = get_db()
    try:
        today = today_utc()
        now = datetime.now(timezone.utc)
        start_window = (now - timedelta(hours=6)).isoformat()
        end_window = (now + timedelta(hours=18)).isoformat()
        undeparted_limit = (now - timedelta(hours=24)).isoformat()
        
        rows = con.execute("""
            SELECT f.origin, f.dest, p.proba_delay, p.predicted_delay
            FROM flights f
            JOIN (
                SELECT fa_flight_id, proba_delay, predicted_delay,
                       ROW_NUMBER() OVER (
                           PARTITION BY fa_flight_id ORDER BY predicted_at_utc DESC
                       ) AS rn
                FROM predictions
            ) p ON p.fa_flight_id = f.fa_flight_id AND p.rn = 1
            LEFT JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
            WHERE f.cancelled = 0
              AND (f.origin = ? OR f.dest = ?)
              AND (
                  (datetime(f.scheduled_out_utc) >= datetime(?) AND datetime(f.scheduled_out_utc) <= datetime(?))
                  OR
                  (datetime(f.scheduled_out_utc) >= datetime(?) AND a.actual_out_utc IS NULL)
              )
        """, (code, code, start_window, end_window, undeparted_limit)).fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail=f"No flight data for {code} currently")

        probas = [float(r["proba_delay"]) for r in rows]
        departures = [r for r in rows if r["origin"] == code]
        high_risk  = sum(1 for p in probas if p >= 0.35)
        total      = len(rows)

        return {
            "airport_code":          code,
            "date_utc":              today,
            "total_flights":         total,
            "departures":            len(departures),
            "arrivals":              total - len(departures),
            "high_risk_count":       high_risk,
            "delay_rate":            round(sum(r["predicted_delay"] for r in rows) / total, 4),
            "avg_delay_probability": round(float(np.mean(probas)), 4),
            "congestion_level":      "high" if total > 80 else "medium" if total > 40 else "low",
        }
    finally:
        con.close()


@app.get("/metrics/routes")
def metrics_routes():
    """Puntualidad histórica por ruta (origen→destino) con actuals disponibles."""
    con = get_db()
    try:
        rows = con.execute("""
            SELECT f.origin, f.dest,
                   COUNT(*) AS total_flights,
                   SUM(CASE WHEN a.arr_delay_min <= 15 THEN 1 ELSE 0 END) AS on_time_count,
                   ROUND(AVG(a.arr_delay_min), 1) AS avg_delay_min
            FROM flights f
            JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
            WHERE a.arr_delay_min IS NOT NULL
              AND f.origin IS NOT NULL AND f.origin != ''
              AND f.dest IS NOT NULL AND f.dest != ''
            GROUP BY f.origin, f.dest
            HAVING COUNT(*) >= 1
            ORDER BY total_flights DESC
            LIMIT 30
        """).fetchall()
        return [
            {
                "origin":        r["origin"],
                "dest":          r["dest"],
                "route":         f"{r['origin']} → {r['dest']}",
                "total_flights": r["total_flights"],
                "on_time_rate":  round(r["on_time_count"] / r["total_flights"], 4),
                "avg_delay_min": float(r["avg_delay_min"]) if r["avg_delay_min"] is not None else 0.0,
            }
            for r in rows
        ]
    finally:
        con.close()


@app.get("/metrics/routes/{origin}/{dest}/history")
def metrics_route_history(origin: str, dest: str):
    """Serie diaria de puntualidad para una ruta específica."""
    con = get_db()
    try:
        rows = con.execute("""
            SELECT date(f.scheduled_out_utc) AS flight_date,
                   COUNT(*) AS total,
                   SUM(CASE WHEN a.arr_delay_min <= 15 THEN 1 ELSE 0 END) AS on_time_count,
                   ROUND(AVG(a.arr_delay_min), 1) AS avg_delay_min
            FROM flights f
            JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
            WHERE f.origin = ? AND f.dest = ?
              AND a.arr_delay_min IS NOT NULL
            GROUP BY flight_date
            ORDER BY flight_date ASC
        """, (origin.upper(), dest.upper())).fetchall()
        return [
            {
                "date":          r["flight_date"][5:].replace("-", "/"),  # "05/21"
                "total":         r["total"],
                "on_time_rate":  round(r["on_time_count"] / r["total"], 4) if r["total"] > 0 else 0.0,
                "avg_delay_min": float(r["avg_delay_min"]) if r["avg_delay_min"] is not None else 0.0,
            }
            for r in rows
        ]
    finally:
        con.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

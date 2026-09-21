"""OnTimeAI — FastAPI backend.

Serves live predictions from live_data.db.  Swap models without restart:
    ACTIVE_MODEL=4year_v9_recal uvicorn api:app --reload

Endpoints:
    GET /flights            — today's scheduled flights + latest prediction
    GET /flights/{id}       — single flight detail + SHAP
    GET /flight-history/{id} — prediction cycles + persisted SHAP/context
    GET /metrics/summary    — today's KPI cards
    GET /metrics/hourly     — predictions grouped by departure hour
    GET /metrics/model      — active model info + live AUC
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import threading
import time
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


def _require_secret(name: str) -> str:
    """
    Lee una variable de entorno obligatoria o aborta el arranque.

    Los secretos no llevan valor por defecto a propósito. Un default convierte
    una variable faltante en un arranque exitoso con una credencial conocida —
    el servicio queda en pie y nada avisa. Es preferible que el contenedor no
    levante: Cloud Run deja la revisión anterior sirviendo y el error queda en
    los logs del despliegue.
    """
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Falta la variable de entorno obligatoria {name}. "
            "Los secretos se inyectan desde Secret Manager con --set-secrets; "
            "ver deploy.sh y docs/SECRETS.md."
        )
    return value


GCS_BUCKET = os.getenv("GCS_BUCKET", "")
_TMP_DB = Path("/tmp/live_data.db")
_BUNDLED_DB = Path(__file__).parent / "live_data.db"
DB_PATH = _TMP_DB if GCS_BUCKET else _BUNDLED_DB
_DB_REFRESH_INTERVAL = 1000  # refresh from GCS every ~16 min
# Segundos para bajar el snapshot completo desde GCS.
#
# El default de google-cloud-storage son 120 s, que con la base en ~670 MB se
# quedaba corto: la descarga se cortaba a mitad con IncompleteRead y el backend
# seguia sirviendo el snapshot anterior sin avisar. Llego a servir datos de dos
# dias atras respondiendo 200.
_DB_DOWNLOAD_TIMEOUT = max(60, int(os.getenv("DB_DOWNLOAD_TIMEOUT", "300")))
# Reintentos cuando un job reemplaza el objeto mientras se lo descarga.
_DB_DOWNLOAD_GENERATION_RETRIES = max(
    0, int(os.getenv("DB_DOWNLOAD_GENERATION_RETRIES", "2"))
)
_db_last_refresh: float = 0.0
_db_last_health_check: float = 0.0
_DB_HEALTH_INTERVAL = 60  # re-verify DB health every 60 s
_DB_REFRESH_LOCK = threading.Lock()
# Salud del refresh. Un fallo dejaba al backend sirviendo el snapshot anterior
# con respuesta 200 y sin rastro fuera de los logs; se expone en
# /admin/db-stats para que la UI pueda mostrar que los datos estan viejos.
_db_last_refresh_ok_utc: str | None = None
_db_last_refresh_error: str | None = None

# Users DB (separate from live_data.db so live job never overwrites it)
USERS_DB_PATH = Path("/tmp/users.db") if GCS_BUCKET else Path(__file__).parent / "users.db"


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return f"{salt}${h.hex()}"


def _unusable_password() -> str:
    """Sentinel for accounts without a local password (Google sign-in).

    Has no '$' separator, so _check_password() always fails for these rows.
    """
    return f"!{secrets.token_hex(16)}"


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
    # Migration: identity provider columns (added for Google sign-in, issue #8)
    existing = {r["name"] for r in con.execute("PRAGMA table_info(users)")}
    for column, ddl in (
        ("email",     "ALTER TABLE users ADD COLUMN email TEXT"),
        ("provider",  "ALTER TABLE users ADD COLUMN provider TEXT NOT NULL DEFAULT 'local'"),
        ("user_type", "ALTER TABLE users ADD COLUMN user_type TEXT"),
    ):
        if column not in existing:
            con.execute(ddl)
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL")
    # Alta inicial de usuarios desde el entorno.
    #
    # Es idempotente: sólo inserta si el usuario no existe. Cambiar estas
    # variables NO rota la contraseña de una cuenta ya creada — users.db
    # persiste en GCS entre despliegues. Para rotar hay que actualizar la fila,
    # vía PATCH /admin/users/{username}. Ver docs/SECRETS.md.
    seeds = [
        (_require_secret("API_USERNAME"), _require_secret("API_PASSWORD"), "superadmin"),
        (_require_secret("API_USERNAME_VIEWER"), _require_secret("API_PASSWORD_VIEWER"), "user"),
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


def _sqlite_readonly_uri(path: Path) -> str:
    """Return an immutable URI so API reads never create WAL sidecars."""
    return f"{path.resolve().as_uri()}?mode=ro&immutable=1"


def _verify_db_snapshot(path: Path) -> bool:
    """Validate a standalone SQLite snapshot before making it visible."""
    try:
        with sqlite3.connect(_sqlite_readonly_uri(path), uri=True) as con:
            result = con.execute("PRAGMA quick_check(1)").fetchone()
        return bool(result and result[0] == "ok")
    except (OSError, sqlite3.Error):
        return False


def _download_db_snapshot(destination: Path) -> int:
    """
    Descarga una generacion completa de GCS.

    Reintenta ante conflicto de generacion. Tres jobs reescriben este objeto
    —live-pull a los :00/:30, live-pull-2 a los :15/:45 y el harvester cada 10
    minutos— y GCS no conserva generaciones viejas sin versionado. Entre el
    reload y el final de la descarga de ~685 MB puede entrar una escritura, y
    entonces la generacion pedida deja de existir:

      404 GET .../live_data.db?ifGenerationMatch=1789392097835849
      No such object

    Los jobs ya reintentan ante el mismo conflicto al escribir; el lector tenia
    que hacer lo propio. La precondicion se conserva porque garantiza que el
    archivo en disco corresponde a una unica generacion y no a dos mezcladas.
    """
    from google.cloud import storage as gcs
    from google.cloud.exceptions import NotFound
    from google.cloud.storage.retry import DEFAULT_RETRY

    blob = gcs.Client().bucket(GCS_BUCKET).blob("live_data.db")

    for attempt in range(_DB_DOWNLOAD_GENERATION_RETRIES + 1):
        blob.reload()
        generation = int(blob.generation)
        try:
            blob.download_to_filename(
                str(destination),
                if_generation_match=generation,
                timeout=_DB_DOWNLOAD_TIMEOUT,
                # `timeout` acota cada request; el deadline de la politica de
                # reintentos acota el total, y tambien vale 120 s por defecto.
                # Subir solo el primero no evitaba el corte.
                retry=DEFAULT_RETRY.with_deadline(_DB_DOWNLOAD_TIMEOUT),
            )
            return generation
        except NotFound:
            if attempt >= _DB_DOWNLOAD_GENERATION_RETRIES:
                raise
            print(
                f"[db_refresh] generation {generation} fue reemplazada durante "
                f"la descarga; reintentando "
                f"({attempt + 2}/{_DB_DOWNLOAD_GENERATION_RETRIES + 1})"
            )

    raise RuntimeError("unreachable")


def _temporary_db_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(filename)


def _install_local_db_copy(source: Path, target: Path) -> None:
    """Atomically install a verified local fallback snapshot."""
    snapshot_path = _temporary_db_path(target)
    try:
        shutil.copyfile(source, snapshot_path)
        if not _verify_db_snapshot(snapshot_path):
            raise sqlite3.DatabaseError(f"fallback snapshot is invalid: {source}")
        os.replace(snapshot_path, target)
    finally:
        snapshot_path.unlink(missing_ok=True)


def _refresh_db_from_gcs(*, force: bool = False) -> bool:
    """Install a verified GCS snapshot without exposing a partial file.

    A regular refresh never waits behind another refresh: it keeps serving the
    previous immutable snapshot. Forced refreshes (startup/recovery) wait for
    the in-flight installer because no known-good snapshot may be available.
    """
    global _db_last_refresh, _db_last_refresh_ok_utc, _db_last_refresh_error
    if not GCS_BUCKET:
        return False

    now = time.monotonic()
    if not force and now - _db_last_refresh < _DB_REFRESH_INTERVAL:
        return False

    blocking = force or not _TMP_DB.exists()
    if not _DB_REFRESH_LOCK.acquire(blocking=blocking):
        # Otro hilo esta descargando. Si esto se repite ciclo tras ciclo, hay
        # un hilo trabado reteniendo el lock y el backend nunca se actualiza.
        print("[db_refresh] abortado: el lock ya esta tomado por otro hilo")
        return False

    snapshot_path: Path | None = None
    try:
        # Otro pedido pudo haber completado el refresh mientras este esperaba.
        now = time.monotonic()
        if not force and now - _db_last_refresh < _DB_REFRESH_INTERVAL:
            print("[db_refresh] abortado: otro hilo refresco mientras esperaba")
            return False

        snapshot_path = _temporary_db_path(_TMP_DB)
        generation = _download_db_snapshot(snapshot_path)
        if not _verify_db_snapshot(snapshot_path):
            raise sqlite3.DatabaseError(
                f"downloaded generation {generation} failed PRAGMA quick_check"
            )

        # POSIX replacement is atomic. Existing immutable readers keep their
        # old file descriptor; new requests open the complete new snapshot.
        os.replace(snapshot_path, _TMP_DB)
        snapshot_path = None
        _db_last_refresh = time.monotonic()
        _db_last_refresh_ok_utc = datetime.now(timezone.utc).isoformat()
        _db_last_refresh_error = None
        print(
            f"[db] refreshed from GCS generation={generation} "
            f"({_TMP_DB.stat().st_size / 1e6:.0f} MB)"
        )
        return True
    except Exception as e:
        _db_last_refresh_error = f"{datetime.now(timezone.utc).isoformat()}: {e}"
        print(f"[db_refresh] failed: {e}")
        return False
    finally:
        if snapshot_path is not None:
            snapshot_path.unlink(missing_ok=True)
        _DB_REFRESH_LOCK.release()


def _verify_db_health(path: Path) -> bool:
    """Read an actual data page to catch corruption beyond the schema."""
    try:
        with sqlite3.connect(_sqlite_readonly_uri(path), uri=True) as chk:
            chk.execute("SELECT fa_flight_id FROM flights LIMIT 1").fetchone()
        return True
    except (OSError, sqlite3.Error):
        return False

MODEL_REGISTRY: dict[str, Path] = {
    "4year_v9":       Path(__file__).parent / "artifacts/4year_v9",
    "4year_v9_recal": Path(__file__).parent / "artifacts/4year_v9_recal",
    "4year_v7_recal": Path(__file__).parent / "artifacts/4year_v7_recal",
}
ACTIVE_MODEL = os.getenv("ACTIVE_MODEL", "4year_v9")
ARTIFACT_PATH = MODEL_REGISTRY.get(ACTIVE_MODEL, MODEL_REGISTRY["4year_v9"])

# ── Auth ───────────────────────────────────────────────────────────────────

JWT_SECRET = _require_secret("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 8

# Google sign-in: audience the ID token must be issued for. Empty disables /auth/google.
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

# Firebase Authentication: el proyecto contra el que se valida el ID token.
#
# Firebase es quien manda los correos de verificacion y de recuperacion, que es
# lo que ni Google Sign-In ni el alta propia pueden hacer: no tenemos servicio
# de envio. Vacio deshabilita /auth/firebase.
#
# Coincide con el id del proyecto de Google Cloud —un proyecto de Firebase ES
# un proyecto de GCP— asi que en produccion vale "ontimeai-prod".
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")
USER_TYPES = ("b2b", "b2c")


_PUBLIC_PATHS = {"/auth/login", "/auth/google", "/auth/firebase", "/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect"}
# Endpoints accesibles sin autenticación para la vista pública /live
_LITE_PUBLIC_PATHS = {"/flights", "/metrics/hourly"}
_LITE_PUBLIC_PREFIXES = ("/weather/",)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        path = request.url.path
        if path in _PUBLIC_PATHS or path.startswith("/redoc"):
            return await call_next(request)
        if request.method == "GET" and (
            path in _LITE_PUBLIC_PATHS
            or any(path.startswith(pfx) for pfx in _LITE_PUBLIC_PREFIXES)
        ):
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


class GoogleLoginRequest(BaseModel):
    id_token: str


class FirebaseLoginRequest(BaseModel):
    id_token: str


class MeUpdate(BaseModel):
    user_type: Optional[str] = None


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
    global _db_last_refresh, _db_last_health_check

    if GCS_BUCKET:
        if not _refresh_db_from_gcs(force=True):
            print("[startup] GCS snapshot unavailable; using bundled DB")
            _install_local_db_copy(_BUNDLED_DB, _TMP_DB)
            # Avoid a download storm if GCS is temporarily unavailable.
            _db_last_refresh = time.monotonic()

    # Trigger database migrations on startup
    try:
        from ontimeai.live import open_db
        conn = open_db(DB_PATH)
        try:
            # The shared snapshot is produced in WAL mode. Checkpoint migrations
            # and switch the API's local copy to DELETE before immutable reads.
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.commit()
        finally:
            conn.close()
        print("[startup] Database migrations executed successfully")
    except Exception as e:
        print(f"[startup] Database migration failed: {e}")

    _db_last_health_check = time.monotonic()
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

    if GCS_BUCKET and not _TMP_DB.exists():
        _refresh_db_from_gcs(force=True)
    else:
        # Sincrono a proposito, no en un hilo de fondo.
        #
        # Cloud Run con throttling —el default— asigna CPU solo mientras se
        # procesa un pedido. Un hilo de fondo arrancado por un request queda
        # sin CPU apenas se envia la respuesta, y la descarga de ~700 MB cae a
        # ~2 MB/s hasta morir en el deadline. Eso dejaba al backend sirviendo
        # datos de horas atras, con respuesta 200 y sin senal alguna.
        #
        # Haciendolo dentro del request hay CPU asignada y la descarga tarda
        # ~20 s. El lock es no bloqueante: solo el primer pedido vencido paga
        # la espera, los concurrentes siguen sirviendo el snapshot anterior.
        _refresh_db_from_gcs()

    if GCS_BUCKET and (time.monotonic() - _db_last_health_check > _DB_HEALTH_INTERVAL):
        if not _verify_db_health(_TMP_DB):
            print("[db] health check failed — forcing re-download from GCS")
            _refresh_db_from_gcs(force=True)
            if not _verify_db_health(_TMP_DB):
                print("[db] GCS DB also corrupt — falling back to bundled DB")
                _install_local_db_copy(_BUNDLED_DB, _TMP_DB)
                _db_last_refresh = time.monotonic()
        _db_last_health_check = time.monotonic()

    con = sqlite3.connect(_sqlite_readonly_uri(DB_PATH), uri=True)
    con.row_factory = sqlite3.Row
    return con


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# Sin umbral en la fila —predicciones anteriores a que se guardara la columna—
# se usa este, que es el del artefacto v9.
_FALLBACK_THRESHOLD = 0.32


def risk_level(proba: float, threshold: float | None = None) -> str:
    """Traduce una probabilidad al nivel que muestra el dashboard.

    Las bandas son relativas al umbral con el que el modelo etiqueto esa misma
    prediccion, no constantes. Antes eran 0.35 y 0.15, elegidas cuando la
    probabilidad media del lote rondaba 0.55 porque la cadena de ajustes la
    inflaba. Con la probabilidad calibrada la media queda en ~0.07 y el umbral
    en ~0.09, asi que esas constantes dejaban un vuelo marcado como demorado
    (`predicted_delay=1`) mostrandose como riesgo bajo.

    Atado al umbral, las tres bandas se sostienen solas:

      alto   p >= 2x umbral   medido: la precision aproximadamente dobla
      medio  p >= umbral      marcado como demorado, pero al filo
      bajo   p <  umbral      no marcado

    De modo que alto + medio es exactamente lo que el modelo marca, y no puede
    volver a desalinearse del label.
    """
    cut = threshold if threshold and threshold > 0 else _FALLBACK_THRESHOLD
    if proba >= 2 * cut:
        return "high"
    if proba >= cut:
        return "medium"
    return "low"


# ── Lazy model loader ──────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_meta() -> dict[str, Any]:
    from ontimeai.model import load_artifact
    return load_artifact(ARTIFACT_PATH)


# ── Helpers ────────────────────────────────────────────────────────────────

def _optional_prediction_column(
    con: sqlite3.Connection, name: str, table_alias: str = "p"
) -> str:
    """`<alias>.<name>` si la columna existe, `NULL AS <name>` si no.

    Las columnas que se fueron sumando a `predictions` las crea una migracion
    `ALTER TABLE` que corre al abrir la base para escritura, o sea en el job.
    La API la abre en modo lectura y puede estar sirviendo una base que todavia
    no paso por ahi: la bundleada en la imagen, que es el fallback de arranque,
    o cualquier snapshot viejo.

    Referenciar una de esas columnas a secas tumba el endpoint con
    `OperationalError: no such column`. Paso el 15/09 con `threshold_used`:
    /metrics/summary y /flights devolvieron 500 hasta el rollback.

    El nombre se interpola en el SQL, asi que solo se llama con literales del
    codigo, nunca con algo que venga de afuera.
    """
    present = {
        row["name"] for row in con.execute("PRAGMA table_info(predictions)").fetchall()
    }
    if name not in present:
        return f"NULL AS {name}"
    return f"{table_alias}.{name}" if table_alias else name

# Cada fuente escribe una tabla distinta y a su propio ritmo. El numero es
# cuantos minutos puede pasar sin escribir antes de considerarla caida: holgado
# respecto de su cadencia real, para que un ciclo lento no dispare ruido.
#
#   (columna de tiempo, minutos tolerados, que la alimenta)
# Cadencia de los predictores. La duracion del ciclo se juzga contra esto.
SCHEDULER_INTERVAL_MIN = 15

# Cuando el archivo empieza a ser el problema. Medido: a 722 MB los ciclos
# daban 6-10 min; a 437 MB, 4-7. El umbral deja margen para reaccionar antes de
# que la transferencia domine el ciclo.
DB_SIZE_WARN_MB = int(os.getenv("DB_SIZE_WARN_MB", "800"))

# Fuentes EXTERNAS: llegan solas, con su propio ritmo, sin importar lo que pase
# en ATL. Si una deja de escribir, algo se rompio afuera.
#
# `predictions` estuvo aca y no corresponde: no es una fuente, es nuestra propia
# salida, y solo se produce cuando hay vuelos que predecir. Atlanta no opera
# salidas entre las 23:00 y las 05:00 locales —verificado contra un tablero
# publico independiente y contra la guia operativa del aeropuerto, cuyos
# controles de seguridad abren recien a las 3:30—, asi que cada madrugada pasaba
# mas de una hora sin escribir y la alarma sonaba sin que pasara nada.
#
# Lo que importa de `predictions` lo cubre `/metrics/summary`: si el ciclo se
# detuvo (`last_run_utc`) y si tenia vuelos y no los predijo
# (`last_run_targeted` contra `last_run_predicted`). Ver el watchdog.
SOURCE_FRESHNESS: dict[str, tuple[str, int, str]] = {
    "actuals":           ("settled_at_utc",   120,  "harvester FR24 + AeroAPI"),
    "weather_obs":       ("valid_utc",        180,  "IEM METAR (publica cada ~1 h)"),
    "nas_status":        ("captured_at_utc",  180,  "NAS status FAA"),
    "aircraft_position": ("captured_at_utc",  120,  "ADS-B airplanes.live / OpenSky"),
}


def _chain_calibrator_status(con: sqlite3.Connection) -> dict:
    """Estado del calibrador post-cadena: cuando se ajusto y si sigue vigente.

    Se expone porque un calibrador vencido no se nota mirando el dashboard: los
    numeros siguen saliendo, solo que exagerados. Es lo que paso con el
    artefacto de agosto durante un mes. Ver ontimeai/chain_calibration.py.
    """
    try:
        from ontimeai.chain_calibration import calibrator_status

        return calibrator_status(con)
    except Exception as exc:
        return {"present": False, "stale": True, "detail": f"no se pudo leer: {exc}"}


def _recent_cycles(con: sqlite3.Connection, limit: int = 8) -> dict:
    """Duracion de los ultimos ciclos completos del job.

    El scheduler dispara cada 15 minutos. Un ciclo que se acerca a esa ventana
    empieza a solaparse con el siguiente: dos jobs escribiendo la misma base y
    pisandose las subidas a GCS. La duracion es la senal que avisa antes de que
    eso pase, y el tamano de la base es su causa principal —el job la baja
    entera y la vuelve a subir en cada ciclo, asi que el costo crece con el
    archivo. Ver issue #4.

    Se prefiere `runs.job_seconds`, que escribe live_job y cubre el ciclo
    entero. `started_utc`/`finished_utc` los escribe live_pull y miden solo el
    pipeline: medido el 15/09, 1,7-3,2 min contra 4,7-7,5 de la ejecucion real
    de Cloud Run. La diferencia es arranque del contenedor, descarga, purga,
    VACUUM y subida. Quedan como respaldo para las filas anteriores a que
    `job_seconds` existiera, con la advertencia de que subestiman.
    """
    tiene_job_seconds = False
    try:
        tiene_job_seconds = "job_seconds" in {
            r["name"] for r in con.execute("PRAGMA table_info(runs)").fetchall()
        }
    except sqlite3.OperationalError:
        return {}

    columna = "job_seconds" if tiene_job_seconds else "NULL AS job_seconds"
    try:
        filas = con.execute(
            f"""SELECT started_utc, finished_utc, {columna} FROM runs
                 WHERE finished_utc IS NOT NULL
                 ORDER BY started_utc DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}

    duraciones: list[float] = []
    completas = 0
    for started, finished, job_seconds in filas:
        if job_seconds is not None:
            duraciones.append(round(float(job_seconds) / 60, 1))
            completas += 1
            continue
        try:
            a = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
            b = datetime.fromisoformat(str(finished).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        duraciones.append(round((b - a).total_seconds() / 60, 1))
    if not duraciones:
        return {}

    ordenadas = sorted(duraciones)
    mediana = ordenadas[len(ordenadas) // 2]
    tolerado = round(SCHEDULER_INTERVAL_MIN * 2 / 3, 1)
    return {
        "recent_minutes": duraciones,
        "median_minutes": mediana,
        "max_minutes": max(duraciones),
        "scheduler_minutes": SCHEDULER_INTERVAL_MIN,
        # Dos tercios de la ventana: deja margen para reaccionar antes del
        # solapamiento, sin gritar por un ciclo lento aislado. Por eso se
        # compara la mediana y no el maximo.
        "tolerated_minutes": tolerado,
        "slow": mediana > tolerado,
        # Cuantas de las filas miden el ciclo completo y no solo el pipeline.
        # Mientras no sean todas, la mediana esta sesgada hacia abajo.
        "full_cycle_samples": completas,
        "total_samples": len(duraciones),
    }


def _source_freshness(con: sqlite3.Connection) -> dict[str, dict]:
    """Cuanto hace que escribio cada fuente, y si eso ya es demasiado.

    Una fuente que deja de escribir no rompe nada: el pipeline sigue, las
    corridas figuran exitosas y la senal desaparece sin ruido. Paso con ADS-B,
    que estuvo 25 dias inerte —`adsb_eta_adjust` y `adsb_holding_adjust`
    recibiendo None en cada ciclo— hasta que alguien miro la tabla. Ver #9.
    """
    now = datetime.now(timezone.utc)
    out: dict[str, dict] = {}
    for table, (column, tolerated_min, fed_by) in SOURCE_FRESHNESS.items():
        try:
            row = con.execute(f"SELECT MAX({column}) FROM {table}").fetchone()
        except sqlite3.OperationalError:
            continue
        last = row[0] if row else None
        age_min: float | None = None
        if last:
            try:
                parsed = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                age_min = round((now - parsed).total_seconds() / 60, 1)
            except ValueError:
                age_min = None
        out[table] = {
            "last_utc": last,
            "age_minutes": age_min,
            "tolerated_minutes": tolerated_min,
            # Sin dato es tan malo como un dato viejo: la tabla vacia es
            # justamente el estado en que quedo aircraft_position.
            "stale": age_min is None or age_min > tolerated_min,
            "fed_by": fed_by,
        }
    return out


def _row_threshold(row) -> float | None:
    """El umbral guardado en la fila, o None si la prediccion es anterior."""
    try:
        value = row["threshold_used"]
    except (IndexError, KeyError):
        return None
    return float(value) if value is not None else None


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

    threshold_col_inner = _optional_prediction_column(
        con, "threshold_used", table_alias="p2"
    )
    rows = con.execute(f"""
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
               p.threshold_used,
               p.predicted_at_utc,
               CASE WHEN a.arr_delay_min IS NOT NULL THEN 1 ELSE 0 END AS has_actual,
               a.arr_delay_min,
               a.departure_delay_min,
               a.actual_out_utc,
               a.actual_off_utc,
               a.actual_on_utc,
               a.actual_in_utc
        FROM flights f
        JOIN (
            SELECT p2.fa_flight_id,
                   p2.proba_delay,
                   p2.predicted_delay,
                   {threshold_col_inner},
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
          AND f.origin = 'ATL'
          AND (
              (datetime(f.scheduled_out_utc) >= datetime(?) AND datetime(f.scheduled_out_utc) <= datetime(?))
              OR
              -- "programado hace poco y todavia sin salir".
              --
              -- Antes esto era `a.actual_out_utc IS NULL` a secas. Ese campo
              -- lo poblaba AeroAPI; FR24, que es el 96% de la muestra desde
              -- Fase 4, no expone gate-out y lo deja nulo siempre. La
              -- condicion se cumplia para todos y la ventana retenia las 24 h
              -- enteras: medido, 492 de 637 vuelos con salida programada hace
              -- mas de 6 h, y el 98% de ellos ya habia aterrizado.
              --
              -- Se exige que no haya NINGUNA senal de salida, en vez de
              -- confiar en una sola columna: un vuelo con despegue o con
              -- aterrizaje obviamente salio, lo diga o no el gate-out.
              (datetime(f.scheduled_out_utc) >= datetime(?)
               AND a.actual_out_utc IS NULL
               AND a.actual_off_utc IS NULL
               AND a.actual_in_utc IS NULL)
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
    act_off = row["actual_off_utc"] if "actual_off_utc" in keys else None
    act_on = row["actual_on_utc"] if "actual_on_utc" in keys else None
    act_in = row["actual_in_utc"] if "actual_in_utc" in keys else None

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
        "actual_off_utc":    act_off,
        "actual_on_utc":     act_on,
        "actual_in_utc":     act_in,
        "aircraft_type":   row["aircraft_type"] or "",
        "risk":            risk_level(proba, _row_threshold(row)),
        "delay_probability": round(proba, 4),
        "predicted_delay": int(row["predicted_delay"]),
        "predicted_at_utc": row["predicted_at_utc"] or "",
        "has_actual":      bool(row["has_actual"]),
        "arr_delay_min":   float(row["arr_delay_min"]) if row["arr_delay_min"] is not None else None,
        "departure_delay_min": float(row["departure_delay_min"]) if row["departure_delay_min"] is not None else None,
    }


# ── Auth routes ────────────────────────────────────────────────────────────

def _issue_token(username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    return jwt.encode(
        {"sub": username, "role": role, "exp": expire},
        JWT_SECRET, algorithm=JWT_ALGORITHM,
    )


@app.post("/auth/login")
def login(body: LoginRequest):
    con = _get_users_con()
    row = con.execute(
        "SELECT password_hash, role, active, user_type FROM users WHERE username=?", (body.username,)
    ).fetchone()
    con.close()
    if not row or not row["active"] or not _check_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")
    return {
        "access_token": _issue_token(body.username, row["role"]),
        "token_type": "bearer",
        "user_type": row["user_type"],
    }


def _verify_google_id_token(raw_token: str) -> dict:
    """Validate a Google ID token against our client ID. Raises HTTPException."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(503, "Login con Google no está configurado en este entorno")
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token as google_id_token
    except ImportError:
        raise HTTPException(503, "Dependencia google-auth no instalada")
    try:
        claims = google_id_token.verify_oauth2_token(
            raw_token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except ValueError as e:
        raise HTTPException(401, f"ID token de Google inválido: {e}")
    if not claims.get("email"):
        raise HTTPException(401, "El ID token no incluye email")
    if not claims.get("email_verified"):
        raise HTTPException(401, "El email de la cuenta de Google no está verificado")
    return claims


@app.post("/auth/google")
def login_google(body: GoogleLoginRequest):
    """Exchange a Google ID token for our own JWT, creating the user on first sign-in."""
    claims = _verify_google_id_token(body.id_token)
    email = claims["email"].lower()

    con = _get_users_con()
    row = con.execute(
        "SELECT username, role, active, user_type, provider FROM users WHERE username=? OR email=?",
        (email, email),
    ).fetchone()

    if row is None:
        con.execute(
            "INSERT INTO users (username, password_hash, role, email, provider) "
            "VALUES (?,?,'user',?, 'google')",
            (email, _unusable_password(), email),
        )
        con.commit()
        con.close()
        _upload_users_db()
        return {
            "access_token": _issue_token(email, "user"),
            "token_type": "bearer",
            "user_type": None,
            "is_new_user": True,
        }

    if not row["active"]:
        con.close()
        raise HTTPException(403, "La cuenta está desactivada")

    # Una cuenta local cuyo usuario ES este correo no se entrega.
    #
    # El alta propia guarda el correo como nombre de usuario y no puede
    # verificarlo: no hay forma de mandar un mail. Sin esta guarda, cualquiera
    # podria registrarse con el correo ajeno y esperar a que su duenio entre
    # con Google, que caeria en la cuenta del otro —cuya contrasena el otro
    # conoce—. Es toma de cuenta.
    #
    # El costo de negarse es que alguien puede ocupar un correo que no es suyo
    # y dejar a su duenio sin poder usar Google. Es molesto y visible; lo otro
    # es silencioso y grave. Se resuelve cuando haya verificacion por mail.
    if row["provider"] == "local" and row["username"] == email:
        con.close()
        raise HTTPException(
            409,
            "Ya existe una cuenta con ese correo creada con contraseña. "
            "Ingresá con tu contraseña.",
        )

    # Existing local account signing in with Google for the first time: link them.
    if row["provider"] == "local":
        con.execute(
            "UPDATE users SET email=?, updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now') "
            "WHERE username=? AND email IS NULL",
            (email, row["username"]),
        )
        con.commit()
    con.close()
    return {
        "access_token": _issue_token(row["username"], row["role"]),
        "token_type": "bearer",
        "user_type": row["user_type"],
        "is_new_user": False,
    }


def _verify_firebase_id_token(raw_token: str) -> dict:
    """Valida un ID token de Firebase y exige que el correo este verificado.

    Firebase firma estos tokens con las claves de Google, asi que `google-auth`
    —ya instalado para Google Sign-In— los valida sin dependencias nuevas.

    Exigir `email_verified` es el punto de todo esto: es la unica prueba que
    tenemos de que el correo le pertenece a quien lo presenta. Sin ella el
    endpoint no aporta nada sobre el alta propia.
    """
    if not FIREBASE_PROJECT_ID:
        raise HTTPException(503, "Firebase no está configurado en este entorno")
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token as google_id_token
    except ImportError:
        raise HTTPException(503, "Dependencia google-auth no instalada")
    try:
        claims = google_id_token.verify_firebase_token(
            raw_token, google_requests.Request(), audience=FIREBASE_PROJECT_ID
        )
    except ValueError as exc:
        raise HTTPException(401, f"ID token de Firebase inválido: {exc}")
    if not claims:
        raise HTTPException(401, "ID token de Firebase inválido")
    if not claims.get("email"):
        raise HTTPException(401, "El ID token no incluye email")
    if not claims.get("email_verified"):
        raise HTTPException(
            403,
            "Falta verificar el correo. Revisá tu casilla y volvé a intentar.",
        )
    return claims


@app.post("/auth/firebase")
def login_firebase(body: FirebaseLoginRequest):
    """Cambia un ID token de Firebase por un JWT propio.

    Firebase se ocupa de las credenciales —alta, verificacion del correo,
    recuperacion de contrasena— y esta tabla sigue siendo la duenia del rol y
    del tipo de cuenta. Es el mismo reparto que con /auth/google, con la
    diferencia de que aca el correo llega probado.
    """
    claims = _verify_firebase_id_token(body.id_token)
    email = claims["email"].lower()

    con = _get_users_con()
    row = con.execute(
        "SELECT username, role, active, user_type, provider FROM users "
        "WHERE username=? OR email=?",
        (email, email),
    ).fetchone()

    if row is None:
        con.execute(
            "INSERT INTO users (username, password_hash, role, email, provider) "
            "VALUES (?,?,'user',?, 'firebase')",
            (email, _unusable_password(), email),
        )
        con.commit()
        con.close()
        _upload_users_db()
        return {
            "access_token": _issue_token(email, "user"),
            "token_type": "bearer",
            "user_type": None,
            "is_new_user": True,
        }

    if not row["active"]:
        con.close()
        raise HTTPException(403, "La cuenta está desactivada")

    # Una cuenta local que reclamaba este correo pasa a manos de quien lo probo,
    # y su contrasena deja de servir.
    #
    # El alta propia no puede verificar el correo, asi que cualquiera pudo
    # haber registrado este y conocer su contrasena. Vincular sin mas le daria
    # al duenio real una cuenta a la que el otro sigue entrando. Con el correo
    # probado, la cuenta es de quien lo probo; el que la ocupaba pierde el
    # acceso y puede recuperarlo por Firebase si el correo era suyo.
    if row["provider"] == "local" and row["username"] == email:
        con.execute(
            "UPDATE users SET password_hash=?, provider='firebase', email=?, "
            "updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE username=?",
            (_unusable_password(), email, row["username"]),
        )
        con.commit()
        con.close()
        _upload_users_db()
        return {
            "access_token": _issue_token(row["username"], row["role"]),
            "token_type": "bearer",
            "user_type": row["user_type"],
            "is_new_user": row["user_type"] is None,
        }

    # Cuenta creada por un administrador que entra por Firebase: se la vincula
    # sin tocarle la contrasena, porque esa si la puso alguien de confianza.
    if row["provider"] == "local":
        con.execute(
            "UPDATE users SET email=?, updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now') "
            "WHERE username=? AND email IS NULL",
            (email, row["username"]),
        )
        con.commit()
    con.close()
    return {
        "access_token": _issue_token(row["username"], row["role"]),
        "token_type": "bearer",
        "user_type": row["user_type"],
        "is_new_user": False,
    }


@app.get("/auth/me")
def auth_me(request: Request):
    payload = _payload_of(request)
    username = payload.get("sub")
    con = _get_users_con()
    row = con.execute(
        "SELECT email, provider, user_type FROM users WHERE username=?", (username,)
    ).fetchone()
    con.close()
    return {
        "username": username,
        "role": payload.get("role", "user"),
        "email": row["email"] if row else None,
        "provider": row["provider"] if row else "local",
        "user_type": row["user_type"] if row else None,
    }


@app.patch("/users/me")
def update_me(request: Request, body: MeUpdate):
    """Set the caller's profile type (B2B/B2C) — used by onboarding and settings."""
    username = _payload_of(request).get("sub")
    if body.user_type is None:
        return {"ok": True}
    if body.user_type not in USER_TYPES:
        raise HTTPException(400, f"user_type inválido. Válidos: {', '.join(USER_TYPES)}")
    con = _get_users_con()
    res = con.execute(
        "UPDATE users SET user_type=?, updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now') "
        "WHERE username=?",
        (body.user_type, username),
    )
    if res.rowcount == 0:
        con.close()
        raise HTTPException(404, "Usuario no encontrado")
    con.commit()
    con.close()
    _upload_users_db()
    return {"ok": True, "user_type": body.user_type}


# ── User management (superadmin only) ──────────────────────────────────────

@app.get("/admin/users")
def list_users(request: Request):
    _require_superadmin(request)
    con = _get_users_con()
    rows = con.execute(
        "SELECT id, username, role, active, provider, user_type, created_at "
        "FROM users ORDER BY created_at"
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
        optional_fields = ", ".join(
            _optional_prediction_column(con, name)
            for name in (
                "proba_raw", "threshold_used", "threshold_strategy",
                "prediction_phase", "gdp_orig_delay_min", "gdp_dest_delay_min",
                "intermediate_dep_delay_min", "adsb_eta_delay_min", "adsb_holding_min",
            )
        )
        identity = con.execute(
            """SELECT stable_id
               FROM predictions
               WHERE fa_flight_id = ?
               ORDER BY predicted_at_utc DESC
               LIMIT 1""",
            (fa_flight_id,),
        ).fetchone()
        stable = identity["stable_id"] if identity else None

        rows = con.execute(
            f"""SELECT p.fa_flight_id, p.predicted_at_utc, p.proba_delay,
                      p.predicted_delay, {optional_fields},
                      s.feature_name, s.shap_value, s.feature_value, s.rank
               FROM predictions p
               LEFT JOIN prediction_shap s
                 ON s.fa_flight_id = p.fa_flight_id
                AND s.predicted_at_utc = p.predicted_at_utc
               WHERE p.fa_flight_id = ?
                  OR (? IS NOT NULL AND p.stable_id = ?)
               ORDER BY p.predicted_at_utc ASC, s.rank ASC""",
            (fa_flight_id, stable, stable),
        ).fetchall()

        cycles: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (row["fa_flight_id"], row["predicted_at_utc"])
            if key not in cycles:
                final_probability = float(row["proba_delay"])
                base_probability = (
                    float(row["proba_raw"])
                    if row["proba_raw"] is not None
                    else final_probability
                )
                cycles[key] = {
                    "predicted_at_utc": row["predicted_at_utc"],
                    "delay_probability": round(final_probability, 4),
                    "base_probability": round(base_probability, 4),
                    "operational_adjustment": round(final_probability - base_probability, 4),
                    "predicted_delay": int(row["predicted_delay"]),
                    "threshold_used": (
                        round(float(row["threshold_used"]), 4)
                        if row["threshold_used"] is not None
                        else None
                    ),
                    "threshold_strategy": row["threshold_strategy"],
                    "prediction_phase": row["prediction_phase"] or "PRE_DEPARTURE",
                    "operational_context": {
                        "gdp_origin_delay_min": row["gdp_orig_delay_min"],
                        "gdp_destination_delay_min": row["gdp_dest_delay_min"],
                        "intermediate_departure_delay_min": row["intermediate_dep_delay_min"],
                        "adsb_eta_delay_min": row["adsb_eta_delay_min"],
                        "adsb_holding_min": row["adsb_holding_min"],
                    },
                    "shap": [],
                }

            if row["feature_name"] is not None:
                contribution = float(row["shap_value"])
                cycles[key]["shap"].append({
                    "feature": row["feature_name"],
                    "label": FEATURE_LABELS.get(
                        row["feature_name"],
                        row["feature_name"].replace("_", " ").title(),
                    ),
                    "contribution": round(abs(contribution), 4),
                    "direction": "positive" if contribution >= 0 else "negative",
                    "value": row["feature_value"],
                })

        return list(cycles.values())
    finally:
        con.close()


def _get_historical_flight(con: sqlite3.Connection, fa_flight_id: str):
    """Fallback: fetch flight+latest prediction without time-window constraints.

    Used when a flight has already departed and is no longer in the active
    sliding window returned by _latest_predictions_active().
    """
    # Sin alias: la subconsulta lee `predictions` directamente.
    threshold_col_inner = _optional_prediction_column(
        con, "threshold_used", table_alias=""
    )
    return con.execute(f"""
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
               p.threshold_used,
               p.predicted_at_utc,
               CASE WHEN a.arr_delay_min IS NOT NULL THEN 1 ELSE 0 END AS has_actual,
               a.arr_delay_min,
               a.departure_delay_min,
               a.actual_out_utc
        FROM flights f
        JOIN (
            SELECT fa_flight_id, proba_delay, predicted_delay, predicted_at_utc,
                   {threshold_col_inner},
                   ROW_NUMBER() OVER (
                       PARTITION BY fa_flight_id ORDER BY predicted_at_utc DESC
                   ) AS rn
            FROM predictions
            WHERE fa_flight_id = ?
        ) p ON p.fa_flight_id = f.fa_flight_id AND p.rn = 1
        LEFT JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
        WHERE f.fa_flight_id = ?
          AND f.origin = 'ATL'
          AND f.cancelled = 0
    """, (fa_flight_id, fa_flight_id)).fetchone()


@app.get("/flights/{fa_flight_id:path}")
def get_flight(fa_flight_id: str):
    con = get_db()
    try:
        rows = _latest_predictions_active(con)
        row = next((r for r in rows if r["fa_flight_id"] == fa_flight_id), None)
        is_historical = row is None
        if is_historical:
            row = _get_historical_flight(con, fa_flight_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Flight not found")

        result = _flight_row_to_dict(row)
        result["is_historical"] = is_historical
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
    except sqlite3.OperationalError as exc:
        # Base sin la tabla: es esperable en una legacy, y el llamador cae al
        # calculo en vivo. Se deja rastro igual, porque si la tabla existe y
        # falla por otra cosa, el sintoma es identico.
        print(f"[shap] cache no disponible para {fa_flight_id}: {exc}")
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
    except Exception as exc:
        # Se loguea antes de devolver vacio. Este es el ultimo recurso: si la
        # cache de `prediction_shap` no tiene nada y esto tampoco, el detalle
        # del vuelo queda sin explicacion. Tragarse la excepcion dejaba el
        # sintoma —`"shap": []`— sin ninguna pista de la causa, que es
        # exactamente lo que hizo dificil de diagnosticar el issue #5.
        print(f"[shap] calculo en vivo fallo para {fa_flight_id}: "
              f"{type(exc).__name__}: {exc}")
        return []


def _last_run(con) -> dict:
    """El ultimo ciclo: cuando corrio, cuantos vuelos tenia y cuantos predijo.

    `flights_targeted` es lo que distingue "no habia nada que hacer" de "habia
    trabajo y no se hizo". Sin ese numero las dos se ven igual desde afuera,
    porque en ambas `flights_predicted` es cero.
    """
    try:
        fila = con.execute(
            """SELECT COALESCE(finished_utc, started_utc) AS cuando,
                      flights_targeted, flights_predicted
                 FROM runs
                WHERE finished_utc IS NOT NULL
                ORDER BY COALESCE(finished_utc, started_utc) DESC LIMIT 1"""
        ).fetchone()
    except sqlite3.OperationalError:
        return {"utc": "", "targeted": None, "predicted": None}
    if not fila:
        return {"utc": "", "targeted": None, "predicted": None}
    return {
        "utc": fila["cuando"] or "",
        "targeted": fila["flights_targeted"],
        "predicted": fila["flights_predicted"],
    }


def _last_run_utc(con) -> str:
    """Cuando corrio el ultimo ciclo, haya predicho algo o no.

    Distinto de `last_tick_utc`, que sale de las predicciones servidas y por lo
    tanto es *el ultimo ciclo que produjo algo*. De madrugada ATL pasa horas sin
    una sola salida programada en la ventana: el ciclo corre, no encuentra nada
    que predecir, y `last_tick_utc` se queda quieto aunque el pipeline este
    perfecto.

    El 16/09 a las 04:38 UTC eso disparo una alerta de datos viejos con los dos
    jobs sanos —80 corridas seguidas sin fallar—. La tabla `runs` recibe una
    fila por ciclo pase lo que pase, asi que es la senial que distingue "el
    pipeline se detuvo" de "no habia nada que hacer".
    """
    try:
        fila = con.execute(
            "SELECT MAX(COALESCE(finished_utc, started_utc)) AS ultimo FROM runs"
        ).fetchone()
    except sqlite3.OperationalError:
        return ""
    return (fila["ultimo"] or "") if fila else ""


@app.get("/metrics/summary")
def metrics_summary():
    con = get_db()
    try:
        ciclo = _last_run(con)
        ultimo_ciclo = ciclo["utc"]
        rows = _latest_predictions_active(con)
        if not rows:
            return {
                "total_flights": 0, "high_risk": 0, "medium_risk": 0,
                "low_risk": 0, "avg_delay_probability": 0.0,
                "predicted_positive_rate": 0.0,
                "model_version": ACTIVE_MODEL, "last_tick_utc": "",
                "last_run_utc": ultimo_ciclo,
                "last_run_targeted": ciclo["targeted"],
                "last_run_predicted": ciclo["predicted"],
            }
        probas = [float(r["proba_delay"]) for r in rows]
        preds  = [int(r["predicted_delay"]) for r in rows]
        # Mismo criterio que /flights: calculados por separado terminarian
        # discrepando entre las fichas y el conteo del encabezado.
        niveles = [risk_level(float(r["proba_delay"]), _row_threshold(r)) for r in rows]
        ticks  = [r["predicted_at_utc"] for r in rows if r["predicted_at_utc"]]
        return {
            "total_flights":           len(rows),
            "high_risk":               niveles.count("high"),
            "medium_risk":             niveles.count("medium"),
            "low_risk":                niveles.count("low"),
            "avg_delay_probability":   round(float(np.mean(probas)), 4),
            "predicted_positive_rate": round(float(np.mean(preds)), 4),
            "model_version":           ACTIVE_MODEL,
            "last_tick_utc":           max(ticks) if ticks else "",
            # El ciclo corre aunque no haya vuelos que predecir; el tick no.
            "last_run_utc":            ultimo_ciclo,
            # Con cuantos vuelos se encontro el ciclo y cuantos predijo. Sin el
            # primero no se distingue la madrugada de una falla.
            "last_run_targeted":       ciclo["targeted"],
            "last_run_predicted":      ciclo["predicted"],
        }
    finally:
        con.close()


@app.get("/metrics/hourly")
def metrics_hourly():
    con = get_db()
    try:
        rows = _latest_predictions_active(con)
        flights = [_flight_row_to_dict(r) for r in rows]
        buckets: dict[str, dict] = {}
        for f in flights:
            sched = f.get("scheduled_out_utc") or ""
            if len(sched) >= 16:
                hour = sched[11:13] + ":00"
            else:
                hour = "??"
            if hour not in buckets:
                buckets[hour] = {"hour": hour, "total": 0, "high_risk": 0, "medium_risk": 0, "low_risk": 0, "sum_proba": 0.0}
            b = buckets[hour]
            b["total"] += 1
            p = float(f.get("delay_probability", 0))
            b["sum_proba"] += p
            risk = f.get("risk", "low")
            if risk == "high":
                b["high_risk"] += 1
            elif risk == "medium":
                b["medium_risk"] += 1
            else:
                b["low_risk"] += 1

        result = []
        for hour in sorted(buckets):
            b = buckets[hour]
            result.append({
                "hour":        hour,
                "total":       b["total"],
                "high_risk":   b["high_risk"],
                "medium_risk": b["medium_risk"],
                "low_risk":    b["low_risk"],
                "avg_proba":   round(b["sum_proba"] / b["total"], 4) if b["total"] else 0.0,
            })
        return result
    finally:
        con.close()


@app.get("/metrics/history")
def metrics_history(segment: str = "all", days: int = 56):
    """Serie diaria de calidad del modelo, mas alla de la retencion de 30 dias.

    Sale de `metrics_daily`, que el job escribe antes de purgar. Las tablas
    crudas se borran a los 30 dias; estos agregados no, asi que la serie crece
    sin limite practico —son kilobytes por semana—. Ver issue #12.

    `segment` acepta 'all', 'carrier:DL' o 'hour:14'. El default de 56 dias son
    las 8 semanas que pide Frontend #3.
    """
    con = get_db()
    try:
        try:
            filas = con.execute(
                """SELECT day, n_flights, n_delayed, n_flagged, tp, fp, tn, fn,
                          auc, brier, ece, mean_proba, mean_threshold, model_version
                     FROM metrics_daily
                    WHERE segment = ?
                      AND day >= date('now', ?)
                    ORDER BY day""",
                (segment, f"-{max(1, min(int(days), 3650))} days"),
            ).fetchall()
        except sqlite3.OperationalError:
            # Base anterior al primer rollup: serie vacia, no un 500.
            return {"segment": segment, "days": days, "points": []}

        puntos = []
        for r in filas:
            tp, fp, fn = r["tp"], r["fp"], r["fn"]
            puntos.append({
                "day": r["day"],
                "n_flights": r["n_flights"],
                "n_delayed": r["n_delayed"],
                "n_flagged": r["n_flagged"],
                "actual_delay_rate": round(r["n_delayed"] / r["n_flights"], 4)
                                     if r["n_flights"] else None,
                "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
                "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
                "auc": round(r["auc"], 4) if r["auc"] is not None else None,
                "brier": round(r["brier"], 4) if r["brier"] is not None else None,
                "ece": round(r["ece"], 4) if r["ece"] is not None else None,
                "mean_proba": round(r["mean_proba"], 4) if r["mean_proba"] is not None else None,
                "mean_threshold": round(r["mean_threshold"], 4)
                                  if r["mean_threshold"] is not None else None,
                "model_version": r["model_version"],
            })
        return {"segment": segment, "days": days, "points": puntos}
    finally:
        con.close()


@app.get("/metrics/breakdown")
def metrics_breakdown(by: str = "carrier", days: int = 28):
    """Calidad del modelo agregada por aerolinea, por hora del dia o por fase.

    Suma las filas diarias de `metrics_daily` sobre la ventana pedida en vez de
    promediar los promedios: un dia con 8 vuelos de una aerolinea chica no puede
    pesar lo mismo que uno con 200. Los conteos se suman y las metricas se
    recalculan sobre el total.

    El AUC no se puede sumar —depende del orden entre vuelos, no de conteos— asi
    que se promedia ponderando por vuelos y se devuelve aparte, dicho como lo
    que es. Ver Frontend #3.
    """
    prefijos = {"carrier": "carrier:", "hour": "hour:", "phase": "phase:"}
    if by not in prefijos:
        raise HTTPException(400, f"`by` debe ser uno de: {', '.join(prefijos)}")
    prefijo = prefijos[by]

    con = get_db()
    try:
        try:
            filas = con.execute(
                """SELECT segment, n_flights, n_delayed, n_flagged, tp, fp, tn, fn,
                          auc, brier, ece
                     FROM metrics_daily
                    WHERE segment LIKE ?
                      AND day >= date('now', ?)""",
                (f"{prefijo}%", f"-{max(1, min(int(days), 3650))} days"),
            ).fetchall()
        except sqlite3.OperationalError:
            return {"by": by, "days": days, "rows": []}

        acumulado: dict[str, dict] = {}
        for r in filas:
            clave = r["segment"][len(prefijo):]
            acc = acumulado.setdefault(clave, {
                "key": clave, "n_flights": 0, "n_delayed": 0, "n_flagged": 0,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "_auc_peso": 0.0, "_auc_suma": 0.0,
                "_brier_suma": 0.0, "_ece_suma": 0.0, "_metrica_peso": 0.0,
                "n_days": 0,
            })
            for c in ("n_flights", "n_delayed", "n_flagged", "tp", "fp", "tn", "fn"):
                acc[c] += r[c]
            acc["n_days"] += 1
            peso = float(r["n_flights"] or 0)
            if r["auc"] is not None and peso:
                acc["_auc_suma"] += float(r["auc"]) * peso
                acc["_auc_peso"] += peso
            if peso:
                if r["brier"] is not None:
                    acc["_brier_suma"] += float(r["brier"]) * peso
                if r["ece"] is not None:
                    acc["_ece_suma"] += float(r["ece"]) * peso
                acc["_metrica_peso"] += peso

        salida = []
        for acc in acumulado.values():
            tp, fp, fn, n = acc["tp"], acc["fp"], acc["fn"], acc["n_flights"]
            aciertos = acc["tp"] + acc["tn"]
            salida.append({
                "key": acc["key"],
                "n_flights": n,
                "n_days": acc["n_days"],
                "n_delayed": acc["n_delayed"],
                "n_flagged": acc["n_flagged"],
                "actual_delay_rate": round(acc["n_delayed"] / n, 4) if n else None,
                "accuracy": round(aciertos / n, 4) if n else None,
                "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
                "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
                "auc": round(acc["_auc_suma"] / acc["_auc_peso"], 4)
                       if acc["_auc_peso"] else None,
                "brier": round(acc["_brier_suma"] / acc["_metrica_peso"], 4)
                         if acc["_metrica_peso"] else None,
                "ece": round(acc["_ece_suma"] / acc["_metrica_peso"], 4)
                       if acc["_metrica_peso"] else None,
            })
        salida.sort(key=lambda x: (-x["n_flights"], x["key"]))
        return {"by": by, "days": days, "rows": salida}
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


@app.get("/metrics/classification")
def metrics_classification():
    """Precision, recall, F1 y confusion matrix sobre predicciones con actuals (últimos 7 días)."""
    try:
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            confusion_matrix, roc_auc_score, brier_score_loss,
        )
        con = get_db()
        import pandas as pd
        df = pd.read_sql("""
            SELECT p.fa_flight_id, p.proba_delay, p.predicted_delay, p.predicted_at_utc,
                   COALESCE(p.prediction_phase, 'PRE_DEPARTURE') AS prediction_phase,
                   CASE WHEN a.arr_delay_min > 15 THEN 1 ELSE 0 END AS delayed
            FROM predictions p
            JOIN actuals a ON p.fa_flight_id = a.fa_flight_id
            WHERE a.arr_delay_min IS NOT NULL AND a.cancelled = 0
              AND p.predicted_at_utc >= datetime('now', '-7 days')
        """, con)
        con.close()

        if df.empty or len(df) < 30:
            return {"error": "Datos insuficientes", "n_actuals": len(df)}

        # PRE_DEPARTURE: primera predicción por vuelo
        pre = (df[df["prediction_phase"] == "PRE_DEPARTURE"]
               .sort_values("predicted_at_utc")
               .groupby("fa_flight_id", as_index=False).first())
        if pre.empty:
            pre = df.sort_values("predicted_at_utc").groupby("fa_flight_id", as_index=False).first()

        y_true = pre["delayed"].to_numpy(dtype=int)
        y_pred = pre["predicted_delay"].fillna(0).to_numpy(dtype=int)
        y_prob = pre["proba_delay"].to_numpy(dtype=float)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        prec   = float(precision_score(y_true, y_pred, zero_division=0))
        rec    = float(recall_score(y_true, y_pred, zero_division=0))
        f1     = float(f1_score(y_true, y_pred, zero_division=0))
        acc    = float((tp + tn) / len(y_true))
        fpr    = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        auc    = float(roc_auc_score(y_true, y_prob)) if y_true.sum() >= 5 else None
        brier  = float(brier_score_loss(y_true, y_prob))
        actual_delay_rate = float(y_true.mean())

        return {
            "n_actuals":         int(len(y_true)),
            "actual_delay_rate": round(actual_delay_rate, 4),
            "predicted_pos_rate": round(float(y_pred.mean()), 4),
            "auc":               round(auc, 4) if auc else None,
            "brier":             round(brier, 4),
            "precision":         round(prec, 4),
            "recall":            round(rec, 4),
            "f1":                round(f1, 4),
            "accuracy":          round(acc, 4),
            "false_positive_rate": round(fpr, 4),
            "confusion_matrix": {
                "TP": int(tp), "FP": int(fp),
                "TN": int(tn), "FN": int(fn),
            },
        }
    except Exception as e:
        return {"error": str(e)}


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
                  -- "programado hace poco y todavia sin salir".
              --
              -- Antes esto era `a.actual_out_utc IS NULL` a secas. Ese campo
              -- lo poblaba AeroAPI; FR24, que es el 96% de la muestra desde
              -- Fase 4, no expone gate-out y lo deja nulo siempre. La
              -- condicion se cumplia para todos y la ventana retenia las 24 h
              -- enteras: medido, 492 de 637 vuelos con salida programada hace
              -- mas de 6 h, y el 98% de ellos ya habia aterrizado.
              --
              -- Se exige que no haya NINGUNA senal de salida, en vez de
              -- confiar en una sola columna: un vuelo con despegue o con
              -- aterrizaje obviamente salio, lo diga o no el gate-out.
              (datetime(f.scheduled_out_utc) >= datetime(?)
               AND a.actual_out_utc IS NULL
               AND a.actual_off_utc IS NULL
               AND a.actual_in_utc IS NULL)
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


_SIZE_SAMPLE_ROWS = 5000


def _table_bytes_exact(con, tables: list[str]) -> dict[str, int] | None:
    """
    Bytes reales por tabla, incluidos sus índices, vía el vtab `dbstat`.

    Sólo está disponible si SQLite fue compilado con SQLITE_ENABLE_DBSTAT_VTAB,
    lo que no está garantizado. Devuelve None si no existe, para que el llamador
    caiga a la estimación por muestreo.
    """
    try:
        con.execute("SELECT 1 FROM dbstat LIMIT 1")
    except sqlite3.OperationalError:
        return None

    sizes: dict[str, int] = {}
    for tbl in tables:
        try:
            # `name` en dbstat cubre tanto la tabla como sus índices; se
            # agrupan bajo la tabla para que el total sea el costo real.
            row = con.execute(
                """
                SELECT COALESCE(SUM(pgsize), 0) FROM dbstat
                 WHERE name = ?
                    OR name IN (SELECT name FROM sqlite_master
                                 WHERE type = 'index' AND tbl_name = ?)
                """,
                (tbl, tbl),
            ).fetchone()
            sizes[tbl] = int(row[0]) if row else 0
        except sqlite3.OperationalError:
            sizes[tbl] = 0
    return sizes


def _table_bytes_estimated(con, tables: list[str]) -> dict[str, int]:
    """
    Estimación por muestreo, para cuando `dbstat` no está disponible.

    Mide el largo en bytes de cada columna sobre una muestra y lo extrapola por
    la cantidad de filas. No contempla el peso de los índices ni el overhead de
    página, así que subestima: sirve para ordenar tablas por peso relativo, que
    es lo que hace falta para decidir qué purgar.
    """
    sizes: dict[str, int] = {}
    for tbl in tables:
        try:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({tbl})")]
            if not cols:
                sizes[tbl] = 0
                continue

            total_rows = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            if total_rows == 0:
                sizes[tbl] = 0
                continue

            # CAST a BLOB para contar bytes y no caracteres.
            expr = " + ".join(
                f'COALESCE(LENGTH(CAST("{c}" AS BLOB)), 0)' for c in cols
            )
            row = con.execute(
                f"SELECT COALESCE(SUM({expr}), 0), COUNT(*) "
                f"FROM (SELECT * FROM {tbl} LIMIT {_SIZE_SAMPLE_ROWS})"
            ).fetchone()
            sampled_bytes, sampled_rows = int(row[0]), int(row[1])
            if sampled_rows == 0:
                sizes[tbl] = 0
                continue

            sizes[tbl] = int(sampled_bytes / sampled_rows * total_rows)
        except sqlite3.OperationalError:
            sizes[tbl] = 0
    return sizes


@app.get("/admin/db-stats")
def db_stats(request: Request):
    _require_superadmin(request)
    db_file = Path(DB_PATH)
    size_mb = db_file.stat().st_size / 1e6 if db_file.exists() else 0.0

    con = get_db()
    try:
        counts = {}
        for tbl in ["predictions", "actuals", "flights", "runs", "weather_obs", "prediction_shap", "aircraft_position"]:
            try:
                row = con.execute(f"SELECT count(*) FROM {tbl}").fetchone()
                counts[tbl] = row[0] if row else 0
            except sqlite3.OperationalError:
                counts[tbl] = 0

        # Get date range of predictions
        date_range = {"first": None, "last": None}
        try:
            row = con.execute("SELECT MIN(predicted_at_utc), MAX(predicted_at_utc) FROM predictions").fetchone()
            if row:
                date_range["first"] = row[0]
                date_range["last"] = row[1]
        except Exception:
            pass

        # Peso por tabla: sin esto no se puede decidir qué purgar, porque la
        # cantidad de filas no dice nada del espacio que ocupan.
        tables = list(counts.keys())
        exact = _table_bytes_exact(con, tables)
        table_bytes = exact if exact is not None else _table_bytes_estimated(con, tables)
        table_sizes_mb = {t: round(b / 1e6, 2) for t, b in table_bytes.items()}

        # Antigüedad por tabla, para saber cuánto libera cada ventana de corte.
        AGE_COLUMNS = {
            "predictions": "predicted_at_utc",
            "prediction_shap": "predicted_at_utc",
            "actuals": "settled_at_utc",
            "weather_obs": "valid_utc",
        }
        table_dates: dict[str, dict[str, str | None]] = {}
        for tbl, col in AGE_COLUMNS.items():
            try:
                row = con.execute(f"SELECT MIN({col}), MAX({col}) FROM {tbl}").fetchone()
                table_dates[tbl] = {"first": row[0], "last": row[1]} if row else {}
            except sqlite3.OperationalError:
                continue

        return {
            "db_size_mb": round(size_mb, 2),
            "table_counts": counts,
            "table_sizes_mb": table_sizes_mb,
            "table_sizes_source": "dbstat" if exact is not None else "sampled",
            "table_dates": table_dates,
            "sources": _source_freshness(con),
            "cycles": _recent_cycles(con),
            "chain_calibrator": _chain_calibrator_status(con),
            "db_size_warn_mb": DB_SIZE_WARN_MB,
            "db_size_over_warn": size_mb > DB_SIZE_WARN_MB,
            "prediction_dates": date_range,
            "refresh": {
                "last_ok_utc": _db_last_refresh_ok_utc,
                "last_error": _db_last_refresh_error,
                # Estado interno del refresh. Sin esto habia que inferirlo de
                # los logs, y un refresh que no se intenta no deja ninguno.
                "seconds_since_last": round(
                    time.monotonic() - _db_last_refresh, 1
                ),
                "interval_seconds": _DB_REFRESH_INTERVAL,
                "is_due": (
                    time.monotonic() - _db_last_refresh >= _DB_REFRESH_INTERVAL
                ),
                # Refresco sincrono: si el lock esta tomado, hay un pedido
                # descargando ahora mismo.
                "lock_held": _DB_REFRESH_LOCK.locked(),
            },
        }
    finally:
        con.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

"""
Vigía del pipeline. Corre fuera del backend y avisa por Telegram cuando algo
se rompe.

Por qué vive aparte: el 13/09 el backend estuvo dos días sirviendo un snapshot
viejo. Respondía 200, el dashboard mostraba datos de hace dos días, y nadie se
enteró. Un chequeo embebido en el backend no habría servido — si el backend es
el que falla, no puede ser él quien avise.

Chequea tres cosas, de afuera hacia adentro:

1. Que los jobs estén escribiendo: la antigüedad del objeto en GCS.
2. Que el backend responda.
3. Que el backend sirva datos frescos, que es distinto de responder. Este es
   el que habría detectado el incidente.

Sólo notifica en los cambios de estado. Un problema que persiste no vuelve a
avisar en cada ciclo, y cuando se resuelve manda un aviso de recuperación.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

GCS_BUCKET = os.environ["GCS_BUCKET"]
DB_OBJECT = os.getenv("DB_OBJECT", "live_data.db")
STATE_OBJECT = os.getenv("WATCHDOG_STATE_OBJECT", "watchdog_state.json")
BACKEND_URL = os.environ["BACKEND_URL"].rstrip("/")

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.environ["API_PASSWORD"]

# Los predictores corren cada 15 min. Tres ciclos perdidos es una señal clara y
# deja margen para un ciclo lento o un reintento.
STALE_AFTER_MIN = int(os.getenv("WATCHDOG_STALE_AFTER_MIN", "45"))
HTTP_TIMEOUT = 30


@dataclass
class Check:
    """Resultado de un chequeo. `key` identifica la alerta entre corridas."""

    key: str
    ok: bool
    detail: str


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _minutes_since(ts: datetime) -> float:
    return (_now() - ts).total_seconds() / 60


def _parse_utc(value: str) -> datetime:
    """Acepta tanto '...Z' como '...+00:00'; el backend usa el segundo."""
    cleaned = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(cleaned)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ── Chequeos ───────────────────────────────────────────────────────────────


def check_gcs_freshness() -> Check:
    """¿Los jobs siguen escribiendo la base?"""
    from google.cloud import storage

    try:
        blob = storage.Client().bucket(GCS_BUCKET).get_blob(DB_OBJECT)
        if blob is None:
            return Check("gcs", False, f"No existe gs://{GCS_BUCKET}/{DB_OBJECT}")
        age = _minutes_since(blob.updated)
        if age > STALE_AFTER_MIN:
            return Check("gcs", False, f"La base en GCS no se actualiza hace {age:.0f} min")
        return Check("gcs", True, f"Base actualizada hace {age:.0f} min")
    except Exception as exc:
        return Check("gcs", False, f"No se pudo leer GCS: {exc}")


def check_backend_up() -> Check:
    """¿El backend responde?"""
    try:
        response = requests.get(f"{BACKEND_URL}/docs", timeout=HTTP_TIMEOUT)
        if response.status_code != 200:
            return Check("backend_up", False, f"/docs devolvió {response.status_code}")
        return Check("backend_up", True, "Backend respondiendo")
    except Exception as exc:
        return Check("backend_up", False, f"Backend inalcanzable: {exc}")


def check_backend_data_fresh() -> Check:
    """
    ¿El backend sirve datos frescos?

    Responder no alcanza: en el incidente del 13/09 devolvía 200 con un
    snapshot de dos días atrás. Este chequeo compara contra el último tick.
    """
    try:
        auth = requests.post(
            f"{BACKEND_URL}/auth/login",
            json={"username": API_USERNAME, "password": API_PASSWORD},
            timeout=HTTP_TIMEOUT,
        )
        if auth.status_code != 200:
            return Check("backend_data", False, f"Login falló con {auth.status_code}")
        token = auth.json()["access_token"]

        summary = requests.get(
            f"{BACKEND_URL}/metrics/summary",
            headers={"Authorization": f"Bearer {token}"},
            timeout=HTTP_TIMEOUT,
        ).json()

        tick = summary.get("last_tick_utc")
        if not tick:
            return Check("backend_data", False, "El backend no reporta ningún tick")

        age = _minutes_since(_parse_utc(tick))
        if age > STALE_AFTER_MIN:
            return Check(
                "backend_data",
                False,
                f"El backend sirve datos de hace {age:.0f} min "
                f"({summary.get('total_flights', 0)} vuelos). "
                "Los jobs pueden estar bien: suele ser el refresh desde GCS.",
            )
        return Check(
            "backend_data",
            True,
            f"Datos de hace {age:.0f} min ({summary.get('total_flights', 0)} vuelos)",
        )
    except Exception as exc:
        return Check("backend_data", False, f"No se pudo verificar la frescura: {exc}")


# ── Estado, para no repetir la misma alerta ────────────────────────────────


def load_state() -> dict:
    from google.cloud import storage

    try:
        blob = storage.Client().bucket(GCS_BUCKET).get_blob(STATE_OBJECT)
        return json.loads(blob.download_as_text()) if blob else {}
    except Exception:
        # Sin estado previo se asume todo sano: a lo sumo se repite un aviso.
        return {}


def save_state(state: dict) -> None:
    from google.cloud import storage

    try:
        storage.Client().bucket(GCS_BUCKET).blob(STATE_OBJECT).upload_from_string(
            json.dumps(state), content_type="application/json"
        )
    except Exception as exc:
        print(f"[watchdog] no se pudo guardar el estado: {exc}")


# ── Telegram ───────────────────────────────────────────────────────────────


def notify(text: str) -> None:
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=HTTP_TIMEOUT,
        )
        if response.status_code != 200:
            print(f"[watchdog] Telegram devolvió {response.status_code}: {response.text}")
    except Exception as exc:
        print(f"[watchdog] no se pudo notificar: {exc}")


def main() -> int:
    checks = [check_gcs_freshness(), check_backend_up(), check_backend_data_fresh()]
    state = load_state()
    new_state = {}
    transitions: list[str] = []

    for check in checks:
        was_ok = state.get(check.key, True)
        new_state[check.key] = check.ok
        status = "OK " if check.ok else "FAIL"
        print(f"[watchdog] {status} {check.key}: {check.detail}")

        if was_ok and not check.ok:
            transitions.append(f"🔴 *{check.key}*\n{check.detail}")
        elif not was_ok and check.ok:
            transitions.append(f"🟢 *{check.key}* recuperado\n{check.detail}")

    if transitions:
        stamp = _now().strftime("%d/%m %H:%M UTC")
        notify("*OnTimeAI*\n\n" + "\n\n".join(transitions) + f"\n\n_{stamp}_")

    save_state(new_state)
    # Salir con 0 aunque haya fallos: el vigía cumplió su trabajo. Un exit
    # distinto marcaría el job como fallido y dispararía ruido sobre el ruido.
    return 0


if __name__ == "__main__":
    sys.exit(main())

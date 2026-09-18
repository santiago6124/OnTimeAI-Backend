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

def _env(name: str, default: str | None = None) -> str:
    """
    Lee una variable y le saca los espacios de los extremos.

    Los secretos de Secret Manager se inyectan como bytes crudos, sin tocar. Un
    valor cargado con `print()` o desde un archivo termina con un salto de
    linea, y ese caracter viaja hasta el destino: la primera corrida de este
    vigia fallo con 401 porque intentaba loguearse con "<password>\n". El
    backend no lo sufria porque ya hacia strip; este proceso tenia que hacer lo
    mismo.
    """
    value = os.environ[name] if default is None else os.getenv(name, default)
    return value.strip()


GCS_BUCKET = _env("GCS_BUCKET")
DB_OBJECT = _env("DB_OBJECT", "live_data.db")
STATE_OBJECT = _env("WATCHDOG_STATE_OBJECT", "watchdog_state.json")
BACKEND_URL = _env("BACKEND_URL").rstrip("/")

TELEGRAM_TOKEN = _env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID")

API_USERNAME = _env("API_USERNAME", "admin")
API_PASSWORD = _env("API_PASSWORD")

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


def _login() -> str:
    """Token del backend. Lanza si no se pudo, para que el chequeo lo reporte."""
    response = requests.post(
        f"{BACKEND_URL}/auth/login",
        json={"username": API_USERNAME, "password": API_PASSWORD},
        timeout=HTTP_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(f"login fallo con {response.status_code}")
    return response.json()["access_token"]


def check_sources_fresh() -> list[Check]:
    """Una senal por fuente de datos, no una sola por el pipeline entero.

    Que el pipeline corra no significa que siga recibiendo todo lo que recibia.
    ADS-B dejo de escribir el 12/08 y estuvo 25 dias inerte: las corridas
    figuraban exitosas, no habia errores, y dos de las cuatro etapas de ajuste
    recibian None en cada ciclo. La causa —403 de airplanes.live pidiendo
    registro, y timeout de OpenSky— estaba en los logs del harvester, en nivel
    WARNING, sin que nada la levantara. Ver #9.

    Cada fuente es su propia alerta: una caida no puede quedar tapada por el
    resto funcionando.
    """
    try:
        stats = requests.get(
            f"{BACKEND_URL}/admin/db-stats",
            headers={"Authorization": f"Bearer {_login()}"},
            timeout=HTTP_TIMEOUT,
        ).json()
    except Exception as exc:
        return [Check("sources", False, f"No se pudo leer la frescura por fuente: {exc}")]

    sources = stats.get("sources") or {}
    if not sources:
        # Backend viejo, anterior a que el endpoint lo expusiera.
        return []

    checks: list[Check] = []
    for table, info in sorted(sources.items()):
        age = info.get("age_minutes")
        fed_by = info.get("fed_by", "")
        if age is None:
            detail = f"`{table}` esta vacia o sin fecha legible ({fed_by})"
        else:
            detail = (
                f"`{table}` no recibe datos hace {age:.0f} min "
                f"(se toleran {info.get('tolerated_minutes')}) — {fed_by}"
            )
        checks.append(Check(f"source:{table}", not info.get("stale", False), detail))
    return checks


def check_capacity() -> list[Check]:
    """Tamano de la base y duracion del ciclo, antes de que sean un problema.

    El job baja la base entera de GCS, la modifica y la vuelve a subir en cada
    ciclo, asi que su duracion crece con el archivo. Los predictores corren cada
    15 minutos: cuando el ciclo se acerca a esa ventana, dos jobs terminan
    escribiendo la misma base y pisandose las subidas.

    Es una senal adelantada, no una caida: avisa mientras todavia hay margen
    para hacer algo —purgar, vacuumear mas seguido, partir el historico—.
    Ver issue #4.
    """
    try:
        stats = requests.get(
            f"{BACKEND_URL}/admin/db-stats",
            headers={"Authorization": f"Bearer {_login()}"},
            timeout=HTTP_TIMEOUT,
        ).json()
    except Exception as exc:
        return [Check("capacity", False, f"No se pudo leer /admin/db-stats: {exc}")]

    checks: list[Check] = []

    size = stats.get("db_size_mb")
    limite = stats.get("db_size_warn_mb")
    if size is not None and limite:
        checks.append(Check(
            "db_size",
            not stats.get("db_size_over_warn", False),
            f"La base pesa {size:.0f} MB (se avisa sobre {limite})",
        ))

    cal = stats.get("chain_calibrator") or {}
    if cal:
        checks.append(Check(
            "chain_calibrator",
            not cal.get("stale", True),
            f"Calibrador post-cadena: {cal.get('detail', 'sin detalle')}",
        ))

    ciclos = stats.get("cycles") or {}
    if ciclos.get("median_minutes") is not None:
        checks.append(Check(
            "cycle_duration",
            not ciclos.get("slow", False),
            f"El ciclo tarda {ciclos['median_minutes']:.1f} min de mediana, "
            f"maximo {ciclos.get('max_minutes', 0):.1f} "
            f"(se toleran {ciclos.get('tolerated_minutes')} contra un scheduler "
            f"de {ciclos.get('scheduler_minutes')} min)",
        ))
    return checks


def check_backend_data_fresh() -> Check:
    """
    ¿El backend sirve datos frescos?

    Responder no alcanza: en el incidente del 13/09 devolvía 200 con un
    snapshot de dos días atrás. Este chequeo compara contra el último ciclo.

    Se mira `last_run_utc` y no `last_tick_utc`. El tick sale de las
    predicciones servidas, así que es *el último ciclo que produjo algo*: de
    madrugada ATL pasa horas sin una sola salida programada en la ventana, el
    ciclo corre, no encuentra nada que predecir, y el tick se queda quieto con
    el pipeline perfecto. El 16/09 a las 04:38 UTC eso disparó una alerta con
    los dos jobs sanos, 80 corridas seguidas sin fallar.

    Un monitor que grita en falso cada madrugada se empieza a ignorar, y así es
    como se pasa por alto el incidente de verdad. `runs` recibe una fila por
    ciclo pase lo que pase: esa es la señal que distingue "el pipeline se
    detuvo" de "no había nada que hacer".

    Que el ciclo tuviera vuelos para predecir y no predijera ninguno sí es un
    problema, y se reporta aparte. Eso se mide con `last_run_targeted`, el
    número que el ciclo mismo calcula, y no con los vuelos servidos: esos son
    los que están en el aire, no los que esperan predicción. Confundir una cosa
    con la otra fue la falsa alarma del 17/09.
    """
    try:
        token = _login()

        summary = requests.get(
            f"{BACKEND_URL}/metrics/summary",
            headers={"Authorization": f"Bearer {token}"},
            timeout=HTTP_TIMEOUT,
        ).json()

        vuelos = summary.get("total_flights", 0)
        corrida = summary.get("last_run_utc")
        tick = summary.get("last_tick_utc")

        if not corrida:
            # Backend anterior a este campo, o base sin tabla `runs`: se cae al
            # criterio viejo en vez de quedarse sin chequeo.
            if not tick:
                return Check("backend_data", False, "El backend no reporta ningún ciclo")
            edad = _minutes_since(_parse_utc(tick))
            if edad > STALE_AFTER_MIN:
                return Check(
                    "backend_data", False,
                    f"El backend sirve datos de hace {edad:.0f} min ({vuelos} vuelos). "
                    "Los jobs pueden estar bien: suele ser el refresh desde GCS.",
                )
            return Check("backend_data", True, f"Datos de hace {edad:.0f} min ({vuelos} vuelos)")

        edad_ciclo = _minutes_since(_parse_utc(corrida))
        if edad_ciclo > STALE_AFTER_MIN:
            return Check(
                "backend_data", False,
                f"El último ciclo que ve el backend es de hace {edad_ciclo:.0f} min "
                f"({vuelos} vuelos). O los jobs se detuvieron, o el backend no "
                "refresca desde GCS.",
            )

        # El ciclo corre. Lo que falta saber es si TENÍA algo que hacer.
        #
        # La versión anterior miraba `total_flights`, los vuelos servidos en la
        # ventana activa, y los tomaba como trabajo pendiente. No lo son: son
        # vuelos ya predichos que todavía no aterrizaron. El 17/09 a las 05:08
        # eso disparó una alerta con el pipeline sano, porque a esa hora ATL no
        # tenía una sola salida por delante y aun así quedaban 185 en el aire.
        #
        # `last_run_targeted` es el número que el ciclo mismo calcula: cuántos
        # vuelos tenía para predecir. Con cero no hay nada que reprochar.
        objetivo = summary.get("last_run_targeted")
        predichos = summary.get("last_run_predicted")

        if objetivo is None:
            # Backend sin el campo todavía: se informa el ciclo y nada más, en
            # vez de emitir un juicio sobre datos que no están.
            return Check(
                "backend_data", True,
                f"Ciclo de hace {edad_ciclo:.0f} min ({vuelos} vuelos)",
            )

        if objetivo > 0 and not predichos:
            # Esto sí es una falla: el ciclo se encontró con vuelos por predecir
            # y no produjo ninguno. Es lo que pasaría si el modelo no cargara o
            # el armado de features se rompiera, y de otro modo es invisible.
            return Check(
                "backend_data", False,
                f"El último ciclo tenía {objetivo} vuelos para predecir y no "
                f"predijo ninguno (hace {edad_ciclo:.0f} min).",
            )

        if objetivo == 0:
            return Check(
                "backend_data", True,
                f"Ciclo de hace {edad_ciclo:.0f} min, sin salidas por delante "
                f"({vuelos} vuelos en el aire)",
            )

        return Check(
            "backend_data", True,
            f"Ciclo de hace {edad_ciclo:.0f} min, {predichos} de {objetivo} "
            f"vuelos predichos ({vuelos} en la ventana)",
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
    # Solo tiene sentido preguntar por las fuentes si el backend responde:
    # si esta caido, cada fuente daria un falso positivo y el aviso serian
    # cinco alertas en vez de una.
    if all(c.ok for c in checks):
        checks.extend(check_sources_fresh())
        checks.extend(check_capacity())
    state = load_state()
    # Se arranca del estado previo en vez de vacio. Una clave que no se pudo
    # evaluar en esta corrida —las fuentes cuando el backend esta caido— tiene
    # que conservar lo que sabiamos, no desaparecer: si desapareciera, al
    # volver el backend `state.get(key, True)` la daria por sana y volveria a
    # avisar de un problema que nunca dejo de estar.
    new_state = dict(state)
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

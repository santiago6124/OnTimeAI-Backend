"""Caracterización de los chequeos del vigía (`watchdog.py`) frente a sus fuentes.

CP-COB-03 del trabajo de Calidad (OnTimeAI-Calidad). `tests/test_watchdog.py`
cubre la lógica de transiciones de `main` y la interpretación de la frescura
del backend. Quedaban sin caracterizar los chequeos que hablan con el mundo:
la antigüedad del objeto en GCS, que el backend responda, el login, la
frescura por fuente, la capacidad (tamaño de la base, calibrador, duración
del ciclo), la persistencia del estado entre corridas y el envío a Telegram.

Cada dependencia externa se reemplaza por un doble en memoria: el `Client`
de `google.cloud.storage`, `requests.get`/`requests.post` y el reloj. Lo que
se verifica es el `Check` que produce cada función —clave, veredicto y
detalle—, que es lo que `main` convierte en alertas.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

# El módulo lee su configuración al importarse.
os.environ.setdefault("GCS_BUCKET", "test-bucket")
os.environ.setdefault("BACKEND_URL", "https://backend.test")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_CHAT_ID", "-100")
os.environ.setdefault("API_PASSWORD", "test-password")

from google.cloud import storage  # noqa: E402

import watchdog  # noqa: E402

AHORA = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)


# ── Dobles ─────────────────────────────────────────────────────────────────


class _Respuesta:
    def __init__(self, status: int = 200, cuerpo: dict | None = None, texto: str = "") -> None:
        self.status_code = status
        self._cuerpo = cuerpo if cuerpo is not None else {}
        self.text = texto

    def json(self) -> dict:
        return self._cuerpo


class _Blob:
    def __init__(self, *, updated: datetime | None = None, texto: str | None = None) -> None:
        self.updated = updated
        self._texto = texto
        self.subido: tuple | None = None

    def download_as_text(self) -> str:
        return self._texto or ""

    def upload_from_string(self, data: str, content_type: str) -> None:
        self.subido = (data, content_type)


class _ClienteGcs:
    """Doble de `storage.Client`: un bucket con los objetos que se le dan."""

    def __init__(self, objetos: dict | None = None, *, falla: Exception | None = None) -> None:
        self.objetos = objetos or {}
        self._falla = falla
        self.buckets: list[str] = []

    def bucket(self, nombre: str):
        self.buckets.append(nombre)
        if self._falla is not None:
            raise self._falla
        return SimpleNamespace(
            get_blob=lambda n: self.objetos.get(n),
            blob=lambda n: self.objetos.setdefault(n, _Blob()),
        )


@pytest.fixture
def gcs(monkeypatch):
    """Instala el doble y lo devuelve para inspeccionar qué se pidió."""
    cliente = _ClienteGcs()
    monkeypatch.setattr(storage, "Client", lambda: cliente)
    monkeypatch.setattr(watchdog, "_now", lambda: AHORA)
    return cliente


@pytest.fixture
def http(monkeypatch):
    """Reemplaza requests.get/post; el test fija qué devuelve cada uno."""
    registro = SimpleNamespace(gets=[], posts=[], get=None, post=None)

    def _get(url, **kwargs):
        registro.gets.append({"url": url, **kwargs})
        if isinstance(registro.get, Exception):
            raise registro.get
        return registro.get

    def _post(url, **kwargs):
        registro.posts.append({"url": url, **kwargs})
        if isinstance(registro.post, Exception):
            raise registro.post
        return registro.post

    monkeypatch.setattr(watchdog.requests, "get", _get)
    monkeypatch.setattr(watchdog.requests, "post", _post)
    return registro


# ── GCS ────────────────────────────────────────────────────────────────────


class TestFrescuraGcs:
    def test_una_base_recien_escrita_esta_sana(self, gcs) -> None:
        # Arrange: el objeto se actualizó hace 10 minutos.
        gcs.objetos["live_data.db"] = _Blob(updated=AHORA - timedelta(minutes=10))
        # Act
        check = watchdog.check_gcs_freshness()
        # Assert
        assert check.key == "gcs"
        assert check.ok is True
        assert "hace 10 min" in check.detail
        assert gcs.buckets == [watchdog.GCS_BUCKET]

    def test_una_base_que_nadie_escribe_dispara(self, gcs) -> None:
        # Arrange: tres ciclos perdidos y algo más.
        gcs.objetos["live_data.db"] = _Blob(
            updated=AHORA - timedelta(minutes=watchdog.STALE_AFTER_MIN + 75),
        )
        # Act
        check = watchdog.check_gcs_freshness()
        # Assert
        assert check.ok is False
        assert f"hace {watchdog.STALE_AFTER_MIN + 75} min" in check.detail

    def test_justo_en_el_umbral_todavia_no_dispara(self, gcs) -> None:
        gcs.objetos["live_data.db"] = _Blob(updated=AHORA - timedelta(minutes=watchdog.STALE_AFTER_MIN))
        assert watchdog.check_gcs_freshness().ok is True

    def test_si_el_objeto_no_existe_lo_dice_con_la_ruta(self, gcs) -> None:
        check = watchdog.check_gcs_freshness()
        assert check.ok is False
        assert check.detail == f"No existe gs://{watchdog.GCS_BUCKET}/{watchdog.DB_OBJECT}"

    def test_un_error_de_gcs_se_reporta_como_falla_sin_propagar(self, monkeypatch) -> None:
        monkeypatch.setattr(storage, "Client", lambda: _ClienteGcs(falla=PermissionError("403")))
        check = watchdog.check_gcs_freshness()
        assert check.ok is False
        assert "No se pudo leer GCS" in check.detail and "403" in check.detail


# ── Backend ────────────────────────────────────────────────────────────────


class TestBackendResponde:
    def test_un_200_en_docs_es_un_backend_vivo(self, http) -> None:
        # Arrange
        http.get = _Respuesta(200)
        # Act
        check = watchdog.check_backend_up()
        # Assert
        assert check == watchdog.Check("backend_up", True, "Backend respondiendo")
        assert http.gets[0]["url"] == f"{watchdog.BACKEND_URL}/docs"
        assert http.gets[0]["timeout"] == watchdog.HTTP_TIMEOUT

    def test_otro_codigo_es_una_falla_con_el_codigo_en_el_detalle(self, http) -> None:
        http.get = _Respuesta(503)
        check = watchdog.check_backend_up()
        assert check.ok is False and "503" in check.detail

    def test_sin_conexion_es_inalcanzable(self, http) -> None:
        http.get = ConnectionError("timeout")
        check = watchdog.check_backend_up()
        assert check.ok is False and "inalcanzable" in check.detail


class TestLogin:
    def test_devuelve_el_token_y_manda_las_credenciales_del_entorno(self, http) -> None:
        # Arrange
        http.post = _Respuesta(200, {"access_token": "jwt-de-prueba", "token_type": "bearer"})
        # Act
        token = watchdog._login()
        # Assert
        assert token == "jwt-de-prueba"
        assert http.posts[0]["url"] == f"{watchdog.BACKEND_URL}/auth/login"
        assert http.posts[0]["json"] == {
            "username": watchdog.API_USERNAME, "password": watchdog.API_PASSWORD,
        }

    def test_un_login_rechazado_lanza_con_el_codigo(self, http) -> None:
        http.post = _Respuesta(401)
        with pytest.raises(RuntimeError, match="401"):
            watchdog._login()


# ── Frescura por fuente ────────────────────────────────────────────────────


class TestFrescuraPorFuente:
    def _con_stats(self, monkeypatch, http, stats: dict) -> list[watchdog.Check]:
        monkeypatch.setattr(watchdog, "_login", lambda: "token")
        http.get = _Respuesta(200, stats)
        return watchdog.check_sources_fresh()

    def test_cada_tabla_es_su_propio_check_ordenado_por_nombre(self, monkeypatch, http) -> None:
        # Arrange: una fuente viva y una vacía.
        stats = {"sources": {
            "weather_obs": {"age_minutes": 12.4, "tolerated_minutes": 60,
                            "fed_by": "IEM", "stale": False},
            "aircraft_position": {"age_minutes": None, "tolerated_minutes": 60,
                                  "fed_by": "ADS-B", "stale": True},
        }}
        # Act
        checks = self._con_stats(monkeypatch, http, stats)
        # Assert
        assert [c.key for c in checks] == ["source:aircraft_position", "source:weather_obs"]
        assert checks[0].ok is False
        assert "esta vacia" in checks[0].detail and "ADS-B" in checks[0].detail
        assert checks[1].ok is True
        assert "hace 12 min" in checks[1].detail and "se toleran 60" in checks[1].detail
        assert http.gets[0]["headers"] == {"Authorization": "Bearer token"}
        assert http.gets[0]["url"].endswith("/admin/db-stats")

    def test_una_fuente_con_datos_viejos_se_marca_caida(self, monkeypatch, http) -> None:
        stats = {"sources": {"actuals": {"age_minutes": 240.0, "tolerated_minutes": 120,
                                         "fed_by": "FR24", "stale": True}}}
        [check] = self._con_stats(monkeypatch, http, stats)
        assert check.ok is False
        assert "hace 240 min" in check.detail

    def test_un_backend_viejo_sin_el_campo_no_produce_checks(self, monkeypatch, http) -> None:
        assert self._con_stats(monkeypatch, http, {}) == []
        assert self._con_stats(monkeypatch, http, {"sources": {}}) == []

    def test_si_no_se_puede_consultar_hay_un_solo_check_caido(self, monkeypatch, http) -> None:
        monkeypatch.setattr(watchdog, "_login", lambda: "token")
        http.get = ConnectionError("sin red")
        [check] = watchdog.check_sources_fresh()
        assert check.key == "sources" and check.ok is False
        assert "sin red" in check.detail

    def test_un_login_fallido_tambien_cae_en_ese_check(self, monkeypatch, http) -> None:
        def _sin_login():
            raise RuntimeError("login fallo con 401")

        monkeypatch.setattr(watchdog, "_login", _sin_login)
        [check] = watchdog.check_sources_fresh()
        assert check.ok is False and "401" in check.detail


# ── Capacidad ──────────────────────────────────────────────────────────────


class TestCapacidad:
    def _con_stats(self, monkeypatch, http, stats: dict) -> list[watchdog.Check]:
        monkeypatch.setattr(watchdog, "_login", lambda: "token")
        http.get = _Respuesta(200, stats)
        return watchdog.check_capacity()

    def test_sin_datos_no_hay_checks(self, monkeypatch, http) -> None:
        assert self._con_stats(monkeypatch, http, {}) == []

    def test_el_tamano_de_la_base_se_informa_contra_su_umbral(self, monkeypatch, http) -> None:
        # Arrange
        stats = {"db_size_mb": 650.4, "db_size_warn_mb": 800, "db_size_over_warn": False}
        # Act
        [check] = self._con_stats(monkeypatch, http, stats)
        # Assert
        assert check.key == "db_size" and check.ok is True
        assert check.detail == "La base pesa 650 MB (se avisa sobre 800)"

    def test_pasado_el_umbral_la_base_es_una_alerta(self, monkeypatch, http) -> None:
        stats = {"db_size_mb": 850.0, "db_size_warn_mb": 800, "db_size_over_warn": True}
        [check] = self._con_stats(monkeypatch, http, stats)
        assert check.ok is False and "850 MB" in check.detail

    def test_sin_umbral_configurado_no_se_juzga_el_tamano(self, monkeypatch, http) -> None:
        assert self._con_stats(monkeypatch, http, {"db_size_mb": 850.0}) == []

    def test_el_calibrador_post_cadena_tiene_su_propio_check(self, monkeypatch, http) -> None:
        stats = {"chain_calibrator": {"stale": False, "detail": "ajustado hace 12 min"}}
        [check] = self._con_stats(monkeypatch, http, stats)
        assert check.key == "chain_calibrator" and check.ok is True
        assert check.detail == "Calibrador post-cadena: ajustado hace 12 min"

    def test_un_calibrador_sin_veredicto_se_toma_como_viejo(self, monkeypatch, http) -> None:
        # Sin la clave `stale` se asume True: ante la duda, se avisa.
        [check] = self._con_stats(monkeypatch, http, {"chain_calibrator": {"detail": "?"}})
        assert check.ok is False

    def test_la_duracion_del_ciclo_se_compara_con_el_scheduler(self, monkeypatch, http) -> None:
        # Arrange: la mediana ya se acerca a la ventana de 15 min.
        stats = {"cycles": {"median_minutes": 11.0, "max_minutes": 14.25, "slow": True,
                            "tolerated_minutes": 10, "scheduler_minutes": 15}}
        # Act
        [check] = self._con_stats(monkeypatch, http, stats)
        # Assert
        assert check.key == "cycle_duration" and check.ok is False
        assert "11.0 min de mediana" in check.detail
        assert "maximo 14.2" in check.detail
        assert "se toleran 10" in check.detail and "de 15 min" in check.detail

    def test_sin_mediana_no_se_juzga_el_ciclo(self, monkeypatch, http) -> None:
        assert self._con_stats(monkeypatch, http, {"cycles": {"slow": True}}) == []

    def test_las_tres_senales_salen_juntas_y_en_orden(self, monkeypatch, http) -> None:
        stats = {
            "db_size_mb": 700.0, "db_size_warn_mb": 800, "db_size_over_warn": False,
            "chain_calibrator": {"stale": False, "detail": "ok"},
            "cycles": {"median_minutes": 4.0, "max_minutes": 6.0, "slow": False,
                       "tolerated_minutes": 10, "scheduler_minutes": 15},
        }
        checks = self._con_stats(monkeypatch, http, stats)
        assert [c.key for c in checks] == ["db_size", "chain_calibrator", "cycle_duration"]
        assert all(c.ok for c in checks)

    def test_si_no_se_puede_consultar_hay_un_solo_check_caido(self, monkeypatch, http) -> None:
        monkeypatch.setattr(watchdog, "_login", lambda: "token")
        http.get = ConnectionError("sin red")
        [check] = watchdog.check_capacity()
        assert check.key == "capacity" and check.ok is False
        assert "/admin/db-stats" in check.detail


class TestFrescuraDelBackendAnteErrores:
    def test_un_login_fallido_se_reporta_sin_propagar(self, monkeypatch) -> None:
        def _sin_login():
            raise RuntimeError("login fallo con 401")

        monkeypatch.setattr(watchdog, "_login", _sin_login)
        check = watchdog.check_backend_data_fresh()
        assert check.key == "backend_data" and check.ok is False
        assert "No se pudo verificar" in check.detail and "401" in check.detail


# ── Estado entre corridas ──────────────────────────────────────────────────


class TestEstado:
    def test_sin_estado_previo_se_arranca_de_vacio(self, gcs) -> None:
        assert watchdog.load_state() == {}

    def test_el_estado_guardado_se_lee_como_json(self, gcs) -> None:
        # Arrange
        gcs.objetos[watchdog.STATE_OBJECT] = _Blob(texto=json.dumps({"gcs": True, "backend_up": False}))
        # Act
        estado = watchdog.load_state()
        # Assert
        assert estado == {"gcs": True, "backend_up": False}

    def test_un_estado_ilegible_se_toma_como_vacio(self, gcs) -> None:
        # JSON roto: a lo sumo se repite un aviso, nunca se cae el vigía.
        gcs.objetos[watchdog.STATE_OBJECT] = _Blob(texto="{no es json")
        assert watchdog.load_state() == {}

    def test_sin_acceso_a_gcs_tambien_se_toma_como_vacio(self, monkeypatch) -> None:
        monkeypatch.setattr(storage, "Client", lambda: _ClienteGcs(falla=PermissionError("403")))
        assert watchdog.load_state() == {}

    def test_guardar_sube_el_json_al_objeto_de_estado(self, gcs) -> None:
        # Act
        watchdog.save_state({"gcs": True, "source:actuals": False})
        # Assert
        blob = gcs.objetos[watchdog.STATE_OBJECT]
        assert blob.subido is not None
        datos, tipo = blob.subido
        assert json.loads(datos) == {"gcs": True, "source:actuals": False}
        assert tipo == "application/json"

    def test_si_no_se_puede_guardar_se_deja_en_el_log_y_sigue(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(storage, "Client", lambda: _ClienteGcs(falla=PermissionError("403")))
        watchdog.save_state({"gcs": True})
        assert "no se pudo guardar el estado" in capsys.readouterr().out


# ── Telegram ───────────────────────────────────────────────────────────────


class TestNotify:
    def test_manda_el_mensaje_al_chat_configurado_en_markdown(self, http, capsys) -> None:
        # Arrange
        http.post = _Respuesta(200, {"ok": True})
        # Act
        watchdog.notify("*OnTimeAI*\n\nprueba")
        # Assert
        [llamada] = http.posts
        assert llamada["url"] == f"https://api.telegram.org/bot{watchdog.TELEGRAM_TOKEN}/sendMessage"
        assert llamada["json"]["chat_id"] == watchdog.TELEGRAM_CHAT_ID
        assert llamada["json"]["text"] == "*OnTimeAI*\n\nprueba"
        assert llamada["json"]["parse_mode"] == "Markdown"
        assert llamada["json"]["disable_web_page_preview"] is True
        assert capsys.readouterr().out == ""

    def test_un_rechazo_de_telegram_queda_en_el_log(self, http, capsys) -> None:
        http.post = _Respuesta(400, texto='{"description":"Bad Request: chat not found"}')
        watchdog.notify("hola")
        salida = capsys.readouterr().out
        assert "Telegram devolvió 400" in salida and "chat not found" in salida

    def test_sin_red_no_se_propaga(self, http, capsys) -> None:
        http.post = ConnectionError("sin red")
        watchdog.notify("hola")
        assert "no se pudo notificar" in capsys.readouterr().out

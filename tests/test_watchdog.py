"""
Tests del vigía.

El foco está en la lógica de transiciones: es la parte que decide si el aviso
llega o no, y la que puede degradar en silencio. Un vigía que avisa en cada
ciclo se vuelve ruido que la gente silencia; uno que no avisa nunca es lo que
ya tuvimos.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

# El módulo lee su configuración al importarse.
os.environ.setdefault("GCS_BUCKET", "test-bucket")
os.environ.setdefault("BACKEND_URL", "https://backend.test")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
os.environ.setdefault("TELEGRAM_CHAT_ID", "-100")
os.environ.setdefault("API_PASSWORD", "test-password")

import watchdog  # noqa: E402


def _run_with(checks, previous_state, monkeypatch, sources=None, capacity=None):
    """Corre main() con chequeos fijos y devuelve (mensajes, estado guardado)."""
    sent: list[str] = []
    saved: dict = {}

    monkeypatch.setattr(watchdog, "check_gcs_freshness", lambda: checks[0])
    monkeypatch.setattr(watchdog, "check_backend_up", lambda: checks[1])
    monkeypatch.setattr(watchdog, "check_backend_data_fresh", lambda: checks[2])
    monkeypatch.setattr(watchdog, "check_sources_fresh", lambda: list(sources or []))
    monkeypatch.setattr(watchdog, "check_capacity", lambda: list(capacity or []))
    monkeypatch.setattr(watchdog, "load_state", lambda: previous_state)
    monkeypatch.setattr(watchdog, "save_state", lambda s: saved.update(s))
    monkeypatch.setattr(watchdog, "notify", lambda text: sent.append(text))

    assert watchdog.main() == 0
    return sent, saved


def _ok(key: str) -> watchdog.Check:
    return watchdog.Check(key, True, "todo bien")


def _fail(key: str, detail: str = "se rompio") -> watchdog.Check:
    return watchdog.Check(key, False, detail)


ALL_OK = [_ok("gcs"), _ok("backend_up"), _ok("backend_data")]


def test_no_avisa_cuando_todo_esta_bien(monkeypatch) -> None:
    sent, _ = _run_with(ALL_OK, {"gcs": True, "backend_up": True, "backend_data": True}, monkeypatch)
    assert sent == []


def test_avisa_la_primera_vez_que_algo_falla(monkeypatch) -> None:
    checks = [_ok("gcs"), _ok("backend_up"), _fail("backend_data", "datos de hace 2880 min")]
    sent, _ = _run_with(checks, {"backend_data": True}, monkeypatch)

    assert len(sent) == 1
    assert "backend_data" in sent[0]
    assert "2880" in sent[0]


def test_no_repite_el_aviso_mientras_el_problema_persiste(monkeypatch) -> None:
    """El caso que vuelve inútil a un sistema de alertas: avisar cada 10 min."""
    checks = [_ok("gcs"), _ok("backend_up"), _fail("backend_data")]
    sent, _ = _run_with(checks, {"gcs": True, "backend_up": True, "backend_data": False}, monkeypatch)
    assert sent == []


def test_avisa_cuando_se_recupera(monkeypatch) -> None:
    sent, _ = _run_with(ALL_OK, {"gcs": True, "backend_up": True, "backend_data": False}, monkeypatch)

    assert len(sent) == 1
    assert "recuperado" in sent[0]
    assert "backend_data" in sent[0]


def test_agrupa_varios_problemas_en_un_solo_mensaje(monkeypatch) -> None:
    checks = [_fail("gcs"), _fail("backend_up"), _ok("backend_data")]
    sent, _ = _run_with(checks, {"gcs": True, "backend_up": True, "backend_data": True}, monkeypatch)

    assert len(sent) == 1
    assert "gcs" in sent[0] and "backend_up" in sent[0]


def test_sin_estado_previo_asume_que_estaba_sano(monkeypatch) -> None:
    """
    Primera corrida, o estado ilegible. Asumir "sano" hace que un problema
    existente se reporte; asumir "roto" lo silenciaría.
    """
    checks = [_ok("gcs"), _ok("backend_up"), _fail("backend_data")]
    sent, _ = _run_with(checks, {}, monkeypatch)
    assert len(sent) == 1


def test_guarda_el_estado_de_los_tres_chequeos(monkeypatch) -> None:
    checks = [_ok("gcs"), _fail("backend_up"), _ok("backend_data")]
    _, saved = _run_with(checks, {}, monkeypatch)
    assert saved == {"gcs": True, "backend_up": False, "backend_data": True}


class TestParseUtc:
    def test_acepta_el_formato_que_emite_el_backend(self) -> None:
        # El backend serializa con offset, no con sufijo Z.
        parsed = watchdog._parse_utc("2026-09-14T02:19:55.289030+00:00")
        assert parsed.tzinfo is not None
        assert parsed.year == 2026 and parsed.hour == 2

    def test_acepta_sufijo_z(self) -> None:
        assert watchdog._parse_utc("2026-09-14T02:19:55Z").tzinfo is not None

    def test_asume_utc_cuando_no_hay_zona(self) -> None:
        # Sin esto, una fecha sin zona se leería como hora local y la
        # antigüedad daría 3 horas de más en Argentina.
        assert watchdog._parse_utc("2026-09-14T02:19:55").tzinfo == timezone.utc


class TestMinutesSince:
    def test_mide_la_antiguedad_en_minutos(self, monkeypatch) -> None:
        ahora = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(watchdog, "_now", lambda: ahora)
        assert watchdog._minutes_since(ahora - timedelta(minutes=45)) == pytest.approx(45)

    def test_el_umbral_por_defecto_tolera_tres_ciclos(self) -> None:
        # Los predictores corren cada 15 min: el umbral tiene que dejar pasar
        # un ciclo lento sin gritar, pero detectar una caída real.
        assert watchdog.STALE_AFTER_MIN >= 30


class TestEnvStrip:
    """
    Secret Manager inyecta los bytes crudos del secreto. Un valor guardado con
    `print()` o desde un archivo arrastra un salto de linea, y ese caracter
    llega al destino. La primera corrida del vigia en produccion fallo con 401
    por exactamente esto.
    """

    def test_saca_el_salto_de_linea_final(self, monkeypatch) -> None:
        monkeypatch.setenv("UNA_VAR", "secreto\n")
        assert watchdog._env("UNA_VAR") == "secreto"

    def test_saca_espacios_de_ambos_extremos(self, monkeypatch) -> None:
        monkeypatch.setenv("UNA_VAR", "  secreto \n")
        assert watchdog._env("UNA_VAR") == "secreto"

    def test_tambien_limpia_los_valores_por_defecto(self, monkeypatch) -> None:
        monkeypatch.delenv("OTRA_VAR", raising=False)
        assert watchdog._env("OTRA_VAR", " fallback\n") == "fallback"

    def test_una_variable_obligatoria_que_falta_corta_el_arranque(self, monkeypatch) -> None:
        monkeypatch.delenv("FALTANTE", raising=False)
        with pytest.raises(KeyError):
            watchdog._env("FALTANTE")


class TestFrescuraPorFuente:
    """
    Una fuente que deja de escribir no rompe nada: el pipeline sigue, las
    corridas figuran exitosas y la senal desaparece sin ruido. ADS-B estuvo
    25 dias inerte asi. Ver #9.
    """

    def test_una_fuente_caida_avisa_por_su_cuenta(self, monkeypatch) -> None:
        sources = [_ok("source:weather_obs"), _fail("source:aircraft_position", "vacia")]
        sent, saved = _run_with(ALL_OK, {}, monkeypatch, sources=sources)

        assert len(sent) == 1
        assert "aircraft_position" in sent[0]
        assert saved["source:aircraft_position"] is False
        assert saved["source:weather_obs"] is True

    def test_cada_fuente_es_su_propia_alerta(self, monkeypatch) -> None:
        """Una caida no puede quedar tapada por el resto funcionando."""
        sources = [_fail("source:weather_obs"), _fail("source:nas_status")]
        _, saved = _run_with(ALL_OK, {}, monkeypatch, sources=sources)
        assert saved["source:weather_obs"] is False
        assert saved["source:nas_status"] is False

    def test_no_repite_mientras_la_fuente_sigue_caida(self, monkeypatch) -> None:
        sources = [_fail("source:aircraft_position")]
        sent, _ = _run_with(
            ALL_OK, {"source:aircraft_position": False}, monkeypatch, sources=sources
        )
        assert sent == []

    def test_no_pregunta_por_las_fuentes_si_el_backend_esta_caido(self, monkeypatch) -> None:
        """
        Con el backend abajo cada fuente daria un falso positivo, y el aviso
        serian cinco alertas en vez de una.
        """
        llamadas = []

        def _no_deberia_llamarse():
            llamadas.append(1)
            return []

        monkeypatch.setattr(watchdog, "check_sources_fresh", _no_deberia_llamarse)
        monkeypatch.setattr(watchdog, "check_gcs_freshness", lambda: _ok("gcs"))
        monkeypatch.setattr(watchdog, "check_backend_up", lambda: _fail("backend_up"))
        monkeypatch.setattr(watchdog, "check_backend_data_fresh", lambda: _ok("backend_data"))
        monkeypatch.setattr(watchdog, "load_state", lambda: {})
        monkeypatch.setattr(watchdog, "save_state", lambda s: None)
        monkeypatch.setattr(watchdog, "notify", lambda t: None)

        assert watchdog.main() == 0
        assert llamadas == []

    def test_una_fuente_no_evaluada_conserva_su_estado(self, monkeypatch) -> None:
        """
        El backend cae mientras una fuente ya estaba rota. Si su clave
        desapareciera del estado, al volver el backend `state.get(key, True)`
        la daria por sana y volveria a avisar de algo que nunca se arreglo.
        """
        previo = {"gcs": True, "backend_up": True, "backend_data": True,
                  "source:aircraft_position": False}
        _, saved = _run_with(ALL_OK, previo, monkeypatch, sources=[])
        assert saved["source:aircraft_position"] is False


class TestCapacidad:
    """
    Tamano de la base y duracion del ciclo: senales adelantadas, no caidas.

    El job baja la base entera de GCS y la vuelve a subir en cada ciclo, asi
    que su duracion crece con el archivo. Los predictores corren cada 15 min;
    cuando el ciclo se acerca a esa ventana, dos jobs terminan escribiendo la
    misma base. Ver issue #4.
    """

    def test_avisa_cuando_la_base_pasa_el_umbral(self, monkeypatch) -> None:
        cap = [_fail("db_size", "La base pesa 850 MB (se avisa sobre 800)")]
        sent, saved = _run_with(ALL_OK, {}, monkeypatch, capacity=cap)
        assert len(sent) == 1 and "850 MB" in sent[0]
        assert saved["db_size"] is False

    def test_avisa_cuando_el_ciclo_se_acerca_al_scheduler(self, monkeypatch) -> None:
        cap = [_fail("cycle_duration", "El ciclo tarda 11.0 min de mediana")]
        sent, _ = _run_with(ALL_OK, {}, monkeypatch, capacity=cap)
        assert len(sent) == 1 and "cycle_duration" in sent[0]

    def test_tamano_y_duracion_son_alertas_distintas(self, monkeypatch) -> None:
        # La base puede crecer sin que el ciclo sufra todavia, y al reves.
        cap = [_fail("db_size"), _ok("cycle_duration")]
        _, saved = _run_with(ALL_OK, {}, monkeypatch, capacity=cap)
        assert saved["db_size"] is False
        assert saved["cycle_duration"] is True

    def test_no_pregunta_si_el_backend_esta_caido(self, monkeypatch) -> None:
        llamadas = []
        monkeypatch.setattr(watchdog, "check_capacity", lambda: llamadas.append(1) or [])
        monkeypatch.setattr(watchdog, "check_sources_fresh", lambda: [])
        monkeypatch.setattr(watchdog, "check_gcs_freshness", lambda: _ok("gcs"))
        monkeypatch.setattr(watchdog, "check_backend_up", lambda: _fail("backend_up"))
        monkeypatch.setattr(watchdog, "check_backend_data_fresh", lambda: _ok("backend_data"))
        monkeypatch.setattr(watchdog, "load_state", lambda: {})
        monkeypatch.setattr(watchdog, "save_state", lambda s: None)
        monkeypatch.setattr(watchdog, "notify", lambda t: None)

        assert watchdog.main() == 0
        assert llamadas == []


# ── Frescura: distinguir "el pipeline se detuvo" de "no habia nada que hacer" ──


class TestFrescuraDelBackend:
    """
    El 16/09 a las 04:38 UTC sono una alerta de datos viejos con los dos jobs
    sanos: 80 corridas seguidas sin fallar. A esa hora ATL —00:38 local— no
    tenia una sola salida programada en la ventana, asi que el ciclo corrio,
    no encontro nada que predecir, y `last_tick_utc` se quedo clavado.

    Un monitor que grita en falso cada madrugada se empieza a ignorar, y asi es
    como se pasa por alto el incidente de verdad: el del 13/09, donde el backend
    devolvia 200 con un snapshot de dos dias atras.
    """

    def _con_resumen(self, monkeypatch, resumen):
        import watchdog

        monkeypatch.setattr(watchdog, "_login", lambda: "token")

        class _Respuesta:
            @staticmethod
            def json():
                return resumen

        monkeypatch.setattr(watchdog.requests, "get", lambda *a, **k: _Respuesta())
        return watchdog.check_backend_data_fresh()

    def _hace(self, minutos):
        from datetime import datetime, timedelta, timezone

        return (datetime.now(timezone.utc) - timedelta(minutes=minutos)).isoformat()

    def test_la_madrugada_de_atl_ya_no_dispara(self, monkeypatch) -> None:
        # El caso exacto del 16/09: ciclo reciente, tick viejo, sin vuelos.
        check = self._con_resumen(monkeypatch, {
            "last_run_utc": self._hace(6),
            "last_tick_utc": self._hace(49),
            "total_flights": 0,
            "last_run_targeted": 0, "last_run_predicted": 0,
        })
        assert check.ok is True

    def test_la_madrugada_CON_vuelos_en_el_aire_tampoco_dispara(self, monkeypatch) -> None:
        """
        El caso del 17/09 a las 05:08, que la version anterior de este chequeo
        marcaba como falla: el ciclo corria hace 7 min, la ultima prediccion era
        de hace 49, y habia 185 vuelos servidos.

        Esos 185 no son trabajo pendiente: son vuelos ya predichos que todavia
        no aterrizaron. El ciclo no tenia ni una salida por delante.
        """
        check = self._con_resumen(monkeypatch, {
            "last_run_utc": self._hace(7),
            "last_tick_utc": self._hace(49),
            "total_flights": 185,
            "last_run_targeted": 0, "last_run_predicted": 0,
        })
        assert check.ok is True
        assert "sin salidas por delante" in check.detail

    def test_habia_vuelos_y_no_predijo_ninguno_es_falla(self, monkeypatch) -> None:
        """
        Lo que este chequeo existe para atrapar: el ciclo se encontro con vuelos
        por predecir y no produjo ninguno. Pasaria si el modelo no cargara o el
        armado de features se rompiera, y de otro modo es invisible.
        """
        check = self._con_resumen(monkeypatch, {
            "last_run_utc": self._hace(5),
            "last_tick_utc": self._hace(120),
            "total_flights": 185,
            "last_run_targeted": 42, "last_run_predicted": 0,
        })
        assert check.ok is False
        assert "42" in check.detail

    def test_operacion_normal_con_vuelos_predichos(self, monkeypatch) -> None:
        check = self._con_resumen(monkeypatch, {
            "last_run_utc": self._hace(5),
            "last_tick_utc": self._hace(6),
            "total_flights": 150,
            "last_run_targeted": 42, "last_run_predicted": 42,
        })
        assert check.ok is True
        assert "42 de 42" in check.detail

    def test_sin_el_campo_nuevo_no_inventa_un_juicio(self, monkeypatch) -> None:
        """Watchdog desplegado antes que la API: informa y no acusa."""
        check = self._con_resumen(monkeypatch, {
            "last_run_utc": self._hace(5),
            "last_tick_utc": self._hace(120),
            "total_flights": 185,
        })
        assert check.ok is True

    def test_el_pipeline_detenido_sigue_disparando(self, monkeypatch) -> None:
        """Lo que el chequeo existe para atrapar tiene que seguir atrapandose."""
        check = self._con_resumen(monkeypatch, {
            "last_run_utc": self._hace(2880),
            "last_tick_utc": self._hace(2880),
            "total_flights": 0,
            "last_run_targeted": 0, "last_run_predicted": 0,
        })
        assert check.ok is False
        assert "ciclo" in check.detail.lower()

    def test_un_backend_viejo_cae_al_criterio_anterior(self, monkeypatch) -> None:
        """
        Sin `last_run_utc` —backend sin desplegar todavia— el chequeo no puede
        quedarse mudo: vuelve a mirar el tick.
        """
        viejo = self._con_resumen(monkeypatch, {
            "last_tick_utc": self._hace(2880), "total_flights": 0,
        })
        assert viejo.ok is False

        sano = self._con_resumen(monkeypatch, {
            "last_tick_utc": self._hace(10), "total_flights": 100,
        })
        assert sano.ok is True

    def test_sin_ningun_dato_reporta_falla(self, monkeypatch) -> None:
        assert self._con_resumen(monkeypatch, {}).ok is False


def test_predictions_no_se_vigila_como_fuente_externa() -> None:
    """
    `predictions` no es una fuente: es nuestra propia salida, y solo se produce
    cuando hay vuelos que predecir.

    Atlanta no opera salidas entre las 23:00 y las 05:00 locales. Verificado
    contra un tablero publico independiente y contra la guia del aeropuerto,
    cuyos controles de seguridad abren recien a las 3:30. Asi que cada madrugada
    pasaban mas de cinco horas sin escribir nada, y con una tolerancia de 60
    minutos la alarma sonaba sin que pasara nada.

    Lo que importa lo cubre el chequeo de frescura del backend, que pregunta si
    el ciclo tenia vuelos y si los predijo.
    """
    import api

    assert "predictions" not in api.SOURCE_FRESHNESS
    # Las externas se quedan: esas si llegan solas y su silencio es una falla.
    for externa in ("actuals", "weather_obs", "nas_status", "aircraft_position"):
        assert externa in api.SOURCE_FRESHNESS

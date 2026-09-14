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


def _run_with(checks, previous_state, monkeypatch):
    """Corre main() con chequeos fijos y devuelve (mensajes, estado guardado)."""
    sent: list[str] = []
    saved: dict = {}

    monkeypatch.setattr(watchdog, "check_gcs_freshness", lambda: checks[0])
    monkeypatch.setattr(watchdog, "check_backend_up", lambda: checks[1])
    monkeypatch.setattr(watchdog, "check_backend_data_fresh", lambda: checks[2])
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

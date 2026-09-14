"""
Tests del cache de METAR entre reintentos.

Ante un conflicto CAS, live_job rehace el pipeline completo dentro del mismo
proceso. Eso es necesario por correctitud —las predicciones se recalculan sobre
la generacion ganadora— pero volver a pedir el clima de ~130 aeropuertos no lo
es: el METAR se publica cada ~1 hora.

Ese trabajo repetido alarga el ciclo, y ciclos mas largos generan mas
conflictos, que generan mas trabajo repetido.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ontimeai import live


@pytest.fixture(autouse=True)
def _clear_cache():
    live._IEM_CACHE = None
    yield
    live._IEM_CACHE = None


def _fake_response(monkeypatch, counter: list[int]):
    """Sustituye la llamada HTTP a IEM por una respuesta valida y contable."""

    class _Resp:
        status_code = 200
        text = (
            "station,valid,tmpc,dwpc,relh,drct,sknt,alti,p01m,vsby,gust,wxcodes\n"
            "ATL,2026-09-14 12:00,20.0,10.0,50.0,180,5.0,30.0,0.0,10.0,M,M\n"
        )

        def raise_for_status(self):
            return None

    def _get(*_args, **_kwargs):
        counter[0] += 1
        return _Resp()

    monkeypatch.setattr(live.requests, "get", _get)
    monkeypatch.setattr(live.time, "sleep", lambda _s: None)


def test_reutiliza_el_fetch_para_el_mismo_conjunto(monkeypatch) -> None:
    calls = [0]
    _fake_response(monkeypatch, calls)
    start = pd.Timestamp("2026-09-14T10:00:00")
    end = pd.Timestamp("2026-09-14T14:00:00")

    primero = live.fetch_iem_obs({"ATL"}, start, end)
    tras_primero = calls[0]
    segundo = live.fetch_iem_obs({"ATL"}, start, end)

    assert calls[0] == tras_primero, "el segundo pedido no deberia salir a la red"
    assert segundo.equals(primero)


def test_no_devuelve_el_mismo_objeto_que_cachea(monkeypatch) -> None:
    """
    Quien recibe el DataFrame lo modifica aguas abajo. Si se devolviera el
    objeto cacheado, la segunda llamada entregaria datos ya mutados.
    """
    calls = [0]
    _fake_response(monkeypatch, calls)
    start = pd.Timestamp("2026-09-14T10:00:00")
    end = pd.Timestamp("2026-09-14T14:00:00")

    primero = live.fetch_iem_obs({"ATL"}, start, end)
    primero.loc[:, "tmpc"] = 99.0

    segundo = live.fetch_iem_obs({"ATL"}, start, end)
    assert segundo.loc[0, "tmpc"] == 20.0, "el cache quedo contaminado"


def test_vuelve_a_pedir_si_cambia_el_conjunto_de_estaciones(monkeypatch) -> None:
    calls = [0]
    _fake_response(monkeypatch, calls)
    start = pd.Timestamp("2026-09-14T10:00:00")
    end = pd.Timestamp("2026-09-14T14:00:00")

    live.fetch_iem_obs({"ATL"}, start, end)
    tras_primero = calls[0]
    live.fetch_iem_obs({"ATL", "MIA"}, start, end)

    assert calls[0] > tras_primero, "un conjunto distinto exige un pedido nuevo"


def test_vuelve_a_pedir_cuando_vence_el_ttl(monkeypatch) -> None:
    calls = [0]
    _fake_response(monkeypatch, calls)
    start = pd.Timestamp("2026-09-14T10:00:00")
    end = pd.Timestamp("2026-09-14T14:00:00")

    live.fetch_iem_obs({"ATL"}, start, end)
    tras_primero = calls[0]

    # El cache guarda time.monotonic(); se simula el paso del TTL.
    key, cached_at, df = live._IEM_CACHE
    live._IEM_CACHE = (key, cached_at - live._IEM_CACHE_TTL_S - 1, df)

    live.fetch_iem_obs({"ATL"}, start, end)
    assert calls[0] > tras_primero, "pasado el TTL hay que volver a pedir"

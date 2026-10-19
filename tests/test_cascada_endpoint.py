"""`GET /cascade/{id}`: a qué vuelos siguientes les llega la demora.

Lo que estos tests protegen son las tres cosas que el endpoint puede decir mal
sin que se note:

  1. de dónde salen los minutos. La demora real de un vuelo aterrizado y una
     derivada de la probabilidad no valen lo mismo, y la respuesta tiene que
     decir cuál es.
  2. dónde corta la cadena. Si el origen de una pierna no sigue al destino
     anterior, el avión hizo algo que no vemos y propagar es inventar.
  3. qué pasa cuando no se puede responder. Devolver una lista vacía sin motivo
     se lee como "no hay cascada", que es distinto de "no sabemos".
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api  # noqa: E402
from ontimeai.live import SCHEMA  # noqa: E402

AHORA = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)


def _iso(min_desde_ahora: float) -> str:
    return (AHORA + timedelta(minutes=min_desde_ahora)).isoformat()


@pytest.fixture
def con(monkeypatch) -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    monkeypatch.setattr(api, "get_db", lambda: c)
    return c


def _vuelo(con, fid, *, tail="N123AB", origen="ATL", destino="MIA",
           sale=0, llega=120, arr_delay=None, proba=None, nro="100"):
    con.execute(
        "INSERT INTO flights (fa_flight_id, stable_id, op_carrier, flight_number,"
        " tail_num, origin, dest, scheduled_out_utc, scheduled_in_utc,"
        " first_seen_utc, last_updated_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (fid, fid, "DL", nro, tail, origen, destino, _iso(sale), _iso(llega),
         _iso(sale - 600), _iso(sale - 600)))
    if arr_delay is not None:
        con.execute(
            "INSERT INTO actuals (fa_flight_id, stable_id, arr_delay_min,"
            " cancelled, diverted, settled_at_utc) VALUES (?,?,?,?,?,?)",
            (fid, fid, arr_delay, 0, 0, _iso(llega)))
    if proba is not None:
        con.execute(
            "INSERT INTO predictions (fa_flight_id, stable_id, predicted_at_utc,"
            " proba_delay, predicted_delay) VALUES (?,?,?,?,?)",
            (fid, fid, _iso(sale - 60), proba, 1 if proba > 0.2 else 0))
    con.commit()


def _rotacion(con, tail="N123AB"):
    """Un vuelo demorado y dos piernas siguientes con turnaround ajustado."""
    _vuelo(con, "A", tail=tail, origen="ATL", destino="MIA", sale=0, llega=120,
           arr_delay=45, nro="100")
    _vuelo(con, "B", tail=tail, origen="MIA", destino="ATL", sale=180, llega=300,
           nro="200")
    _vuelo(con, "C", tail=tail, origen="ATL", destino="JFK", sale=360, llega=480,
           nro="300")


# ── de dónde salen los minutos ─────────────────────────────────────────────

def test_usa_la_demora_real_cuando_el_vuelo_ya_aterrizo(con) -> None:
    _rotacion(con)
    r = api.cascade_de_vuelo("A")
    assert r["demora_origen"] == "real"
    assert r["demora_llegada_min"] == 45.0
    assert r["aviso"] is None


def test_la_demora_pasada_a_mano_gana_sobre_la_real(con) -> None:
    """Es el modo de simulación que pide el anteproyecto: «dado un vuelo
    retrasado, propagar el impacto». Tiene que poder contradecir lo que pasó."""
    _rotacion(con)
    r = api.cascade_de_vuelo("A", demora_min=200)
    assert r["demora_origen"] == "simulada"
    assert r["demora_llegada_min"] == 200.0


def test_sin_demora_real_la_deriva_de_la_probabilidad_y_avisa(con) -> None:
    """La tabla que convierte probabilidad en minutos tiene pocos casos en la
    banda alta, que es justo la que propaga. Si la respuesta no lo dijera, ese
    número se leería igual que una demora medida."""
    _vuelo(con, "A", origen="ATL", destino="MIA", sale=0, llega=120, proba=0.5)
    _vuelo(con, "B", origen="MIA", destino="ATL", sale=180, llega=300)
    r = api.cascade_de_vuelo("A")
    assert r["demora_origen"] == "estimada"
    assert r["aviso"] and "probabilidad" in r["aviso"]


def test_sin_demora_ni_prediccion_dice_por_que(con) -> None:
    _vuelo(con, "A", origen="ATL", destino="MIA", sale=0, llega=120)
    _vuelo(con, "B", origen="MIA", destino="ATL", sale=180, llega=300)
    r = api.cascade_de_vuelo("A")
    assert r["piernas"] == []
    assert "demora_min" in r["motivo"], "tiene que decir que se puede simular"


# ── la cadena ──────────────────────────────────────────────────────────────

def test_propaga_por_la_rotacion_en_orden(con) -> None:
    _rotacion(con)
    r = api.cascade_de_vuelo("A")
    assert [x["orden"] for x in r["piernas"]] == [1, 2]
    assert [x["fa_flight_id"] for x in r["piernas"]] == ["B", "C"]
    assert all(x["demorado"] for x in r["piernas"])


def test_la_holgura_se_mide_contra_la_llegada_programada(con) -> None:
    """Contra la llegada del vuelo anterior, no contra su salida. Medirla mal
    da holguras de horas donde hay minutos, y la tabla devuelve que no propaga
    nada."""
    _rotacion(con)
    r = api.cascade_de_vuelo("A")
    assert r["piernas"][0]["holgura_min"] == 60.0   # sale 180, llega 120
    assert r["piernas"][1]["holgura_min"] == 60.0   # sale 360, llega 300


def test_corta_donde_el_origen_no_sigue_al_destino(con) -> None:
    """Si la siguiente pierna sale de otro aeropuerto, el avión hizo algo que
    no capturamos —una pierna que falta o un cambio de aeronave— y seguir
    propagando sería inventar."""
    _vuelo(con, "A", origen="ATL", destino="MIA", sale=0, llega=120, arr_delay=45)
    _vuelo(con, "B", origen="MCO", destino="ATL", sale=180, llega=300)  # no sale de MIA
    r = api.cascade_de_vuelo("A")
    assert r["piernas"] == []
    assert "próxima pierna" in r["motivo"]


def test_una_pierna_anterior_no_entra_en_la_cadena(con) -> None:
    """Sólo las que salen DESPUÉS de que este vuelo aterriza."""
    _rotacion(con)
    _vuelo(con, "Z", origen="JFK", destino="ATL", sale=-600, llega=-480, nro="900")
    r = api.cascade_de_vuelo("A")
    assert "Z" not in [x["fa_flight_id"] for x in r["piernas"]]


def test_respeta_el_maximo_de_piernas(con) -> None:
    _rotacion(con)
    r = api.cascade_de_vuelo("A", max_piernas=1)
    assert len(r["piernas"]) == 1


# ── cuando no se puede responder ───────────────────────────────────────────

def test_un_vuelo_sin_matricula_lo_dice(con) -> None:
    """Sin matrícula no hay rotación que seguir. Devolver lista vacía sin
    motivo se leería como 'no hay cascada'."""
    con.execute(
        "INSERT INTO flights (fa_flight_id, stable_id, origin, dest,"
        " scheduled_out_utc, scheduled_in_utc, first_seen_utc, last_updated_utc)"
        " VALUES ('A','A','ATL','MIA',?,?,?,?)",
        (_iso(0), _iso(120), _iso(-600), _iso(-600)))
    con.commit()
    r = api.cascade_de_vuelo("A")
    assert r["piernas"] == []
    assert "matrícula" in r["motivo"]


def test_un_vuelo_que_no_existe_da_404(con) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as e:
        api.cascade_de_vuelo("no-existe")
    assert e.value.status_code == 404


def test_sin_piernas_siguientes_lo_dice(con) -> None:
    """El harvester descubre las rotaciones caminando aviones y no siempre
    llega antes de que el vuelo salga: medido, en el 66% de los casos."""
    _vuelo(con, "A", origen="ATL", destino="MIA", sale=0, llega=120, arr_delay=45)
    r = api.cascade_de_vuelo("A")
    assert r["piernas"] == []
    assert r["piernas_conocidas"] == 0
    assert "harvester" in r["motivo"]


# ── la respuesta ───────────────────────────────────────────────────────────

def test_la_respuesta_avisa_que_es_una_proyeccion(con) -> None:
    """18 minutos de error absoluto medio por paso. Sin la nota, los minutos se
    leen como una hora prometida."""
    _rotacion(con)
    r = api.cascade_de_vuelo("A")
    assert "proyección" in r["nota"]
    assert "gate-out" in r["nota"], "la semántica de la demora de salida importa"


def test_marca_la_cadena_rota_con_demoras_enormes(con) -> None:
    _rotacion(con)
    r = api.cascade_de_vuelo("A", demora_min=300)
    assert r["piernas"][0]["cadena_rota"] is True

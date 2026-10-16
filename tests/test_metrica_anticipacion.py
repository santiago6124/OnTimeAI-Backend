"""Cuánto antes avisa el modelo que la aerolínea.

Es el argumento central de la tesis (Frontend #5) y hasta el 16/10/2026 no se
podía calcular: `flights.estimated_out_utc` guarda el estado actual y cada ciclo
lo sobreescribe, así que faltaba el otro lado de la comparación. El harvester
ahora registra cada revisión en `flight_estimate_log`.

Lo que estos tests fijan es sobre todo las DEFINICIONES, porque son lo que un
tribunal puede discutir: qué cuenta como "la aerolínea reconoció la demora", qué
cuenta como "el modelo avisó", y que los casos en que perdemos no se escondan.
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

AHORA = datetime.now(timezone.utc)


def _iso(delta_horas: float) -> str:
    return (AHORA + timedelta(hours=delta_horas)).isoformat()


@pytest.fixture
def con(monkeypatch) -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    c.executescript(
        """CREATE TABLE IF NOT EXISTS flight_estimate_log (
               fa_flight_id TEXT NOT NULL,
               observed_at_utc TEXT NOT NULL,
               estimated_out_utc TEXT
           );"""
    )
    # El registro tiene que venir corriendo desde antes que los vuelos de la
    # prueba, porque la metrica solo cuenta vuelos que pudo haber cubierto
    # enteros. Esta fila no pertenece a ningun vuelo: solo establece desde
    # cuando hay cobertura, que es lo que pasa en produccion con los vuelos
    # viejos de la base.
    c.execute(
        "INSERT INTO flight_estimate_log (fa_flight_id, observed_at_utc,"
        " estimated_out_utc) VALUES ('__cobertura__', ?, NULL)",
        (_iso(-48),),
    )
    c.commit()
    monkeypatch.setattr(api, "get_db", lambda: c)
    return c


def _vuelo(
    con, fid, *, sale_en=-3.0, arr_delay=40.0, cancelled=0, diverted=0,
    marcado_en=None, publicado_en=None, demora_publicada_min=30,
):
    """Un vuelo con resultado, opcionalmente marcado por el modelo y/o
    publicado como demorado por la aerolínea.

    `marcado_en` y `publicado_en` son horas relativas a ahora.
    """
    sale = _iso(sale_en)
    con.execute(
        "INSERT INTO flights (fa_flight_id, stable_id, op_carrier, origin, dest,"
        " scheduled_out_utc, first_seen_utc, last_updated_utc)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (fid, fid, "DL", "ATL", "MIA", sale, sale, sale),
    )
    con.execute(
        "INSERT INTO actuals (fa_flight_id, stable_id, arr_delay_min, cancelled,"
        " diverted, settled_at_utc) VALUES (?,?,?,?,?,?)",
        (fid, fid, arr_delay, cancelled, diverted, sale),
    )
    if marcado_en is not None:
        # Dos ciclos: uno sin marcar antes, para comprobar que se toma el
        # primero MARCADO y no el primero a secas.
        con.execute(
            "INSERT INTO predictions (fa_flight_id, stable_id, predicted_at_utc,"
            " proba_delay, predicted_delay) VALUES (?,?,?,?,?)",
            (fid, fid, _iso(marcado_en - 1), 0.02, 0),
        )
        con.execute(
            "INSERT INTO predictions (fa_flight_id, stable_id, predicted_at_utc,"
            " proba_delay, predicted_delay) VALUES (?,?,?,?,?)",
            (fid, fid, _iso(marcado_en), 0.40, 1),
        )
    if publicado_en is not None:
        estimada = (
            datetime.fromisoformat(sale) + timedelta(minutes=demora_publicada_min)
        ).isoformat()
        con.execute(
            "INSERT INTO flight_estimate_log (fa_flight_id, observed_at_utc,"
            " estimated_out_utc) VALUES (?,?,?)",
            (fid, _iso(publicado_en), estimada),
        )
    con.commit()


# -------- cuando no se puede calcular, decirlo --------

def test_sin_la_tabla_del_harvester_avisa_en_vez_de_tumbarse(monkeypatch) -> None:
    """La API puede estar sirviendo una base que no pasó por el harvester —la
    bundleada en la imagen es el fallback de arranque—. Referenciar la tabla a
    secas devolvía 500, que fue lo que pasó el 15/09 con `threshold_used`."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    monkeypatch.setattr(api, "get_db", lambda: c)
    r = api.metrics_lead_time()
    assert r["disponible"] is False
    assert "flight_estimate_log" in r["motivo"]


def test_sin_vuelos_comparables_no_inventa_un_numero(con) -> None:
    """El registro arrancó el 16/10 y no hay dato anterior. Devolver una media
    sobre cero vuelos, o un cero, haría ver la métrica como si existiera."""
    _vuelo(con, "A", arr_delay=40, marcado_en=-6, publicado_en=None)
    r = api.metrics_lead_time()
    assert r["disponible"] is False
    assert r["n_demorados"] == 1
    assert "16/10" in r["motivo"]


# -------- las definiciones --------

def test_la_ventaja_es_cuanto_antes_marco_el_modelo(con) -> None:
    _vuelo(con, "A", arr_delay=40, marcado_en=-8, publicado_en=-5)
    r = api.metrics_lead_time()
    assert r["disponible"] is True
    assert r["n_comparables"] == 1
    assert r["ventaja_promedio_min"] == pytest.approx(180, abs=1)


def test_toma_el_primer_ciclo_MARCADO_no_el_primero_a_secas(con) -> None:
    """`_vuelo` siembra un ciclo sin marcar una hora antes del marcado. Si se
    tomara el primero a secas, la ventaja saldría una hora más grande."""
    _vuelo(con, "A", arr_delay=40, marcado_en=-8, publicado_en=-5)
    assert api.metrics_lead_time()["ventaja_promedio_min"] == pytest.approx(180, abs=1)


def test_una_revision_chica_no_cuenta_como_demora_publicada(con) -> None:
    """Quince minutos es el umbral del DOT, el mismo con el que se define el
    target. Si una revisión de tres minutos contara, la aerolínea parecería
    avisar muchísimo antes de lo que avisa."""
    _vuelo(con, "A", arr_delay=40, marcado_en=-8, publicado_en=-5,
           demora_publicada_min=3)
    r = api.metrics_lead_time()
    assert r["disponible"] is False, "una revisión de 3 min no es reconocer demora"


def test_el_borde_de_15_minutos(con) -> None:
    _vuelo(con, "A", arr_delay=40, marcado_en=-8, publicado_en=-5,
           demora_publicada_min=15)
    assert api.metrics_lead_time()["disponible"] is False
    _vuelo(con, "B", arr_delay=40, marcado_en=-8, publicado_en=-5,
           demora_publicada_min=16)
    assert api.metrics_lead_time()["disponible"] is True


def test_solo_cuentan_los_vuelos_que_se_demoraron(con) -> None:
    """Haber avisado antes sobre un vuelo que salió bien no es una ventaja."""
    _vuelo(con, "A", arr_delay=2, marcado_en=-8, publicado_en=-5)
    r = api.metrics_lead_time()
    assert r["disponible"] is False
    assert r["n_demorados"] == 0


def test_ignora_cancelados_y_desviados(con) -> None:
    _vuelo(con, "A", arr_delay=40, cancelled=1, marcado_en=-8, publicado_en=-5)
    _vuelo(con, "B", arr_delay=40, diverted=1, marcado_en=-8, publicado_en=-5)
    assert api.metrics_lead_time()["n_vuelos_con_resultado"] == 0


# -------- lo que no hay que esconder --------

def test_cuando_la_aerolinea_avisa_primero_la_ventaja_es_negativa(con) -> None:
    """Esconder los casos perdidos haría ver la métrica mejor de lo que es, y es
    exactamente lo que un tribunal va a buscar."""
    _vuelo(con, "A", arr_delay=40, marcado_en=-4, publicado_en=-6)
    r = api.metrics_lead_time()
    assert r["ventaja_promedio_min"] == pytest.approx(-120, abs=1)
    perdidos = next(t for t in r["histograma"]
                    if t["tramo"] == "la aerolínea avisó primero")
    assert perdidos["n"] == 1


def test_el_win_rate_pide_60_minutos_completos(con) -> None:
    _vuelo(con, "A", arr_delay=40, marcado_en=-6.0, publicado_en=-5.0)   # 60 min
    _vuelo(con, "B", arr_delay=40, marcado_en=-6.0, publicado_en=-5.2)   # 48 min
    r = api.metrics_lead_time()
    assert r["n_comparables"] == 2
    assert r["win_rate_60min"] == pytest.approx(0.5)


def test_cuenta_los_puntos_ciegos(con) -> None:
    """El vuelo se demoró, el modelo lo marcó y la aerolínea nunca publicó
    demora. Es el caso que el issue pide contar y no entra en el promedio,
    porque no hay con qué comparar."""
    _vuelo(con, "A", arr_delay=40, marcado_en=-8, publicado_en=-5)
    _vuelo(con, "B", arr_delay=40, marcado_en=-8, publicado_en=None)
    r = api.metrics_lead_time()
    assert r["n_comparables"] == 1
    assert r["n_puntos_ciegos"] == 1


def test_un_demorado_que_el_modelo_nunca_marco_no_cuenta_como_ventaja(con) -> None:
    """Sin marca del modelo no hay anticipación que medir, y contarlo como punto
    ciego lo haría ver como un acierto."""
    _vuelo(con, "A", arr_delay=40, marcado_en=None, publicado_en=-5)
    r = api.metrics_lead_time()
    assert r["disponible"] is False
    assert r["n_puntos_ciegos"] == 0


def test_la_respuesta_avisa_que_no_mide_la_misma_cantidad(con) -> None:
    """El modelo predice demora de LLEGADA y la aerolínea publica su estimación
    de SALIDA. Sin la advertencia el número se lee como si fuera lo mismo."""
    _vuelo(con, "A", arr_delay=40, marcado_en=-8, publicado_en=-5)
    r = api.metrics_lead_time()
    assert "llegada" in r["advertencia"] and "salida" in r["advertencia"]


def test_no_cuenta_vuelos_anteriores_al_inicio_del_registro(con) -> None:
    """Para un vuelo anterior al registro, la aerolinea no tiene marca porque no
    se estaba registrando, no porque no haya avisado. Contarlo como punto ciego
    convierte dato faltante en un acierto.

    Medido contra la base real el 16/10, sin este filtro daba 1.044 puntos
    ciegos sobre 1.757 vuelos demorados: un 59% inventado.
    """
    # Este vuelo empieza a predecirse 60 h antes de ahora, o sea 12 h antes de
    # que arranque la cobertura del fixture.
    _vuelo(con, "VIEJO", arr_delay=40, marcado_en=-59, publicado_en=None)
    _vuelo(con, "NUEVO", arr_delay=40, marcado_en=-8, publicado_en=None)
    r = api.metrics_lead_time()
    assert r["n_puntos_ciegos"] == 1, (
        "el vuelo anterior al registro no puede contar como punto ciego")
    assert r["n_vuelos_con_resultado"] == 1

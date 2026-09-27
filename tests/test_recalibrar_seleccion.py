"""Contra que prediccion se calibra.

`recalibrate_live.py` tomaba una prediccion por vuelo con

    df.sort_values("proba_delay").groupby("fa_flight_id").last()

El comentario decia "la mas reciente" pero ordenaba por PROBABILIDAD, asi que
se quedaba con la mas pesimista de cada vuelo. Y `predicted_at_utc` ni siquiera
estaba en el SELECT.

Medido el 27/09 sobre 8.372 vuelos de ATL: las dos selecciones eligen distinto
en el 86% de los casos, y la probabilidad media pasa de 9,6% (la ultima) a
18,0% (la maxima). El calibrador se ajustaba contra una distribucion con casi
el doble de media que la que despues ve en produccion.

Consecuencia medida: el artefacto `4year_v9_recal` empuja la probabilidad media
a 4,5% cuando la tasa real es 12,7%, con Brier 0,1106 contra 0,1094 sin
recalibrar. Es peor que no recalibrar, y por eso nunca se activo.
"""
from __future__ import annotations

import re
from pathlib import Path

FUENTE = (Path(__file__).resolve().parents[1] / "recalibrate_live.py").read_text("utf-8")


def _bloque_seleccion() -> str:
    """La linea que se queda con una prediccion por vuelo."""
    m = re.search(r'^.*groupby\("fa_flight_id".*$', FUENTE, re.M)
    assert m, "no se encontro la seleccion de una prediccion por vuelo"
    return m.group(0)


def test_ordena_por_fecha_y_no_por_probabilidad() -> None:
    linea = _bloque_seleccion()
    assert 'sort_values("predicted_at_utc")' in linea, linea
    assert 'sort_values("proba_delay")' not in linea, linea


def test_la_consulta_trae_la_fecha_de_la_prediccion() -> None:
    """Sin esta columna no hay forma de ordenar por tiempo."""
    assert "p.predicted_at_utc" in FUENTE


def test_solo_entran_predicciones_anteriores_a_la_salida() -> None:
    """
    Una prediccion posterior al horario de salida no es la que operaciones
    tuvo en la mano: para entonces el vuelo ya se fue. Calibrar contra esas
    mide algo que nadie usa.
    """
    normalizado = " ".join(FUENTE.split())
    assert "p.predicted_at_utc <= f.scheduled_out_utc" in normalizado
    assert "JOIN flights" in normalizado


def test_el_comentario_describe_lo_que_hace_el_codigo() -> None:
    """
    El defecto original sobrevivio porque el comentario decia "la mas reciente"
    y nadie leyo la linea de abajo. Que vuelvan a contradecirse es la forma en
    que esto reaparece.
    """
    i = FUENTE.index('groupby("fa_flight_id"')
    contexto = FUENTE[max(0, i - 1400):i]
    assert "última predicción anterior a la salida" in contexto

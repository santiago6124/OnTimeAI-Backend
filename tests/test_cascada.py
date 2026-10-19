"""Propagación de demoras por la rotación del avión.

Es la mitad del título de la tesis. Lo que estos tests protegen no es que los
números sean los correctos —eso lo dice la medición sobre 779.777 pares de BTS,
y está documentada en el módulo— sino las propiedades que una tabla de
propagación tiene que cumplir, porque un dedazo en una celda no se nota a ojo y
rompe el sentido del resultado.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ontimeai.cascade import (  # noqa: E402
    UMBRAL_PROPAGACION_MIN,
    BORDES_DEMORA_MIN,
    BORDES_HOLGURA_MIN,
    DEMORA_CADENA_ROTA_MIN,
    RECUPERACION_EN_RUTA_MIN,
    TABLA_PROPAGACION,
    Pierna,
    demora_siguiente_pierna,
    proyectar_rotacion,
)


# ── la tabla ───────────────────────────────────────────────────────────────

def test_la_tabla_tiene_la_forma_de_los_bordes() -> None:
    assert len(TABLA_PROPAGACION) == len(BORDES_HOLGURA_MIN) - 1
    for fila in TABLA_PROPAGACION:
        assert len(fila) == len(BORDES_DEMORA_MIN) - 1


def test_mas_holgura_propaga_menos() -> None:
    """Columna por columna, bajando por los tramos de holgura, la demora que
    sale tiene que caer. Es la propiedad central de lo medido y un dedazo en
    una celda la rompe."""
    for j in range(len(TABLA_PROPAGACION[0])):
        columna = [fila[j] for fila in TABLA_PROPAGACION]
        assert columna == sorted(columna, reverse=True), (
            f"columna {j} no decrece con la holgura: {columna}")


def test_mas_demora_que_llega_propaga_mas() -> None:
    for i, fila in enumerate(TABLA_PROPAGACION):
        assert list(fila) == sorted(fila), f"fila {i} no crece con la demora: {fila}"


def test_con_turnaround_ajustado_la_demora_crece() -> None:
    """El hallazgo que no es obvio: con holgura de 20 a 70 minutos la demora no
    se traslada, se AMPLIFICA. Medido al 136% y al 124%. Si alguien 'corrige'
    la tabla para que nunca supere lo que entra, borra el resultado."""
    for demora, j in ((22, 1), (37, 2), (57, 3)):
        assert TABLA_PROPAGACION[0][j] > demora, (
            "con 20-45 min de holgura la salida tiene que superar a la llegada")
        assert TABLA_PROPAGACION[1][j] > demora, (
            "con 45-70 min de holgura también amplifica")


def test_con_mucha_holgura_casi_no_propaga() -> None:
    """Con más de 3 horas la demora se absorbe: medido entre 5% y 7%."""
    assert TABLA_PROPAGACION[-1][1] < 10
    assert TABLA_PROPAGACION[-1][3] < 15


# ── un paso ────────────────────────────────────────────────────────────────

def test_una_llegada_en_horario_no_propaga_nada() -> None:
    """La tabla se midió sobre piernas anteriores DEMORADAS. Con el avión en
    horario la siguiente sale 3 minutos antes, así que propagar la celda sería
    inventar una demora que no existe."""
    for d in (0, -5, -40):
        assert demora_siguiente_pierna(60, d) == 0.0


def test_por_debajo_de_15_minutos_no_propaga() -> None:
    """La primera columna de la tabla es demora de salida de BASE, no
    propagación: con el avión en horario la siguiente ya sale ~5 min tarde.
    Encadenando sin este corte esos minutos se regeneran en cada paso y la
    cadena no muere nunca."""
    for d in (1, 10, 15):
        assert demora_siguiente_pierna(60, d) == 0.0
    assert demora_siguiente_pierna(60, 16) > 0.0


def test_la_primera_columna_queda_sin_usar_a_proposito() -> None:
    """Se conserva porque es parte de la medición, pero ninguna demora que
    propaga cae en ella. Si alguien baja el umbral, este test avisa."""
    assert UMBRAL_PROPAGACION_MIN == BORDES_DEMORA_MIN[1]


def test_holgura_por_debajo_del_rango_usa_el_tramo_mas_ajustado() -> None:
    """Menos de 20 minutos no es un turnaround real y no hay dato. Se usa el
    tramo que más propaga: subestimar en el caso más apretado sería el error
    caro."""
    assert demora_siguiente_pierna(5, 40) == demora_siguiente_pierna(25, 40)


def test_holgura_por_encima_del_rango_satura_en_el_ultimo_tramo() -> None:
    # 40 min cae en el tramo 30-45, que es la columna 2.
    assert demora_siguiente_pierna(5000, 40) == TABLA_PROPAGACION[-1][2]


def test_los_bordes_de_los_tramos_no_dejan_huecos() -> None:
    """Un valor exactamente en el borde tiene que caer en el tramo de arriba,
    no quedar sin tramo ni duplicarse."""
    for borde in BORDES_HOLGURA_MIN[1:-1]:
        antes = demora_siguiente_pierna(borde - 0.1, 40)
        justo = demora_siguiente_pierna(borde, 40)
        assert antes != justo, f"el borde {borde} no cambia de tramo"


# ── la cadena ──────────────────────────────────────────────────────────────

def _pierna(id_vuelo, sale, llega, origen="ATL", destino="MIA"):
    return Pierna(id_vuelo, origen, destino, sale, llega)


def test_la_demora_se_arrastra_por_varias_piernas() -> None:
    piernas = [_pierna("B", 600, 720), _pierna("C", 780, 900)]
    r = proyectar_rotacion(piernas, demora_llegada_inicial_min=45,
                           llegada_programada_inicial_min=540)
    assert len(r) == 2
    assert all(p.demorado for p in r)
    assert r[0].holgura_min == 60
    assert r[1].holgura_min == 60


def test_una_escala_larga_corta_la_cadena() -> None:
    """La holgura de cada pierna se evalúa sola, así que una escala larga en el
    medio detiene la propagación sin que haga falta una regla aparte."""
    piernas = [_pierna("B", 600, 720), _pierna("C", 1200, 1320)]
    r = proyectar_rotacion(piernas, demora_llegada_inicial_min=45,
                           llegada_programada_inicial_min=540)
    assert r[0].demorado
    assert not r[1].demorado, "con 480 min de holgura no debería quedar demora"


def test_la_cadena_se_detiene_cuando_la_demora_muere() -> None:
    """Sin esto se devolverían piernas con cero demora, que ensucian la
    pantalla con filas que no dicen nada."""
    piernas = [_pierna("B", 3000, 3120), _pierna("C", 3200, 3320),
               _pierna("D", 3400, 3520)]
    r = proyectar_rotacion(piernas, demora_llegada_inicial_min=20,
                           llegada_programada_inicial_min=540)
    assert len(r) == 1, "la primera holgura enorme ya absorbe todo"


def test_con_holgura_ajustada_la_demora_se_sostiene() -> None:
    """No es un defecto de la proyección: con 60 minutos de turnaround el
    traslado medido es del 124%, o sea que la demora se autosostiene. En la
    realidad la aerolínea la rompe cambiando el avión, y para eso está
    `max_piernas`.

    Este test estaba escrito al revés —afirmaba que la cadena se cortaba— y lo
    corrigió la medición, no el código."""
    piernas = [_pierna(f"P{i}", 600 + i * 150, 690 + i * 150) for i in range(6)]
    r = proyectar_rotacion(piernas, demora_llegada_inicial_min=16,
                           llegada_programada_inicial_min=540, max_piernas=6)
    assert len(r) == 6
    assert all(p.demorado for p in r)
    # Se estabiliza, no explota.
    assert r[-1].demora_llegada_min == pytest.approx(r[1].demora_llegada_min, abs=1)


def test_no_devuelve_piernas_con_cero_demora() -> None:
    """Si el corte fuera después de agregar, la última pierna saldría con cero
    y ensuciaría la pantalla con una fila que no dice nada."""
    piernas = [_pierna("B", 3000, 3120), _pierna("C", 3200, 3320)]
    r = proyectar_rotacion(piernas, demora_llegada_inicial_min=20,
                           llegada_programada_inicial_min=540)
    assert all(p.demora_salida_min > 0 for p in r)


def test_respeta_el_maximo_de_piernas() -> None:
    """El error se acumula: 18 minutos por paso. A la cuarta pierna el
    intervalo es más ancho que el número."""
    piernas = [_pierna(f"P{i}", 600 + i * 180, 720 + i * 180) for i in range(8)]
    r = proyectar_rotacion(piernas, demora_llegada_inicial_min=120,
                           llegada_programada_inicial_min=540, max_piernas=2)
    assert len(r) <= 2


def test_marca_la_cadena_rota_con_demoras_enormes() -> None:
    """Arriba de 4 horas la aerolínea cambia el avión y la tabla sobreestima
    (179 contra 157 reales). El número se devuelve, pero marcado."""
    piernas = [_pierna("B", 600, 720)]
    normal = proyectar_rotacion(piernas, demora_llegada_inicial_min=60,
                                llegada_programada_inicial_min=540)
    enorme = proyectar_rotacion(piernas, demora_llegada_inicial_min=DEMORA_CADENA_ROTA_MIN + 1,
                                llegada_programada_inicial_min=540)
    assert not normal[0].cadena_rota
    assert enorme[0].cadena_rota


def test_la_llegada_descuenta_lo_que_se_recupera_en_ruta() -> None:
    piernas = [_pierna("B", 600, 720)]
    r = proyectar_rotacion(piernas, demora_llegada_inicial_min=60,
                           llegada_programada_inicial_min=540)[0]
    assert r.demora_llegada_min == pytest.approx(
        r.demora_salida_min - RECUPERACION_EN_RUTA_MIN)


def test_la_recuperacion_no_vuelve_negativa_la_demora() -> None:
    """Con una demora de salida chica, restar los 4,5 minutos daría negativo y
    'llega antes de lo programado' no es algo que la cascada pueda afirmar."""
    piernas = [_pierna("B", 3000, 3120)]
    r = proyectar_rotacion(piernas, demora_llegada_inicial_min=16,
                           llegada_programada_inicial_min=540)[0]
    assert r.demora_llegada_min >= 0.0


def test_el_umbral_de_demorado_es_el_del_modelo() -> None:
    """15 minutos, igual que el target del clasificador. Si la cascada usara
    otro, la pantalla diría 'demorado' para un vuelo que el resto del sistema
    cuenta como puntual."""
    piernas = [_pierna("B", 600, 720)]
    r = proyectar_rotacion(piernas, demora_llegada_inicial_min=45,
                           llegada_programada_inicial_min=540)[0]
    assert r.demorado == (r.demora_llegada_min > 15)


def test_sin_piernas_devuelve_vacio() -> None:
    assert proyectar_rotacion([], demora_llegada_inicial_min=60,
                              llegada_programada_inicial_min=540) == []

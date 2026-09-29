"""El modelo tiene que saber cómo viene el día.

La tasa real de demora en ATL va de 8% a 39% según el día —medido del 14 al
26/09— y el modelo asigna ~9,5% todos los días. Le pega en los tranquilos y se
queda cortísimo en los malos.

No es que sea ciego: en días malos sube la probabilidad media de 7,2% a 14,6%,
así que la cadena del avión, la congestión y el clima llevan algo de señal.
Pero no alcanza. Dentro del tramo donde dice "entre 10 y 20%", la realidad es
12,1% un día calmo y 30,7% uno malo.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ontimeai.live import (  # noqa: E402
    ESTADO_DIA_BASE,
    ESTADO_DIA_FACTOR_MAX,
    ESTADO_DIA_FACTOR_MIN,
    ESTADO_DIA_MIN_VUELOS,
    estado_del_dia,
    estado_dia_adjust,
    open_db,
)

AHORA = datetime(2026, 9, 27, 18, 0, tzinfo=timezone.utc)


def _base(n_vuelos: int, tasa_demora: float, horas_atras: float = 6.0) -> sqlite3.Connection:
    """Salidas de ATL ya resueltas, `horas_atras` antes de `AHORA`."""
    con = open_db(":memory:")
    sale = (AHORA - timedelta(hours=horas_atras)).strftime("%Y-%m-%dT%H:%M:%S")
    ahora_iso = AHORA.isoformat()
    for i in range(n_vuelos):
        fid = f"F{i}"
        con.execute(
            """INSERT INTO flights (fa_flight_id, origin, dest, scheduled_out_utc,
                                    first_seen_utc, last_updated_utc)
               VALUES (?,?,?,?,?,?)""",
            (fid, "ATL", "BOS", sale, ahora_iso, ahora_iso))
        con.execute(
            """INSERT INTO actuals (fa_flight_id, departure_delay_min, cancelled,
                                    settled_at_utc)
               VALUES (?,?,?,?)""",
            (fid, 40.0 if i < round(n_vuelos * tasa_demora) else 2.0, 0, ahora_iso))
    con.commit()
    return con


def test_un_dia_tranquilo_baja_la_probabilidad() -> None:
    con = _base(400, tasa_demora=0.25)
    d = estado_del_dia(con, AHORA)
    assert d["n"] == 400
    assert d["tasa"] == pytest.approx(0.25, abs=0.01)
    assert d["factor"] < 1.0


def test_un_dia_malo_la_sube() -> None:
    """El 22/09 hubo 76,8% de salidas demoradas contra una base de 44,7%."""
    con = _base(400, tasa_demora=0.77)
    d = estado_del_dia(con, AHORA)
    assert d["factor"] > 1.0
    assert d["factor"] <= ESTADO_DIA_FACTOR_MAX


def test_un_dia_normal_no_mueve_nada() -> None:
    con = _base(400, tasa_demora=ESTADO_DIA_BASE)
    d = estado_del_dia(con, AHORA)
    assert d["factor"] == pytest.approx(1.0, abs=0.02)


def test_con_pocos_vuelos_no_se_arriesga() -> None:
    """
    Medido el 27/09: una ventana de las últimas 3 h deja 38 vuelos, porque el
    dato de salida tarda unas 3 h en llegar. Con esa muestra la tasa es ruido,
    y mover todas las predicciones del día por ruido es peor que no tocarlas.
    """
    con = _base(ESTADO_DIA_MIN_VUELOS - 1, tasa_demora=0.9)
    d = estado_del_dia(con, AHORA)
    assert d["factor"] == 1.0


def test_no_mira_la_franja_donde_el_dato_todavia_no_llego() -> None:
    """
    De los vuelos que salieron hace menos de 1 h, el 0% tiene el dato; hace 2-3
    h, el 43%; recién a las 3-6 h llega al 94%. Por eso la ventana arranca tres
    horas atrás. Un vuelo de hace una hora no tiene que contar.
    """
    con = _base(400, tasa_demora=0.9, horas_atras=1.0)
    d = estado_del_dia(con, AHORA)
    assert d["n"] == 0
    assert d["factor"] == 1.0


def test_tampoco_mira_lo_demasiado_viejo() -> None:
    """Lo de ayer no dice cómo viene hoy."""
    con = _base(400, tasa_demora=0.9, horas_atras=30.0)
    d = estado_del_dia(con, AHORA)
    assert d["n"] == 0


def test_el_factor_esta_acotado_en_los_dos_extremos() -> None:
    assert estado_del_dia(_base(400, 0.999), AHORA)["factor"] <= ESTADO_DIA_FACTOR_MAX
    assert estado_del_dia(_base(400, 0.0), AHORA)["factor"] >= ESTADO_DIA_FACTOR_MIN


def test_el_ajuste_no_se_sale_de_rango_ni_en_el_extremo() -> None:
    """
    Multiplicar la probabilidad directamente rompe arriba: 0,8 × 1,8 = 1,44,
    que hay que recortar a 1 y aplasta las diferencias entre los vuelos más
    riesgosos. Sobre los odds no hay nada que recortar.
    """
    for p in (0.01, 0.1, 0.5, 0.8, 0.95, 0.999):
        for f in (ESTADO_DIA_FACTOR_MIN, 1.0, ESTADO_DIA_FACTOR_MAX):
            q = estado_dia_adjust(p, f)
            assert 0.0 <= q <= 0.999, (p, f, q)


def test_el_ajuste_conserva_el_orden() -> None:
    """Corre el nivel, no reordena: un vuelo más riesgoso lo sigue siendo."""
    ps = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9]
    for f in (0.7, 1.0, 1.3, 1.8):
        qs = [estado_dia_adjust(p, f) for p in ps]
        assert qs == sorted(qs), (f, qs)


def test_factor_uno_no_toca_nada() -> None:
    for p in (0.0, 0.03, 0.5, 0.97, 1.0):
        assert estado_dia_adjust(p, 1.0) == pytest.approx(min(p, 0.999), abs=1e-9)


def test_sube_y_baja_en_la_direccion_correcta() -> None:
    p = 0.15
    assert estado_dia_adjust(p, 1.8) > p
    assert estado_dia_adjust(p, 0.7) < p

"""
Cuantos dias de historia cruda se conservan en la base caliente.

Se bajo de 30 a 14 el 25/09. La base llego a 813 MB y el harvester —que la baja
entera, la modifica y la sube en cada ciclo— empezo a morir en su timeout de
900 s: dos horas sin recolectar, y sin que ninguna alarma avisara, porque
`cycle_duration` solo mira el job de prediccion.

El crecimiento es consecuencia de descubrir el horario futuro, que cuadruplico
los vuelos que seguimos. No se pierde nada: el historico completo vive en
BigQuery, verificado fila por fila antes de bajar este numero.
"""
from __future__ import annotations

import importlib
import os

import pytest


def _recargar(valor: str | None):
    if valor is None:
        os.environ.pop("RETENTION_DAYS", None)
    else:
        os.environ["RETENTION_DAYS"] = valor
    import live_job
    return importlib.reload(live_job)


@pytest.fixture(autouse=True)
def _limpiar():
    previo = os.environ.get("RETENTION_DAYS")
    yield
    _recargar(previo)


def test_el_default_es_catorce_dias() -> None:
    assert _recargar(None).RETENTION_DAYS == 14


def test_se_puede_ajustar_sin_redesplegar() -> None:
    """Si 14 resulta poco, subirlo no puede exigir un deploy."""
    assert _recargar("21").RETENTION_DAYS == 21


def test_la_purga_usa_la_constante_y_no_un_numero_fijo() -> None:
    """
    Estaba escrito `days=30` en la llamada. Un numero fijo ahi obliga a
    redesplegar para ajustarlo, justo cuando la base se esta llenando y hay
    apuro.
    """
    from pathlib import Path

    fuente = Path(__file__).resolve().parent.parent / "live_job.py"
    texto = fuente.read_text()
    assert "days=RETENTION_DAYS" in texto
    assert "days=30," not in texto


def test_el_valor_deja_margen_sobre_la_ventana_de_prediccion() -> None:
    """
    La ventana de prediccion son 6 h y el rollup mira dias hacia atras: la
    retencion tiene que cubrir holgadamente ambos para que no se evaluen
    predicciones cuyo vuelo ya se borro.
    """
    assert _recargar(None).RETENTION_DAYS >= 7

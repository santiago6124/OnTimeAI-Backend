"""Valores límite y clases de equivalencia de las bandas de riesgo (CP-RISK-01..03).

`risk_level(proba, umbral)` tiene dos límites: el umbral (de "bajo" a "medio")
y su doble (de "medio" a "alto"). Para cada límite se prueba el valor exacto y
sus dos vecinos inmediatos, como pide el análisis de valores límite: el error
típico es un `>` donde iba un `>=`, y sólo el valor exacto lo detecta.

El umbral llega de la fila de predicción y puede faltar o ser inválido; esas
entradas forman una clase de equivalencia propia que cae al umbral de
respaldo.
"""
from __future__ import annotations

import pytest

from api import _FALLBACK_THRESHOLD, risk_level

UMBRAL = 0.09
EPS = 1e-9


@pytest.mark.parametrize("proba, esperado", [
    (UMBRAL - EPS, "low"),       # justo debajo del umbral
    (UMBRAL, "medium"),          # el umbral mismo: el modelo marca con >=
    (UMBRAL + EPS, "medium"),    # justo encima
])
def test_risk_01_limite_entre_bajo_y_medio(proba, esperado):
    assert risk_level(proba, UMBRAL) == esperado


@pytest.mark.parametrize("proba, esperado", [
    (2 * UMBRAL - EPS, "medium"),  # justo debajo del doble
    (2 * UMBRAL, "high"),          # el doble exacto
    (2 * UMBRAL + EPS, "high"),    # justo encima
])
def test_risk_02_limite_entre_medio_y_alto(proba, esperado):
    assert risk_level(proba, UMBRAL) == esperado


@pytest.mark.parametrize("umbral_invalido", [None, 0.0, -0.5])
def test_risk_03_umbral_ausente_o_invalido_usa_el_de_respaldo(umbral_invalido):
    # Clase de equivalencia: cualquier umbral no positivo se trata igual.
    assert risk_level(_FALLBACK_THRESHOLD, umbral_invalido) == "medium"
    assert risk_level(_FALLBACK_THRESHOLD - EPS, umbral_invalido) == "low"
    assert risk_level(2 * _FALLBACK_THRESHOLD, umbral_invalido) == "high"


@pytest.mark.parametrize("proba", [0.0, 1.0])
def test_risk_04_extremos_del_dominio(proba):
    # 0 y 1 son los bordes físicos de una probabilidad: nunca deben romper.
    assert risk_level(proba, UMBRAL) in {"low", "high"}

"""
El umbral se elige sobre la misma distribucion contra la que se compara.

El pipeline live calculaba el percentil 78 de la probabilidad calibrada y
despues etiquetaba comparando contra la probabilidad ya ajustada. Los cuatro
ajustes post-prediccion son noisy-OR —`p' = 1 - (1-p)(1-p_x)`, que solo puede
subir la probabilidad—, asi que la tasa de positivos superaba la pedida sin
importar los datos. Medido en produccion sobre 3.980 vuelos con resultado real:
78% del lote marcado como demorado contra el 22% de la estrategia, y la tasa
real de retraso en 7,7%.

Ninguna de las dos funciones que intervenian estaba mal por separado. Por eso
estos tests miran el resultado conjunto y no cada una.
"""
from __future__ import annotations

import numpy as np
import pytest

from ontimeai.model import select_threshold_and_label


def _noisy_or(proba: np.ndarray, p_x: float) -> np.ndarray:
    """La forma de todos los ajustes post-prediccion del pipeline."""
    return 1.0 - (1.0 - proba) * (1.0 - p_x)


@pytest.fixture
def calibrada() -> np.ndarray:
    """Un lote parecido al real: casi todo bajo, con cola a la derecha."""
    rng = np.random.default_rng(20260915)
    return np.clip(rng.beta(1.4, 14.0, size=4000), 0.0, 1.0)


def _tasa(labels: np.ndarray) -> float:
    return float(labels.mean())


def test_la_tasa_de_positivos_da_la_pedida(calibrada: np.ndarray) -> None:
    _, _, labels = select_threshold_and_label(
        calibrada, target_pos_rate=0.22, artifact_threshold=0.5
    )
    assert _tasa(labels) == pytest.approx(0.22, abs=0.01)


def test_la_tasa_se_sostiene_despues_de_un_ajuste_noisy_or(calibrada: np.ndarray) -> None:
    """
    El caso que fallaba. Un ajuste que empuja al 65% del lote a p >= 0.75 movia
    la tasa de positivos al 78% cuando el umbral venia de la distribucion previa.
    """
    ajustada = calibrada.copy()
    empujados = slice(0, int(len(ajustada) * 0.65))
    ajustada[empujados] = _noisy_or(ajustada[empujados], 0.75)

    _, _, labels = select_threshold_and_label(
        ajustada, target_pos_rate=0.22, artifact_threshold=0.5
    )
    assert _tasa(labels) == pytest.approx(0.22, abs=0.01)


def test_usar_el_umbral_de_la_distribucion_previa_rompe_la_tasa(
    calibrada: np.ndarray,
) -> None:
    """
    Deja constancia de la magnitud del bug: es lo que hacia el codigo viejo.
    Si algun dia esta asercion falla, el noisy-OR dejo de ser monotono y vale
    la pena mirar por que.
    """
    umbral_previo, _, _ = select_threshold_and_label(
        calibrada, target_pos_rate=0.22, artifact_threshold=0.5
    )
    ajustada = _noisy_or(calibrada, 0.75)

    tasa_cruzada = float((ajustada >= umbral_previo).mean())
    assert tasa_cruzada > 0.9, "el ajuste deberia empujar casi todo el lote"


def test_el_umbral_devuelto_es_el_que_produce_las_etiquetas(
    calibrada: np.ndarray,
) -> None:
    umbral, _, labels = select_threshold_and_label(
        calibrada, target_pos_rate=0.22, artifact_threshold=0.5
    )
    np.testing.assert_array_equal(labels, (calibrada >= umbral).astype(labels.dtype))


def test_un_umbral_absoluto_gana_sobre_el_cuantil(calibrada: np.ndarray) -> None:
    umbral, strategy, labels = select_threshold_and_label(
        calibrada, target_pos_rate=0.22, artifact_threshold=0.5, abs_threshold=0.30
    )
    assert umbral == pytest.approx(0.30)
    assert strategy == "abs@0.30"
    # Con umbral absoluto la tasa flota con las condiciones, que es el objetivo.
    assert _tasa(labels) == pytest.approx(float((calibrada >= 0.30).mean()))


def test_un_lote_chico_cae_al_umbral_del_artefacto() -> None:
    proba = np.array([0.1, 0.2, 0.9])
    umbral, strategy, labels = select_threshold_and_label(
        proba, target_pos_rate=0.22, artifact_threshold=0.44
    )
    assert umbral == pytest.approx(0.44)
    assert strategy == "artifact"
    np.testing.assert_array_equal(labels, np.array([0, 0, 1], dtype=labels.dtype))


def test_un_lote_vacio_no_explota() -> None:
    umbral, strategy, labels = select_threshold_and_label(
        np.array([], dtype=float), target_pos_rate=0.22, artifact_threshold=0.44
    )
    assert umbral == pytest.approx(0.44)
    assert strategy == "artifact"
    assert labels.size == 0

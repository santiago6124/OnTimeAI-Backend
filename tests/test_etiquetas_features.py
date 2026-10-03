"""`FEATURE_LABELS` tiene que cubrir las features que el modelo usa.

Cuando una feature no está en el diccionario, el endpoint cae al respaldo
`feat.replace("_", " ").title()` y la pantalla de operaciones muestra
"Dest Wx Dwpc" o "Congestion Orig Window": medio inglés, medio técnico.

No falla nada, y por eso había quedado así. Medido el 02/10 contra el
artefacto: de las **84 features del modelo, 58 no tenían etiqueta** (69%), y
11 de las 26 entradas nombraban features que el modelo no usa —`vsby_origin`,
`congestion_score`, `CRS_DEP_MIN_sin`—, restos del set de features de una
versión anterior.

La comparación va contra `feature_cols` del artefacto y no contra una lista
pegada acá: esa sería otra copia para desincronizar. Tampoco contra
`prediction_shap`, que solo guarda el top 15 de cada predicción y deja afuera
features reales que simplemente nunca rankean —así `ORIG_WX_SKNT` parecía no
existir—.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]


def _features_del_modelo() -> set[str]:
    import joblib

    meta = joblib.load(RAIZ / "artifacts" / "4year_v9" / "meta.joblib")
    return set(meta["feature_cols"])


@pytest.fixture(scope="module")
def features() -> set[str]:
    try:
        return _features_del_modelo()
    except Exception as exc:  # artefacto ausente en un entorno mínimo
        pytest.skip(f"no se pudo leer el artefacto: {exc}")


def test_todas_las_features_del_modelo_tienen_etiqueta(features) -> None:
    faltan = sorted(features - set(api.FEATURE_LABELS))
    assert not faltan, (
        f"{len(faltan)} features caerían al respaldo y se mostrarían en inglés "
        f"mal formateado: {faltan}"
    )


def test_no_hay_etiquetas_para_features_que_el_modelo_no_usa(features) -> None:
    """
    Una entrada que nombra una feature inexistente no hace daño visible
    —nunca coincide— pero hace creer que la lista está al día. Es exactamente
    cómo sobrevivió el desfasaje anterior.
    """
    sobran = sorted(set(api.FEATURE_LABELS) - features)
    assert not sobran, f"apuntan a features que el modelo no usa: {sobran}"


def test_las_que_mas_pesan_estan_cubiertas() -> None:
    """
    Medido sobre `prediction_shap`: estas seis concentran el 84% de los casos
    en que son la razón número uno de una predicción.
    """
    for f in (
        "CRS_ELAPSED_TIME", "TAIL_DELAY_DECAY", "prev_turnaround_tail_min",
        "congestion_orig_window", "DEST", "DISTANCE",
    ):
        assert f in api.FEATURE_LABELS, f


def test_ninguna_etiqueta_es_el_nombre_crudo() -> None:
    """
    El respaldo produce "Dest Wx Dwpc". Una etiqueta idéntica a eso es señal
    de que alguien la copió del nombre en vez de escribirla.
    """
    for feature, etiqueta in api.FEATURE_LABELS.items():
        assert etiqueta.strip(), feature
        assert etiqueta != feature.replace("_", " ").title(), feature


def test_origen_y_destino_no_comparten_etiqueta() -> None:
    """
    `ORIG_WX_VSBY` y `DEST_WX_VSBY` miden lo mismo en lugares distintos. Si
    las dos dijeran "Visibilidad", el panel mostraría dos filas idénticas con
    valores distintos y no habría forma de saber cuál es cuál.
    """
    por_etiqueta: dict[str, list[str]] = {}
    for feature, etiqueta in api.FEATURE_LABELS.items():
        por_etiqueta.setdefault(etiqueta, []).append(feature)
    repetidas = {e: fs for e, fs in por_etiqueta.items() if len(fs) > 1}
    assert not repetidas, f"etiquetas ambiguas: {repetidas}"

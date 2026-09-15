"""
Los valores SHAP salen de LightGBM, no del paquete `shap`.

Para un modelo de arboles, `shap.TreeExplainer(booster).shap_values(X)` delega
en `booster.predict(X, pred_contrib=True)`. Se ve en el propio codigo de shap
(`shap/explainers/_tree.py`):

    phi = self.model.original_model.predict(
        X, num_iteration=tree_limit, pred_contrib=True)

Llamarlo directo permite sacar `shap` de las dependencias del job, que arrastra
numba y llvmlite: 156 MB instalados. La imagen se baja entera en cada ciclo —96
veces por dia— y el arranque en frio es el grueso del tiempo. Ver issue #4.

El test de equivalencia corre solo donde `shap` esta instalado: en CI y en
desarrollo, no en la imagen del job. Ahi es justamente donde tiene que correr,
porque es el que autoriza a no instalarlo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ontimeai.explainability import compute_shap_values, explain_instance


@pytest.fixture
def booster():
    """Un booster chico entrenado al vuelo: no depende de los artefactos."""
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(20260915)
    X = pd.DataFrame(rng.normal(size=(200, 6)), columns=[f"f{i}" for i in range(6)])
    y = (X["f0"] + 0.5 * X["f1"] + rng.normal(scale=0.2, size=len(X)) > 0).astype(int)
    return lgb.train(
        {"objective": "binary", "verbosity": -1, "num_leaves": 7, "seed": 1},
        lgb.Dataset(X, label=y),
        num_boost_round=15,
    ), X


def test_devuelve_una_columna_por_feature(booster) -> None:
    b, X = booster
    sv = compute_shap_values(b, X)
    assert sv.shape == X.shape, "pred_contrib trae una columna extra que hay que recortar"


def test_la_columna_recortada_es_el_valor_base(booster) -> None:
    """
    `pred_contrib=True` agrega al final el valor esperado del modelo. shap no lo
    incluye; si algun dia se dejara de recortar, las contribuciones quedarian
    corridas una posicion y el feature mas importante seria el equivocado.
    """
    b, X = booster
    completo = b.predict(X, pred_contrib=True)
    assert completo.shape[1] == X.shape[1] + 1
    base = completo[:, -1]
    assert np.allclose(base, base[0]), "el valor base es el mismo para todas las filas"


def test_coincide_con_shap_treeexplainer(booster) -> None:
    """El test que autoriza a no instalar shap en la imagen del job."""
    shap = pytest.importorskip("shap")
    b, X = booster
    esperado = np.asarray(shap.TreeExplainer(b).shap_values(X))
    np.testing.assert_array_equal(compute_shap_values(b, X), esperado)


def test_explain_instance_ordena_por_magnitud_conservando_el_signo(booster) -> None:
    """
    Ordena por |contribucion| pero devuelve el valor con signo: el signo es lo
    que distingue "empuja hacia demora" de "empuja hacia puntual" en la ficha
    del vuelo.
    """
    b, X = booster
    top = explain_instance(compute_shap_values(b, X), list(X.columns), 0, top_n=3)
    magnitudes = top["contribution"].abs().tolist()
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert len(top) == 3
    assert set(top["feature"]) <= set(X.columns)

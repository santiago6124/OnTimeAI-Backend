"""Valores SHAP de un booster LightGBM, con su implementacion nativa."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_shap_values(booster, X: pd.DataFrame):
    """Valores SHAP del booster, con la implementacion nativa de LightGBM.

    Es exactamente lo que hacia `shap.TreeExplainer(booster).shap_values(X)`:
    para un modelo de arboles, shap delega en esta misma llamada. Se ve en su
    propio codigo (`shap/explainers/_tree.py`):

        phi = self.model.original_model.predict(
            X, num_iteration=tree_limit, pred_contrib=True)

    Verificado ademas numericamente sobre el artefacto 4year_v9: identicos bit
    a bit, diferencia maxima 0.000e+00. Hay un test que lo comprueba cuando
    shap esta instalado.

    Se hace asi para sacar `shap` de las dependencias del job: arrastra numba y
    llvmlite, que son 156 MB instalados. La imagen del job se baja entera en
    cada ciclo —96 veces por dia— y el arranque en frio es el grueso del
    tiempo. Ver issue #4.

    `pred_contrib=True` devuelve una columna extra al final con el valor
    esperado del modelo; shap no la incluye, asi que se recorta.
    """
    contribuciones = booster.predict(X, pred_contrib=True)
    return np.asarray(contribuciones)[:, :-1]


def global_feature_importance(shap_values, feature_names: list[str]) -> pd.Series:
    if isinstance(shap_values, list):
        stacked = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            stacked = np.abs(sv).mean(axis=(0, 2))
        else:
            stacked = np.abs(sv).mean(axis=0)
    return pd.Series(stacked, index=feature_names).sort_values(ascending=False)


def explain_instance(
    shap_values, feature_names: list[str], row_idx: int, top_n: int = 10
) -> pd.DataFrame:
    if isinstance(shap_values, list):
        contrib = np.mean([np.abs(sv[row_idx]) for sv in shap_values], axis=0)
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            contrib = np.abs(sv[row_idx]).mean(axis=-1)
        else:
            contrib = sv[row_idx]
    df = pd.DataFrame({"feature": feature_names, "contribution": contrib})
    df["abs_contribution"] = df["contribution"].abs()
    return (
        df.sort_values("abs_contribution", ascending=False)
        .head(top_n)
        .drop(columns=["abs_contribution"])
        .reset_index(drop=True)
    )

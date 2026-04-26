"""SHAP TreeExplainer wrappers for LightGBM boosters."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def compute_shap_values(booster, X: pd.DataFrame):
    explainer = shap.TreeExplainer(booster)
    return explainer.shap_values(X)


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

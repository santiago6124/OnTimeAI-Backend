"""Juzgar un recalibrado contra datos que no vio.

Las métricas en muestra no sirven para decidir: un calibrador que vio las
respuestas no puede empeorar el Brier del set con que se ajustó. El isotónico
llega a ECE exactamente 0,0000 en muestra. Así pasó los controles el artefacto
`4year_v9_recal`, que en producción empujaba la probabilidad media a 4,5%
contra una tasa real de 12,7%.

Un holdout único tampoco alcanza. Medido el 11/10 sobre 12.413 vuelos, el
veredicto cambia de signo según cuántos días se reserven:

    2 días  -0,2%     4 días  -1,0%
    3 días  -0,6%     5 días  -0,6%
    6 días  +0,1%  <- con este el portón rechaza

El recalibrado ayuda mucho los días con demoras y casi nada los calmos, así
que un holdout corto mide sobre todo qué días le tocaron. Por eso se camina
sobre varios cortes y se pide ganar en la mayoría, no en uno elegido a dedo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from .calibration import fit_calibrator

# Un vuelo que aterriza no tiene su `actual` cargado al instante. Al ajustar en
# el corte T solo se puede usar lo que salió antes de T menos este margen.
MARGEN_ASENTAMIENTO = timedelta(days=1)


@dataclass(frozen=True)
class Corte:
    """Un corte del camino: qué se midió sobre los vuelos posteriores."""

    fecha: pd.Timestamp
    n_ajuste: int
    n_prueba: int
    brier_cruda: float
    brier_recal: float
    # Los vectores del tramo de prueba, para que quien llame pueda calcular
    # AUC, ECE o la tabla por tramos sin repetir el camino.
    y: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.empty(0))
    p_cruda: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.empty(0))
    p_recal: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.empty(0))

    @property
    def mejora(self) -> bool:
        return self.brier_recal < self.brier_cruda


def caminar_cortes(
    salida: pd.Series,
    y: np.ndarray,
    p: np.ndarray,
    *,
    metodo: str = "isotonic",
    horizonte_dias: int = 2,
    ventana_dias: int | None = None,
    min_ajuste: int = 400,
    min_prueba: int = 100,
    desde: pd.Timestamp | None = None,
) -> list[Corte]:
    """Ajusta en cada corte con lo ya asentado y mide sobre lo que vino después.

    Args:
        salida: hora de salida programada de cada vuelo, con zona.
        y: resultado real (0/1), alineado con `salida`.
        p: probabilidad que el modelo dio, alineada con `salida`.
        ventana_dias: días de historia para ajustar; None usa todo lo asentado.
        desde: primer corte. Obligatorio para comparar varias estrategias entre
            sí: si cada una arranca donde le alcanza su ventana, terminan
            midiendo sobre períodos distintos y los Brier no se pueden
            comparar. Con ventanas largas y poca historia, los primeros cortes
            ajustan con menos días —que es lo que pasaría en producción el día
            que se enciendan—.

    Returns:
        Un `Corte` por fecha evaluable, en orden. Vacío si no hay historia.
    """
    salida = pd.to_datetime(salida, utc=True)
    orden = np.argsort(salida.values)
    salida, y, p = salida.iloc[orden], np.asarray(y)[orden], np.asarray(p)[orden]

    cortes: list[Corte] = []
    if salida.empty:
        return cortes

    paso = ventana_dias if ventana_dias else 1
    t = desde if desde is not None else (
        salida.min().normalize() + timedelta(days=paso + MARGEN_ASENTAMIENTO.days))
    fin = salida.max().normalize() - timedelta(days=horizonte_dias)
    while t <= fin:
        es_prueba = (salida >= t) & (salida < t + timedelta(days=horizonte_dias))
        asentado = salida < t - MARGEN_ASENTAMIENTO
        if ventana_dias:
            asentado &= salida >= t - MARGEN_ASENTAMIENTO - timedelta(days=ventana_dias)
        n_a, n_p = int(asentado.sum()), int(es_prueba.sum())
        if n_p >= min_prueba and n_a >= min_ajuste and len(np.unique(y[asentado])) > 1:
            cal = fit_calibrator(p[asentado], y[asentado], method=metodo)
            pr = np.clip(cal.transform(p[es_prueba]), 1e-6, 1 - 1e-6)
            cortes.append(Corte(
                fecha=t, n_ajuste=n_a, n_prueba=n_p,
                brier_cruda=float(brier_score_loss(y[es_prueba], p[es_prueba])),
                brier_recal=float(brier_score_loss(y[es_prueba], pr)),
                y=y[es_prueba], p_cruda=p[es_prueba], p_recal=pr,
            ))
        t += timedelta(days=1)
    return cortes


def veredicto(cortes: list[Corte], *, minimo_ganados: float = 0.7) -> tuple[bool, str]:
    """¿Vale la pena guardar este recalibrado?

    Pide ganar en al menos `minimo_ganados` de los cortes. Con el promedio solo
    alcanzaba que un día malo compensara varios días peores.
    """
    if not cortes:
        return False, "sin cortes evaluables: hace falta más historia asentada"
    ganados = sum(c.mejora for c in cortes)
    frac = ganados / len(cortes)
    cruda = float(np.mean([c.brier_cruda for c in cortes]))
    recal = float(np.mean([c.brier_recal for c in cortes]))
    delta = (recal - cruda) / cruda * 100 if cruda else float("nan")
    resumen = (f"gana en {ganados}/{len(cortes)} cortes ({frac:.0%}); "
               f"Brier medio {cruda:.4f} -> {recal:.4f} ({delta:+.1f}%)")
    if frac < minimo_ganados:
        return False, f"{resumen} — por debajo del mínimo de {minimo_ganados:.0%}"
    return True, resumen


def elegir_threshold(proba: np.ndarray, objetivo: float) -> float:
    """El corte que deja la tasa positiva más cerca de `objetivo`.

    Delega en `ontimeai.model.quantile_threshold`, que es el que corre en
    producción: tener dos implementaciones de esto fue justamente cómo el
    desborde por empates sobrevivió en el camino productivo mientras el script
    de recalibrado ya lo tenía resuelto.
    """
    from .model import quantile_threshold

    proba = np.asarray(proba, dtype=float)
    if proba.size == 0:
        return float("nan")
    return quantile_threshold(proba, objetivo)

"""Módulo de demostración del TP de Calidad: NO mergear.

Existe para mostrar en vivo que el gate de complejidad corta el CI. La
función de abajo tiene complejidad ciclomática 11 (una decisión por cada
rango de demora); el run tiene que fallar en "Complejidad ciclomática" con
su nombre y su valor, y pasar al revertir este commit.
"""
from __future__ import annotations


def clasificar_demora_para_la_demo(minutos: float | None, cancelado: bool, desviado: bool) -> str:
    if cancelado:
        return "cancelado"
    if desviado:
        return "desviado"
    if minutos is None:
        return "sin dato"
    if minutos < -15:
        return "muy adelantado"
    if minutos < 0:
        return "adelantado"
    if minutos == 0:
        return "en horario"
    if minutos <= 15:
        return "leve"
    if minutos <= 45:
        return "moderada"
    if minutos <= 120:
        return "severa"
    if minutos <= 240:
        return "muy severa"
    return "excepcional"

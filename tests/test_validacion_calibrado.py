"""El control que decide si un recalibrado se guarda.

Sin esto el criterio eran las métricas en muestra, y el isotónico da ECE
exactamente 0,0000 contra los datos con que se ajustó. Así pasó el artefacto
`4year_v9_recal` del 06/08, cuyo reporte declaraba AUC 0,9063 —inflado por el
bug de selección, que elegía por vuelo la predicción de probabilidad MÁS ALTA,
o sea la que mejor había adivinado—.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ontimeai import validacion_calibrado as vc  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]


def _datos(dias: int = 14, por_dia: int = 300, semilla: int = 0):
    """Vuelos sintéticos donde el modelo subestima, como en producción."""
    rng = np.random.default_rng(semilla)
    base = datetime(2026, 10, 1, tzinfo=timezone.utc)
    salida, p, y = [], [], []
    for d in range(dias):
        for _ in range(por_dia):
            prob = float(rng.beta(1.5, 20))          # media ~7%
            salida.append(base + timedelta(days=d, minutes=int(rng.integers(0, 1440))))
            p.append(prob)
            # La realidad pasa 1,7x más seguido que lo que promete el modelo.
            y.append(int(rng.random() < min(prob * 1.7, 1.0)))
    return pd.Series(salida), np.array(y), np.array(p)


def test_no_ajusta_con_vuelos_posteriores_al_corte(monkeypatch) -> None:
    """La prueba de fuga temporal.

    Si el ajuste viera vuelos que salen después del corte, el control estaría
    midiendo contra datos que el calibrador ya conoce y volveríamos justo al
    problema que este módulo existe para evitar.
    """
    salida, y, p = _datos()
    vistos: list[tuple] = []
    real = vc.fit_calibrator

    def espia(proba, objetivo, method="isotonic"):
        vistos.append((len(proba), float(proba.min()), float(proba.max())))
        return real(proba, objetivo, method=method)

    monkeypatch.setattr(vc, "fit_calibrator", espia)
    cortes = vc.caminar_cortes(salida, y, p, min_ajuste=300)
    assert cortes, "el escenario debería producir cortes"
    assert len(vistos) == len(cortes)

    # Reconstruido: para cada corte, el conjunto de ajuste no puede incluir
    # ningún vuelo que salga en o después de (corte - margen).
    s = pd.to_datetime(salida, utc=True)
    for c, (n_ajuste, _, _) in zip(cortes, vistos):
        limite = c.fecha - vc.MARGEN_ASENTAMIENTO
        assert n_ajuste == int((s < limite).sum()), (
            f"corte {c.fecha:%d/%m}: ajustó con {n_ajuste} vuelos, "
            f"pero antes de {limite:%d/%m %H:%M} hay {(s < limite).sum()}")
        assert c.n_prueba == int(((s >= c.fecha) & (s < c.fecha + timedelta(days=2))).sum())


def test_la_prueba_de_cada_corte_es_posterior_al_corte() -> None:
    salida, y, p = _datos()
    s = pd.to_datetime(salida, utc=True)
    for c in vc.caminar_cortes(salida, y, p, min_ajuste=300):
        assert c.n_ajuste > 0 and c.n_prueba > 0
        assert c.fecha - vc.MARGEN_ASENTAMIENTO > s.min()


def test_desde_alinea_los_cortes_entre_estrategias() -> None:
    """Sin `desde`, cada ventana arranca donde le alcanza y las estrategias
    terminan midiendo sobre períodos distintos. Medido el 11/10, iso_7d parecía
    ganar 20,7% porque le tocaban días calmos, contra una cruda calculada sobre
    el doble de vuelos."""
    salida, y, p = _datos()
    desde = pd.to_datetime(salida, utc=True).min().normalize() + timedelta(days=4)
    corridas = [
        vc.caminar_cortes(salida, y, p, ventana_dias=v, min_ajuste=300, desde=desde)
        for v in (None, 3, 7)
    ]
    fechas = {tuple(c.fecha for c in cs) for cs in corridas}
    assert len(fechas) == 1, "todas las estrategias deben caminar los mismos cortes"
    tamanos = {tuple(c.n_prueba for c in cs) for cs in corridas}
    assert len(tamanos) == 1, "el conjunto de prueba debe ser idéntico"


def test_veredicto_rechaza_cuando_gana_pocas_veces() -> None:
    peor = [vc.Corte(fecha=pd.Timestamp("2026-10-0%d" % d, tz="UTC"), n_ajuste=1000,
                     n_prueba=500, brier_cruda=0.09, brier_recal=0.10)
            for d in range(1, 6)]
    ok, motivo = vc.veredicto(peor)
    assert not ok
    assert "0/5" in motivo


def test_veredicto_rechaza_aunque_el_promedio_mejore() -> None:
    """Un día en que mejora muchísimo no compensa cuatro en que empeora: el
    promedio se vería bien y en producción andaría mal cuatro días de cada
    cinco."""
    cortes = [vc.Corte(fecha=pd.Timestamp("2026-10-01", tz="UTC"), n_ajuste=1000,
                       n_prueba=500, brier_cruda=0.30, brier_recal=0.10)]
    cortes += [vc.Corte(fecha=pd.Timestamp("2026-10-0%d" % d, tz="UTC"), n_ajuste=1000,
                        n_prueba=500, brier_cruda=0.09, brier_recal=0.095)
               for d in range(2, 6)]
    promedio_mejora = (np.mean([c.brier_recal for c in cortes])
                       < np.mean([c.brier_cruda for c in cortes]))
    assert promedio_mejora, "el escenario debe tener promedio favorable"
    ok, _ = vc.veredicto(cortes)
    assert not ok


def test_veredicto_acepta_cuando_gana_casi_siempre() -> None:
    cortes = [vc.Corte(fecha=pd.Timestamp("2026-10-0%d" % d, tz="UTC"), n_ajuste=1000,
                       n_prueba=500, brier_cruda=0.09, brier_recal=0.085)
              for d in range(1, 6)]
    ok, motivo = vc.veredicto(cortes)
    assert ok, motivo
    assert "5/5" in motivo


def test_sin_historia_no_hay_veredicto_favorable() -> None:
    ok, motivo = vc.veredicto([])
    assert not ok
    assert "historia" in motivo


def test_recalibrar_mejora_cuando_el_modelo_subestima() -> None:
    """Un calibrador ajustado sobre días anteriores tiene que mejorar el Brier
    de los días siguientes si el sesgo es estable.

    No se afirma acá que isotonic le gane a sigmoid: eso depende de la FORMA de
    la mala calibración real, que este escenario sintético —donde la relación
    es lineal y Platt la ajusta igual de bien— no reproduce. La evidencia de
    esa comparación vive en `recal_report.json`, medida sobre producción.
    """
    salida, y, p = _datos(dias=20, por_dia=400, semilla=7)
    desde = pd.to_datetime(salida, utc=True).min().normalize() + timedelta(days=4)
    cs = vc.caminar_cortes(salida, y, p, min_ajuste=1000, desde=desde)
    assert cs
    ok, motivo = vc.veredicto(cs)
    assert ok, motivo


def test_elegir_threshold_respeta_el_objetivo_con_valores_empatados() -> None:
    """El caso que rompía: isotonic empata muchos valores, el cuantil cae
    dentro de un bloque y `>=` se lleva el bloque entero (34,5% con objetivo
    22%).
    """
    # Tres bloques grandes de valores idénticos: lo que produce isotonic.
    p = np.concatenate([np.full(7000, 0.03), np.full(2000, 0.12), np.full(1000, 0.40)])
    for objetivo in (0.1, 0.22, 0.3, 0.5):
        c = vc.elegir_threshold(p, objetivo)
        tasa = float((p >= c).mean())
        # Con bloques de 10%, 20% y 70% no toda tasa es alcanzable; se pide que
        # sea la más cercana entre las posibles, no que coincida.
        posibles = [float((p >= v).mean()) for v in np.unique(p)]
        mejor = min(posibles, key=lambda t: abs(t - objetivo))
        assert tasa == pytest.approx(mejor), (
            f"objetivo {objetivo}: eligió {tasa:.1%}, la más cercana era {mejor:.1%}")


def test_elegir_threshold_sin_empates_se_porta_como_el_cuantil() -> None:
    rng = np.random.default_rng(11)
    p = rng.random(5000)
    c = vc.elegir_threshold(p, 0.22)
    assert float((p >= c).mean()) == pytest.approx(0.22, abs=0.01)


# -------- el artefacto generado --------

ARTEFACTO = RAIZ / "artifacts" / "4year_v9_recal_20261011"


@pytest.fixture(scope="module")
def artefacto():
    if not (ARTEFACTO / "meta.joblib").exists():
        pytest.skip(f"no está generado: {ARTEFACTO.name}")
    import joblib
    return joblib.load(ARTEFACTO / "meta.joblib")


def test_la_segunda_capa_sube_las_probabilidades(artefacto) -> None:
    """El modelo subestima —media 5,7% contra una tasa real de 9,7%—, así que
    un recalibrado correcto tiene que EMPUJAR HACIA ARRIBA el tramo bajo.

    El artefacto del 06/08 hacía lo contrario: llevaba la media a 4,5% contra
    una realidad de 12,7%. Por eso nunca se activó.
    """
    cal2 = artefacto["calibrator"].cal2
    for x in (0.01, 0.03, 0.05, 0.08, 0.12, 0.20):
        y = float(cal2.transform(np.array([x]))[0])
        assert y > x, f"con proba {x} debería subir, dio {y}"


def test_la_segunda_capa_es_monotona(artefacto) -> None:
    """Si rompiera el orden, dos vuelos cambiarían de posición relativa y el
    ranking que la pantalla usa para ordenar dejaría de tener sentido."""
    cal2 = artefacto["calibrator"].cal2
    xs = np.linspace(0.001, 0.999, 500)
    ys = cal2.transform(xs)
    assert np.all(np.diff(ys) >= -1e-9)
    assert ys.min() >= 0.0 and ys.max() <= 1.0


def test_el_threshold_del_artefacto_sale_de_la_distribucion_real(artefacto) -> None:
    """El threshold es específico de la distribución sobre la que se eligió, no
    una propiedad del calibrador: medirlo contra probabilidades sintéticas da
    cualquier cosa. Acá solo se comprueba que sea un valor usable y que esté
    entre los que el calibrador puede devolver.
    """
    cal2 = artefacto["calibrator"].cal2
    t = artefacto["threshold"]
    assert 0.0 < t < 1.0
    salidas = cal2.transform(np.linspace(0.0, 1.0, 2001))
    assert salidas.min() <= t <= salidas.max(), (
        f"threshold {t} fuera del rango que el calibrador produce "
        f"[{salidas.min():.3f}, {salidas.max():.3f}]")


def test_el_reporte_guarda_la_evidencia_fuera_de_muestra() -> None:
    """Las métricas en muestra no alcanzan para decidir, y el reporte anterior
    solo tenía esas."""
    import json
    rep = ARTEFACTO / "recal_report.json"
    if not rep.exists():
        pytest.skip("artefacto no generado")
    d = json.loads(rep.read_text())
    assert "fuera_de_muestra" in d, "el reporte debe cargar su propia evidencia"
    f = d["fuera_de_muestra"]
    assert f["n_cortes"] >= 3
    assert f["cortes_ganados"] / f["n_cortes"] >= 0.7
    assert f["brier_recal"] < f["brier_cruda"]
    assert d["method"] == "isotonic"

"""
El nivel de riesgo que muestra el dashboard tiene que coincidir con lo que el
modelo marca.

Las bandas eran 0.35 y 0.15, constantes elegidas cuando la probabilidad media
del lote rondaba 0.55 porque la cadena de ajustes post-prediccion la inflaba.
Corregida esa cadena, la media quedo en ~0.07 y el umbral operativo en ~0.09:
con las constantes viejas un vuelo marcado como demorado se mostraba en verde.
"""
from __future__ import annotations

import pytest

from api import _FALLBACK_THRESHOLD, risk_level


class TestBandas:
    def test_por_encima_del_doble_del_umbral_es_alto(self) -> None:
        assert risk_level(0.20, 0.09) == "high"

    def test_entre_el_umbral_y_su_doble_es_medio(self) -> None:
        assert risk_level(0.10, 0.09) == "medium"

    def test_por_debajo_del_umbral_es_bajo(self) -> None:
        assert risk_level(0.05, 0.09) == "low"

    def test_justo_en_el_umbral_ya_no_es_bajo(self) -> None:
        # El modelo lo marca con >=, asi que la ficha no puede decir lo contrario.
        assert risk_level(0.09, 0.09) == "medium"

    def test_justo_en_el_doble_es_alto(self) -> None:
        assert risk_level(0.18, 0.09) == "high"


class TestCoherenciaConLaEtiqueta:
    """
    La propiedad que motiva todo esto: alto + medio es exactamente el conjunto
    que el modelo marca como demorado. Si algun dia deja de valer, el dashboard
    volvio a contar una cosa distinta de la que decide el modelo.
    """

    @pytest.mark.parametrize("umbral", [0.05, 0.09, 0.22, 0.32, 0.5])
    def test_marcado_equivale_a_alto_o_medio(self, umbral: float) -> None:
        for paso in range(0, 101):
            proba = paso / 100.0
            marcado = proba >= umbral
            nivel = risk_level(proba, umbral)
            assert marcado == (nivel in ("high", "medium")), (
                f"p={proba:.2f} umbral={umbral}: marcado={marcado} nivel={nivel}"
            )

    def test_el_caso_que_fallaba_con_las_constantes_viejas(self) -> None:
        """
        Probabilidad calibrada tipica (0.12) con el umbral operativo actual.
        Las bandas viejas —alto 0.35, medio 0.15— lo daban "low" pese a estar
        marcado como demorado.
        """
        assert risk_level(0.12, 0.09) == "medium"
        assert not (0.12 >= 0.15), "la banda vieja lo mandaba a low"


class TestUmbralAusente:
    """Predicciones anteriores a que se guardara la columna."""

    def test_sin_umbral_usa_el_del_artefacto(self) -> None:
        assert risk_level(0.70, None) == "high"
        assert risk_level(0.40, None) == "medium"
        assert risk_level(0.10, None) == "low"

    @pytest.mark.parametrize("degenerado", [0.0, -1.0, None])
    def test_un_umbral_degenerado_cae_al_del_artefacto(self, degenerado) -> None:
        # Sin esto un umbral 0 mandaria el lote entero a "high": todo p >= 0.
        assert risk_level(0.10, degenerado) == risk_level(0.10, _FALLBACK_THRESHOLD)
        assert risk_level(0.10, degenerado) == "low"

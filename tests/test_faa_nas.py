"""Caracterización del cliente de estado NAS de la FAA (`ontimeai.faa_nas`).

CP-COB-01 del trabajo de Calidad (OnTimeAI-Calidad): el módulo estaba
desplegado con 0 % de cobertura y `_parse_nas_xml` es la función de mayor
complejidad ciclomática del paquete (27). Estos tests fijan el comportamiento
actual del parser frente a cada sección del feed —programas de parada en
tierra, programas de demora en tierra, demoras de llegada/salida y cierres—
y de la capa de cliente que lo envuelve: caché, límite de tasa, degradación
ante errores de red y traducción a las features `GDP_*` que consume el modelo.

No se toca la red: la sesión HTTP se reemplaza por un doble y el reloj por un
contador controlado. Los XML son sintéticos, con la forma del feed real; el
más completo vive en `tests/fixtures/faa_nas_status.xml`.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ontimeai import faa_nas
from ontimeai.faa_nas import FaaNasClient, _parse_delay_string, _parse_nas_xml

FIXTURE_XML = Path(__file__).resolve().parent / "fixtures" / "faa_nas_status.xml"


def _xml(cuerpo: str) -> bytes:
    """Envuelve fragmentos en la raíz del feed para armar fixtures inline."""
    return f"<AIRPORT_STATUS_INFORMATION>{cuerpo}</AIRPORT_STATUS_INFORMATION>".encode()


# ── _parse_delay_string ────────────────────────────────────────────────────


class TestParseDelayString:
    """La FAA escribe las demoras en prosa: '1 hour and 15 minutes'."""

    def test_horas_y_minutos_se_suman_en_minutos(self) -> None:
        # Arrange: la forma más larga que emite el feed.
        texto = "1 hour and 15 minutes"
        # Act
        minutos = _parse_delay_string(texto)
        # Assert
        assert minutos == 75.0

    def test_solo_minutos(self) -> None:
        assert _parse_delay_string("54 minutes") == 54.0

    def test_solo_horas_en_plural(self) -> None:
        assert _parse_delay_string("2 hours") == 120.0

    def test_un_numero_suelto_se_toma_como_minutos(self) -> None:
        assert _parse_delay_string("30") == 30.0

    def test_mayusculas_y_espacios_no_importan(self) -> None:
        assert _parse_delay_string("  1 Hour AND 5 Minutes ") == 65.0

    @pytest.mark.parametrize("texto", ["", "   ", None])
    def test_vacio_devuelve_none(self, texto) -> None:
        assert _parse_delay_string(texto) is None

    def test_texto_sin_numeros_devuelve_none(self) -> None:
        assert _parse_delay_string("unknown") is None

    @pytest.mark.parametrize("texto", ["0 minutes", "0"])
    def test_cero_se_trata_como_sin_demora(self, texto) -> None:
        # Una demora de cero no es información: se devuelve None, no 0.0.
        assert _parse_delay_string(texto) is None


# ── _parse_nas_xml ─────────────────────────────────────────────────────────


class TestParseNasXml:
    def test_un_programa_de_parada_en_tierra_se_lee_con_motivo_y_hora_de_fin(self) -> None:
        # Arrange
        xml = _xml(
            "<Delay_type><Name>Ground Stop Programs</Name><Ground_Stop_List>"
            "<Program><ARPT>DTW</ARPT><Reason>thunderstorms</Reason>"
            "<End_Time>3:30 pm EDT</End_Time></Program>"
            "</Ground_Stop_List></Delay_type>"
        )
        # Act
        estado = _parse_nas_xml(xml)
        # Assert
        assert estado == {
            "DTW": {"ground_stop": True, "reason": "thunderstorms", "end_time": "3:30 pm EDT"},
        }

    def test_un_programa_de_demora_en_tierra_traduce_promedio_y_maximo_a_minutos(self) -> None:
        # Arrange
        xml = _xml(
            "<Delay_type><Name>Ground Delay Programs</Name><Ground_Delay_List>"
            "<Ground_Delay><ARPT>BNA</ARPT><Reason>other</Reason>"
            "<Avg>54 minutes</Avg><Max>1 hour and 46 minutes</Max></Ground_Delay>"
            "</Ground_Delay_List></Delay_type>"
        )
        # Act
        estado = _parse_nas_xml(xml)
        # Assert
        assert estado["BNA"] == {
            "gdp": True, "reason": "other", "avg_delay_min": 54.0, "max_delay_min": 106.0,
        }

    def test_un_gdp_sin_promedio_ni_maximo_no_inventa_minutos(self) -> None:
        xml = _xml(
            "<Delay_type><Name>Ground Delay Programs</Name><Ground_Delay_List>"
            "<Ground_Delay><ARPT>BNA</ARPT><Reason>other</Reason><Avg></Avg></Ground_Delay>"
            "</Ground_Delay_List></Delay_type>"
        )
        estado = _parse_nas_xml(xml)
        assert estado["BNA"]["gdp"] is True
        assert "avg_delay_min" not in estado["BNA"]
        assert "max_delay_min" not in estado["BNA"]

    def test_la_seccion_de_llegadas_y_salidas_marca_ambas_banderas(self) -> None:
        # El feed real titula la sección "Arrival/Departure Delay Information":
        # el parser enciende las dos banderas para cada aeropuerto listado, y
        # lee Min/Max cuando son hijos directos de la entrada.
        xml = _xml(
            "<Delay_type><Name>Arrival/Departure Delay Information</Name>"
            "<Arrival_Departure_Delay_List><Delay><ARPT>SFO</ARPT><Reason>low ceilings</Reason>"
            "<Min>31 minutes</Min><Max>45 minutes</Max>"
            "</Delay></Arrival_Departure_Delay_List></Delay_type>"
        )
        estado = _parse_nas_xml(xml)
        assert estado["SFO"]["arrival_delay"] is True
        assert estado["SFO"]["departure_delay"] is True
        assert estado["SFO"]["reason"] == "low ceilings"
        assert estado["SFO"]["min_delay_min"] == 31.0
        assert estado["SFO"]["max_delay_min"] == 45.0

    def test_los_minutos_anidados_del_feed_real_no_se_leen(self) -> None:
        # Caracterización, no deseo: en el feed real Min/Max van dentro de
        # <Arrival_Departure Type="...">, y el parser los busca como hijos
        # directos de la entrada (`findtext("Min")`). Con la forma real quedan
        # las banderas y el motivo, pero sin minutos. Si alguien lo arregla,
        # este test tiene que cambiar a propósito.
        xml = _xml(
            "<Delay_type><Name>Arrival/Departure Delay Information</Name>"
            "<Arrival_Departure_Delay_List><Delay><ARPT>SFO</ARPT><Reason>low ceilings</Reason>"
            "<Arrival_Departure Type=\"Arrival\"><Min>31 minutes</Min><Max>45 minutes</Max>"
            "</Arrival_Departure></Delay></Arrival_Departure_Delay_List></Delay_type>"
        )
        estado = _parse_nas_xml(xml)
        assert estado["SFO"] == {
            "arrival_delay": True, "departure_delay": True, "reason": "low ceilings",
        }
        assert "min_delay_min" not in estado["SFO"]

    def test_una_seccion_solo_de_llegadas_no_enciende_salidas(self) -> None:
        xml = _xml(
            "<Delay_type><Name>Arrival Delay</Name><Lista><Delay><ARPT>LGA</ARPT>"
            "<Min>20 minutes</Min></Delay></Lista></Delay_type>"
        )
        estado = _parse_nas_xml(xml)
        assert estado["LGA"] == {"arrival_delay": True, "min_delay_min": 20.0}

    def test_una_seccion_solo_de_salidas_no_enciende_llegadas(self) -> None:
        xml = _xml(
            "<Delay_type><Name>Departure Delay</Name><Lista><Delay><ARPT>ORD</ARPT>"
            "<Reason></Reason><Max>1 hour</Max></Delay></Lista></Delay_type>"
        )
        estado = _parse_nas_xml(xml)
        # Un <Reason> vacío no deja clave; el máximo sí se traduce.
        assert estado["ORD"] == {"departure_delay": True, "max_delay_min": 60.0}

    def test_un_cierre_de_aeropuerto_solo_marca_closure(self) -> None:
        xml = _xml(
            "<Delay_type><Name>Airport Closures</Name><Airport_Closure_List><Airport>"
            "<ARPT>JFK</ARPT><Reason>AD AP CLSD</Reason><Reopen>22:00 UTC</Reopen>"
            "</Airport></Airport_Closure_List></Delay_type>"
        )
        assert _parse_nas_xml(xml) == {"JFK": {"closure": True}}

    def test_el_codigo_de_aeropuerto_se_normaliza_a_mayusculas(self) -> None:
        xml = _xml(
            "<Delay_type><Name>Ground Stop Programs</Name><Ground_Stop_List>"
            "<Program><ARPT> ewr </ARPT></Program></Ground_Stop_List></Delay_type>"
        )
        estado = _parse_nas_xml(xml)
        assert list(estado) == ["EWR"]
        # Sin <Reason> ni <End_Time> quedan cadenas vacías, no None.
        assert estado["EWR"] == {"ground_stop": True, "reason": "", "end_time": ""}

    def test_un_aeropuerto_en_dos_secciones_acumula_ambas_marcas(self) -> None:
        # Arrange: ATL con GDP y además demora de salida.
        xml = _xml(
            "<Delay_type><Name>Ground Delay Programs</Name><Ground_Delay_List>"
            "<Ground_Delay><ARPT>ATL</ARPT><Reason>low ceilings</Reason><Avg>75</Avg>"
            "</Ground_Delay></Ground_Delay_List></Delay_type>"
            "<Delay_type><Name>Departure Delay Information</Name><Lista><Delay>"
            "<ARPT>ATL</ARPT><Reason>wind</Reason><Max>30 minutes</Max></Delay></Lista>"
            "</Delay_type>"
        )
        # Act
        estado = _parse_nas_xml(xml)
        # Assert: el motivo lo pisa la última sección; las banderas se suman.
        assert estado["ATL"]["gdp"] is True
        assert estado["ATL"]["departure_delay"] is True
        assert estado["ATL"]["avg_delay_min"] == 75.0
        assert estado["ATL"]["max_delay_min"] == 30.0
        assert estado["ATL"]["reason"] == "wind"

    @pytest.mark.parametrize(
        "cuerpo",
        [
            (
                "<Delay_type><Name>Ground Stop Programs</Name><Ground_Stop_List>"
                "<Program><ARPT></ARPT><Reason>x</Reason></Program></Ground_Stop_List></Delay_type>"
            ),
            (
                "<Delay_type><Name>Ground Delay Programs</Name><Ground_Delay_List>"
                "<Ground_Delay><ARPT>  </ARPT></Ground_Delay></Ground_Delay_List></Delay_type>"
            ),
            "<Delay_type><Name>Arrival Delay</Name><L><D><ARPT/></D></L></Delay_type>",
            "<Delay_type><Name>Airport Closures</Name><L><A><ARPT/></A></L></Delay_type>",
        ],
        ids=["ground_stop", "ground_delay", "arrival", "closure"],
    )
    def test_una_entrada_sin_codigo_de_aeropuerto_se_ignora(self, cuerpo: str) -> None:
        assert _parse_nas_xml(_xml(cuerpo)) == {}

    def test_una_seccion_sin_nombre_se_salta(self) -> None:
        xml = _xml(
            "<Delay_type><Ground_Stop_List><Program><ARPT>DTW</ARPT></Program>"
            "</Ground_Stop_List></Delay_type>"
        )
        assert _parse_nas_xml(xml) == {}

    def test_una_seccion_de_tipo_desconocido_se_ignora(self) -> None:
        xml = _xml(
            "<Delay_type><Name>Special Traffic Management</Name><Lista><Item>"
            "<ARPT>DEN</ARPT></Item></Lista></Delay_type>"
        )
        assert _parse_nas_xml(xml) == {}

    def test_un_feed_sin_eventos_devuelve_vacio(self) -> None:
        # Un cielo sin programas activos: raíz legítima y sin secciones.
        assert _parse_nas_xml(b"<AIRPORT_STATUS_INFORMATION/>") == {}

    def test_xml_malformado_devuelve_vacio_y_lo_deja_en_el_log(self, capsys) -> None:
        # Arrange
        roto = b"<AIRPORT_STATUS_INFORMATION><Delay_type>"
        # Act
        estado = _parse_nas_xml(roto)
        # Assert: degrada a "sin datos" en vez de propagar la excepción.
        assert estado == {}
        assert "XML parse error" in capsys.readouterr().out

    def test_bytes_vacios_tambien_degradan_a_vacio(self) -> None:
        assert _parse_nas_xml(b"") == {}

    def test_el_fixture_completo_se_lee_entero(self) -> None:
        # Arrange: las cuatro secciones del feed, con ATL en dos de ellas.
        xml = FIXTURE_XML.read_bytes()
        # Act
        estado = _parse_nas_xml(xml)
        # Assert
        assert set(estado) == {"DTW", "EWR", "BNA", "ATL", "SFO", "JFK"}
        assert estado["DTW"] == {
            "ground_stop": True, "reason": "thunderstorms", "end_time": "3:30 pm EDT",
        }
        assert estado["EWR"]["reason"] == "volume"
        assert estado["BNA"] == {
            "gdp": True, "reason": "other", "avg_delay_min": 54.0, "max_delay_min": 106.0,
        }
        # ATL aparece en el GDP y en la sección de llegadas/salidas: las marcas
        # se acumulan, el motivo lo pisa la última sección y los minutos del
        # GDP sobreviven (los de la demora van anidados y no se leen).
        assert estado["ATL"] == {
            "gdp": True, "avg_delay_min": 75.0, "max_delay_min": 130.0,
            "arrival_delay": True, "departure_delay": True, "reason": "wind",
        }
        assert estado["SFO"] == {
            "arrival_delay": True, "departure_delay": True, "reason": "low ceilings",
        }
        assert estado["JFK"] == {"closure": True}

    def test_un_aeropuerto_presente_en_las_cuatro_secciones(self) -> None:
        # Arrange: la sección de demoras va primero para que las otras tres
        # encuentren al aeropuerto ya creado y sólo le agreguen marcas.
        xml = _xml(
            "<Delay_type><Name>Arrival Delay</Name><L><D><ARPT>ATL</ARPT>"
            "<Min>10 minutes</Min></D></L></Delay_type>"
            "<Delay_type><Name>Ground Stop Programs</Name><Ground_Stop_List>"
            "<Program><ARPT>ATL</ARPT><Reason>storms</Reason></Program></Ground_Stop_List>"
            "</Delay_type>"
            "<Delay_type><Name>Ground Delay Programs</Name><Ground_Delay_List>"
            "<Ground_Delay><ARPT>ATL</ARPT><Avg>45 minutes</Avg></Ground_Delay>"
            "</Ground_Delay_List></Delay_type>"
            "<Delay_type><Name>Airport Closures</Name><L><A><ARPT>ATL</ARPT></A></L>"
            "</Delay_type>"
        )
        # Act
        estado = _parse_nas_xml(xml)
        # Assert: una sola entrada con todo acumulado.
        assert list(estado) == ["ATL"]
        assert estado["ATL"] == {
            "arrival_delay": True, "min_delay_min": 10.0,
            "ground_stop": True, "end_time": "",
            "gdp": True, "avg_delay_min": 45.0, "reason": "",
            "closure": True,
        }


# ── FaaNasClient ───────────────────────────────────────────────────────────


class _RespuestaFalsa:
    def __init__(self, contenido: bytes, *, falla: Exception | None = None) -> None:
        self.content = contenido
        self._falla = falla

    def raise_for_status(self) -> None:
        if self._falla is not None:
            raise self._falla


class _SesionFalsa:
    """Doble de `requests.Session`: registra las llamadas y devuelve lo pactado."""

    def __init__(self, respuestas: list) -> None:
        self._respuestas = list(respuestas)
        self.llamadas: list[dict] = []
        self.headers: dict = {}

    def get(self, url: str, **kwargs):
        self.llamadas.append({"url": url, **kwargs})
        respuesta = self._respuestas.pop(0)
        if isinstance(respuesta, Exception):
            raise respuesta
        return respuesta


@pytest.fixture
def reloj(monkeypatch):
    """Reloj controlado para `faa_nas.time`: avanza sólo cuando el test lo pide."""
    estado = SimpleNamespace(ahora=1_000.0, dormido=[])

    def _time() -> float:
        return estado.ahora

    def _sleep(segundos: float) -> None:
        estado.dormido.append(segundos)
        estado.ahora += segundos

    monkeypatch.setattr(faa_nas, "time", SimpleNamespace(time=_time, sleep=_sleep))
    return estado


def _cliente_con(respuestas: list) -> tuple[FaaNasClient, _SesionFalsa]:
    cliente = FaaNasClient()
    sesion = _SesionFalsa(respuestas)
    cliente._session = sesion
    return cliente, sesion


class TestFaaNasClient:
    def test_la_sesion_se_identifica_como_uso_academico(self) -> None:
        cliente = FaaNasClient()
        assert cliente._session.headers["User-Agent"] == "OnTimeAI-Academic/1.0"
        assert cliente._cache == {}

    def test_pide_el_feed_oficial_en_xml_y_lo_parsea(self, reloj) -> None:
        # Arrange
        cliente, sesion = _cliente_con([_RespuestaFalsa(FIXTURE_XML.read_bytes())])
        # Act
        estado = cliente.get_nas_status()
        # Assert
        assert sesion.llamadas[0]["url"] == faa_nas.FAA_STATUS_URL
        assert sesion.llamadas[0]["headers"] == {"Accept": "application/xml"}
        assert sesion.llamadas[0]["timeout"] == 15
        assert estado["BNA"]["gdp"] is True

    def test_la_segunda_consulta_dentro_del_ttl_sale_del_cache(self, reloj) -> None:
        # Arrange: una sola respuesta disponible; la segunda llamada rompería.
        cliente, sesion = _cliente_con([_RespuestaFalsa(FIXTURE_XML.read_bytes())])
        primero = cliente.get_nas_status()
        reloj.ahora += 60
        # Act
        segundo = cliente.get_nas_status()
        # Assert
        assert segundo is primero
        assert len(sesion.llamadas) == 1

    def test_pasado_el_ttl_vuelve_a_consultar(self, reloj) -> None:
        cliente, sesion = _cliente_con([
            _RespuestaFalsa(FIXTURE_XML.read_bytes()),
            _RespuestaFalsa(b"<AIRPORT_STATUS_INFORMATION/>"),
        ])
        cliente.get_nas_status()
        reloj.ahora += cliente._cache_ttl + 1
        assert cliente.get_nas_status() == {}
        assert len(sesion.llamadas) == 2

    def test_respeta_dos_segundos_entre_consultas(self, reloj) -> None:
        # Arrange: la última consulta fue hace medio segundo.
        cliente, _ = _cliente_con([_RespuestaFalsa(b"<AIRPORT_STATUS_INFORMATION/>")])
        cliente._last_request = reloj.ahora - 0.5
        # Act
        cliente.get_nas_status()
        # Assert: dormió lo que faltaba y dejó marcada la hora de la consulta.
        assert reloj.dormido == [pytest.approx(1.5)]
        assert cliente._last_request == reloj.ahora

    def test_no_duerme_si_ya_pasaron_los_dos_segundos(self, reloj) -> None:
        cliente, _ = _cliente_con([_RespuestaFalsa(b"<AIRPORT_STATUS_INFORMATION/>")])
        cliente._last_request = reloj.ahora - 10
        cliente.get_nas_status()
        assert reloj.dormido == []

    def test_un_error_de_red_degrada_a_vacio_sin_cachear(self, reloj, capsys) -> None:
        # Arrange: primero falla la red, después responde.
        cliente, sesion = _cliente_con([
            ConnectionError("sin red"),
            _RespuestaFalsa(FIXTURE_XML.read_bytes()),
        ])
        # Act
        caido = cliente.get_nas_status()
        recuperado = cliente.get_nas_status()
        # Assert: la falla no se cachea, así que la siguiente consulta sale.
        assert caido == {}
        assert "fetch failed" in capsys.readouterr().out
        assert recuperado["DTW"]["ground_stop"] is True
        assert len(sesion.llamadas) == 2

    def test_un_http_de_error_tambien_degrada_a_vacio(self, reloj) -> None:
        cliente, _ = _cliente_con([
            _RespuestaFalsa(b"", falla=RuntimeError("503 Service Unavailable")),
        ])
        assert cliente.get_nas_status() == {}
        assert cliente._cache == {}


FEATURES_EN_CERO = {
    "GDP_FLAG": 0, "GDP_DELAY_MIN": 0.0, "GROUND_STOP": 0,
    "ARRIVAL_DELAY": 0, "DEPARTURE_DELAY": 0, "CLOSURE": 0,
}


class TestGdpFeatures:
    """La traducción del estado NAS a las columnas que consume el modelo."""

    def _features(self, estado: dict, iata: str) -> dict:
        cliente = FaaNasClient()
        cliente.get_nas_status = lambda: estado
        return cliente.get_gdp_features(iata)

    def test_un_aeropuerto_sin_eventos_recibe_todo_en_cero(self) -> None:
        assert self._features({"BNA": {"gdp": True}}, "ATL") == FEATURES_EN_CERO

    def test_sin_respuesta_de_la_faa_todo_queda_en_cero(self) -> None:
        assert self._features({}, "ATL") == FEATURES_EN_CERO

    def test_el_codigo_se_busca_en_mayusculas(self) -> None:
        assert self._features({"ATL": {"closure": True}}, "atl")["CLOSURE"] == 1

    def test_un_gdp_con_promedio_informa_bandera_y_minutos(self) -> None:
        # Arrange
        estado = {"BNA": {"gdp": True, "avg_delay_min": 54.0, "max_delay_min": 106.0}}
        # Act
        features = self._features(estado, "BNA")
        # Assert
        assert features["GDP_FLAG"] == 1
        assert features["GDP_DELAY_MIN"] == 54.0
        assert features["GROUND_STOP"] == 0

    def test_sin_promedio_usa_el_punto_medio_entre_minimo_y_maximo(self) -> None:
        estado = {"SFO": {"arrival_delay": True, "min_delay_min": 30.0, "max_delay_min": 50.0}}
        features = self._features(estado, "SFO")
        assert features["ARRIVAL_DELAY"] == 1
        assert features["GDP_FLAG"] == 0
        assert features["GDP_DELAY_MIN"] == 40.0

    def test_un_promedio_ilegible_no_cae_al_punto_medio(self) -> None:
        # `avg_delay_min` presente pero None (texto que no se pudo parsear):
        # la implementación actual no cae al rango min/max y deja 0.0.
        estado = {"BNA": {"gdp": True, "avg_delay_min": None, "min_delay_min": 10.0,
                          "max_delay_min": 20.0}}
        assert self._features(estado, "BNA")["GDP_DELAY_MIN"] == 15.0

    def test_solo_un_extremo_del_rango_no_alcanza_para_estimar(self) -> None:
        estado = {"SFO": {"arrival_delay": True, "max_delay_min": 50.0}}
        assert self._features(estado, "SFO")["GDP_DELAY_MIN"] == 0.0

    def test_todas_las_banderas_a_la_vez(self) -> None:
        estado = {"ATL": {"ground_stop": True, "gdp": True, "arrival_delay": True,
                          "departure_delay": True, "closure": True, "avg_delay_min": 90.0}}
        assert self._features(estado, "ATL") == {
            "GDP_FLAG": 1, "GDP_DELAY_MIN": 90.0, "GROUND_STOP": 1,
            "ARRIVAL_DELAY": 1, "DEPARTURE_DELAY": 1, "CLOSURE": 1,
        }

    def test_de_punta_a_punta_desde_el_xml(self, reloj) -> None:
        # Arrange: cliente real con la sesión reemplazada.
        cliente, _ = _cliente_con([_RespuestaFalsa(FIXTURE_XML.read_bytes())])
        # Act
        atl = cliente.get_gdp_features("ATL")
        jfk = cliente.get_gdp_features("JFK")
        # Assert
        assert atl["GDP_FLAG"] == 1 and atl["GDP_DELAY_MIN"] == 75.0
        assert atl["DEPARTURE_DELAY"] == 1 and atl["ARRIVAL_DELAY"] == 1
        assert atl["GROUND_STOP"] == 0 and atl["CLOSURE"] == 0
        assert jfk["CLOSURE"] == 1 and jfk["GDP_FLAG"] == 0

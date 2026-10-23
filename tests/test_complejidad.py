"""scripts/complejidad.py: el trinquete de la baseline.

La regla: una función nueva por encima del umbral falla aunque lleve `# noqa`;
una exenta que crece falla; una exenta que baja se puede sacar; la baseline
sólo puede achicarse.
"""
from __future__ import annotations

import json

import pytest

from scripts import complejidad


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    ruta = tmp_path / ".complejidad-baseline.json"
    monkeypatch.setattr(complejidad, "BASELINE", ruta)
    return ruta


def _con_violaciones(monkeypatch, datos: dict[str, int]) -> None:
    monkeypatch.setattr(complejidad, "violaciones", lambda: dict(datos))


def test_sin_baseline_toda_violacion_es_nueva(baseline, monkeypatch, capsys):
    # Arrange: dos funciones por encima del umbral y ninguna exenta
    _con_violaciones(monkeypatch, {"a.py::f": 12, "b.py::g": 30})
    # Act
    codigo = complejidad.check()
    # Assert
    assert codigo == 1
    salida = capsys.readouterr().out
    assert "2 función(es) nueva(s)" in salida and "b.py::g  CC=30" in salida


def test_exentas_iguales_pasan(baseline, monkeypatch):
    complejidad.guardar_baseline({"a.py::f": 12})
    _con_violaciones(monkeypatch, {"a.py::f": 12})
    assert complejidad.check() == 0


def test_exenta_que_crece_falla(baseline, monkeypatch, capsys):
    complejidad.guardar_baseline({"a.py::f": 12})
    _con_violaciones(monkeypatch, {"a.py::f": 13})
    assert complejidad.check() == 1
    assert "12 → 13" in capsys.readouterr().out


def test_exenta_que_baja_pasa_y_avisa(baseline, monkeypatch, capsys):
    complejidad.guardar_baseline({"a.py::f": 12})
    _con_violaciones(monkeypatch, {})
    assert complejidad.check() == 0
    assert "Ya no superan el umbral" in capsys.readouterr().out


def test_noqa_no_exime_porque_violaciones_ignora_noqa(baseline, monkeypatch):
    # El check compara contra la baseline, no contra las marcas: una función
    # nueva con noqa sigue apareciendo en violaciones() y falla.
    complejidad.guardar_baseline({})
    _con_violaciones(monkeypatch, {"nuevo.py::h": 11})
    assert complejidad.check() == 1


def test_actualizar_baseline_solo_achica(baseline, monkeypatch):
    complejidad.guardar_baseline({"a.py::f": 12, "b.py::g": 20})
    # Arrange: una desapareció y otra creció
    _con_violaciones(monkeypatch, {"b.py::g": 25})
    # Act: se rechaza porque hay crecimiento
    assert complejidad.main(["--actualizar-baseline"]) == 1
    assert json.loads(baseline.read_text())["exentas"] == {"b.py::g": 20, "a.py::f": 12}
    # Act: sin crecimiento, se reescribe más chica
    _con_violaciones(monkeypatch, {"b.py::g": 20})
    assert complejidad.main(["--actualizar-baseline"]) == 0
    assert json.loads(baseline.read_text())["exentas"] == {"b.py::g": 20}


def test_baseline_con_otro_umbral_se_rechaza(baseline):
    baseline.write_text(json.dumps({"umbral": complejidad.UMBRAL + 5, "exentas": {}}))
    with pytest.raises(SystemExit):
        complejidad.cargar_baseline()


def test_resumen_markdown_lista_el_top(baseline):
    rep = {"bloques": 3, "promedio": 7.5, "sobre_umbral": 1, "sobre_20": 0,
           "por_rango": {"A": 2, "C": 1},
           "detalle": [{"nombre": "f", "cc": 14, "archivo": "a.py", "linea": 3},
                       {"nombre": "g", "cc": 4, "archivo": "b.py", "linea": 9}]}
    md = complejidad.resumen_markdown(rep, {"a.py::f": 14})
    assert "| 3 | 7.5 | 1 | 0 | 1 |" in md and "| `f` | 14 | `a.py:3` |" in md and "A=2, C=1" in md


def test_reporte_radon_corre_sobre_el_alcance_real():
    # Prueba de humo contra el código real: radon tiene que poder leer todo el
    # alcance productivo y devolver bloques.
    rep = complejidad.reporte_radon()
    assert rep["bloques"] > 100 and rep["promedio"] > 0
    assert set(rep["por_rango"]) <= set("ABCDEF")

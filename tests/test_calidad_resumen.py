"""scripts/calidad_resumen.py: el resumen refleja los insumos y tolera que falten."""
from __future__ import annotations

import json

from scripts import calidad_resumen as cr

COBERTURA_XML = """<?xml version="1.0" ?>
<coverage line-rate="0.5" branch-rate="0.25" lines-covered="50" lines-valid="100"
          branches-covered="5" branches-valid="20" version="7.0">
  <packages><package name="."><classes>
    <class name="grande.py" filename="grande.py" line-rate="0.2"><lines>{lineas_grande}</lines></class>
    <class name="chico.py" filename="chico.py" line-rate="0.0"><lines><line number="1" hits="0"/></lines></class>
  </classes></package></packages>
</coverage>
"""


def _xml(tmp_path):
    lineas = "".join(f'<line number="{i}" hits="{1 if i < 12 else 0}"/>' for i in range(1, 61))
    p = tmp_path / "coverage.xml"
    p.write_text(COBERTURA_XML.format(lineas_grande=lineas))
    return p


def test_cobertura_totales_y_peor_cubiertos(tmp_path):
    # Arrange
    xml = _xml(tmp_path)
    # Act
    cob = cr.leer_cobertura(xml)
    # Assert: totales del nodo raíz y sólo el archivo con ≥ 50 líneas en la lista
    assert cob["lineas_pct"] == 50.0 and cob["ramas_pct"] == 25.0
    assert cob["lineas"] == 100 and cob["ramas_cubiertas"] == 5
    assert cob["peor_cubiertos"] == [(20.0, "grande.py", 60)]


def test_duplicacion_lee_el_total_de_jscpd(tmp_path):
    p = tmp_path / "jscpd-report.json"
    p.write_text(json.dumps({"statistics": {"total": {
        "lines": 7172, "duplicatedLines": 64, "percentage": 0.89, "clones": 7, "sources": 34}}}))
    dup = cr.leer_duplicacion(p)
    assert dup == {"pct_lineas": 0.89, "lineas_duplicadas": 64, "lineas": 7172, "clones": 7, "archivos": 34}


def test_markdown_con_todo(tmp_path):
    cob = cr.leer_cobertura(_xml(tmp_path))
    dup = {"pct_lineas": 0.89, "lineas_duplicadas": 64, "lineas": 7172, "clones": 7, "archivos": 34}
    md = cr.markdown(cob, dup, "| Diff | 90 % |")
    assert "| 50.0 % (50/100) | 25.0 % (5/20) | 0.89 % (64/7172 líneas, 7 clones) |" in md
    assert "`grande.py`: 20.0 % de 60 líneas" in md
    assert "Cobertura sobre las líneas cambiadas" in md and "| Diff | 90 % |" in md


def test_markdown_sin_insumos_no_explota():
    md = cr.markdown(None, None, None)
    assert "sin reporte" in md and "líneas cambiadas" not in md


def test_main_agrega_al_archivo_de_salida(tmp_path):
    salida = tmp_path / "resumen.md"
    salida.write_text("previo\n")
    codigo = cr.main(["--cobertura", str(_xml(tmp_path)), "--jscpd", str(tmp_path / "no.json"),
                      "--diff-cover", str(tmp_path / "no.md"), "--salida", str(salida)])
    assert codigo == 0
    texto = salida.read_text()
    assert texto.startswith("previo\n") and "### Calidad del run" in texto and "50.0 %" in texto

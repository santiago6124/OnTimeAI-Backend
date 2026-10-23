"""feature_engineering_v7/adsb_lineage.py: el setup del CLI (regresión de #90).

`main()` llamaba a `download_faa_registry`, que nunca existió: el módulo nació
con la descarga llamada `download_aircraft_db` (base de OpenSky) y el CLI quedó
con el nombre de un borrador. Nadie volvió a correr `--setup` y ningún test
pasaba por `main()`, así que el NameError vivió desde el primer commit hasta
que el gate de análisis estático lo marcó (F821).
"""
from __future__ import annotations

import sys

import pandas as pd

from feature_engineering_v7 import adsb_lineage


def test_setup_descarga_y_construye_el_mapa_sin_explotar(tmp_path, monkeypatch, capsys):
    # Arrange: dobles de la descarga y del mapa, directorio de datos temporal
    llamadas: list = []
    db = tmp_path / "opensky_aircraft_db.csv"
    monkeypatch.setattr(adsb_lineage, "DATA_DIR", tmp_path / "data_raw")
    monkeypatch.setattr(adsb_lineage, "download_aircraft_db",
                        lambda *a, **k: llamadas.append("descarga") or db)
    monkeypatch.setattr(adsb_lineage, "build_tail_icao24_map",
                        lambda ruta: llamadas.append(("mapa", ruta))
                        or pd.DataFrame({"tail_num": ["N999DL"], "icao24": ["a1b2c3"]}))
    monkeypatch.setattr(sys, "argv", ["adsb_lineage", "--setup"])

    # Act: criterio de aceptación fijado en #90 antes de corregir
    adsb_lineage.main()

    # Assert: descarga, construye el mapa con esa ruta, y sale sin tocar el parquet
    assert llamadas == ["descarga", ("mapa", db)]
    salida = capsys.readouterr().out
    assert "Mapeo construido: 1 aviones" in salida
    assert "Setup completo" in salida
    assert (tmp_path / "data_raw").is_dir()

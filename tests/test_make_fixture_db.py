"""`scripts/make_fixture_db.py`: la base contra la que corre k6."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ontimeai.live import open_db
from scripts import make_fixture_db


@pytest.fixture
def origen(tmp_path: Path) -> Path:
    """Una base chica con el esquema real, en lugar de los 81 MB del repo."""
    ruta = tmp_path / "origen.db"
    con = open_db(ruta)
    con.execute(
        "INSERT INTO weather_obs (station, valid_utc, tmpc, sknt, vsby) VALUES (?,?,?,?,?)",
        ("ATL", "2026-10-22T14:52:00", 21.0, 7.0, 10.0),
    )
    con.commit()
    con.close()
    return ruta


def test_copia_y_siembra_lo_pedido(tmp_path: Path, origen: Path, capsys) -> None:
    destino = tmp_path / "salida" / "fixture.db"

    codigo = make_fixture_db.main([
        "--destino", str(destino), "--vuelos", "12", "--semilla", "5",
        "--origen", str(origen),
    ])

    assert codigo == 0
    assert destino.is_file()
    salida = capsys.readouterr().out
    assert "12 vuelos de hoy sembrados" in salida
    assert "12 predicciones" in salida and "36 filas SHAP" in salida
    assert "semilla 5" in salida

    con = sqlite3.connect(destino)
    try:
        assert con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 12
        # Lo que había en el origen sigue ahí: es una copia, no una base nueva.
        assert con.execute("SELECT COUNT(*) FROM weather_obs").fetchone()[0] == 1
    finally:
        con.close()
    # El origen no se toca.
    con = sqlite3.connect(origen)
    try:
        assert con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
    finally:
        con.close()


def test_sobreescribe_un_destino_anterior(tmp_path: Path, origen: Path) -> None:
    destino = tmp_path / "fixture.db"
    make_fixture_db.armar_fixture(destino, vuelos=3, semilla=1, origen=origen)
    make_fixture_db.armar_fixture(destino, vuelos=5, semilla=1, origen=origen)
    con = sqlite3.connect(destino)
    try:
        assert con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 5
    finally:
        con.close()


def test_falla_si_el_origen_no_existe(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        make_fixture_db.armar_fixture(
            tmp_path / "f.db", vuelos=1, semilla=1, origen=tmp_path / "no-existe.db"
        )


def test_rechaza_cero_vuelos(tmp_path: Path, origen: Path) -> None:
    with pytest.raises(SystemExit):
        make_fixture_db.main([
            "--destino", str(tmp_path / "f.db"), "--vuelos", "0", "--origen", str(origen),
        ])


def test_el_default_apunta_a_la_base_del_repo() -> None:
    assert make_fixture_db.BASE_BUNDLEADA.name == "live_data.db"
    assert make_fixture_db.BASE_BUNDLEADA.parent == make_fixture_db.RAIZ

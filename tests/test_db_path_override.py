"""`DB_PATH_OVERRIDE` / `USERS_DB_PATH_OVERRIDE`: la API contra una base fixture.

Se prueba la función de resolución con un diccionario en vez de recargar el
módulo `api`: `importlib.reload(api)` resetea `DB_PATH` y rompe los tests que
corren después (ver el historial de `fa7b94f`).
"""
from __future__ import annotations

from pathlib import Path

import api

DEFAULT = Path("/donde/estaba/live_data.db")


def test_sin_la_variable_queda_el_default() -> None:
    assert api._ruta_de_base({}, "DB_PATH_OVERRIDE", DEFAULT) == DEFAULT


def test_la_variable_vacia_o_en_blanco_no_cuenta() -> None:
    assert api._ruta_de_base({"DB_PATH_OVERRIDE": ""}, "DB_PATH_OVERRIDE", DEFAULT) == DEFAULT
    assert api._ruta_de_base({"DB_PATH_OVERRIDE": "   "}, "DB_PATH_OVERRIDE", DEFAULT) == DEFAULT


def test_con_la_variable_gana_la_ruta_del_entorno() -> None:
    entorno = {"DB_PATH_OVERRIDE": " /tmp/fixture.db ", "USERS_DB_PATH_OVERRIDE": "/tmp/u.db"}
    assert api._ruta_de_base(entorno, "DB_PATH_OVERRIDE", DEFAULT) == Path("/tmp/fixture.db")
    assert api._ruta_de_base(entorno, "USERS_DB_PATH_OVERRIDE", DEFAULT) == Path("/tmp/u.db")


def test_cada_variable_mira_solo_la_suya() -> None:
    entorno = {"USERS_DB_PATH_OVERRIDE": "/tmp/u.db"}
    assert api._ruta_de_base(entorno, "DB_PATH_OVERRIDE", DEFAULT) == DEFAULT


def test_el_modulo_arranca_con_la_base_bundleada_si_nadie_la_pisa(monkeypatch) -> None:
    """Sin GCS_BUCKET ni override, la API sirve el live_data.db del repo."""
    monkeypatch.delenv("DB_PATH_OVERRIDE", raising=False)
    assert api._ruta_de_base({}, "DB_PATH_OVERRIDE", api._BUNDLED_DB) == api._BUNDLED_DB
    assert api._BUNDLED_DB.name == "live_data.db"

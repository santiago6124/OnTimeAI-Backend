"""Lo que necesita el modo viajero del backend.

Dos superficies nuevas:

  - `/users/me/flights`: los vuelos que el viajero guarda. Tienen que quedar
    **históricamente**, y eso obliga a guardarlos en la base de usuarios y no
    en `live_data.db`, que los jobs reescriben entera cada ciclo y que purga a
    los 14 días.

  - `/weather/alerts`: los aeropuertos con mal tiempo ahora, filtrables por
    severidad.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import api  # noqa: E402


# ── severidad del clima ──────────────────────────────────────────────────────

def _obs(**campos):
    base = {"vsby": 10.0, "sknt": 5.0, "gust": None, "wx_precip_flag": 0}
    base.update(campos)
    return base


def test_cielo_despejado_es_bajo() -> None:
    nivel, motivos = api._severidad_clima(_obs())
    assert nivel == "bajo"
    assert motivos == []


def test_visibilidad_muy_baja_es_alto() -> None:
    nivel, motivos = api._severidad_clima(_obs(vsby=0.5))
    assert nivel == "alto"
    assert any("visibilidad" in m for m in motivos)


def test_visibilidad_intermedia_es_medio() -> None:
    assert api._severidad_clima(_obs(vsby=2.0))[0] == "medio"


def test_viento_fuerte_es_alto() -> None:
    assert api._severidad_clima(_obs(sknt=35.0))[0] == "alto"


def test_la_rafaga_cuenta_aunque_el_sostenido_sea_bajo() -> None:
    """Una ráfaga de 40 kt cierra una pista igual que un viento sostenido."""
    assert api._severidad_clima(_obs(sknt=8.0, gust=40.0))[0] == "alto"


def test_un_motivo_alto_manda_sobre_uno_medio() -> None:
    nivel, motivos = api._severidad_clima(_obs(vsby=0.5, sknt=22.0))
    assert nivel == "alto"
    assert len(motivos) == 2


def test_la_precipitacion_sola_no_llega_a_alto() -> None:
    """Llueve en medio aeropuerto del país todos los días."""
    assert api._severidad_clima(_obs(wx_precip_flag=1))[0] == "medio"


def test_tolera_mediciones_ausentes() -> None:
    """El METAR puede no traer visibilidad ni viento; eso no es una alerta."""
    nivel, _ = api._severidad_clima(
        {"vsby": None, "sknt": None, "gust": None, "wx_precip_flag": 0})
    assert nivel == "bajo"


# ── ruteo ────────────────────────────────────────────────────────────────────

def test_alerts_se_declara_antes_que_el_aeropuerto() -> None:
    """
    `/weather/alerts` y `/weather/{airport_code}` compiten por la misma forma.
    FastAPI resuelve por orden de declaración, así que si el parametrizado
    quedara primero, "alerts" llegaría como código de aeropuerto y la respuesta
    sería un 404 —o peor, un 400 de validación— en vez de las alertas.
    """
    fuente = (Path(api.__file__)).read_text("utf-8")
    assert fuente.index('@app.get("/weather/alerts")') < fuente.index(
        '@app.get("/weather/{airport_code}")')


def test_el_filtro_de_severidad_rechaza_lo_que_no_conoce() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        api.weather_alerts(severity="catastrofico")
    assert exc.value.status_code == 400


# ── persistencia de los guardados ────────────────────────────────────────────

def test_los_guardados_viven_en_la_base_de_usuarios() -> None:
    """
    No en `live_data.db`: esa la reescriben los jobs entera en cada ciclo y
    tiene retención de 14 días. Un vuelo guardado tiene que sobrevivir a eso.
    """
    fuente = (Path(api.__file__)).read_text("utf-8")
    i = fuente.index("CREATE TABLE IF NOT EXISTS user_saved_flights")
    j = fuente.index("CREATE TABLE IF NOT EXISTS users")
    # Las dos están en el mismo bloque de creación de la base de usuarios.
    assert abs(i - j) < 4000


def test_guarda_copia_de_lo_que_el_viajero_reconoce() -> None:
    """
    Cuando el vuelo se purgue de `flights` a los 14 días, la lista tiene que
    seguir mostrando número, ruta y horario. Sin la copia queda un
    identificador y nada más.
    """
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    fuente = (Path(api.__file__)).read_text("utf-8")
    ini = fuente.index("CREATE TABLE IF NOT EXISTS user_saved_flights")
    fin = fuente.index(");", ini) + 2
    con.executescript(fuente[ini:fin])

    cols = {r["name"] for r in con.execute("PRAGMA table_info(user_saved_flights)")}
    for necesaria in ("ident_iata", "origin", "dest", "scheduled_out_utc"):
        assert necesaria in cols, f"falta {necesaria}: el guardado quedaría mudo"


def test_la_clave_impide_guardar_dos_veces_el_mismo_vuelo() -> None:
    con = sqlite3.connect(":memory:")
    fuente = (Path(api.__file__)).read_text("utf-8")
    ini = fuente.index("CREATE TABLE IF NOT EXISTS user_saved_flights")
    fin = fuente.index(");", ini) + 2
    con.executescript(fuente[ini:fin])

    con.execute("INSERT INTO user_saved_flights (username, fa_flight_id, saved_at_utc)"
                " VALUES ('ana','ABC123','2026-10-02T00:00:00Z')")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO user_saved_flights (username, fa_flight_id, saved_at_utc)"
                    " VALUES ('ana','ABC123','2026-10-02T01:00:00Z')")

    # Pero dos personas sí pueden guardar el mismo vuelo.
    con.execute("INSERT INTO user_saved_flights (username, fa_flight_id, saved_at_utc)"
                " VALUES ('beto','ABC123','2026-10-02T01:00:00Z')")
    assert con.execute("SELECT COUNT(*) FROM user_saved_flights").fetchone()[0] == 2

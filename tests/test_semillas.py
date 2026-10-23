"""La siembra de vuelos de hoy, que alimenta el fixture de la API y la base de k6.

Lo que importa proteger: que lo sembrado caiga en la ventana que `/flights`
considera activa —si no, la base fixture no ejercita nada y la prueba de carga
mide una lista vacía—, que sea reproducible por semilla y que la cascada
encuentre la pierna siguiente.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import api
from ontimeai.live import open_db
from ontimeai.semillas import Siembra, sembrar_vuelo_de_hoy, sembrar_vuelos_de_hoy

AHORA = datetime(2026, 10, 22, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def base(tmp_path: Path) -> Path:
    ruta = tmp_path / "live_data.db"
    open_db(ruta).close()
    return ruta


def _conexion(ruta: Path) -> sqlite3.Connection:
    con = sqlite3.connect(ruta)
    con.row_factory = sqlite3.Row
    return con


def _filas(ruta: Path, consulta: str, *params) -> list[sqlite3.Row]:
    con = _conexion(ruta)
    try:
        return con.execute(consulta, params).fetchall()
    finally:
        con.close()


def test_siembra_lo_que_dice_que_siembra(base: Path) -> None:
    siembra = sembrar_vuelos_de_hoy(base, n=25, semilla=7)

    assert isinstance(siembra, Siembra)
    assert siembra.vuelos == siembra.predicciones == siembra.piernas_siguientes == 25
    assert siembra.shap == 75
    assert len(siembra.ids) == 25 and siembra.ids[0] == "SEED-0001"

    contar = lambda tabla, filtro="": _filas(  # noqa: E731
        base, f"SELECT COUNT(*) AS n FROM {tabla} {filtro}")[0]["n"]
    assert contar("flights", "WHERE fa_flight_id LIKE 'SEED-%' AND origin = 'ATL'") == 25
    assert contar("flights", "WHERE fa_flight_id LIKE 'SEED-%-R'") == 25
    assert contar("predictions") == 25
    assert contar("prediction_shap") == 75
    assert contar("actuals") == siembra.resultados


def test_todos_los_vuelos_caen_en_la_ventana_activa_de_flights(base: Path) -> None:
    """Es la razón de ser de la siembra: `/flights` tiene que verlos a todos."""
    siembra = sembrar_vuelos_de_hoy(base, n=40, semilla=1)

    con = _conexion(base)
    try:
        activos = {r["fa_flight_id"] for r in api._latest_predictions_active(con)}
    finally:
        con.close()

    assert activos == set(siembra.ids)


def test_solo_los_que_ya_aterrizaron_tienen_resultado(base: Path) -> None:
    sembrar_vuelos_de_hoy(base, n=60, semilla=3, ahora=AHORA)
    filas = _filas(
        base,
        """SELECT f.fa_flight_id, f.scheduled_in_utc, a.arr_delay_min, a.actual_in_utc
             FROM flights f LEFT JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
            WHERE f.fa_flight_id LIKE 'SEED-%' AND f.fa_flight_id NOT LIKE '%-R'""",
    )
    con_resultado = [f for f in filas if f["arr_delay_min"] is not None]
    assert con_resultado, "con 60 vuelos repartidos en 22 h alguno tiene que haber aterrizado"
    assert len(con_resultado) < len(filas), "y alguno tiene que estar todavía por salir"
    limite = AHORA.replace(microsecond=0).isoformat()
    for f in con_resultado:
        assert f["scheduled_in_utc"] < limite
        assert f["actual_in_utc"] is not None
    # Los que todavía no aterrizaron (más 20 min de gracia) no tienen resultado.
    gracia = (AHORA + timedelta(minutes=20)).replace(microsecond=0).isoformat()
    for f in filas:
        if f["scheduled_in_utc"] >= gracia:
            assert f["arr_delay_min"] is None


def test_la_misma_semilla_da_la_misma_base(tmp_path: Path) -> None:
    rutas = []
    for nombre in ("a.db", "b.db"):
        ruta = tmp_path / nombre
        open_db(ruta).close()
        sembrar_vuelos_de_hoy(ruta, n=15, semilla=42, ahora=AHORA)
        rutas.append(ruta)

    consulta = """SELECT f.fa_flight_id, f.tail_num, f.dest, f.scheduled_out_utc,
                         p.proba_delay, s.feature_name, s.shap_value
                    FROM flights f
                    JOIN predictions p ON p.fa_flight_id = f.fa_flight_id
                    JOIN prediction_shap s ON s.fa_flight_id = f.fa_flight_id
                   ORDER BY f.fa_flight_id, s.rank"""
    a = [tuple(r) for r in _filas(rutas[0], consulta)]
    b = [tuple(r) for r in _filas(rutas[1], consulta)]
    assert a == b and len(a) == 45


def test_semillas_distintas_dan_vuelos_distintos(tmp_path: Path) -> None:
    probas = []
    for semilla in (1, 2):
        ruta = tmp_path / f"{semilla}.db"
        open_db(ruta).close()
        sembrar_vuelos_de_hoy(ruta, n=10, semilla=semilla, ahora=AHORA)
        probas.append([r["proba_delay"] for r in _filas(
            ruta, "SELECT proba_delay FROM predictions ORDER BY fa_flight_id")])
    assert probas[0] != probas[1]


def test_rechaza_sembrar_cero_vuelos(base: Path) -> None:
    with pytest.raises(ValueError):
        sembrar_vuelos_de_hoy(base, n=0)


def test_la_cascada_encuentra_la_pierna_siguiente(base: Path, monkeypatch) -> None:
    """Sin la pierna de vuelta `/cascade/{id}` contesta "no conocemos la próxima
    pierna" y la prueba de carga no ejercitaría la proyección."""
    siembra = sembrar_vuelos_de_hoy(base, n=5, semilla=9)
    con = _conexion(base)
    monkeypatch.setattr(api, "get_db", lambda: con)
    try:
        for fid in siembra.ids:
            r = api.cascade_de_vuelo(fid)
            assert r["piernas_conocidas"] == 1, r.get("motivo")
            assert r["demora_origen"] in ("real", "estimada")
            # Con menos de 15 min de demora la cadena corta antes de la primera
            # pierna por diseño; simulada una demora grande, tiene que llegar.
            simulada = api.cascade_de_vuelo(fid, demora_min=60)
            assert [p["fa_flight_id"] for p in simulada["piernas"]] == [f"{fid}-R"]
    finally:
        con.close()


def test_el_vuelo_unico_del_fixture_de_la_api_sigue_igual(base: Path) -> None:
    sembrar_vuelo_de_hoy(base)
    vuelo = _filas(base, "SELECT * FROM flights WHERE fa_flight_id = 'TEST-SEED-0001'")
    assert len(vuelo) == 1 and vuelo[0]["ident_iata"] == "DL9999"
    pred = _filas(base, "SELECT * FROM predictions WHERE fa_flight_id = 'TEST-SEED-0001'")
    assert len(pred) == 1 and pred[0]["proba_delay"] == 0.42 and pred[0]["predicted_delay"] == 1
    assert len(_filas(base, "SELECT 1 FROM prediction_shap WHERE fa_flight_id = 'TEST-SEED-0001'")) == 3

    con = _conexion(base)
    try:
        assert [r["fa_flight_id"] for r in api._latest_predictions_active(con)] == ["TEST-SEED-0001"]
    finally:
        con.close()

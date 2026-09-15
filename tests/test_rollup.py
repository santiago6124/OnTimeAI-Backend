"""
Los agregados diarios tienen que sobrevivir a la purga y contar bien.

`prune_db` borra `predictions`, `flights` y `actuals` a los 30 dias, y con eso
se lleva el historico que necesitan Frontend #3, #5 y #11. La salida es
persistir agregados —kilobytes por semana— fuera del ciclo de purga. Ver #12.
"""
from __future__ import annotations

import sqlite3

import pytest

from ontimeai import rollup
from ontimeai.live import SCHEMA


@pytest.fixture
def con() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.executescript(SCHEMA)
    rollup.ensure_schema(c)
    return c


def _vuelo(con, fid, *, day="2026-09-10", hour="12", carrier="DL",
           proba=0.1, flagged=0, threshold=0.09, arr_delay=None,
           cancelled=0, diverted=0, phase="PRE_DEPARTURE", predicted_at=None):
    sched = f"{day}T{hour}:00:00+00:00"
    con.execute(
        "INSERT INTO flights (fa_flight_id, op_carrier, origin, dest,"
        " scheduled_out_utc, first_seen_utc, last_updated_utc)"
        " VALUES (?,?,?,?,?,?,?)",
        (fid, carrier, "ATL", "MIA", sched, sched, sched),
    )
    con.execute(
        "INSERT INTO predictions (fa_flight_id, stable_id, predicted_at_utc,"
        " proba_delay, predicted_delay, threshold_used, prediction_phase)"
        " VALUES (?,?,?,?,?,?,?)",
        (fid, fid, predicted_at or sched, proba, flagged, threshold, phase),
    )
    if arr_delay is not None or cancelled or diverted:
        con.execute(
            "INSERT INTO actuals (fa_flight_id, stable_id, arr_delay_min,"
            " cancelled, diverted, settled_at_utc) VALUES (?,?,?,?,?,?)",
            (fid, fid, arr_delay, cancelled, diverted, sched),
        )
    con.commit()


class TestConteos:
    def test_cuenta_vuelos_demorados_y_marcados(self, con) -> None:
        _vuelo(con, "A", proba=0.8, flagged=1, arr_delay=40)   # TP
        _vuelo(con, "B", proba=0.7, flagged=1, arr_delay=2)    # FP
        _vuelo(con, "C", proba=0.02, flagged=0, arr_delay=1)   # TN
        _vuelo(con, "D", proba=0.03, flagged=0, arr_delay=60)  # FN

        todos = next(f for f in rollup.compute_daily_rollup(con) if f["segment"] == "all")
        assert todos["n_flights"] == 4
        assert todos["n_delayed"] == 2
        assert todos["n_flagged"] == 2
        assert (todos["tp"], todos["fp"], todos["tn"], todos["fn"]) == (1, 1, 1, 1)

    def test_el_umbral_de_15_minutos_es_el_del_target(self, con) -> None:
        _vuelo(con, "A", arr_delay=15.0)   # exactamente 15: NO es demora
        _vuelo(con, "B", arr_delay=15.1)
        todos = next(f for f in rollup.compute_daily_rollup(con) if f["segment"] == "all")
        assert todos["n_delayed"] == 1

    def test_ignora_cancelados_y_desviados(self, con) -> None:
        _vuelo(con, "A", arr_delay=40)
        _vuelo(con, "B", arr_delay=40, cancelled=1)
        _vuelo(con, "C", arr_delay=40, diverted=1)
        todos = next(f for f in rollup.compute_daily_rollup(con) if f["segment"] == "all")
        assert todos["n_flights"] == 1

    def test_ignora_vuelos_sin_resultado(self, con) -> None:
        _vuelo(con, "A", arr_delay=40)
        _vuelo(con, "B", arr_delay=None)
        todos = next(f for f in rollup.compute_daily_rollup(con) if f["segment"] == "all")
        assert todos["n_flights"] == 1


class TestUnidadDeAgregacion:
    def test_un_vuelo_cuenta_una_vez_aunque_tenga_muchas_predicciones(self, con) -> None:
        """
        La razon de existir de este modulo aparte de live_metrics.py: aquel
        agrega sobre todas las filas de `predictions`, asi que un vuelo que
        estuvo diez ciclos en la ventana pesa diez veces mas que uno que
        entro en el ultimo.
        """
        _vuelo(con, "A", proba=0.9, flagged=1, arr_delay=40,
               predicted_at="2026-09-10T08:00:00+00:00")
        for h in ("09", "10", "11"):
            con.execute(
                "INSERT INTO predictions (fa_flight_id, stable_id, predicted_at_utc,"
                " proba_delay, predicted_delay, threshold_used, prediction_phase)"
                " VALUES (?,?,?,?,?,?,?)",
                ("A", "A", f"2026-09-10T{h}:00:00+00:00", 0.5, 1, 0.09, "PRE_DEPARTURE"),
            )
        con.commit()

        todos = next(f for f in rollup.compute_daily_rollup(con) if f["segment"] == "all")
        assert todos["n_flights"] == 1

    def test_toma_la_ultima_prediccion_anterior_al_aterrizaje(self, con) -> None:
        _vuelo(con, "A", proba=0.20, flagged=1, arr_delay=40,
               predicted_at="2026-09-10T08:00:00+00:00")
        con.execute(
            "INSERT INTO predictions (fa_flight_id, stable_id, predicted_at_utc,"
            " proba_delay, predicted_delay, threshold_used, prediction_phase)"
            " VALUES (?,?,?,?,?,?,?)",
            ("A", "A", "2026-09-10T11:00:00+00:00", 0.60, 1, 0.09, "PRE_DEPARTURE"),
        )
        # Posterior al aterrizaje: no la vio nadie, no cuenta.
        con.execute(
            "INSERT INTO predictions (fa_flight_id, stable_id, predicted_at_utc,"
            " proba_delay, predicted_delay, threshold_used, prediction_phase)"
            " VALUES (?,?,?,?,?,?,?)",
            ("A", "A", "2026-09-10T23:00:00+00:00", 0.99, 1, 0.09, "POST_LANDING"),
        )
        con.commit()

        todos = next(f for f in rollup.compute_daily_rollup(con) if f["segment"] == "all")
        assert todos["mean_proba"] == pytest.approx(0.60)


class TestSegmentos:
    def test_separa_por_aerolinea_y_por_hora(self, con) -> None:
        _vuelo(con, "A", carrier="DL", hour="08", arr_delay=40)
        _vuelo(con, "B", carrier="AA", hour="08", arr_delay=1)
        _vuelo(con, "C", carrier="DL", hour="17", arr_delay=1)

        por = {f["segment"]: f for f in rollup.compute_daily_rollup(con)}
        assert por["all"]["n_flights"] == 3
        assert por["carrier:DL"]["n_flights"] == 2
        assert por["carrier:AA"]["n_flights"] == 1
        assert por["hour:08"]["n_flights"] == 2
        assert por["hour:17"]["n_flights"] == 1

    def test_separa_por_dia(self, con) -> None:
        _vuelo(con, "A", day="2026-09-10", arr_delay=40)
        _vuelo(con, "B", day="2026-09-11", arr_delay=1)
        dias = {f["day"] for f in rollup.compute_daily_rollup(con) if f["segment"] == "all"}
        assert dias == {"2026-09-10", "2026-09-11"}


class TestMetricas:
    def test_el_auc_queda_en_null_con_una_sola_clase(self, con) -> None:
        # Un dia donde no se demoro nadie no tiene AUC definido. Guardar 0.5
        # o 0.0 ahi seria inventar un dato que despues se grafica.
        _vuelo(con, "A", proba=0.1, arr_delay=1)
        _vuelo(con, "B", proba=0.2, arr_delay=2)
        todos = next(f for f in rollup.compute_daily_rollup(con) if f["segment"] == "all")
        assert todos["auc"] is None
        assert todos["brier"] is not None

    def test_un_ranking_perfecto_da_auc_1(self, con) -> None:
        _vuelo(con, "A", proba=0.9, arr_delay=40)
        _vuelo(con, "B", proba=0.8, arr_delay=30)
        _vuelo(con, "C", proba=0.1, arr_delay=1)
        todos = next(f for f in rollup.compute_daily_rollup(con) if f["segment"] == "all")
        assert todos["auc"] == pytest.approx(1.0)

    def test_el_brier_castiga_la_confianza_equivocada(self, con) -> None:
        _vuelo(con, "A", proba=1.0, arr_delay=1)   # segurisimo y erro
        todos = next(f for f in rollup.compute_daily_rollup(con) if f["segment"] == "all")
        assert todos["brier"] == pytest.approx(1.0)

    def test_un_umbral_ausente_no_rompe_el_promedio(self, con) -> None:
        _vuelo(con, "A", arr_delay=40, threshold=None)
        _vuelo(con, "B", arr_delay=1, threshold=0.10)
        todos = next(f for f in rollup.compute_daily_rollup(con) if f["segment"] == "all")
        assert todos["mean_threshold"] == pytest.approx(0.10)

    def test_sin_ningun_umbral_el_promedio_es_null(self, con) -> None:
        _vuelo(con, "A", arr_delay=40, threshold=None)
        todos = next(f for f in rollup.compute_daily_rollup(con) if f["segment"] == "all")
        assert todos["mean_threshold"] is None


class TestPersistencia:
    def test_guarda_y_relee(self, con) -> None:
        _vuelo(con, "A", arr_delay=40)
        n = rollup.rollup_daily_metrics(con, model_version="4year_v9")
        assert n > 0
        fila = con.execute(
            "SELECT n_flights, model_version FROM metrics_daily WHERE segment='all'"
        ).fetchone()
        assert fila == (1, "4year_v9")

    def test_recalcular_reemplaza_en_vez_de_duplicar(self, con) -> None:
        """
        Los actuals llegan tarde: un dia calculado antes de que settleen sus
        vuelos queda corto, y la siguiente corrida tiene que corregirlo.
        """
        _vuelo(con, "A", arr_delay=40)
        rollup.rollup_daily_metrics(con)
        _vuelo(con, "B", arr_delay=1)
        rollup.rollup_daily_metrics(con)

        filas = con.execute(
            "SELECT n_flights FROM metrics_daily WHERE segment='all'"
        ).fetchall()
        assert len(filas) == 1, "una sola fila por (dia, segmento)"
        assert filas[0][0] == 2

    def test_la_purga_no_toca_la_tabla(self, con) -> None:
        """
        `prune_db` borra con una lista explicita de DELETE, asi que una tabla
        nueva queda afuera sola. El test lo fija: si alguien la agrega a esa
        lista, el historico se pierde y este test avisa.
        """
        import scripts.prune_db as prune_mod
        import inspect

        fuente = inspect.getsource(prune_mod)
        assert "DELETE FROM metrics_daily" not in fuente

    def test_sin_datos_no_escribe_nada(self, con) -> None:
        assert rollup.rollup_daily_metrics(con) == 0


@pytest.fixture
def api_con_base(tmp_path, monkeypatch):
    """Base en archivo + `get_db` que abre una conexion nueva por llamada.

    El endpoint cierra su conexion en `finally`, como corresponde, asi que
    devolver siempre el mismo objeto haria fallar la segunda llamada. Ademas
    asi se parece a produccion, donde cada request abre la suya.
    """
    import sqlite3

    import api
    from ontimeai.live import SCHEMA

    db = tmp_path / "live.db"
    semilla = sqlite3.connect(db)
    semilla.executescript(SCHEMA)
    rollup.ensure_schema(semilla)
    semilla.close()

    def _abrir():
        c = sqlite3.connect(db)
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr(api, "get_db", _abrir)
    return api, _abrir


class TestEndpointDeHistoria:
    """`/metrics/history` es lo que consume Frontend #3."""

    def test_sin_la_tabla_devuelve_serie_vacia(self, tmp_path, monkeypatch) -> None:
        """
        Una base anterior al primer rollup no puede dar 500. Es la leccion del
        incidente del 15/09: referenciar algo que la base todavia no tiene
        tumbo /metrics/summary y /flights.
        """
        import sqlite3

        import api
        from ontimeai.live import SCHEMA

        db = tmp_path / "vieja.db"
        semilla = sqlite3.connect(db)
        semilla.executescript(SCHEMA)     # sin metrics_daily
        semilla.close()

        def _abrir():
            c = sqlite3.connect(db)
            c.row_factory = sqlite3.Row
            return c

        monkeypatch.setattr(api, "get_db", _abrir)
        r = api.metrics_history()
        assert r["points"] == []
        assert r["segment"] == "all"

    def test_devuelve_la_serie_con_precision_y_recall(self, api_con_base) -> None:
        api, abrir = api_con_base
        con = abrir()
        _vuelo(con, "A", proba=0.8, flagged=1, arr_delay=40)   # TP
        _vuelo(con, "B", proba=0.7, flagged=1, arr_delay=2)    # FP
        _vuelo(con, "C", proba=0.03, flagged=0, arr_delay=60)  # FN
        rollup.rollup_daily_metrics(con, model_version="4year_v9")
        con.close()

        puntos = api.metrics_history(days=3650)["points"]
        assert len(puntos) == 1
        p = puntos[0]
        assert p["n_flights"] == 3
        assert p["precision"] == pytest.approx(0.5)    # 1 de 2 marcados
        assert p["recall"] == pytest.approx(0.5)       # 1 de 2 demorados
        assert p["model_version"] == "4year_v9"

    def test_filtra_por_segmento(self, api_con_base) -> None:
        api, abrir = api_con_base
        con = abrir()
        _vuelo(con, "A", carrier="DL", arr_delay=40)
        _vuelo(con, "B", carrier="AA", arr_delay=1)
        rollup.rollup_daily_metrics(con)
        con.close()

        assert api.metrics_history(segment="carrier:DL", days=3650)["points"][0]["n_flights"] == 1
        assert api.metrics_history(segment="all", days=3650)["points"][0]["n_flights"] == 2
        assert api.metrics_history(segment="carrier:UA", days=3650)["points"] == []

    def test_la_serie_excede_la_retencion_de_30_dias(self, api_con_base) -> None:
        """
        El punto del issue: el agregado sobrevive aunque las filas crudas ya no
        esten. Se simula escribiendo un dia viejo directo en metrics_daily,
        que es como quedaria despues de que la purga se lleve su materia prima.
        """
        api, abrir = api_con_base
        con = abrir()
        rollup.upsert_daily_rollup(con, [{
            "day": "2026-07-01", "segment": "all", "n_flights": 500,
            "n_delayed": 40, "n_flagged": 110, "tp": 25, "fp": 85, "tn": 375,
            "fn": 15, "auc": 0.72, "brier": 0.07, "ece": 0.02,
            "mean_proba": 0.08, "mean_threshold": 0.10,
            "model_version": "4year_v9", "computed_at_utc": "2026-07-02T00:00:00+00:00",
        }])
        assert con.execute("SELECT COUNT(*) FROM flights").fetchone()[0] == 0
        con.close()

        puntos = api.metrics_history(days=3650)["points"]
        assert len(puntos) == 1
        assert puntos[0]["day"] == "2026-07-01"
        assert puntos[0]["n_flights"] == 500

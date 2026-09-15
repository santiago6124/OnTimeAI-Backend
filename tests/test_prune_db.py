"""Tests for bounded database pruning maintenance."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

from scripts import prune_db as prune_module


def _prunable_db(path: Path, *, old_predictions: int) -> None:
    old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    con = sqlite3.connect(path)
    con.executescript(
        """
        -- `fa_flight_id` existe en el esquema real y es de lo que cuelga
        -- el barrido de huerfanos.
        CREATE TABLE predictions(fa_flight_id TEXT, predicted_at_utc TEXT);
        CREATE TABLE prediction_shap(fa_flight_id TEXT, predicted_at_utc TEXT);
        -- `first_seen_utc` es NOT NULL en el esquema real; la purga lo usa
        -- como respaldo cuando `scheduled_out_utc` viene vacio.
        CREATE TABLE flights(fa_flight_id TEXT, scheduled_out_utc TEXT,
                             first_seen_utc TEXT);
        CREATE TABLE actuals(fa_flight_id TEXT);
        CREATE TABLE weather_obs(valid_utc TEXT);
        CREATE TABLE runs(started_utc TEXT);
        CREATE TABLE harvester_runs(run_at_utc TEXT);
        CREATE TABLE nas_status(captured_at_utc TEXT);
        CREATE TABLE aircraft_position(captured_at_utc TEXT);
        """
    )
    con.executemany(
        "INSERT INTO predictions(predicted_at_utc) VALUES (?)",
        [(old,)] * old_predictions,
    )
    con.commit()
    con.close()


def test_prune_skips_vacuum_below_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "skip.db"
    _prunable_db(db_path, old_predictions=2)
    vacuum_calls: list[bool] = []
    monkeypatch.setattr(
        prune_module,
        "_vacuum",
        lambda _con: vacuum_calls.append(True),
    )

    result = prune_module.prune_db(
        db_path,
        days=30,
        vacuum_min_deleted=2,
    )

    assert result == 0
    assert vacuum_calls == []


def test_prune_runs_vacuum_above_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "vacuum.db"
    _prunable_db(db_path, old_predictions=3)
    vacuum_calls: list[bool] = []
    monkeypatch.setattr(
        prune_module,
        "_vacuum",
        lambda _con: vacuum_calls.append(True),
    )

    result = prune_module.prune_db(
        db_path,
        days=30,
        vacuum_min_deleted=2,
    )

    assert result == 0
    assert vacuum_calls == [True]


def _db_con_vuelos(tmp_path, filas):
    """Base minima con flights + actuals, para probar que la purga alcance todo."""
    import sqlite3

    db = tmp_path / "live.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE prediction_shap(fa_flight_id TEXT, predicted_at_utc TEXT);
        CREATE TABLE predictions(fa_flight_id TEXT, predicted_at_utc TEXT);
        CREATE TABLE flights(fa_flight_id TEXT, scheduled_out_utc TEXT,
                             first_seen_utc TEXT);
        CREATE TABLE actuals(fa_flight_id TEXT);
        CREATE TABLE weather_obs(valid_utc TEXT);
        CREATE TABLE runs(started_utc TEXT);
        CREATE TABLE harvester_runs(run_at_utc TEXT);
        CREATE TABLE nas_status(captured_at_utc TEXT);
        CREATE TABLE aircraft_position(captured_at_utc TEXT);
        """
    )
    for fid, sched, first_seen in filas:
        con.execute("INSERT INTO flights VALUES (?,?,?)", (fid, sched, first_seen))
        con.execute("INSERT INTO actuals VALUES (?)", (fid,))
    con.commit()
    con.close()
    return db


class TestVuelosSinHorarioProgramado:
    """
    `scheduled_out_utc < ?` no alcanza a las filas donde ese campo es NULL o
    vacio: en SQL `NULL < 'x'` es NULL, no verdadero. Esas filas quedaban vivas
    para siempre, y como los `actuals` se borran por orfandad contra `flights`,
    cada vuelo inmortal mantenia vivo el suyo.

    Medido en produccion el 15/09: `actuals` arrastraba filas del 26/05 con una
    retencion de 30 dias. Ver issue #12.
    """

    def test_borra_los_que_tienen_horario_viejo(self, tmp_path) -> None:
        import sqlite3

        from scripts.prune_db import prune_db

        db = _db_con_vuelos(tmp_path, [("VIEJO", "2020-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00")])
        prune_db(db, days=30, dry_run=False)
        con = sqlite3.connect(db)
        assert con.execute("SELECT COUNT(*) FROM flights").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM actuals").fetchone()[0] == 0
        con.close()

    def test_borra_los_que_tienen_horario_nulo_pero_son_viejos(self, tmp_path) -> None:
        import sqlite3

        from scripts.prune_db import prune_db

        db = _db_con_vuelos(tmp_path, [("NULO", None, "2020-01-01T00:00:00+00:00")])
        prune_db(db, days=30, dry_run=False)
        con = sqlite3.connect(db)
        assert con.execute("SELECT COUNT(*) FROM flights").fetchone()[0] == 0, (
            "un vuelo sin scheduled_out_utc era inmortal"
        )
        assert con.execute("SELECT COUNT(*) FROM actuals").fetchone()[0] == 0, (
            "y mantenia vivo su actual"
        )
        con.close()

    def test_borra_los_que_tienen_horario_vacio(self, tmp_path) -> None:
        import sqlite3

        from scripts.prune_db import prune_db

        # Cadena vacia, no NULL: `'' < '2026-...'` SI es verdadero, asi que esta
        # ya se borraba. Se fija igual para que el CASE no la rompa.
        db = _db_con_vuelos(tmp_path, [("VACIO", "", "2020-01-01T00:00:00+00:00")])
        prune_db(db, days=30, dry_run=False)
        con = sqlite3.connect(db)
        assert con.execute("SELECT COUNT(*) FROM flights").fetchone()[0] == 0
        con.close()

    def test_conserva_los_recientes_sin_horario(self, tmp_path) -> None:
        import sqlite3
        from datetime import datetime, timezone

        from scripts.prune_db import prune_db

        ahora = datetime.now(timezone.utc).isoformat()
        db = _db_con_vuelos(tmp_path, [("NUEVO", None, ahora)])
        prune_db(db, days=30, dry_run=False)
        con = sqlite3.connect(db)
        assert con.execute("SELECT COUNT(*) FROM flights").fetchone()[0] == 1, (
            "un vuelo sin horario pero visto hoy todavia sirve"
        )
        assert con.execute("SELECT COUNT(*) FROM actuals").fetchone()[0] == 1
        con.close()

    def test_conserva_los_recientes_con_horario(self, tmp_path) -> None:
        import sqlite3
        from datetime import datetime, timezone

        from scripts.prune_db import prune_db

        ahora = datetime.now(timezone.utc).isoformat()
        db = _db_con_vuelos(tmp_path, [("NUEVO", ahora, ahora)])
        prune_db(db, days=30, dry_run=False)
        con = sqlite3.connect(db)
        assert con.execute("SELECT COUNT(*) FROM flights").fetchone()[0] == 1
        con.close()


class TestBarridoDeHuerfanos:
    """
    Filas que apuntan a un vuelo que ya no existe.

    `actuals` ya se limpiaba asi; `predictions` y `prediction_shap` no, y ahi se
    acumulaba la mayor parte. Los placeholders SYN- de captura de legs futuros
    se borraban de `flights` al vencer su TTL y dejaban todo lo suyo colgando.

    Medido el 15/09 sobre produccion: 33.469 predicciones huerfanas (16,1% de la
    tabla, 100% con id SYN-, ninguna con label) y 623.505 filas de SHAP sin
    prediccion correspondiente. El origen se corrige en el harvester; esto
    limpia lo acumulado y queda como red de seguridad.
    """

    def _base(self, tmp_path):
        import sqlite3
        from datetime import datetime, timezone

        db = tmp_path / "live.db"
        con = sqlite3.connect(db)
        con.executescript(
            """
            CREATE TABLE prediction_shap(fa_flight_id TEXT, predicted_at_utc TEXT);
            CREATE TABLE predictions(fa_flight_id TEXT, predicted_at_utc TEXT);
            CREATE TABLE flights(fa_flight_id TEXT, scheduled_out_utc TEXT,
                                 first_seen_utc TEXT);
            CREATE TABLE actuals(fa_flight_id TEXT);
            CREATE TABLE weather_obs(valid_utc TEXT);
            CREATE TABLE runs(started_utc TEXT);
            CREATE TABLE harvester_runs(run_at_utc TEXT);
            CREATE TABLE nas_status(captured_at_utc TEXT);
            CREATE TABLE aircraft_position(captured_at_utc TEXT);
            """
        )
        ahora = datetime.now(timezone.utc).isoformat()
        # Un vuelo vivo con todo lo suyo.
        con.execute("INSERT INTO flights VALUES ('REAL', ?, ?)", (ahora, ahora))
        con.execute("INSERT INTO predictions VALUES ('REAL', ?)", (ahora,))
        con.execute("INSERT INTO prediction_shap VALUES ('REAL', ?)", (ahora,))
        con.execute("INSERT INTO actuals VALUES ('REAL')")
        # Un placeholder que ya no esta en flights, con todo colgando.
        con.execute("INSERT INTO predictions VALUES ('SYN-DL1-ATL-JFK-2026-09-01', ?)", (ahora,))
        con.execute("INSERT INTO prediction_shap VALUES ('SYN-DL1-ATL-JFK-2026-09-01', ?)", (ahora,))
        con.execute("INSERT INTO actuals VALUES ('SYN-DL1-ATL-JFK-2026-09-01')")
        con.commit()
        con.close()
        return db

    def test_barre_predicciones_y_shap_sin_vuelo(self, tmp_path) -> None:
        import sqlite3

        from scripts.prune_db import prune_db

        db = self._base(tmp_path)
        prune_db(db, days=30, dry_run=False)
        con = sqlite3.connect(db)
        for tabla in ("predictions", "prediction_shap", "actuals"):
            ids = [r[0] for r in con.execute(f"SELECT fa_flight_id FROM {tabla}")]
            assert ids == ["REAL"], f"{tabla} conservo una fila huerfana: {ids}"
        con.close()

    def test_no_toca_lo_que_tiene_vuelo(self, tmp_path) -> None:
        import sqlite3

        from scripts.prune_db import prune_db

        db = self._base(tmp_path)
        prune_db(db, days=30, dry_run=False)
        con = sqlite3.connect(db)
        assert con.execute(
            "SELECT COUNT(*) FROM predictions WHERE fa_flight_id='REAL'"
        ).fetchone()[0] == 1
        con.close()

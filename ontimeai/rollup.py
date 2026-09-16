"""
Agregados diarios que sobreviven a la purga.

`prune_db` borra `predictions`, `flights`, `actuals` y el resto a los 30 dias.
Eso mantiene la base acotada, pero se lleva la materia prima de tres issues del
Frontend: el historico de accuracy (#3), la serie para medir lead time (#5) y
los escenarios de la demo (#11). Ver issue #12.

La salida no es subir la retencion —la base ya pesa 720 MB— sino separar lo que
se purga de lo que se conserva. Los agregados son kilobytes por semana:

    ~57 filas por dia (todos + por aerolinea + por hora + por fase)
    ~21.000 filas por anio

`prune_db` borra con una lista explicita de DELETE, asi que esta tabla queda
afuera sin necesidad de excluirla. Aun asi conviene no agregarla nunca a esa
lista: ese es justamente el punto.

La unidad es **un vuelo**, no una prediccion. `live_metrics.py` agrega sobre
todas las filas de `predictions`, lo que multiplica cada vuelo por la cantidad
de ciclos en que fue predicho y le da mas peso a los vuelos que estuvieron mas
tiempo en la ventana. Aca se toma la ultima prediccion anterior al aterrizaje,
que es la que el operador llego a ver.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import numpy as np

# Cuantos dias se recalculan en cada corrida.
#
# Mas que la retencion de 30 dias a proposito: los `actuals` llegan tarde —un
# vuelo predicho hoy puede settlear manana— y un dia calculado antes de que
# lleguen sus resultados queda con menos muestra de la que le corresponde.
# Recalcular la ventana entera cada vez lo corrige solo. Es barato: son miles
# de filas, no millones.
DEFAULT_DAYS_BACK = 35

DELAY_THRESHOLD_MIN = 15.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics_daily (
    day TEXT NOT NULL,              -- YYYY-MM-DD, dia operativo (scheduled_out)
    segment TEXT NOT NULL,          -- 'all'|'carrier:DL'|'hour:14'|'phase:PRE_DEPARTURE'
    n_flights INTEGER NOT NULL,     -- vuelos con resultado real
    n_delayed INTEGER NOT NULL,     -- los que efectivamente llegaron tarde
    n_flagged INTEGER NOT NULL,     -- los que el modelo marco
    tp INTEGER NOT NULL,
    fp INTEGER NOT NULL,
    tn INTEGER NOT NULL,
    fn INTEGER NOT NULL,
    auc REAL,                       -- NULL si el dia tiene una sola clase
    brier REAL,
    ece REAL,
    mean_proba REAL,
    mean_threshold REAL,
    model_version TEXT,
    computed_at_utc TEXT NOT NULL,
    PRIMARY KEY (day, segment)
);
CREATE INDEX IF NOT EXISTS idx_metrics_daily_day ON metrics_daily(day);
"""


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)


def _auc(p: np.ndarray, y: np.ndarray) -> float | None:
    """Mann-Whitney con empates promediados. None si hay una sola clase."""
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ordenadas = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and ordenadas[j + 1] == ordenadas[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    idx = np.clip(np.digitize(p, np.linspace(0.0, 1.0, bins + 1)[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def _metrics(p: np.ndarray, y: np.ndarray, flagged: np.ndarray,
             thresholds: np.ndarray) -> dict:
    tp = int(((flagged == 1) & (y == 1)).sum())
    fp = int(((flagged == 1) & (y == 0)).sum())
    tn = int(((flagged == 0) & (y == 0)).sum())
    fn = int(((flagged == 0) & (y == 1)).sum())
    finitos = thresholds[np.isfinite(thresholds)]
    return {
        "n_flights": int(len(y)),
        "n_delayed": int(y.sum()),
        "n_flagged": int(flagged.sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "auc": _auc(p, y),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": _ece(p, y),
        "mean_proba": float(p.mean()),
        "mean_threshold": float(finitos.mean()) if finitos.size else None,
    }


def _load_flights(con: sqlite3.Connection, days_back: int) -> list[sqlite3.Row]:
    """Un vuelo por fila: su ultima prediccion anterior al aterrizaje y el resultado."""
    previo = con.row_factory
    con.row_factory = sqlite3.Row
    try:
        return con.execute(
            """
            SELECT substr(f.scheduled_out_utc, 1, 10)               AS day,
                   f.op_carrier                                     AS carrier,
                   CAST(substr(f.scheduled_out_utc, 12, 2) AS INTEGER) AS hour,
                   p.proba_delay                                    AS proba,
                   p.predicted_delay                                AS flagged,
                   p.threshold_used                                 AS threshold,
                   CASE WHEN a.arr_delay_min > ? THEN 1 ELSE 0 END  AS y
              FROM flights f
              JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
              JOIN predictions p ON p.fa_flight_id = f.fa_flight_id
             WHERE a.arr_delay_min IS NOT NULL
               AND COALESCE(a.cancelled, 0) = 0
               AND COALESCE(a.diverted, 0) = 0
               AND f.scheduled_out_utc >= datetime('now', ?)
               AND COALESCE(p.prediction_phase, '') IS NOT 'POST_LANDING'
               AND p.predicted_at_utc = (
                     SELECT MAX(p2.predicted_at_utc) FROM predictions p2
                      WHERE p2.fa_flight_id = p.fa_flight_id
                        AND COALESCE(p2.prediction_phase, '') IS NOT 'POST_LANDING')
            """,
            (DELAY_THRESHOLD_MIN, f"-{int(days_back)} days"),
        ).fetchall()
    finally:
        con.row_factory = previo


def _load_by_phase(con: sqlite3.Connection, days_back: int) -> list[sqlite3.Row]:
    """Una fila por (vuelo, fase): la ultima prediccion que hizo en esa fase.

    Distinto de `_load_flights`, que toma una sola por vuelo —la ultima de
    todas— y por eso casi siempre mide al avion ya en el aire.

    La diferencia entre las dos fases es la pregunta que importa: predecir con
    el avion volando es facil, porque media hora despues se sabe solo. El valor
    esta en acertar **antes de que salga**, que es cuando todavia se puede hacer
    algo. Medirlas juntas esconde exactamente eso.
    """
    previo = con.row_factory
    con.row_factory = sqlite3.Row
    try:
        return con.execute(
            """
            SELECT substr(f.scheduled_out_utc, 1, 10)               AS day,
                   p.prediction_phase                               AS phase,
                   p.proba_delay                                    AS proba,
                   p.predicted_delay                                AS flagged,
                   p.threshold_used                                 AS threshold,
                   CASE WHEN a.arr_delay_min > ? THEN 1 ELSE 0 END  AS y
              FROM flights f
              JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
              JOIN predictions p ON p.fa_flight_id = f.fa_flight_id
             WHERE a.arr_delay_min IS NOT NULL
               AND COALESCE(a.cancelled, 0) = 0
               AND COALESCE(a.diverted, 0) = 0
               AND f.scheduled_out_utc >= datetime('now', ?)
               AND p.prediction_phase IN ('PRE_DEPARTURE', 'EN_ROUTE')
               AND p.predicted_at_utc = (
                     SELECT MAX(p2.predicted_at_utc) FROM predictions p2
                      WHERE p2.fa_flight_id = p.fa_flight_id
                        AND p2.prediction_phase = p.prediction_phase)
            """,
            (DELAY_THRESHOLD_MIN, f"-{int(days_back)} days"),
        ).fetchall()
    finally:
        con.row_factory = previo


def compute_daily_rollup(
    con: sqlite3.Connection, *, days_back: int = DEFAULT_DAYS_BACK,
    model_version: str = "",
) -> list[dict]:
    """Una fila por (dia, segmento): todos, por aerolinea, por hora y por fase.

    No se segmenta por ruta: ATL tiene cientos, y multiplicarlas por dia haria
    crecer la tabla mas rapido de lo que justifica lo que se consulta hoy.
    """
    filas = _load_flights(con, days_back)
    por_fase = _load_by_phase(con, days_back)
    if not filas and not por_fase:
        return []

    por_clave: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for r in filas:
        day = r["day"]
        if not day:
            continue
        for segmento in ("all", f"carrier:{r['carrier']}", f"hour:{r['hour']:02d}"):
            por_clave.setdefault((day, segmento), []).append(r)

    # Las fases van aparte porque un mismo vuelo aporta a las dos, con una
    # prediccion distinta en cada una. Sumarlas a los segmentos de arriba lo
    # contaria dos veces.
    for r in por_fase:
        day = r["day"]
        if not day or not r["phase"]:
            continue
        por_clave.setdefault((day, f"phase:{r['phase']}"), []).append(r)

    ahora = datetime.now(timezone.utc).isoformat()
    salida: list[dict] = []
    for (day, segment), grupo in sorted(por_clave.items()):
        p = np.array([float(r["proba"]) for r in grupo])
        y = np.array([int(r["y"]) for r in grupo])
        flagged = np.array([int(r["flagged"]) for r in grupo])
        thresholds = np.array(
            [float(r["threshold"]) if r["threshold"] is not None else np.nan
             for r in grupo]
        )
        fila = {"day": day, "segment": segment,
                "model_version": model_version, "computed_at_utc": ahora}
        fila.update(_metrics(p, y, flagged, thresholds))
        salida.append(fila)
    return salida


def upsert_daily_rollup(con: sqlite3.Connection, filas: list[dict]) -> int:
    if not filas:
        return 0
    ensure_schema(con)
    columnas = ("day", "segment", "n_flights", "n_delayed", "n_flagged",
                "tp", "fp", "tn", "fn", "auc", "brier", "ece",
                "mean_proba", "mean_threshold", "model_version", "computed_at_utc")
    con.executemany(
        f"""INSERT OR REPLACE INTO metrics_daily ({','.join(columnas)})
            VALUES ({','.join('?' * len(columnas))})""",
        [tuple(f[c] for c in columnas) for f in filas],
    )
    con.commit()
    return len(filas)


def rollup_daily_metrics(
    con: sqlite3.Connection, *, days_back: int = DEFAULT_DAYS_BACK,
    model_version: str = "",
) -> int:
    """Calcula y persiste. Se llama antes de purgar, no despues."""
    ensure_schema(con)
    return upsert_daily_rollup(
        con, compute_daily_rollup(con, days_back=days_back, model_version=model_version)
    )

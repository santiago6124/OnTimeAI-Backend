"""
Calibrador de la salida de la cadena de ajustes, que se reajusta solo.

El modelo pasa por su calibrador y produce una probabilidad honesta. Despues la
cadena de ajustes post-prediccion la modifica con constantes puestas a mano
—GDP usa `1 - exp(min/60)`— y el resultado deja de ser una probabilidad.
Medido: mostraba 89,5% para vuelos cuya tasa real era 39%.

La salida es tratar a la cadena como un puntaje, no como una probabilidad, y
aprender la traduccion de los vuelos que ya aterrizaron:

    cuando la cadena dice 0,85  ->  en realidad pasa el 41%  ->  mostrar 0,41

Medido fuera de muestra —ajustado con el 70% mas viejo, evaluado con el 30% mas
nuevo, que es como funciona en produccion:

    cadena sola          Brier 0,0832   AUC 0,7634   ECE 0,0640
    cadena + calibrador  Brier 0,0682   AUC 0,7579   ECE 0,0193

El AUC no se mueve porque una transformacion monotona no cambia el orden. Eso
es tambien la respuesta al tercer criterio del issue #7: ninguna calibracion
sube el AUC, por construccion.

## Por que se reajusta solo

`4year_v9_recal` es un artefacto congelado del 6 de agosto. Medido hoy,
EMPEORA las cosas —ECE 0,0164 sin el, 0,0490 con el— porque aprendio a corregir
una exageracion que ya no existe (ver issue #15). Un corrector desactualizado
corrige en la direccion equivocada.

Este vive en la base, se rehace con los ultimos `window_days` y se APAGA si no
puede hacerlo bien. Sin calibrador se muestra un numero exagerado; con uno
vencido se muestra un numero exagerado en la otra direccion, y eso es peor
porque nadie lo sospecha.

## Por que no se realimenta

Se ajusta sobre `predictions.proba_chain`, la salida cruda de la cadena, y no
sobre `proba_delay`, que es lo que finalmente se sirve. Si se ajustara sobre lo
servido, cada reajuste aprenderia sobre valores ya calibrados y la correccion
se aplicaria dos veces, tres, n veces.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

# Ventana de la que se aprende. Treinta dias es lo que retiene la base.
DEFAULT_WINDOW_DAYS = 30

# Por debajo de esto no se ajusta. Un isotonico con pocas muestras sigue el
# ruido: inventaria una traduccion que no se sostiene.
DEFAULT_MIN_SAMPLES = 500

# Pasado esto se ignora, aunque exista. Es el error de #15 automatizado para que
# no dependa de que alguien se acuerde.
DEFAULT_MAX_AGE_DAYS = 7

DELAY_THRESHOLD_MIN = 15.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS chain_calibrator (
    id            INTEGER PRIMARY KEY CHECK (id = 1),  -- una sola fila, siempre
    fitted_at_utc TEXT NOT NULL,
    n_samples     INTEGER NOT NULL,
    window_days   INTEGER NOT NULL,
    base_rate     REAL,
    x_json        TEXT NOT NULL,   -- quiebres del isotonico
    y_json        TEXT NOT NULL
);
"""


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)


@dataclass(frozen=True)
class ChainCalibrator:
    """Traduccion aprendida, lista para aplicar. Solo necesita numpy."""

    xs: np.ndarray
    ys: np.ndarray
    fitted_at_utc: str
    n_samples: int
    age_days: float

    def __call__(self, proba: float) -> float:
        # `np.interp` recorta fuera de rango, que es el mismo comportamiento
        # que `IsotonicRegression(out_of_bounds="clip")`.
        return float(np.interp(float(proba), self.xs, self.ys))

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(proba, dtype=float), self.xs, self.ys)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _training_rows(con: sqlite3.Connection, window_days: int) -> list[tuple[float, int]]:
    """Salida cruda de la cadena y resultado real, una fila por vuelo.

    Se toma la ultima prediccion anterior al aterrizaje: es la que el operador
    llego a ver, y la unica sobre la que tiene sentido calibrar.
    """
    try:
        filas = con.execute(
            """
            SELECT p.proba_chain,
                   CASE WHEN a.arr_delay_min > ? THEN 1 ELSE 0 END AS y
              FROM predictions p
              JOIN actuals a ON a.stable_id = p.stable_id
             WHERE p.proba_chain IS NOT NULL
               AND a.arr_delay_min IS NOT NULL
               AND COALESCE(a.cancelled, 0) = 0
               AND COALESCE(a.diverted, 0) = 0
               AND p.predicted_at_utc >= datetime('now', ?)
               AND COALESCE(p.prediction_phase, '') IS NOT 'POST_LANDING'
               AND p.predicted_at_utc = (
                     SELECT MAX(p2.predicted_at_utc) FROM predictions p2
                      WHERE p2.stable_id = p.stable_id
                        AND p2.proba_chain IS NOT NULL
                        AND COALESCE(p2.prediction_phase, '') IS NOT 'POST_LANDING')
            """,
            (DELAY_THRESHOLD_MIN, f"-{int(window_days)} days"),
        ).fetchall()
    except sqlite3.OperationalError:
        # Base sin `proba_chain`: todavia no corrio ningun ciclo que la escriba.
        return []
    return [(float(x), int(y)) for x, y in filas]


def fit_chain_calibrator(
    con: sqlite3.Connection,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict | None:
    """Aprende la traduccion y la guarda. Devuelve None si no se pudo.

    Devolver None es un resultado valido, no un error: sin muestra suficiente
    conviene no calibrar antes que calibrar mal.
    """
    from sklearn.isotonic import IsotonicRegression

    ensure_schema(con)
    filas = _training_rows(con, window_days)
    if len(filas) < min_samples:
        return None

    x = np.array([f[0] for f in filas], dtype=float)
    y = np.array([f[1] for f in filas], dtype=int)
    if y.min() == y.max():
        # Una sola clase en la ventana: no hay nada que aprender, y el
        # isotonico devolveria una constante.
        return None

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x, y)
    xs = np.asarray(iso.X_thresholds_, dtype=float)
    ys = np.asarray(iso.y_thresholds_, dtype=float)

    fila = {
        "fitted_at_utc": _now().isoformat(),
        "n_samples": len(filas),
        "window_days": int(window_days),
        "base_rate": float(y.mean()),
        "x_json": json.dumps([round(v, 6) for v in xs.tolist()]),
        "y_json": json.dumps([round(v, 6) for v in ys.tolist()]),
    }
    con.execute(
        """INSERT OR REPLACE INTO chain_calibrator
           (id, fitted_at_utc, n_samples, window_days, base_rate, x_json, y_json)
           VALUES (1,?,?,?,?,?,?)""",
        (fila["fitted_at_utc"], fila["n_samples"], fila["window_days"],
         fila["base_rate"], fila["x_json"], fila["y_json"]),
    )
    con.commit()
    return fila


def load_chain_calibrator(
    con: sqlite3.Connection, *, max_age_days: float = DEFAULT_MAX_AGE_DAYS
) -> ChainCalibrator | None:
    """El calibrador vigente, o None si no hay o esta vencido."""
    try:
        fila = con.execute(
            """SELECT fitted_at_utc, n_samples, x_json, y_json
                 FROM chain_calibrator WHERE id = 1"""
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if not fila:
        return None

    fitted_at, n_samples, x_json, y_json = fila
    try:
        edad = (_now() - _parse_utc(fitted_at)).total_seconds() / 86400
    except ValueError:
        return None
    if edad > max_age_days:
        return None

    xs = np.array(json.loads(x_json), dtype=float)
    ys = np.array(json.loads(y_json), dtype=float)
    if xs.size < 2 or xs.size != ys.size:
        return None
    return ChainCalibrator(
        xs=xs, ys=ys, fitted_at_utc=str(fitted_at),
        n_samples=int(n_samples), age_days=round(edad, 2),
    )


def calibrator_status(con: sqlite3.Connection) -> dict:
    """Lo que hace falta para verlo en /admin/db-stats y alertarlo."""
    try:
        fila = con.execute(
            """SELECT fitted_at_utc, n_samples, window_days, base_rate
                 FROM chain_calibrator WHERE id = 1"""
        ).fetchone()
    except sqlite3.OperationalError:
        fila = None
    if not fila:
        return {"present": False, "stale": True,
                "detail": "todavia no se ajusto ninguna vez"}

    fitted_at, n_samples, window_days, base_rate = fila
    try:
        edad = (_now() - _parse_utc(fitted_at)).total_seconds() / 86400
    except ValueError:
        return {"present": False, "stale": True,
                "detail": f"fecha ilegible: {fitted_at}"}
    vencido = edad > DEFAULT_MAX_AGE_DAYS
    return {
        "present": True,
        "fitted_at_utc": str(fitted_at),
        "age_days": round(edad, 2),
        "max_age_days": DEFAULT_MAX_AGE_DAYS,
        "n_samples": int(n_samples),
        "window_days": int(window_days),
        "base_rate": round(float(base_rate), 4) if base_rate is not None else None,
        "stale": vencido,
        "detail": (
            f"ajustado hace {edad:.1f} dias con {n_samples:,} vuelos"
            + (" — VENCIDO, no se aplica" if vencido else "")
        ),
    }

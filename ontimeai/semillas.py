"""Siembra de vuelos "de hoy" en una copia de `live_data.db`.

El `live_data.db` versionado es un snapshot fijo: no tiene vuelos de hoy, así
que `/flights` devuelve una lista vacía contra él y todo lo que cuelga de un
vuelo —detalle, historial, cascada— no se puede ejercitar. Este módulo
siembra vuelos dentro de la ventana activa de la API (ver
`api._latest_predictions_active`), con predicción, SHAP y, para los que ya
aterrizaron, resultado.

Lo usan dos cosas:

  - el fixture `client` de `tests/test_api.py`, que siembra UN vuelo para que
    los tests de esquema no se salten con "No flights in DB today";
  - `scripts/make_fixture_db.py`, que siembra N vuelos para armar la base
    contra la que corren las pruebas de carga con k6 (`docs/STRESS_K6.md`).

Siempre sobre una copia: nunca sobre el archivo del repo.
"""
from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Rutas desde ATL. La ventana activa de la API filtra `origin = 'ATL'`, así
# que todo lo que se siembra para aparecer en /flights sale de ATL.
DESTINOS = ("MIA", "LGA", "ORD", "DFW", "LAX", "MCO", "JFK", "BOS", "DEN", "SEA")
AEROLINEAS = ("DL", "WN", "AA", "UA", "F9")
TIPOS_DE_AERONAVE = ("B738", "A321", "B752", "A320", "B739", "CRJ9")
FEATURES_SHAP = (
    "DEP_HOUR", "ORIG_WX_SKNT", "CARRIER", "DEST",
    "CRS_ELAPSED_TIME", "DAY_OF_WEEK", "ORIG_WX_VSBY",
)
UMBRAL = 0.10
ESTRATEGIA = "quantile@0.22"


@dataclass(frozen=True)
class Siembra:
    """Cuántas filas se insertaron, para informarlo y para los tests."""

    vuelos: int
    predicciones: int
    shap: int
    resultados: int
    piernas_siguientes: int
    ids: tuple[str, ...]


def _iso(momento: datetime) -> str:
    return momento.replace(microsecond=0).isoformat()


def _insertar_vuelo(con, *, fid, ident, carrier, numero, tail, origen, destino,
                    sale, llega, tipo, visto) -> None:
    con.execute(
        """INSERT OR REPLACE INTO flights
           (fa_flight_id, stable_id, ident_iata, op_carrier, flight_number,
            tail_num, origin, dest, fl_date, scheduled_out_utc,
            scheduled_in_utc, aircraft_type, cancelled, diverted,
            first_seen_utc, last_updated_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (fid, fid, ident, carrier, numero, tail, origen, destino,
         sale[:10], sale, llega, tipo, 0, 0, visto, visto),
    )


def _insertar_prediccion(con, *, fid, predicho, proba, proba_raw=None) -> None:
    con.execute(
        """INSERT OR REPLACE INTO predictions
           (fa_flight_id, stable_id, predicted_at_utc, proba_delay, proba_raw,
            predicted_delay, threshold_used, threshold_strategy,
            prediction_phase)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (fid, fid, predicho, proba, proba_raw, int(proba >= UMBRAL),
         UMBRAL, ESTRATEGIA, "PRE_DEPARTURE"),
    )


def _insertar_shap(con, *, fid, predicho, contribuciones) -> int:
    for rank, (feature, valor, valor_feature) in enumerate(contribuciones, start=1):
        con.execute(
            """INSERT OR REPLACE INTO prediction_shap
               (fa_flight_id, predicted_at_utc, feature_name, shap_value,
                feature_value, rank)
               VALUES (?,?,?,?,?,?)""",
            (fid, predicho, feature, valor, valor_feature, rank),
        )
    return len(contribuciones)


def _insertar_resultado(con, *, fid, sale, llega, demora_salida, demora_llegada,
                        asentado) -> None:
    salio = sale + timedelta(minutes=demora_salida)
    aterrizo = llega + timedelta(minutes=demora_llegada)
    con.execute(
        """INSERT OR REPLACE INTO actuals
           (fa_flight_id, stable_id, actual_out_utc, actual_off_utc,
            actual_on_utc, actual_in_utc, arr_delay_min, departure_delay_min,
            cancelled, diverted, settled_at_utc, source_provider)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (fid, fid, _iso(salio), _iso(salio + timedelta(minutes=14)),
         _iso(aterrizo - timedelta(minutes=8)), _iso(aterrizo),
         round(demora_llegada, 1), round(demora_salida, 1), 0, 0,
         _iso(asentado), "fr24"),
    )


def sembrar_vuelo_de_hoy(db: Path | str, fid: str = "TEST-SEED-0001") -> None:
    """Un vuelo dentro de la ventana activa, con predicción, SHAP y resultado.

    Es la siembra mínima de `tests/test_api.py`: valores fijos, sin azar, para
    que el fixture sea el mismo en cada corrida.
    """
    ahora = datetime.now(timezone.utc)
    sale = _iso(ahora + timedelta(hours=2))
    llega = _iso(ahora + timedelta(hours=4))
    predicho = _iso(ahora - timedelta(minutes=10))

    con = sqlite3.connect(str(db))
    try:
        _insertar_vuelo(
            con, fid=fid, ident="DL9999", carrier="DL", numero="9999",
            tail="N999DL", origen="ATL", destino="MIA", sale=sale, llega=llega,
            tipo="B738", visto=predicho,
        )
        _insertar_prediccion(con, fid=fid, predicho=predicho, proba=0.42)
        _insertar_shap(
            con, fid=fid, predicho=predicho,
            contribuciones=(("DEP_HOUR", 0.21, "12"), ("ORIG_WX_SKNT", -0.08, "12"),
                            ("CARRIER", 0.05, "12")),
        )
        con.commit()
    finally:
        con.close()


def sembrar_vuelos_de_hoy(
    db: Path | str,
    n: int = 200,
    semilla: int = 2026,
    *,
    ahora: datetime | None = None,
    prefijo: str = "SEED",
) -> Siembra:
    """Siembra `n` vuelos de hoy desde ATL, reproducibles por `semilla`.

    Cada vuelo lleva una predicción pre-salida con tres contribuciones SHAP.
    Los que según su horario ya aterrizaron reciben además un resultado en
    `actuals`, y todos tienen una pierna siguiente del mismo avión
    (destino → ATL, sin predicción) para que `/cascade/{id}` tenga por dónde
    propagar en vez de devolver "no conocemos la próxima pierna".

    Los horarios se reparten entre 5 horas atrás y 17 adelante, que es la
    ventana que `/flights` considera activa (6 h atrás a 18 h adelante), con
    margen para que una corrida larga no los deje afuera.

    `ahora` se puede fijar para que la siembra sea determinista también en el
    tiempo; por defecto es el instante de la llamada.
    """
    if n < 1:
        raise ValueError("hay que sembrar al menos un vuelo")
    rng = random.Random(semilla)
    ahora = (ahora or datetime.now(timezone.utc)).astimezone(timezone.utc)

    ids: list[str] = []
    filas_shap = resultados = 0
    con = sqlite3.connect(str(db))
    try:
        for i in range(1, n + 1):
            fid = f"{prefijo}-{i:04d}"
            carrier = rng.choice(AEROLINEAS)
            numero = str(rng.randint(100, 4999))
            tail = f"N{rng.randint(100, 999)}{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{i % 26 + 65:c}"
            destino = rng.choice(DESTINOS)
            sale = ahora + timedelta(minutes=rng.uniform(-5 * 60, 17 * 60))
            llega = sale + timedelta(minutes=rng.randrange(75, 300, 5))
            predicho = sale - timedelta(minutes=rng.randint(90, 180))

            _insertar_vuelo(
                con, fid=fid, ident=f"{carrier}{numero}", carrier=carrier,
                numero=numero, tail=tail, origen="ATL", destino=destino,
                sale=_iso(sale), llega=_iso(llega),
                tipo=rng.choice(TIPOS_DE_AERONAVE), visto=_iso(predicho),
            )

            # Beta(2, 8): mayoría baja, cola de vuelos con riesgo alto. Es la
            # forma de la distribución servida, no un valor medido.
            proba = round(rng.betavariate(2, 8), 4)
            _insertar_prediccion(
                con, fid=fid, predicho=_iso(predicho), proba=proba,
                proba_raw=round(proba * rng.uniform(0.8, 1.0), 4),
            )
            features = rng.sample(FEATURES_SHAP, 3)
            filas_shap += _insertar_shap(
                con, fid=fid, predicho=_iso(predicho),
                contribuciones=tuple(
                    (f, round(rng.uniform(-0.15, 0.25), 4), str(rng.randint(0, 23)))
                    for f in features
                ),
            )

            if llega < ahora - timedelta(minutes=20):
                demora_llegada = max(-25.0, rng.gauss(2.0, 20.0))
                _insertar_resultado(
                    con, fid=fid, sale=sale, llega=llega,
                    demora_salida=max(-10.0, demora_llegada + rng.gauss(5.0, 8.0)),
                    demora_llegada=demora_llegada,
                    asentado=llega + timedelta(minutes=25),
                )
                resultados += 1

            # La pierna de vuelta: mismo avión, sale de donde aterrizó.
            vuelta_sale = llega + timedelta(minutes=rng.randint(45, 90))
            _insertar_vuelo(
                con, fid=f"{fid}-R", ident=f"{carrier}{int(numero) + 1}",
                carrier=carrier, numero=str(int(numero) + 1), tail=tail,
                origen=destino, destino="ATL", sale=_iso(vuelta_sale),
                llega=_iso(vuelta_sale + timedelta(minutes=rng.randrange(75, 300, 5))),
                tipo=None, visto=_iso(predicho),
            )
            ids.append(fid)
        con.commit()
    finally:
        con.close()

    return Siembra(
        vuelos=n, predicciones=n, shap=filas_shap, resultados=resultados,
        piernas_siguientes=n, ids=tuple(ids),
    )

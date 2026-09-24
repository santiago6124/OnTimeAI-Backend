"""Per-tick pipeline: pull schedules + actuals + weather, predict, store.

Designed to be cron'd every 30 min. Idempotent: safe to re-run.

Data source modes (env var `LIVE_DATA_SOURCE`):
    - aeroapi (default): pulls all data from AeroAPI per tick (~$0.10/tick)
    - harvester: skips AeroAPI; reads flights/actuals already populated by
                 OnTimeAI-Scrapper into the shared bucket DB (~$0.001/tick).

Usage:
    python3 live_pull.py                            # default: AeroAPI mode
    LIVE_DATA_SOURCE=harvester python3 live_pull.py # uses harvester buffer
    python3 live_pull.py --schedule-hours 6         # look 6h ahead instead
    python3 live_pull.py --no-weather               # skip IEM pull
    python3 live_pull.py --dry-run                  # print plan, skip writes
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ontimeai.live import (
    open_db, fetch_airport_flights, fetch_iem_obs,
    aeroapi_to_flight_row, upsert_flights, upsert_actuals_from_aeroapi, upsert_weather,
    build_inference_frame, chain_walk_inbound, AIRPORTS, stable_id,
    snapshot_nas_status, latest_nas_status, gdp_post_prediction_adjust,
    compute_atl_arrival_congestion, carrier_delay_rate_bayesian,
    intermediate_dep_delay_adjust, estimated_dep_delay_adjust,
    compute_adsb_eta_delay, adsb_eta_adjust,
    compute_adsb_holding_min, adsb_holding_adjust,
)
from ontimeai.lineage_fallback import load_lookups, build_live_turnaround_lookups
from ontimeai.model import (
    load_artifact,
    predict_proba,
    select_threshold,
    select_threshold_and_label,
)
from ontimeai.training_store import (
    enqueue_prediction_snapshots,
    enqueue_recent_outcomes,
    training_store_enabled,
    training_store_required,
)
from predict import prepare_inference_frame
from ontimeai.config import ARTIFACTS_DIR


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# Cuanto vale una observacion antes de volver a pedirla. IEM publica METAR una
# vez por hora; 50 minutos deja margen para que el ciclo siguiente la renueve
# sin pedirla cuatro veces de gusto.
WEATHER_FRESH_MINUTES = int(os.getenv("WEATHER_FRESH_MINUTES", "50"))


def _airports_with_fresh_weather(conn, minutos: int) -> set[str]:
    """Estaciones con una observacion de menos de `minutos`.

    `valid_utc` se guarda sin offset, asi que se compara contra `datetime('now')`
    que en SQLite tambien es UTC. Leerlo como hora local daria tres horas de mas
    en Argentina y ningun aeropuerto figuraria fresco.
    """
    try:
        filas = conn.execute(
            """SELECT station FROM weather_obs
                GROUP BY station
               HAVING MAX(valid_utc) > datetime('now', ?)""",
            (f"-{int(minutos)} minutes",),
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    return {r[0] for r in filas if r[0]}


def load_gate_departure_delays(
    conn, stable_ids: list[str]
) -> tuple[dict[str, float], set[str]]:
    """Demora de PUERTA de los vuelos que ya despegaron, y cuales ya aterrizaron.

    Filtra `source_provider = 'aeroapi'` a proposito.
    `intermediate_dep_delay_adjust` usa bandas validadas contra BTS para demora
    de puerta, y solo aeroapi la mide: expone `actual_out` y su
    `departure_delay` es gate-out.

    FR24 guarda en esa misma columna `actual_off - scheduled_out`, o sea demora
    de puerta MAS rodaje —no expone gate-out—. Medido sobre 410.650 filas fr24,
    la mediana de `departure_delay_min - arr_delay_min` da +31,2 min contra
    +7,0 en aeroapi: ese delta es el rodaje de ATL. Sin el filtro, un vuelo que
    empujaba en horario y rodaba media hora entraba en la banda 30-60 y saltaba
    a p=0.90. Ver issue #11.

    El costo es cobertura: fr24 es el 96% de la muestra desde Fase 4, asi que
    el ajuste queda casi inactivo. Es el resultado correcto mientras no exista
    una demora de puerta comparable entre proveedores.
    """
    if not stable_ids:
        return {}, set()
    placeholders = ",".join("?" for _ in stable_ids)
    dep_delay_map = {
        row[0]: float(row[1])
        for row in conn.execute(
            f"""SELECT stable_id, departure_delay_min FROM actuals
               WHERE stable_id IN ({placeholders})
                 AND departure_delay_min IS NOT NULL
                 AND source_provider = 'aeroapi'
                 AND actual_in_utc IS NULL""",  # despego pero todavia no aterrizo
            stable_ids,
        ).fetchall()
    }
    landed_ids = {
        row[0]
        for row in conn.execute(
            f"""SELECT stable_id FROM actuals
               WHERE stable_id IN ({placeholders})
                 AND actual_in_utc IS NOT NULL""",
            stable_ids,
        ).fetchall()
    }
    return dep_delay_map, landed_ids


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", default=ARTIFACTS_DIR / "4year_v9")
    p.add_argument("--airport", default="KATL")
    p.add_argument("--schedule-hours", type=int, default=6,
                   help="Pull scheduled departures for the next N hours")
    p.add_argument("--actuals-hours", type=int, default=4,
                   help="Pull arrivals from the past N hours to settle actuals")
    p.add_argument("--actuals-offset-hours", type=int, default=0,
                   help="Shift the actuals window back by N hours (e.g. --actuals-hours 3 "
                        "--actuals-offset-hours 6 covers the window 9h..6h ago)")
    p.add_argument("--max-pages", type=int, default=2,
                   help="Max client-side cursor pages per endpoint (1 page ≈ 15 flights)")
    p.add_argument("--skip-arrivals-sched", action="store_true",
                   help="Skip the scheduled_arrivals endpoint (saves API calls)")
    p.add_argument("--skip-actuals", action="store_true",
                   help="Skip the arrivals endpoint (saves API calls)")
    p.add_argument("--no-weather", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--chain-walk-max",
        type=int,
        default=20,
        help=(
            "Max AeroAPI calls per tick to chain-walk `inbound_fa_flight_id` "
            "→ hydrates prev_arr_delay_tail without waiting for backfill. "
            "Set to 0 to disable. Each call costs ~$0.005."
        ),
    )
    p.add_argument(
        "--target-pos-rate",
        type=float,
        default=0.22,
        help=(
            "Target predicted-positive rate for quantile threshold (default 0.22, "
            "matches v4_full test base rate). Set to 0 to fall back to artifact threshold."
        ),
    )
    p.add_argument(
        "--abs-threshold",
        type=float,
        default=0.0,
        help=(
            "Fixed absolute probability cutoff. When > 0 it OVERRIDES --target-pos-rate: "
            "a flight is flagged delayed iff proba >= this value. Lets the predicted-positive "
            "rate track live conditions (more alerts on storm days, fewer on calm days) instead "
            "of forcing a constant ~22%. Backtest on v9 live actuals: abs@0.50 ~doubles precision "
            "(0.47->0.64) at similar recall vs quantile@0.22. Default 0 = keep quantile behavior."
        ),
    )
    args = p.parse_args()

    # LIVE_DATA_SOURCE: env-var switch entre AeroAPI (default) y harvester (buffer GCS).
    # Fase 4 del plan: depreca AeroAPI cuando el harvester demuestre paridad de AUC.
    data_source = os.getenv("LIVE_DATA_SOURCE", "aeroapi").lower()
    if data_source not in ("aeroapi", "harvester"):
        raise SystemExit(f"LIVE_DATA_SOURCE='{data_source}' inválido; usar 'aeroapi' o 'harvester'")
    is_harvester_mode = data_source == "harvester"

    now = datetime.now(timezone.utc)
    sched_start = now
    sched_end = now + timedelta(hours=args.schedule_hours)
    arr_end   = now - timedelta(hours=args.actuals_offset_hours)
    arr_start = arr_end - timedelta(hours=args.actuals_hours)

    print(f"Tick {now.isoformat()}")
    print(f"  data source:     {data_source}")
    print(f"  schedule window: {_iso(sched_start)} → {_iso(sched_end)}")
    print(f"  actuals window:  {_iso(arr_start)} → {_iso(arr_end)}")

    if args.dry_run:
        if is_harvester_mode:
            print("\n[dry-run] would read flights/actuals from shared bucket DB + IEM weather")
        else:
            print("\n[dry-run] would call AeroAPI scheduled_departures + arrivals + IEM")
        return 0

    conn = open_db()

    # Limpieza: eliminar predicciones realizadas después del scheduled_out_utc del vuelo.
    # Estas predicciones son ruido — el vuelo ya debía haber salido cuando se predijo.
    # Es idempotente; el pipeline ya no genera nuevas (corte en scheduled_out_utc).
    deleted = conn.execute(
        """
        DELETE FROM predictions
        WHERE EXISTS (
            SELECT 1 FROM flights f
            WHERE f.fa_flight_id = predictions.fa_flight_id
              AND f.scheduled_out_utc IS NOT NULL
              AND datetime(predictions.predicted_at_utc) > datetime(f.scheduled_out_utc)
        )
        """
    ).rowcount
    conn.commit()
    if deleted:
        print(f"   cleanup: {deleted} predicciones post-salida eliminadas")

    cur = conn.execute(
        "INSERT INTO runs (started_utc) VALUES (?)", (now.isoformat(),),
    )
    run_id = cur.lastrowid
    conn.commit()

    sched: list[dict] = []
    arr_sched: list[dict] = []
    arrived: list[dict] = []
    sched_rows: list[dict] = []
    arr_sched_rows: list[dict] = []
    n_act = 0
    n_chain_calls = 0
    n_chain_actuals = 0

    if is_harvester_mode:
        # ---- harvester mode: leemos del buffer poblado por OnTimeAI-Scrapper ----
        print("\n[1-3b] harvester mode: leyendo flights del buffer GCS (no AeroAPI calls)")
        # Vuelos KATL en la ventana sched — el harvester ya los pobló
        # (origin='ATL' OR dest='ATL' en schema IATA del scrapper)
        # SQLite datetime() normaliza ambos formatos:
        #   '2026-05-21T20:25:00'       (AeroAPI, sin TZ)
        #   '2026-05-22T22:00:00+00:00' (FR24, con TZ)
        cur = conn.execute(
            """
            SELECT fa_flight_id, origin, dest, tail_num, op_carrier,
                   scheduled_out_utc, scheduled_in_utc
            FROM flights
            WHERE (
                (origin = 'ATL'
                 AND scheduled_out_utc IS NOT NULL
                 AND datetime(scheduled_out_utc) >= datetime(?)
                 AND datetime(scheduled_out_utc) < datetime(?))
                OR
                (dest = 'ATL'
                 AND scheduled_in_utc IS NOT NULL
                 AND datetime(scheduled_in_utc) >= datetime(?)
                 AND datetime(scheduled_in_utc) < datetime(?))
            )
            ORDER BY COALESCE(scheduled_out_utc, scheduled_in_utc)
            """,
            (_iso(sched_start), _iso(sched_end), _iso(sched_start), _iso(sched_end)),
        )
        cols = [d[0] for d in cur.description]
        sched_rows = [dict(zip(cols, r)) for r in cur.fetchall() if r[0]]
        # Heurística para mantener compat con el código downstream:
        # tratamos los vuelos cuyo destino es ATL como "arr_sched" y los demás como sched.
        arr_sched_rows = [r for r in sched_rows if r.get("dest") == "ATL"]
        sched_rows = [r for r in sched_rows if r.get("origin") == "ATL"]
        print(f"   buffer hits: {len(sched_rows)} departures + {len(arr_sched_rows)} arrivals (ventana sched)")
        # n_act y chain_walk: ambos los hace el harvester, no replicamos
    else:
        # ---- AeroAPI mode (default, comportamiento original) ----
        print("\n[1] AeroAPI scheduled_departures...")
        try:
            sched = fetch_airport_flights(args.airport, "scheduled_departures",
                                          _iso(sched_start), _iso(sched_end), args.max_pages)
            print(f"   pulled {len(sched)} scheduled departures")
            sched_rows = [r for r in (aeroapi_to_flight_row(rec) for rec in sched) if r]
            n_sched = upsert_flights(conn, sched_rows)
            print(f"   upserted {n_sched} flights to DB (after ATL+known-airports filter)")
        except RuntimeError as _e:
            print(f"   [1] skipped (rate-limited): {_e}")
            sched = []

        if not args.skip_arrivals_sched:
            # ---- 2. scheduled arrivals (KATL) — captures FLOW=ARR_TO_ATL ----
            time.sleep(5)
            print("\n[2] AeroAPI scheduled_arrivals...")
            try:
                arr_sched = fetch_airport_flights(args.airport, "scheduled_arrivals",
                                                  _iso(sched_start), _iso(sched_end), args.max_pages)
                print(f"   pulled {len(arr_sched)} scheduled arrivals")
                arr_sched_rows = [r for r in (aeroapi_to_flight_row(rec) for rec in arr_sched) if r]
                n_arr_sched = upsert_flights(conn, arr_sched_rows)
                print(f"   upserted {n_arr_sched} flights to DB")
            except RuntimeError as _e:
                print(f"   [2] skipped (rate-limited): {_e}")
                arr_sched = []
        else:
            print("\n[2] (skipped scheduled_arrivals)")
            arr_sched = []

        # ---- 2b. Intermediate actuals (Tier 3 #1): some scheduled_* records
        # already have actual_off because the target took off but hasn't
        # arrived yet. Persist them so the prediction phase can boost proba
        # using the empirical P(arr_delay|dep_delay) relationship.
        intermediate_recs = [
            r for r in (sched + arr_sched)
            if r.get("actual_off")
        ]
        if intermediate_recs:
            n_intermediate = upsert_actuals_from_aeroapi(conn, intermediate_recs)
            print(f"\n[2b] intermediate dep_delay captures: {n_intermediate} flights already departed")

        if not args.skip_actuals:
            # Cap actuals at 4 pages — sufficient for last 4h of ATL arrivals,
            # and avoids burning API quota needed for scheduled endpoints.
            actuals_pages = min(args.max_pages, 4)

            # ---- 3. completed arrivals to KATL → actuals (settles ARR_TO_ATL preds) ----
            time.sleep(5)
            print("\n[3] AeroAPI arrivals (completed at KATL)...")
            try:
                arrived = fetch_airport_flights(args.airport, "arrivals",
                                                _iso(arr_start), _iso(arr_end), actuals_pages)
                print(f"   pulled {len(arrived)} arrivals")
                arrived_filt = [r for r in arrived if r.get("actual_in")]
                n_act_arr = upsert_actuals_from_aeroapi(conn, arrived_filt)
                print(f"   wrote {n_act_arr} actuals")
            except RuntimeError as _e:
                print(f"   [3] skipped (rate-limited): {_e}")
                n_act_arr = 0

            # ---- 3a. completed departures from KATL → actuals (settles DEP_FROM_ATL preds)
            # Non-fatal: if rate-limited after steps 1+2+3, log and continue.
            time.sleep(5)
            print("\n[3a] AeroAPI departures (completed from KATL)...")
            try:
                departed = fetch_airport_flights(args.airport, "departures",
                                                 _iso(arr_start), _iso(arr_end), actuals_pages)
                landed = [r for r in departed if r.get("actual_in")]
                en_route = [r for r in departed if r.get("actual_off") and not r.get("actual_in")]
                print(f"   pulled {len(departed)} departures, {len(landed)} landed, {len(en_route)} en route")
                # Save arr_delay for landed flights
                n_act_dep = upsert_actuals_from_aeroapi(conn, landed)
                # Also save actual_off for en-route flights so they leave "Programado" state
                if en_route:
                    n_dep_out = upsert_actuals_from_aeroapi(conn, en_route)
                    print(f"   wrote {n_act_dep} actuals (landed) + {n_dep_out} actual_off (en route)")
                else:
                    print(f"   wrote {n_act_dep} actuals")
            except RuntimeError as _e:
                print(f"   [3a] skipped (rate-limited): {_e}")
                n_act_dep = 0
            n_act = n_act_arr + n_act_dep
        else:
            print("\n[3] (skipped actuals)")

        # ---- 3b. chain-walk inbound_fa_flight_id → hydrate lineage on demand ----
        if args.chain_walk_max > 0:
            print("\n[3b] Chain-walk inbound_fa_flight_id...")
            n_chain_calls, n_chain_actuals = chain_walk_inbound(
                conn,
                sched_rows + arr_sched_rows,
                max_calls=args.chain_walk_max,
            )
            print(
                f"   chain-walk: {n_chain_calls} AeroAPI calls "
                f"(~${n_chain_calls * 0.005:.2f} USD), {n_chain_actuals} actuals hydrated"
            )
        else:
            print("\n[3b] (chain-walk disabled)")

    # ---- 4. weather ----
    n_wx = 0
    if not args.no_weather:
        print("\n[4] IEM METAR refresh...")
        _t_weather = time.monotonic()
        # Only fetch weather for airports that appear in today's flights (~40-50),
        # not the full 326-airport universe (would exceed 300s task timeout).
        active_airports: set[str] = set()
        for r in sched_rows + arr_sched_rows:
            if r.get("origin"):
                active_airports.add(r["origin"])
            if r.get("dest"):
                active_airports.add(r["dest"])
        active_airports &= AIRPORTS  # keep only ones we have IEM network info for

        # No volver a pedir el clima que ya tenemos fresco.
        #
        # IEM publica METAR una vez por hora, y el ciclo corre cada 15 minutos:
        # tres de cada cuatro pedidos traian exactamente el mismo dato. Eso era
        # gratis mientras eran ~6 aeropuertos. Al descubrir el horario futuro
        # pasaron a ser 135, con 46 pedidos de red, y como IEM limita por tasa
        # cada uno paga 5 segundos de espera: el ciclo salto de 4,8 a 13,5
        # minutos contra un scheduler de 15.
        frescos = _airports_with_fresh_weather(conn, WEATHER_FRESH_MINUTES)
        pendientes = active_airports - frescos
        # Se informan los frescos ENTRE LOS ACTIVOS, no todos los de la base:
        # `frescos` incluye estaciones de vuelos viejos y el conteo salia mayor
        # que el de activos, que no tiene sentido leerlo.
        print(f"   {len(active_airports)} aeropuertos activos, "
              f"{len(active_airports & frescos)} ya frescos "
              f"(<{WEATHER_FRESH_MINUTES} min), {len(pendientes)} a consultar")
        if pendientes:
            wx = fetch_iem_obs(pendientes, sched_start - timedelta(hours=2),
                               sched_end + timedelta(hours=2))
            if not wx.empty:
                n_wx = upsert_weather(conn, wx)
                print(f"   upserted {n_wx} weather observations")

    # ---- 4a. Bootstrap unseen tails (Layer 2 lineage fix) ----
    # When a tail appears in today's schedule but is not in `tail_lineage_cache`,
    # insert a placeholder row with hydrated_until in the past. The harvester's
    # `select_tails_to_hydrate` prioritizes never > expired, so unseen tails get
    # chain-walked within ~30 min of first appearance instead of relying on
    # the fallback for that tail's entire schedule.
    target_tails = set()
    for r in sched_rows + arr_sched_rows:
        t = r.get("tail_num") or r.get("registration")
        if t and str(t).strip():
            target_tails.add(str(t).strip())
    if target_tails:
        placeholders = ",".join(["?"] * len(target_tails))
        cur = conn.execute(
            f"SELECT tail FROM tail_lineage_cache WHERE tail IN ({placeholders})",
            list(target_tails),
        )
        cached_tails = {r[0] for r in cur.fetchall()}
        new_tails = target_tails - cached_tails
        if new_tails:
            conn.executemany(
                "INSERT OR IGNORE INTO tail_lineage_cache(tail, hydrated_until, "
                "last_pull_source, last_pull_ok, consecutive_failures) "
                "VALUES (?, '1970-01-01T00:00:00+00:00', 'bootstrap-request', 0, 0)",
                [(t,) for t in new_tails],
            )
            conn.commit()
            print(f"[4] clima listo en {time.monotonic() - _t_weather:.0f} s")
            print(f"\n[4a] bootstrap: queued {len(new_tails)} unseen tails for next harvester tick")
        else:
            print(f"[4] clima listo en {time.monotonic() - _t_weather:.0f} s")
            print(f"\n[4a] bootstrap: all {len(target_tails)} target tails already cached")

    # ---- 4b. FAA NAS Status snapshot (Tier 2 #K) ----
    # Capture programs (GDP/GS/Closure) active right now. Persists into
    # nas_status table for both historical dataset building and immediate
    # post-prediction adjustment below.
    n_nas = snapshot_nas_status(conn)
    if n_nas > 0:
        print(f"\n[4b] NAS snapshot: {n_nas} airports under active programs")
    else:
        print("\n[4b] NAS snapshot: no active programs (or fetch failed)")

    # ---- 5. predict scheduled flights ----
    print("\n[5] Building features and predicting...")
    _t_predict = time.monotonic()
    # Decouple the prediction target from this cycle's fetch. Predicting only the
    # flights pulled this tick means each flight is scored ~once, minutes before
    # departure (the scheduled_departures feed only surfaces near-term flights).
    # Instead, every cycle re-predict ALL upcoming ATL departures whose departure
    # is still AHEAD — so a fresh PRE-departure prediction always exists and
    # refines as the gate nears.
    #
    # Two cases qualify as "still needs a prediction":
    #   1. Upcoming: estimated_out is still in the future (AeroAPI updated the estimate).
    #   2. Delayed on ground: scheduled_out passed but no actual_off yet AND
    #      the scheduled time is within DELAY_GRACE_H hours — covers flights that
    #      AeroAPI hasn't updated the estimated_out for (shows stale scheduled time)
    #      but hasn't departed either (stuck at gate due to delay).
    # Without case 2, a delayed flight like F93512 stops getting predictions the
    # moment its scheduled time passes, leaving "Programado" with a stale score.
    # The grace window excludes departed-but-unsettled flights older than DELAY_GRACE_H
    # (AeroAPI typically settles actual_off within 30-60 min; beyond 2h it's safe to
    # assume the lag is too large or the flight is actually airborne and needs no update).
    horizon_h = int(os.getenv("PREDICT_HORIZON_HOURS", str(args.schedule_hours)))
    # Only predict ATL departures whose scheduled_out is still in the future.
    # Arrivals (dest=ATL) are fetched for actuals/tail-lineage only — not predicted.
    # Delayed departures that already passed their scheduled_out are excluded:
    # once the scheduled window closes, the pre-departure prediction is final.
    db_dep_rows = conn.execute(
        """
        SELECT f.fa_flight_id
        FROM flights f
        LEFT JOIN actuals a ON a.fa_flight_id = f.fa_flight_id
        WHERE f.origin = 'ATL'
          AND f.scheduled_out_utc IS NOT NULL
          AND datetime(f.scheduled_out_utc) > datetime(?)
          AND datetime(f.scheduled_out_utc) <= datetime(?, ?)
          AND COALESCE(f.cancelled, 0) = 0
        """,
        (_iso(now), _iso(now), f"+{horizon_h} hours"),
    ).fetchall()
    # Departures only — arr_sched_rows fetched for lineage/actuals, not predicted.
    target_ids = list(
        {r[0] for r in db_dep_rows}
        | {r["fa_flight_id"] for r in sched_rows}
    )
    print(f"   target: {len(db_dep_rows)} standing ATL deps (scheduled_out ahead) + "
          f"{len(sched_rows)} freshly fetched deps → {len(target_ids)} unique "
          f"(arrivals: {len(arr_sched_rows)} fetched for lineage only)")

    # Export labels independently from feature snapshots.  The rolling lookback
    # makes the outbox self-healing after scheduler gaps, while deterministic
    # outcome event IDs keep retries idempotent.
    if training_store_enabled():
        try:
            outcome_lookback_days = max(
                1, int(os.getenv("TRAINING_OUTCOME_LOOKBACK_DAYS", "7")),
            )
            n_outcomes_queued = enqueue_recent_outcomes(
                conn,
                # Use a fresh timestamp after API sleeps/upserts.  The tick's
                # start time can precede actuals.settled_at_utc (or even cross
                # midnight), which would invert revision causality/partitioning.
                observed_at_utc=datetime.now(timezone.utc),
                lookback_days=outcome_lookback_days,
            )
            print(f"   training store: queued {n_outcomes_queued} new outcome revisions")
        except Exception as exc:
            print(
                "   ⚠ training outcome enqueue failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if training_store_required():
                raise
    if not target_ids:
        print("   no flights to predict")
        conn.execute(
            "UPDATE runs SET finished_utc=?, flights_pulled=?, flights_predicted=?,"
            " flights_targeted=?, actuals_updated=?, weather_obs_added=? WHERE run_id=?",
            (datetime.now(timezone.utc).isoformat(), len(sched) + len(arr_sched), 0,
             0, n_act, n_wx, run_id),
        )
        conn.commit()
        conn.close()
        return 0

    df = build_inference_frame(conn, target_ids, history_days=7)
    if df.empty:
        print("   inference frame empty")
        conn.execute(
            """UPDATE runs SET finished_utc=?, flights_pulled=?, flights_predicted=?,
                      flights_targeted=?, actuals_updated=?, weather_obs_added=?
                WHERE run_id=?""",
            (
                datetime.now(timezone.utc).isoformat(),
                len(sched) + len(arr_sched), 0, len(target_ids), n_act, n_wx, run_id,
            ),
        )
        conn.commit()
        conn.close()
        return 0

    target_mask = df["fa_flight_id"].isin(target_ids) & df["ARR_DELAY"].isna()
    print(f"   {target_mask.sum()} target rows | {(~target_mask).sum()} history rows for lineage")

    meta = load_artifact(args.artifact)

    fallback_path = ARTIFACTS_DIR / "lineage_fallback.joblib"
    fallback = None
    # Lineage fallback is default-on. The DISABLE_LINEAGE_FALLBACK env var was a
    # diagnostic flag during the SIGSEGV hunt (root cause turned out to be CRLF
    # in model.lgb, not the fallback). Keep the override path for repro
    # scenarios but never default to bypassing — without the fallback, NaN
    # lineage features force the model to extrapolate via TAIL_DELAY_DECAY and
    # other proxies, which sobre-estimates and degrades live AUC.
    if os.getenv("DISABLE_LINEAGE_FALLBACK", "").lower() in ("1", "true", "yes"):
        print("   lineage fallback disabled via DISABLE_LINEAGE_FALLBACK env (DIAGNOSTIC ONLY — remove this env var in production)")
    elif fallback_path.exists():
        try:
            fallback = load_lookups(fallback_path)
        except Exception as e:
            print(f"   ⚠ fallback load failed (pickle/pandas version mismatch): {e}")
            print("   proceeding without lineage fallback")
    else:
        print(f"   ⚠ lineage fallback artifact missing at {fallback_path} — predictions will use NaN priors (degrades AUC)")
    if fallback is not None:
        print(f"   loaded cold-deck fallback ({fallback_path.name})")

    # Layer 3 fallback: enrich `fallback` with live-DB turnaround aggregations
    # so prev_turnaround_tail_min cascades through (carrier, route, hour)
    # buckets rather than collapsing to the 60-min constant. Computed each tick
    # from the local copy of `flights` -- cheap (~50ms) and self-updating.
    if fallback is None:
        fallback = {}
    try:
        live_turnaround = build_live_turnaround_lookups(conn, days=14)
        fallback.update(live_turnaround)
        meta_t = live_turnaround.get("_meta", {})
        print(
            f"   live turnaround lookups: n_rows={meta_t.get('n_rows', 0)} "
            f"global={live_turnaround.get('global_turnaround_mean', 60.0):.1f}min "
            f"carrier_buckets={len(live_turnaround.get('turnaround_carrier_mean', []))} "
            f"route_buckets={len(live_turnaround.get('turnaround_carrier_route_mean', []))}"
        )
    except Exception as e:
        print(f"   ⚠ live turnaround lookup build failed: {e} (cascade disabled, falls back to 60.0)")

    # ---- Feature NaN logging & Quality assertions ----
    X_raw = prepare_inference_frame(
        df,
        meta["feature_cols"],
        meta["cat_mapping"],
        fallback_lookup=None,
        apply_category_mapping=False,
    )

    target_idx = df.index[target_mask]
    if not target_idx.empty:
        X_raw_target = X_raw.loc[target_idx].copy()

        # Ensure numeric columns are parsed as numeric for accurate NaN checks
        cat_cols_set = set(meta.get("cat_cols", []))
        for c in X_raw_target.columns:
            if c not in cat_cols_set:
                X_raw_target[c] = pd.to_numeric(X_raw_target[c], errors="coerce")

        print("   Raw Feature NaN Rates (before cold-deck fallback) for target flights:")
        col_nan_rates = X_raw_target.isna().mean()
        sorted_col_nans = sorted(col_nan_rates.items(), key=lambda x: x[1], reverse=True)
        for col, rate in sorted_col_nans:
            if rate > 0.0:
                print(f"     - {col}: {rate:.1%}")

        nan_rates_per_flight = X_raw_target.isna().mean(axis=1)
        too_many_nans = nan_rates_per_flight[nan_rates_per_flight > 0.6]
        if not too_many_nans.empty:
            print(f"   Quality Assertion: Skipping prediction for {len(too_many_nans)} flights with >60% NaN features:")
            for idx, rate in too_many_nans.items():
                fl_id = df.loc[idx, "fa_flight_id"]
                print(f"     - {fl_id}: {rate:.1%} NaN features")

            # Exclude flights that failed quality assertion from prediction and database insert
            target_mask = target_mask & (~df.index.isin(too_many_nans.index))
            print(f"   {target_mask.sum()} target rows remaining after quality filtering")
    # --------------------------------------------------

    X = prepare_inference_frame(
        df, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=fallback,
    )
    # Coerce non-categorical object columns to numeric (live path may have all-NaN
    # weather columns that pd.NA leaves as object, which LightGBM rejects).
    cat_cols_set = set(meta.get("cat_cols", []))
    for c in X.columns:
        if c in cat_cols_set:
            continue
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    booster_proba = predict_proba(meta["booster"], X)
    calibrated_proba = booster_proba.copy()
    if meta.get("calibrator") is not None and meta["target"] == "binary":
        calibrated_proba = meta["calibrator"].transform(calibrated_proba)
    # Keep the existing variable name for the operational adjustment path.  The
    # training snapshot stores booster/calibrated/final probabilities separately.
    proba = calibrated_proba

    # Umbral de referencia sobre la probabilidad calibrada.
    #
    # NO es el que etiqueta. El operativo se calcula despues de la cadena de
    # ajustes, sobre la misma distribucion contra la que se compara; este queda
    # para poder medir cuanto la corre la cadena, que es informacion util en el
    # log cuando algo se desalinea.
    target_proba = proba[target_mask.to_numpy()]
    threshold_calibrated, strategy_calibrated = select_threshold(
        target_proba,
        target_pos_rate=args.target_pos_rate,
        artifact_threshold=float(meta["threshold"]),
        abs_threshold=args.abs_threshold,
    )
    print(
        f"   threshold(calibrada) strategy={strategy_calibrated} "
        f"value={threshold_calibrated:.4f} | proba_target n={target_proba.size} "
        f"mean={target_proba.mean():.3f} std={target_proba.std():.3f}"
        if target_proba.size > 0
        else f"   threshold(calibrada) strategy={strategy_calibrated} "
             f"value={threshold_calibrated:.4f} (no targets)"
    )

    # Post-prediction GDP adjustment (Tier 2 #K). The v9 model was trained
    # without GDP_FLAG in its feature_cols, so even if the live feature
    # pipeline computes GDP_FLAG it doesn't reach the booster. We compensate
    # by adjusting `proba` after the fact when ORIGIN or DEST is under a
    # program. `proba_raw` is preserved so we can A/B test the lift.
    nas_state = latest_nas_status(conn, max_age_minutes=30)
    gdp_adjust_enabled = os.getenv("GDP_ADJUST", "1").lower() in ("1", "true", "yes")
    if nas_state and gdp_adjust_enabled:
        affected = [a for a in nas_state if a in {df.loc[i, "origin"] for i in df.index[target_mask]} | {df.loc[i, "dest"] for i in df.index[target_mask]}]
        if affected:
            print(f"   GDP adjustment: {len(affected)} target-relevant airports under program: {affected}")
    elif not gdp_adjust_enabled:
        print("   GDP adjustment disabled via GDP_ADJUST env")

    # Pre-fetch intermediate dep_delay for all target stable_ids in one query.
    target_indices = list(df.index[target_mask])
    target_stable_ids = [stable_id(df.loc[i, "fa_flight_id"]) for i in target_indices]
    dep_delay_map, landed_ids = load_gate_departure_delays(conn, target_stable_ids)

    dep_adjust_enabled = os.getenv("DEP_DELAY_ADJUST", "1").lower() in ("1", "true", "yes")

    # Apagado por defecto: hay que pedirlo explicitamente.
    #
    # `estimated_dep_delay_adjust` heredo las bandas de su hermano
    # `intermediate_dep_delay_adjust`, que estan validadas contra BTS para
    # demoras de salida YA OCURRIDAS. Aplicadas a una estimacion de horario
    # —que todavia puede recuperarse— no describen nada.
    #
    # Medido sobre 3.979 vuelos con resultado real: el ajuste empujaba 2.678
    # vuelos (67% del lote) y les mostraba una probabilidad media del 69,4%
    # cuando la tasa real de ese grupo era 6,2%. El lote sin empujar daba 4,2%,
    # asi que la senal existe pero vale un factor 1,5, no un factor 10. Que
    # dos tercios de los vuelos tengan "demora estimada significativa" tambien
    # sugiere que `estimated_out_utc` no significa lo que el ajuste asume.
    #
    # El reemplazo correcto es que el modelo aprenda la relacion, no fijarla a
    # mano: ver issue #7.
    est_adjust_enabled = os.getenv("EST_DELAY_ADJUST", "0").lower() in ("1", "true", "yes")
    if dep_delay_map and dep_adjust_enabled:
        print(f"   intermediate dep_delay available for {len(dep_delay_map)} targets")

    # Calibrador post-cadena. Puede no existir —base nueva— o estar vencido, y
    # en ambos casos se sirve la salida cruda de la cadena, que es exagerada
    # pero conocida. Un calibrador viejo corrige en la direccion equivocada y
    # nadie lo sospecha; ver el issue #15 y ontimeai/chain_calibration.py.
    from ontimeai.chain_calibration import load_chain_calibrator

    chain_calibrator = load_chain_calibrator(conn)
    if chain_calibrator is None:
        print("   calibrador post-cadena: ausente o vencido, se sirve sin calibrar")
    else:
        print(
            f"   calibrador post-cadena: ajustado hace "
            f"{chain_calibrator.age_days:.1f} dias con "
            f"{chain_calibrator.n_samples:,} vuelos"
        )

    adsb_enabled = os.getenv("ADSB_ADJUST", "1").lower() in ("1", "true", "yes")
    adsb_capture_by_tail: dict[str, str] = {}
    target_tails = sorted(
        {
            str(df.loc[i, "tail_num"]).strip().upper()
            for i in target_indices
            if pd.notna(df.loc[i, "tail_num"]) and str(df.loc[i, "tail_num"]).strip()
        }
    )
    if target_tails:
        try:
            placeholders = ",".join("?" for _ in target_tails)
            adsb_capture_by_tail = {
                str(row[0]).strip().upper(): row[1]
                for row in conn.execute(
                    f"""SELECT UPPER(TRIM(registration)), MAX(captured_at_utc)
                        FROM aircraft_position
                        WHERE UPPER(TRIM(registration)) IN ({placeholders})
                        GROUP BY UPPER(TRIM(registration))""",
                    target_tails,
                ).fetchall()
                if row[0] and row[1]
            }
        except Exception:
            # Older DB snapshots may not contain aircraft_position yet.
            adsb_capture_by_tail = {}

    # Persist only target predictions
    pred_now = datetime.now(timezone.utc).isoformat()
    pred_rows: list[tuple] = []
    snapshot_contexts: dict[object, dict[str, object]] = {}
    for i in target_indices:
        proba_raw = float(proba[i])
        origin = df.loc[i, "origin"]
        dest = df.loc[i, "dest"]
        gdp_orig = nas_state.get(origin, {}).get("delay_min", 0.0) if nas_state else 0.0
        gdp_dest = nas_state.get(dest, {}).get("delay_min", 0.0) if nas_state else 0.0

        # Adjustment chain: raw → GDP → estimated dep_delay (if not departed) OR intermediate dep_delay (if departed) → ADS-B ETA.
        proba_after_gdp = (
            gdp_post_prediction_adjust(proba_raw, gdp_orig, gdp_dest)
            if gdp_adjust_enabled
            else proba_raw
        )
        sid = stable_id(df.loc[i, "fa_flight_id"])
        dep_delay = dep_delay_map.get(sid)  # None if target hasn't departed yet
        est_delay = None
        if dep_delay is None:
            sched_out = df.loc[i, "scheduled_out_utc"]
            est_out = df.loc[i, "estimated_out_utc"]
            if sched_out and est_out:
                try:
                    dt_sched = pd.to_datetime(sched_out, utc=True)
                    dt_est = pd.to_datetime(est_out, utc=True)
                    est_delay = float((dt_est - dt_sched).total_seconds() / 60.0)
                except Exception:
                    pass
            proba_after_dep = (
                estimated_dep_delay_adjust(proba_after_gdp, est_delay)
                if est_adjust_enabled
                else proba_after_gdp
            )
        else:
            proba_after_dep = (
                intermediate_dep_delay_adjust(proba_after_gdp, dep_delay)
                if dep_adjust_enabled
                else proba_after_gdp
            )
        # Tier 3 #3 — ADS-B ETA boost (only for arrivals to ATL in air now)
        adsb_delay = (
            compute_adsb_eta_delay(
                conn,
                tail_num=df.loc[i, "tail_num"],
                scheduled_in_utc=df.loc[i, "scheduled_in_utc"],
                dest=df.loc[i, "dest"],
            )
            if adsb_enabled
            else None
        )
        proba_after_eta = (
            adsb_eta_adjust(proba_after_dep, adsb_delay) if adsb_enabled else proba_after_dep
        )

        # Mid win #5 — ADS-B holding pattern detection (orbiting near ATL)
        holding_min = (
            compute_adsb_holding_min(
                conn,
                tail_num=df.loc[i, "tail_num"],
                dest=df.loc[i, "dest"],
            )
            if adsb_enabled
            else None
        )
        proba_chain = (
            adsb_holding_adjust(proba_after_eta, holding_min) if adsb_enabled else proba_after_eta
        )
        # La cadena produce un puntaje, no una probabilidad: sus constantes
        # estan puestas a mano. El calibrador aprende la traduccion de los
        # vuelos que ya aterrizaron. Se guardan los dos: `proba_chain` es sobre
        # lo que se reajusta, `proba_adj` es lo que se sirve.
        proba_adj = (
            chain_calibrator(proba_chain) if chain_calibrator is not None else proba_chain
        )
        # La etiqueta no se puede decidir todavia: el umbral se calcula sobre
        # el conjunto completo de probabilidades ya ajustadas, que recien
        # termina de armarse cuando termina este loop.

        # Prediction phase: classify based on flight departure/landing status
        if sid in landed_ids:
            phase = "POST_LANDING"
        elif sid in dep_delay_map:
            phase = "EN_ROUTE"
        else:
            phase = "PRE_DEPARTURE"

        # Diagnostic features (Tier 2 #I, #J) — computed live, NOT in
        # feature_cols of v9. Persisted for future v9.1 retrain analysis.
        atl_window = compute_atl_arrival_congestion(
            conn, df.loc[i, "scheduled_in_utc"], window_minutes=30,
        )
        carrier_smooth = carrier_delay_rate_bayesian(
            conn, df.loc[i, "op_carrier"], df.loc[i, "scheduled_off_utc"],
            window_hours=24.0, alpha=20.0, prior_window_hours=168.0,
        )
        pred_rows.append([
            df.loc[i, "fa_flight_id"], sid,
            pred_now, float(proba_adj), None,   # etiqueta: se completa abajo
            None, None,                         # umbral y estrategia: idem
            proba_raw, float(gdp_orig), float(gdp_dest),
            float(proba_chain),
            int(atl_window), (float(carrier_smooth) if carrier_smooth is not None else None),
            (float(dep_delay) if dep_delay is not None else None),
            (float(adsb_delay) if adsb_delay is not None else None),
            (float(holding_min) if holding_min is not None else None),
            phase,
        ])
        snapshot_contexts[i] = {
            "prediction_phase": phase,
            "booster_probability": float(booster_proba[i]),
            "calibrated_probability": float(calibrated_proba[i]),
            "chain_probability": float(proba_chain),
            "final_probability": float(proba_adj),
            "probability_after_gdp": float(proba_after_gdp),
            "probability_after_departure": float(proba_after_dep),
            "probability_after_adsb_eta": float(proba_after_eta),
            "gdp_orig_delay_min": float(gdp_orig),
            "gdp_dest_delay_min": float(gdp_dest),
            "estimated_dep_delay_min": est_delay,
            "intermediate_dep_delay_min": dep_delay,
            "adsb_eta_delay_min": adsb_delay,
            "adsb_holding_min": holding_min,
            "atl_arrivals_in_window_30min": int(atl_window),
            "carrier_delay_rate_smooth": carrier_smooth,
            "fallback_applied": fallback is not None,
            "gdp_adjust_enabled": gdp_adjust_enabled,
            "estimated_delay_adjust_enabled": (
                est_adjust_enabled if dep_delay is None else False
            ),
            "departure_delay_adjust_enabled": dep_adjust_enabled,
            "adsb_adjust_enabled": adsb_enabled,
            "nas_snapshot_captured_at_utc": max(
                (
                    state.get("captured_at_utc")
                    for state in (
                        nas_state.get(origin, {}) if nas_state else {},
                        nas_state.get(dest, {}) if nas_state else {},
                    )
                    if state.get("captured_at_utc")
                ),
                default=None,
            ),
            "adsb_latest_captured_at_utc": adsb_capture_by_tail.get(
                str(df.loc[i, "tail_num"]).strip().upper()
                if pd.notna(df.loc[i, "tail_num"])
                else ""
            ),
        }

    # ── Umbral operativo ────────────────────────────────────────────────────
    #
    # Se calcula aca, sobre las probabilidades YA ajustadas, porque es contra
    # esas que se compara.
    #
    # Antes salia del percentil 78 de `calibrated_proba` y despues se comparaba
    # contra `proba_adj`. Los cuatro ajustes son noisy-OR —`p' = 1-(1-p)(1-p_x)`,
    # que solo puede subir la probabilidad—, asi que la tasa de positivos era
    # por construccion mayor a la pretendida, sin importar el dato. Medido sobre
    # 3.980 vuelos con resultado real: 78% del lote marcado como demorado contra
    # el 22% que pide la estrategia, con la tasa real en 7,7%.
    adjusted_proba = np.array([row[3] for row in pred_rows], dtype=float)
    threshold_used, threshold_strategy, adjusted_labels = select_threshold_and_label(
        adjusted_proba,
        target_pos_rate=args.target_pos_rate,
        artifact_threshold=float(meta["threshold"]),
        abs_threshold=args.abs_threshold,
    )
    for row, i, label in zip(pred_rows, target_indices, adjusted_labels):
        row[4] = int(label)
        row[5] = float(threshold_used)
        row[6] = threshold_strategy
        ctx = snapshot_contexts.get(i)
        if ctx is not None:
            ctx["predicted_label"] = row[4]
            ctx["threshold_used"] = float(threshold_used)
            ctx["threshold_strategy"] = threshold_strategy
    if adjusted_proba.size:
        print(
            f"   threshold(operativo) strategy={threshold_strategy} "
            f"value={threshold_used:.4f} | pos_pred_rate="
            f"{(adjusted_proba >= threshold_used).mean():.3f} | la cadena lo "
            f"corrio {threshold_used - threshold_calibrated:+.4f} respecto de "
            f"la calibrada"
        )

    conn.executemany(
        """INSERT OR REPLACE INTO predictions
           (fa_flight_id, stable_id, predicted_at_utc, proba_delay, predicted_delay,
            threshold_used, threshold_strategy,
            proba_raw, gdp_orig_delay_min, gdp_dest_delay_min, proba_chain,
            atl_arrivals_in_window_30min, carrier_delay_rate_smooth,
            intermediate_dep_delay_min, adsb_eta_delay_min, adsb_holding_min,
            prediction_phase)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        pred_rows,
    )

    # The exact X rows and prediction metadata enter the same SQLite transaction
    # as the serving predictions.  live_job publishes only the winning DB
    # generation, so a generation-conflict retry cannot leak orphan snapshots.
    if training_store_enabled() and pred_rows:
        try:
            n_snapshots_queued = enqueue_prediction_snapshots(
                conn,
                flight_frame=df,
                raw_features=X_raw,
                model_features=X,
                target_indices=target_indices,
                contexts=snapshot_contexts,
                predicted_at_utc=pred_now,
                run_id=run_id,
                data_source=data_source,
                artifact_dir=args.artifact,
                feature_cols=meta["feature_cols"],
                cat_cols=meta.get("cat_cols", []),
                cat_mapping=meta.get("cat_mapping", {}),
            )
            print(f"   training store: queued {n_snapshots_queued} causal feature snapshots")
        except Exception as exc:
            print(
                "   ⚠ training snapshot enqueue failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if training_store_required():
                raise
    conn.commit()
    # Las filas se arman posicionalmente y estos indices tienen que seguir al
    # INSERT de arriba. Se nombran para que agregar una columna no obligue a
    # recontar a mano, que es como se rompio al sumar `proba_chain`.
    I_PROBA_FINAL, I_PROBA_RAW, I_PROBA_CHAIN = 3, 7, 10
    I_DEP_DELAY, I_ADSB_ETA, I_ADSB_HOLDING = 13, 14, 15

    n_any = sum(
        1 for r in pred_rows
        if r[I_PROBA_RAW] is not None
        and abs(r[I_PROBA_RAW] - r[I_PROBA_CHAIN]) > 1e-6
    )
    n_calibrado = sum(
        1 for r in pred_rows if abs(r[I_PROBA_CHAIN] - r[I_PROBA_FINAL]) > 1e-6
    )
    n_dep = sum(1 for r in pred_rows if r[I_DEP_DELAY] is not None and r[I_DEP_DELAY] > 5)
    n_adsb_available = sum(1 for r in pred_rows if r[I_ADSB_ETA] is not None)
    n_adsb_boost = sum(
        1 for r in pred_rows if r[I_ADSB_ETA] is not None and r[I_ADSB_ETA] > 5
    )
    n_holding = sum(
        1 for r in pred_rows if r[I_ADSB_HOLDING] is not None and r[I_ADSB_HOLDING] >= 5
    )
    print(f"   wrote {len(pred_rows)} predictions ({n_any} adjusted, "
          f"{n_calibrado} calibrated, "
          f"{n_dep} via dep_delay, {n_adsb_available} with adsb_eta "
          f"of which {n_adsb_boost} boosted, {n_holding} in holding pattern)")

    # ---- SHAP top-K persistence (Fix D in FIXES_PLAN.md) ----
    # Compute SHAP values for target rows only, persist top-15 by |shap|.
    # Disabled if SHAP_TOPK=0. Failure is non-fatal (predictions already written).
    shap_topk = int(os.getenv("SHAP_TOPK", "15"))
    if shap_topk > 0 and pred_rows:
        try:
            X_target = X.loc[target_indices]
            # LightGBM native: pred_contrib=True returns per-row SHAP vector + bias term.
            contribs = meta["booster"].predict(X_target, pred_contrib=True)
            # contribs shape: (n_targets, n_features + 1) — last col is bias
            feat_names = list(X_target.columns)
            shap_matrix = contribs[:, :-1]

            shap_rows: list[tuple] = []
            for row_idx, df_idx in enumerate(target_indices):
                fa_id = df.loc[df_idx, "fa_flight_id"]
                row_shap = shap_matrix[row_idx]
                # Rank by absolute contribution
                abs_order = np.argsort(-np.abs(row_shap))[:shap_topk]
                for rank, feat_idx in enumerate(abs_order, start=1):
                    fname = feat_names[feat_idx]
                    sval = float(row_shap[feat_idx])
                    fval = X_target.iloc[row_idx][fname]
                    fval_str = "NaN" if pd.isna(fval) else str(fval)
                    shap_rows.append((fa_id, pred_now, fname, sval, fval_str, rank))

            conn.executemany(
                """INSERT OR REPLACE INTO prediction_shap
                   (fa_flight_id, predicted_at_utc, feature_name, shap_value,
                    feature_value, rank)
                   VALUES (?,?,?,?,?,?)""",
                shap_rows,
            )
            conn.commit()
            print(f"   wrote {len(shap_rows)} SHAP values (top-{shap_topk} × {len(pred_rows)} preds)")
        except Exception as e:
            print(f"   ⚠ SHAP persistence failed (non-fatal): {type(e).__name__}: {e}")

    conn.execute(
        "UPDATE runs SET finished_utc=?, flights_pulled=?, flights_predicted=?,"
        " flights_targeted=?, actuals_updated=?, weather_obs_added=? WHERE run_id=?",
        (datetime.now(timezone.utc).isoformat(),
         len(sched) + len(arr_sched) + len(arrived),
         len(pred_rows), len(target_ids), n_act, n_wx, run_id),
    )
    conn.commit()
    # Close the connection so SQLite flushes cached pages to disk BEFORE the
    # subsequent GCS upload in live_job.py reads /tmp/live_data.db.
    conn.close()

    print(f"[5] features + inferencia en {time.monotonic() - _t_predict:.0f} s")
    print(f"\nDone. run_id={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

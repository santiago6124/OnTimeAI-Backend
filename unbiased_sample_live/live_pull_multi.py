"""Multi-airport live pull for unbiased delay prediction evaluation.

Replaces the KATL-only live_pull.py with a weighted rotation over the top-30
US airports. Selection weight ∝ sqrt(BTS flight volume), so high-volume hubs
appear more frequently and the live DB mirrors the full training distribution.

Cost model (defaults: k=5 airports, 2 pages/call):
  5 airports × 2 endpoints × 1 call each = 10 calls/tick
  10 × 48 ticks/day = 480 calls/day ≈ $2.40/day at $0.005/call

Shares live_data.db with the original live_pull.py — both scripts are additive
and safe to run on the same database. If cron'd together, offset by 15 minutes
to avoid SQLite write contention.

Usage:
    python3 live_pull_multi.py                   # 5 airports, dry-run safe
    python3 live_pull_multi.py --k-airports 10   # 10 airports (~$4.80/day)
    python3 live_pull_multi.py --status          # show sampler state, exit
    python3 live_pull_multi.py --dry-run         # print plan, no I/O

Key differences from live_pull.py:
  - No ATL gate: accepts any flight between the 326-airport universe
  - FLOW_ATL encoded as NON_ATL for non-ATL-touching flights (v6 model has 3 categories)
  - Weather pulled only for the airports queried this tick (not all 326)
  - No scheduled_arrivals endpoint (departures from both sides cover the route)
  - Chain-walk disabled (multi-airport history buffer fills itself over time)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Resolve repo root so this script works when invoked from any directory
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ontimeai.live import (
    AIRPORTS,
    open_db,
    fetch_airport_flights,
    aeroapi_to_flight_row,
    upsert_flights,
    upsert_actuals_from_aeroapi,
    upsert_weather,
    fetch_iem_obs,
    build_inference_frame,
    stable_id,
)
from ontimeai.lineage_fallback import load_lookups
from ontimeai.model import load_artifact, predict_label, predict_proba, quantile_threshold
from ontimeai.config import ARTIFACTS_DIR
from predict import prepare_inference_frame
from sampler import AirportSampler

# AeroAPI uses ICAO codes; all top-30 airports are continental US (K+IATA)
# except the edge cases below (none are in the default weight table).
_IATA_ICAO_EXCEPTIONS: dict[str, str] = {
    "HNL": "PHNL",
    "ANC": "PANC",
    "FAI": "PAFA",
    "JNU": "PAJN",
}


def _iata_to_icao(iata: str) -> str:
    return _IATA_ICAO_EXCEPTIONS.get(iata.upper(), f"K{iata.upper()}")


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _flight_row_multi(rec: dict) -> dict | None:
    """Like aeroapi_to_flight_row but without the ATL gate.

    Accepts any flight where both origin and dest are in the 326-airport
    universe. strict_airport_filter=False bypasses the ATL-only check in
    live.py; we reapply the universe membership check ourselves.
    """
    row = aeroapi_to_flight_row(rec, strict_airport_filter=False)
    if row is None:
        return None
    if row["origin"] not in AIRPORTS or row["dest"] not in AIRPORTS:
        return None
    return row


def _fix_flow_atl(df: pd.DataFrame) -> pd.DataFrame:
    """Correct FLOW_ATL / PAR_AIRPORT for flights that don't touch ATL.

    build_inference_frame() defaults all non-ATL-origin flights to
    "ARR_TO_ATL", which is wrong for e.g. DFW→ORD. The v6_full model was
    trained with a third category "NON_ATL" and PAR_AIRPORT="" for these
    flights. Without this fix the model sees a distribution mismatch for
    every non-ATL flight it predicts.

    Do NOT apply this to the original live_pull.py — it only ever sees
    ATL-touching flights, so the two-value encoding is correct there.
    """
    out = df.copy()
    atl_touch = out["ORIGIN"].eq("ATL") | out["DEST"].eq("ATL")
    out.loc[~atl_touch, "FLOW_ATL"] = "NON_ATL"
    out.loc[~atl_touch, "PAR_AIRPORT"] = ""
    return out


def _choose_threshold(
    target_proba, meta: dict, target_pos_rate: float
) -> tuple[float, str]:
    if target_pos_rate > 0 and target_proba.size >= 5:
        return quantile_threshold(target_proba, target_pos_rate), f"quantile@{target_pos_rate:.2f}"
    return float(meta["threshold"]), "artifact"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Multi-airport live pull: weighted rotation over top-30 US airports."
    )
    p.add_argument(
        "--artifact",
        default=str(ARTIFACTS_DIR / "4year_v7_recal"),
        help="Path to model artifact directory",
    )
    p.add_argument(
        "--k-airports", type=int, default=5,
        help=(
            "Airports to pull per tick. Cost ≈ k×2 AeroAPI calls/tick. "
            "Default 5 → ~$2.40/day at 30-min cadence."
        ),
    )
    p.add_argument(
        "--budget-calls", type=int, default=30,
        help="Hard cap on AeroAPI calls per tick (safety net, default 30)",
    )
    p.add_argument(
        "--schedule-hours", type=int, default=2,
        help="Pull scheduled departures for next N hours (default 2)",
    )
    p.add_argument(
        "--actuals-hours", type=int, default=4,
        help="Pull completed departures from past N hours to settle actuals (default 4)",
    )
    p.add_argument(
        "--max-pages", type=int, default=2,
        help="Max cursor pages per AeroAPI call — 1 page ≈ 15 flights (default 2)",
    )
    p.add_argument("--no-weather", action="store_true", help="Skip IEM METAR pull")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print selected airports and cost estimate, skip all I/O",
    )
    p.add_argument(
        "--status", action="store_true",
        help="Print sampler overdue-score table and exit (no DB connection needed)",
    )
    p.add_argument(
        "--cost-per-call", type=float, default=0.005,
        help="AeroAPI cost per call in USD for budget reporting (default $0.005)",
    )
    p.add_argument(
        "--target-pos-rate", type=float, default=0.22,
        help=(
            "Target predicted-positive rate for quantile threshold (default 0.22). "
            "Set to 0 to fall back to artifact threshold."
        ),
    )
    p.add_argument(
        "--state-file", type=Path,
        default=Path(__file__).parent / "sampler_state.json",
        help="Path to sampler state JSON (default: <script_dir>/sampler_state.json)",
    )
    args = p.parse_args()

    sampler = AirportSampler(state_file=args.state_file)

    if args.status:
        sampler.report()
        return 0

    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()
    selected = sampler.select(args.k_airports, now_ts)

    est_calls = len(selected) * 2  # sched_departures + departures per airport
    print(f"Tick {now.isoformat()}")
    print(f"  selected airports : {', '.join(selected)}")
    print(f"  estimated calls   : {est_calls} (~${est_calls * args.cost_per_call:.3f})")
    print(f"  coverage          : {len(selected)}/{len(sampler.weights)} airports ({100*len(selected)/len(sampler.weights):.1f}%)")

    if args.dry_run:
        print("\n[dry-run] no API calls or DB writes")
        sampler.report(now_ts)
        return 0

    sched_start = now
    sched_end = now + timedelta(hours=args.schedule_hours)
    act_start = now - timedelta(hours=args.actuals_hours)
    act_end = now

    conn = open_db()
    cur = conn.execute(
        "INSERT INTO runs (started_utc) VALUES (?)", (now.isoformat(),)
    )
    run_id = cur.lastrowid
    conn.commit()

    meta = load_artifact(args.artifact)
    fallback_path = ARTIFACTS_DIR / "lineage_fallback.joblib"
    fallback = load_lookups(fallback_path) if fallback_path.exists() else None

    total_calls = 0
    all_sched_rows: list[dict] = []
    airports_pulled: list[str] = []
    n_actuals = 0

    # ---- per-airport pull loop ----
    for iata in selected:
        if total_calls >= args.budget_calls:
            print(f"\n  [budget] {total_calls} calls reached cap of {args.budget_calls}, stopping")
            break

        icao = _iata_to_icao(iata)
        ap_sched_rows: list[dict] = []

        # -- scheduled departures → prediction targets --
        print(f"\n[{iata}] scheduled_departures...")
        try:
            sched = fetch_airport_flights(
                icao, "scheduled_departures",
                _iso(sched_start), _iso(sched_end), args.max_pages,
            )
            total_calls += 1
            ap_sched_rows = [r for r in (_flight_row_multi(rec) for rec in sched) if r]
            n_up = upsert_flights(conn, ap_sched_rows)
            print(f"  pulled {len(sched)} → {n_up} upserted (universe filter)")
        except RuntimeError as e:
            print(f"  FAIL: {e}")

        # -- completed departures → settle actuals --
        if total_calls < args.budget_calls:
            print(f"[{iata}] departures (actuals)...")
            try:
                deps = fetch_airport_flights(
                    icao, "departures",
                    _iso(act_start), _iso(act_end), args.max_pages,
                )
                total_calls += 1
                # Upsert flight rows (updates tail_num, times, etc.)
                dep_rows = [r for r in (_flight_row_multi(rec) for rec in deps) if r]
                upsert_flights(conn, dep_rows)
                # Settle actuals for flights that have already landed
                landed = [r for r in deps if r.get("actual_in")]
                n_act = upsert_actuals_from_aeroapi(conn, landed)
                n_actuals += n_act
                print(f"  pulled {len(deps)}, {len(landed)} landed → {n_act} actuals")
            except RuntimeError as e:
                print(f"  FAIL: {e}")

        all_sched_rows.extend(ap_sched_rows)
        airports_pulled.append(iata)

    sampler.mark_pulled(airports_pulled, now_ts)

    # ---- weather for pulled airports only ----
    n_wx = 0
    if not args.no_weather and airports_pulled:
        print(f"\n[wx] IEM METAR for {', '.join(airports_pulled)}...")
        wx = fetch_iem_obs(
            set(airports_pulled),
            sched_start - timedelta(hours=2),
            sched_end + timedelta(hours=2),
        )
        if not wx.empty:
            n_wx = upsert_weather(conn, wx)
            print(f"  upserted {n_wx} observations")

    # ---- predict scheduled flights ----
    n_predicted = 0
    if all_sched_rows:
        print(f"\n[predict] Building features for {len(all_sched_rows)} target flights...")
        target_ids = [r["fa_flight_id"] for r in all_sched_rows]
        df = build_inference_frame(conn, target_ids, history_days=7)

        if not df.empty:
            # Fix FLOW_ATL / PAR_AIRPORT for non-ATL flights before inference
            df = _fix_flow_atl(df)

            target_mask = df["fa_flight_id"].isin(target_ids) & df["ARR_DELAY"].isna()
            print(f"  {target_mask.sum()} targets | {(~target_mask).sum()} history rows")

            X = prepare_inference_frame(
                df, meta["feature_cols"], meta["cat_mapping"], fallback_lookup=fallback,
            )
            cat_cols_set = set(meta.get("cat_cols", []))
            for c in X.columns:
                if c not in cat_cols_set and X[c].dtype == object:
                    X[c] = pd.to_numeric(X[c], errors="coerce")

            proba = predict_proba(meta["booster"], X)
            if meta.get("calibrator") is not None and meta["target"] == "binary":
                proba = meta["calibrator"].transform(proba)

            target_proba = proba[target_mask.to_numpy()]
            threshold_used, threshold_strategy = _choose_threshold(
                target_proba, meta, args.target_pos_rate
            )
            labels = predict_label(proba, threshold_used, "binary")

            if target_proba.size > 0:
                print(
                    f"  threshold={threshold_strategy} val={threshold_used:.4f} "
                    f"pos_rate={( target_proba >= threshold_used).mean():.3f}"
                )

            pred_now = datetime.now(timezone.utc).isoformat()
            pred_rows = [
                (
                    df.loc[i, "fa_flight_id"],
                    stable_id(df.loc[i, "fa_flight_id"]),
                    pred_now,
                    float(proba[i]),
                    int(labels[i]),
                    float(threshold_used),
                    threshold_strategy,
                )
                for i in df.index[target_mask]
            ]
            conn.executemany(
                """INSERT OR REPLACE INTO predictions
                   (fa_flight_id, stable_id, predicted_at_utc, proba_delay, predicted_delay,
                    threshold_used, threshold_strategy)
                   VALUES (?,?,?,?,?,?,?)""",
                pred_rows,
            )
            conn.commit()
            n_predicted = len(pred_rows)
            print(f"  wrote {n_predicted} predictions")

    # ---- close tick ----
    est_cost = total_calls * args.cost_per_call
    conn.execute(
        "UPDATE runs SET finished_utc=?, flights_pulled=?, flights_predicted=?, "
        "actuals_updated=?, weather_obs_added=?, notes=? WHERE run_id=?",
        (
            datetime.now(timezone.utc).isoformat(),
            len(all_sched_rows),
            n_predicted,
            n_actuals,
            n_wx,
            f"airports={','.join(airports_pulled)}",
            run_id,
        ),
    )
    conn.commit()

    print(f"\n=== Tick Summary (run_id={run_id}) ===")
    print(f"  airports pulled : {len(airports_pulled)}/{len(sampler.weights)} — {', '.join(airports_pulled)}")
    print(f"  flights upserted: {len(all_sched_rows)}")
    print(f"  predictions     : {n_predicted}")
    print(f"  actuals settled : {n_actuals}")
    print(f"  weather obs     : {n_wx}")
    print(f"  AeroAPI calls   : {total_calls} (est ${est_cost:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

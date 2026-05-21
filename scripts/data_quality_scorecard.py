"""Per-day data quality scorecard for the live pipeline.

Computes for each day in the live_data.db:
  - Flight count (expected ~1000-1200 ATL-touching/day)
  - Prediction coverage (% of flights with at least one prediction)
  - Actuals settlement rate (% of predictions with matched actuals)
  - Weather coverage (% of flights with non-null weather nearby)
  - Lineage hit rate (estimated from chain-walk actual counts)
  - Overall quality grade (A/B/C/F)

Usage:
    python3 scripts/data_quality_scorecard.py
    python3 scripts/data_quality_scorecard.py --out artifacts/data_quality_report.json
    python3 scripts/data_quality_scorecard.py --min-flights 500  # flag days below threshold
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ontimeai.live import open_db


def compute_scorecard(conn, min_flights: int = 500) -> dict:
    """Compute per-day quality metrics from live_data.db."""

    # --- flights per day ---
    flights_per_day = pd.read_sql_query(
        "SELECT fl_date, count(*) as n_flights, "
        "       count(distinct op_carrier) as n_carriers, "
        "       count(distinct origin || '-' || dest) as n_routes "
        "FROM flights GROUP BY fl_date ORDER BY fl_date",
        conn,
    )

    # --- predictions per day ---
    preds_per_day = pd.read_sql_query(
        "SELECT date(predicted_at_utc) as pred_date, "
        "       count(distinct fa_flight_id) as n_predicted_flights, "
        "       count(*) as n_prediction_rows "
        "FROM predictions GROUP BY pred_date ORDER BY pred_date",
        conn,
    )

    # --- actuals per day (by settled date) ---
    actuals_per_day = pd.read_sql_query(
        "SELECT date(settled_at_utc) as settled_date, "
        "       count(*) as n_actuals, "
        "       sum(CASE WHEN arr_delay_min IS NOT NULL THEN 1 ELSE 0 END) as n_with_delay, "
        "       avg(CASE WHEN arr_delay_min IS NOT NULL THEN arr_delay_min END) as avg_delay, "
        "       sum(CASE WHEN arr_delay_min > 15 THEN 1 ELSE 0 END) as n_delayed "
        "FROM actuals GROUP BY settled_date ORDER BY settled_date",
        conn,
    )

    # --- prediction-actual join rate (the metric that matters for eval) ---
    match_per_day = pd.read_sql_query(
        """SELECT f.fl_date,
                  count(distinct p.fa_flight_id) as n_predicted,
                  count(distinct CASE WHEN a.arr_delay_min IS NOT NULL
                                      THEN p.fa_flight_id END) as n_matched
           FROM predictions p
           JOIN flights f ON f.fa_flight_id = p.fa_flight_id
           LEFT JOIN actuals a ON a.stable_id = p.stable_id
           GROUP BY f.fl_date ORDER BY f.fl_date""",
        conn,
    )

    # --- weather coverage per day ---
    wx_per_day = pd.read_sql_query(
        "SELECT date(valid_utc) as wx_date, "
        "       count(*) as n_wx_obs, "
        "       count(distinct station) as n_stations "
        "FROM weather_obs GROUP BY wx_date ORDER BY wx_date",
        conn,
    )

    # --- runs per day (pipeline health) ---
    runs_per_day = pd.read_sql_query(
        "SELECT date(started_utc) as run_date, "
        "       count(*) as n_runs, "
        "       sum(CASE WHEN flights_predicted IS NOT NULL THEN 1 ELSE 0 END) as n_successful, "
        "       sum(CASE WHEN flights_predicted IS NULL THEN 1 ELSE 0 END) as n_failed, "
        "       avg(flights_predicted) as avg_flights_predicted, "
        "       avg(actuals_updated) as avg_actuals_updated "
        "FROM runs GROUP BY run_date ORDER BY run_date",
        conn,
    )

    # --- lineage proxy: how many flights have inbound_fa_flight_id + settled actual ---
    lineage_per_day = pd.read_sql_query(
        """SELECT f.fl_date,
                  count(*) as n_total,
                  sum(CASE WHEN f.inbound_fa_flight_id IS NOT NULL THEN 1 ELSE 0 END) as n_has_inbound,
                  sum(CASE WHEN f.inbound_fa_flight_id IS NOT NULL
                            AND a_inb.arr_delay_min IS NOT NULL THEN 1 ELSE 0 END) as n_lineage_available
           FROM flights f
           LEFT JOIN actuals a_inb ON a_inb.fa_flight_id = f.inbound_fa_flight_id
                                   OR a_inb.stable_id = (
                                       CASE WHEN f.inbound_fa_flight_id IS NOT NULL
                                            THEN substr(f.inbound_fa_flight_id, 1,
                                                        instr(f.inbound_fa_flight_id || '-', '-') - 1
                                                        + 1 +
                                                        instr(substr(f.inbound_fa_flight_id,
                                                              instr(f.inbound_fa_flight_id, '-') + 1) || '-', '-') - 1)
                                       END)
           GROUP BY f.fl_date ORDER BY f.fl_date""",
        conn,
    )

    # --- assemble per-day scorecard ---
    all_dates = set()
    for df in [flights_per_day, preds_per_day, actuals_per_day, wx_per_day, runs_per_day]:
        date_col = [c for c in df.columns if "date" in c.lower()][0] if len(df) > 0 else None
        if date_col and len(df) > 0:
            all_dates |= set(df[date_col].dropna().tolist())
    if flights_per_day is not None and len(flights_per_day) > 0:
        all_dates |= set(flights_per_day["fl_date"].dropna().tolist())

    all_dates = sorted(all_dates)

    days = []
    for d in all_dates:
        row = {"date": d}

        # flights
        f = flights_per_day[flights_per_day["fl_date"] == d]
        row["n_flights"] = int(f["n_flights"].iloc[0]) if len(f) > 0 else 0
        row["n_carriers"] = int(f["n_carriers"].iloc[0]) if len(f) > 0 else 0
        row["n_routes"] = int(f["n_routes"].iloc[0]) if len(f) > 0 else 0

        # predictions
        p = preds_per_day[preds_per_day["pred_date"] == d]
        row["n_predicted_flights"] = int(p["n_predicted_flights"].iloc[0]) if len(p) > 0 else 0
        row["prediction_coverage"] = (
            round(row["n_predicted_flights"] / max(row["n_flights"], 1), 3)
        )

        # actuals
        a = actuals_per_day[actuals_per_day["settled_date"] == d]
        row["n_actuals"] = int(a["n_actuals"].iloc[0]) if len(a) > 0 else 0
        row["n_with_delay"] = int(a["n_with_delay"].iloc[0]) if len(a) > 0 else 0
        row["avg_delay_min"] = round(float(a["avg_delay"].iloc[0]), 1) if len(a) > 0 and pd.notna(a["avg_delay"].iloc[0]) else None
        row["delay_rate"] = round(int(a["n_delayed"].iloc[0]) / max(row["n_with_delay"], 1), 3) if len(a) > 0 else None

        # match rate
        m = match_per_day[match_per_day["fl_date"] == d]
        row["n_pred_with_actual"] = int(m["n_matched"].iloc[0]) if len(m) > 0 else 0
        row["settlement_rate"] = (
            round(row["n_pred_with_actual"] / max(row["n_predicted_flights"], 1), 3)
        )

        # weather
        w = wx_per_day[wx_per_day["wx_date"] == d]
        row["n_wx_obs"] = int(w["n_wx_obs"].iloc[0]) if len(w) > 0 else 0
        row["n_wx_stations"] = int(w["n_stations"].iloc[0]) if len(w) > 0 else 0
        row["has_weather"] = row["n_wx_obs"] > 0

        # runs
        r = runs_per_day[runs_per_day["run_date"] == d]
        row["n_runs"] = int(r["n_runs"].iloc[0]) if len(r) > 0 else 0
        row["n_successful_runs"] = int(r["n_successful"].iloc[0]) if len(r) > 0 else 0
        row["n_failed_runs"] = int(r["n_failed"].iloc[0]) if len(r) > 0 else 0

        # lineage proxy
        ln = lineage_per_day[lineage_per_day["fl_date"] == d]
        row["n_has_inbound_id"] = int(ln["n_has_inbound"].iloc[0]) if len(ln) > 0 else 0
        row["lineage_hit_rate"] = (
            round(int(ln["n_lineage_available"].iloc[0]) / max(row["n_has_inbound_id"], 1), 3)
            if len(ln) > 0 else 0.0
        )

        # quality grade
        grade = "A"
        issues = []
        if row["n_flights"] < min_flights:
            grade = "F"
            issues.append(f"flights={row['n_flights']} < {min_flights}")
        elif row["n_flights"] < min_flights * 1.5:
            grade = max(grade, "C")
            issues.append(f"low flight count ({row['n_flights']})")
        if row["prediction_coverage"] < 0.3:
            grade = "F" if grade != "F" else grade
            issues.append(f"prediction coverage {row['prediction_coverage']:.0%}")
        elif row["prediction_coverage"] < 0.6:
            grade = max(grade, "C")
            issues.append(f"low prediction coverage ({row['prediction_coverage']:.0%})")
        if not row["has_weather"]:
            grade = max(grade, "C")
            issues.append("no weather data")
        if row["n_runs"] == 0:
            grade = "F"
            issues.append("no pipeline runs")
        elif row["n_failed_runs"] > row["n_successful_runs"]:
            grade = max(grade, "C")
            issues.append(f"{row['n_failed_runs']} failed runs")

        row["grade"] = grade
        row["issues"] = issues
        days.append(row)

    # --- aggregates ---
    total_flights = sum(d["n_flights"] for d in days)
    total_predicted = sum(d["n_predicted_flights"] for d in days)
    total_actuals = sum(d["n_actuals"] for d in days)
    total_matched = sum(d["n_pred_with_actual"] for d in days)
    grade_counts = {}
    for d in days:
        grade_counts[d["grade"]] = grade_counts.get(d["grade"], 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_days": len(days),
            "total_flights": total_flights,
            "total_predicted_flights": total_predicted,
            "total_actuals": total_actuals,
            "total_matched_predictions": total_matched,
            "overall_prediction_coverage": round(total_predicted / max(total_flights, 1), 3),
            "overall_settlement_rate": round(total_matched / max(total_predicted, 1), 3),
            "grade_distribution": grade_counts,
            "days_grade_F": grade_counts.get("F", 0),
            "days_grade_A": grade_counts.get("A", 0),
        },
        "per_day": days,
    }


def print_markdown(scorecard: dict) -> None:
    """Print a human-readable markdown summary."""
    s = scorecard["summary"]
    print("# Data Quality Scorecard\n")
    print(f"Generated: {scorecard['generated_at']}\n")
    print("## Summary\n")
    print(f"| Metric | Value |")
    print(f"|---|---|")
    print(f"| Days covered | {s['total_days']} |")
    print(f"| Total flights | {s['total_flights']:,} |")
    print(f"| Predicted flights | {s['total_predicted_flights']:,} |")
    print(f"| Matched (pred+actual) | {s['total_matched_predictions']:,} |")
    print(f"| Prediction coverage | {s['overall_prediction_coverage']:.1%} |")
    print(f"| Settlement rate | {s['overall_settlement_rate']:.1%} |")
    print(f"| Grade A days | {s['days_grade_A']} |")
    print(f"| Grade F days | {s['days_grade_F']} |")

    print("\n## Per-Day Breakdown\n")
    print("| Date | Flights | Predicted | Matched | Wx Obs | Lineage % | Grade | Issues |")
    print("|---|---|---|---|---|---|---|---|")
    for d in scorecard["per_day"]:
        issues_str = "; ".join(d["issues"]) if d["issues"] else "—"
        print(
            f"| {d['date']} | {d['n_flights']:,} | {d['n_predicted_flights']:,} "
            f"| {d['n_pred_with_actual']:,} | {d['n_wx_obs']:,} "
            f"| {d['lineage_hit_rate']:.0%} | {d['grade']} | {issues_str} |"
        )

    # Flag critical issues
    f_days = [d for d in scorecard["per_day"] if d["grade"] == "F"]
    if f_days:
        print(f"\n## ⚠ Critical: {len(f_days)} days with grade F\n")
        for d in f_days:
            print(f"- **{d['date']}**: {'; '.join(d['issues'])}")


def main() -> int:
    p = argparse.ArgumentParser(description="Data quality scorecard for live_data.db")
    p.add_argument("--out", type=Path, default=None, help="Save JSON report to file")
    p.add_argument("--min-flights", type=int, default=500,
                   help="Minimum flights per day for grade A (default 500)")
    args = p.parse_args()

    conn = open_db()
    scorecard = compute_scorecard(conn, min_flights=args.min_flights)
    print_markdown(scorecard)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(scorecard, indent=2, default=str))
        print(f"\nJSON report → {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Post-period analysis of live OnTimeAI predictions vs AeroAPI actuals.

Reads either the live SQLite DB or the snapshot CSV ledger and produces:

  artifacts/live_period_report_<since>_<until>.json   # all numbers
  artifacts/live_period_report_<since>_<until>.md     # narrative for thesis
  artifacts/live_period_plots/*.png                   # figures (if matplotlib)

Pure offline — zero AeroAPI cost.

Usage:
    python scripts/analyze_live_period.py --since 2026-05-05 --until 2026-05-09
    python scripts/analyze_live_period.py --since 2026-05-05 --until 2026-05-09 --source snapshots
    python scripts/analyze_live_period.py --since 2026-05-05 --until 2026-05-09 --no-plots
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "live_data.db"
SNAPSHOT_DIR = REPO / "artifacts" / "live_snapshots"

DELAY_THRESHOLD_MIN = 15.0
MIN_DAILY_N = 5
MIN_CARRIER_N = 10
MIN_HOUR_N = 3
N_CALIBRATION_BINS = 10


# ----------------------------- data loading ------------------------------


def load_from_db(conn: sqlite3.Connection, since_iso: str, until_iso: str) -> pd.DataFrame:
    """Predictions joined with actuals; only settled, non-cancelled, non-diverted.

    Uses `stable_id` for the join because AeroAPI returns slightly different
    `fa_flight_id` suffixes from `/scheduled_*` (predictions) vs `/departures` /
    `/arrivals` (actuals) for the same physical flight.
    """
    return pd.read_sql_query(
        """SELECT
              p.fa_flight_id, p.stable_id, p.predicted_at_utc, p.proba_delay,
              p.predicted_delay, p.threshold_used, p.threshold_strategy,
              a.actual_in_utc, a.arr_delay_min,
              f.scheduled_in_utc, f.op_carrier, f.origin, f.dest, f.tail_num
           FROM predictions p
           INNER JOIN actuals a ON a.stable_id = p.stable_id
           LEFT JOIN flights f ON f.fa_flight_id = p.fa_flight_id
           WHERE p.predicted_at_utc >= ?
             AND p.predicted_at_utc <= ?
             AND p.stable_id IS NOT NULL
             AND a.actual_in_utc IS NOT NULL
             AND a.arr_delay_min IS NOT NULL
             AND COALESCE(a.cancelled, 0) = 0
             AND COALESCE(a.diverted, 0) = 0""",
        conn, params=(since_iso, until_iso),
    )


def load_from_snapshots(snapshot_dir: Path, since_iso: str, until_iso: str) -> pd.DataFrame:
    pred_files = sorted(snapshot_dir.glob("predictions_*.csv"))
    act_files = sorted(snapshot_dir.glob("actuals_*.csv"))
    if not pred_files or not act_files:
        return pd.DataFrame()

    preds = pd.concat([pd.read_csv(f) for f in pred_files], ignore_index=True)
    preds = preds.drop_duplicates(subset=["fa_flight_id", "predicted_at_utc"])

    acts = pd.concat([pd.read_csv(f) for f in act_files], ignore_index=True)
    acts = acts.drop_duplicates(subset=["fa_flight_id"])
    # avoid column collisions on merge — keep only the new fields from actuals
    keep = ["fa_flight_id", "actual_in_utc", "arr_delay_min", "cancelled", "diverted"]
    acts = acts[[c for c in keep if c in acts.columns]]

    df = preds.merge(acts, on="fa_flight_id", how="inner")
    df = df.dropna(subset=["actual_in_utc", "arr_delay_min"])
    if "cancelled" in df.columns:
        df = df[pd.to_numeric(df["cancelled"], errors="coerce").fillna(0).astype(int) == 0]
    if "diverted" in df.columns:
        df = df[pd.to_numeric(df["diverted"], errors="coerce").fillna(0).astype(int) == 0]

    df["predicted_at_utc"] = pd.to_datetime(
        df["predicted_at_utc"], utc=True, errors="coerce", format="ISO8601",
    )
    since_ts = pd.to_datetime(since_iso, utc=True)
    until_ts = pd.to_datetime(until_iso, utc=True)
    return df[(df["predicted_at_utc"] >= since_ts) & (df["predicted_at_utc"] <= until_ts)]


# ----------------------------- metrics core ------------------------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {"n": 0}

    metrics: dict[str, Any] = {
        "n": int(n),
        "positive_rate_truth": float(y_true.mean()),
        "positive_rate_pred": float(y_pred.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_proba)),
    }
    if pd.Series(y_true).nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["log_loss"] = float(log_loss(y_true, np.clip(y_proba, 1e-9, 1 - 1e-9)))
    else:
        metrics["roc_auc"] = None
        metrics["log_loss"] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    return metrics


def compute_calibration(y_true: np.ndarray, y_proba: np.ndarray,
                        n_bins: int = N_CALIBRATION_BINS) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    bin_idx = np.clip(np.digitize(y_proba, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            rows.append({
                "bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                "n": 0, "mean_proba": None, "observed_pos_rate": None,
            })
            continue
        mean_proba = float(y_proba[mask].mean())
        observed = float(y_true[mask].mean())
        rows.append({
            "bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
            "n": n_b, "mean_proba": mean_proba, "observed_pos_rate": observed,
        })
        ece += (n_b / n) * abs(mean_proba - observed)
    return {"bins": rows, "ece": float(ece)}


def threshold_by_hour(df: pd.DataFrame) -> list[dict[str, Any]]:
    df = df.copy()
    df["hour_utc"] = pd.to_datetime(df["predicted_at_utc"], utc=True, format="ISO8601").dt.hour
    rows = []
    for h, sub in df.groupby("hour_utc"):
        if len(sub) < MIN_HOUR_N:
            continue
        rows.append({
            "hour_utc": int(h),
            "n": int(len(sub)),
            "median_threshold": float(sub["threshold_used"].median()),
            "mean_proba": float(sub["proba_delay"].mean()),
            "observed_pos_rate": float((sub["arr_delay_min"] > DELAY_THRESHOLD_MIN).mean()),
            "predicted_pos_rate": float(pd.to_numeric(sub["predicted_delay"], errors="coerce").mean()),
        })
    return rows


def per_carrier(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "op_carrier" not in df.columns:
        return []
    out = []
    for carrier, sub in df.groupby("op_carrier"):
        if pd.isna(carrier) or len(sub) < MIN_CARRIER_N:
            continue
        y_t = (sub["arr_delay_min"] > DELAY_THRESHOLD_MIN).astype(int).to_numpy()
        y_p = pd.to_numeric(sub["predicted_delay"], errors="coerce").fillna(0).astype(int).to_numpy()
        y_pr = sub["proba_delay"].astype(float).to_numpy()
        m = compute_metrics(y_t, y_p, y_pr)
        m["carrier"] = str(carrier)
        out.append(m)
    return sorted(out, key=lambda r: -r["n"])


def per_day(df: pd.DataFrame) -> list[dict[str, Any]]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["predicted_at_utc"], utc=True, format="ISO8601").dt.date
    out = []
    for date, sub in df.groupby("date"):
        if len(sub) < MIN_DAILY_N:
            continue
        y_t = (sub["arr_delay_min"] > DELAY_THRESHOLD_MIN).astype(int).to_numpy()
        y_p = pd.to_numeric(sub["predicted_delay"], errors="coerce").fillna(0).astype(int).to_numpy()
        y_pr = sub["proba_delay"].astype(float).to_numpy()
        m = compute_metrics(y_t, y_p, y_pr)
        m["date"] = str(date)
        out.append(m)
    return sorted(out, key=lambda r: r["date"])


def per_strategy(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for strategy, sub in df.groupby(df["threshold_strategy"].fillna("unknown")):
        y_t = (sub["arr_delay_min"] > DELAY_THRESHOLD_MIN).astype(int).to_numpy()
        y_p = pd.to_numeric(sub["predicted_delay"], errors="coerce").fillna(0).astype(int).to_numpy()
        y_pr = sub["proba_delay"].astype(float).to_numpy()
        m = compute_metrics(y_t, y_p, y_pr)
        m["strategy"] = str(strategy)
        out.append(m)
    return sorted(out, key=lambda r: -r["n"])


# ----------------------------- markdown rendering ------------------------


def _fmt(x: Any, fmt: str = ".4f") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:{fmt}}"


def render_md(report: dict[str, Any]) -> str:
    o = report["overall"]
    cm = o.get("confusion_matrix", {})
    lines: list[str] = [
        f"# Live period report — {report['period']['since']} → {report['period']['until']}",
        "",
        f"Generated UTC: {report['period']['generated_at_utc']}  ",
        f"Source: `{report['period']['source']}`  ",
        f"Settled predictions: **{report['n_settled_predictions']}**",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **ROC AUC** | **{_fmt(o.get('roc_auc'))}** |",
        f"| F1 | {_fmt(o.get('f1'))} |",
        f"| Accuracy | {_fmt(o.get('accuracy'))} |",
        f"| Brier score | {_fmt(o.get('brier'))} |",
        f"| Log loss | {_fmt(o.get('log_loss'))} |",
        f"| Precision | {_fmt(o.get('precision'))} |",
        f"| Recall | {_fmt(o.get('recall'))} |",
        f"| Pos rate (truth) | {_fmt(o.get('positive_rate_truth'))} |",
        f"| Pos rate (pred) | {_fmt(o.get('positive_rate_pred'))} |",
        f"| ECE (calibration) | {_fmt(report['calibration']['ece'])} |",
        "",
        "**Confusion matrix:**",
        "",
        f"| | pred 0 | pred 1 |",
        f"|---|---|---|",
        f"| **true 0** | {cm.get('tn', 0)} | {cm.get('fp', 0)} |",
        f"| **true 1** | {cm.get('fn', 0)} | {cm.get('tp', 0)} |",
        "",
        "## Daily breakdown",
        "",
        "| Date | n | AUC | F1 | Brier | Recall | Precision | pos_rate_truth |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in report["daily"]:
        lines.append(
            f"| {d['date']} | {d['n']} | {_fmt(d.get('roc_auc'), '.3f')} | "
            f"{_fmt(d['f1'], '.3f')} | {_fmt(d['brier'], '.3f')} | "
            f"{_fmt(d['recall'], '.3f')} | {_fmt(d['precision'], '.3f')} | "
            f"{_fmt(d['positive_rate_truth'], '.3f')} |"
        )

    lines += [
        "",
        "## By threshold strategy",
        "",
        "| Strategy | n | AUC | F1 | Brier | pos_rate_pred |",
        "|---|---|---|---|---|---|",
    ]
    for s in report["by_strategy"]:
        lines.append(
            f"| `{s['strategy']}` | {s['n']} | {_fmt(s.get('roc_auc'), '.3f')} | "
            f"{_fmt(s['f1'], '.3f')} | {_fmt(s['brier'], '.3f')} | "
            f"{_fmt(s['positive_rate_pred'], '.3f')} |"
        )

    if report["by_carrier"]:
        lines += [
            "",
            "## By carrier (n ≥ {} only)".format(MIN_CARRIER_N),
            "",
            "| Carrier | n | AUC | F1 | Recall | pos_rate_truth |",
            "|---|---|---|---|---|---|",
        ]
        for c in report["by_carrier"]:
            lines.append(
                f"| {c['carrier']} | {c['n']} | {_fmt(c.get('roc_auc'), '.3f')} | "
                f"{_fmt(c['f1'], '.3f')} | {_fmt(c['recall'], '.3f')} | "
                f"{_fmt(c['positive_rate_truth'], '.3f')} |"
            )

    lines += [
        "",
        "## Calibration (10 bins)",
        "",
        f"**ECE = {_fmt(report['calibration']['ece'])}** "
        "(closer to 0 = better calibrated. <0.05 is good).",
        "",
        "| Bin | Range | n | Mean proba | Observed pos rate |",
        "|---|---|---|---|---|",
    ]
    for b in report["calibration"]["bins"]:
        if b["n"] == 0:
            lines.append(f"| {b['bin']} | [{b['lo']:.1f}, {b['hi']:.1f}) | 0 | — | — |")
        else:
            lines.append(
                f"| {b['bin']} | [{b['lo']:.1f}, {b['hi']:.1f}) | {b['n']} | "
                f"{_fmt(b['mean_proba'], '.3f')} | {_fmt(b['observed_pos_rate'], '.3f')} |"
            )

    if report["threshold_by_hour"]:
        lines += [
            "",
            "## Threshold and observed pos rate by UTC hour",
            "",
            "Validates that the dynamic quantile-target threshold tracks operational rhythm.",
            "",
            "| Hour UTC | n | Median threshold | Mean proba | Observed pos rate |",
            "|---|---|---|---|---|",
        ]
        for h in report["threshold_by_hour"]:
            lines.append(
                f"| {h['hour_utc']:02d} | {h['n']} | {h['median_threshold']:.3f} | "
                f"{h['mean_proba']:.3f} | {h['observed_pos_rate']:.3f} |"
            )

    lines += [
        "",
        "## Notes for the thesis defense",
        "",
        f"- AUC live = **{_fmt(o.get('roc_auc'))}** vs offline test 0.813 vs OOT 2021 0.788.",
        f"- Brier live = **{_fmt(o.get('brier'))}** vs offline 0.135.",
        f"- ECE = **{_fmt(report['calibration']['ece'])}** — flags whether rolling calibration is needed.",
        "- The system was operated with the full Tier S stack: quantile-target threshold + chain-walk inbound + cold-deck fallback.",
        "",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------- plotting ----------------------------------


def generate_plots(report: dict[str, Any], out_dir: Path) -> None:
    import matplotlib  # noqa: WPS433
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. daily metrics
    daily = report["daily"]
    if daily:
        dates = [d["date"] for d in daily]
        aucs = [d.get("roc_auc") if d.get("roc_auc") is not None else np.nan for d in daily]
        f1s = [d["f1"] for d in daily]
        briers = [d["brier"] for d in daily]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, aucs, label="AUC", marker="o")
        ax.plot(dates, f1s, label="F1", marker="s")
        ax.plot(dates, briers, label="Brier", marker="^")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Date")
        ax.set_ylabel("Metric")
        ax.set_title(
            f"Daily metrics - {report['period']['since']} to {report['period']['until']}"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(out_dir / "daily_metrics.png", dpi=110)
        plt.close(fig)

    # 2. reliability diagram
    bins = [b for b in report["calibration"]["bins"] if b["n"] > 0]
    if bins:
        x = [b["mean_proba"] for b in bins]
        y = [b["observed_pos_rate"] for b in bins]
        sizes = [max(20, b["n"] * 2) for b in bins]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.scatter(x, y, s=sizes, alpha=0.7, color="C2", label="Observed bins")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive rate")
        ax.set_title(f"Reliability diagram (ECE = {report['calibration']['ece']:.3f})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "reliability_diagram.png", dpi=110)
        plt.close(fig)

    # 3. threshold by hour
    hours = report["threshold_by_hour"]
    if hours:
        h_x = [h["hour_utc"] for h in hours]
        thr = [h["median_threshold"] for h in hours]
        obs_rate = [h["observed_pos_rate"] for h in hours]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(h_x, thr, marker="o", color="C0", label="Median threshold (quantile)")
        ax.set_xlabel("UTC Hour")
        ax.set_ylabel("Threshold", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax2 = ax.twinx()
        ax2.plot(h_x, obs_rate, marker="s", color="C1", label="Observed pos rate")
        ax2.set_ylabel("Observed positive rate", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax.set_title("Dynamic threshold and observed pos rate by UTC hour")
        ax.set_xticks(range(0, 24))
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "threshold_by_hour.png", dpi=110)
        plt.close(fig)

    # 4. confusion matrix
    cm = report["overall"].get("confusion_matrix")
    if cm:
        cm_arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(cm_arr, cmap="Blues")
        max_val = cm_arr.max() if cm_arr.size else 1
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, str(cm_arr[i, j]),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm_arr[i, j] > max_val / 2 else "black",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticklabels(["true 0", "true 1"])
        ax.set_title(f"Confusion matrix - n={report['n_settled_predictions']}")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=110)
        plt.close(fig)


# ----------------------------- entry point -------------------------------


def build_report(df: pd.DataFrame, source: str, since: str, until: str) -> dict[str, Any]:
    y_true = (df["arr_delay_min"].astype(float) > DELAY_THRESHOLD_MIN).astype(int).to_numpy()
    y_pred = pd.to_numeric(df["predicted_delay"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_proba = df["proba_delay"].astype(float).to_numpy()

    return {
        "period": {
            "since": since,
            "until": until,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
        },
        "n_settled_predictions": int(len(df)),
        "overall": compute_metrics(y_true, y_pred, y_proba),
        "calibration": compute_calibration(y_true, y_proba),
        "daily": per_day(df),
        "by_strategy": per_strategy(df),
        "by_carrier": per_carrier(df),
        "threshold_by_hour": threshold_by_hour(df),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", required=True, help="Start date YYYY-MM-DD (UTC)")
    parser.add_argument("--until", required=True, help="End date YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--source", choices=["db", "snapshots"], default="db",
                        help="Data source (default: live SQLite DB)")
    parser.add_argument("--out-dir", type=Path, default=REPO / "artifacts")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip matplotlib figures")
    args = parser.parse_args()

    since_iso = f"{args.since}T00:00:00+00:00"
    until_iso = f"{args.until}T23:59:59+00:00"

    if args.source == "db":
        if not DB_PATH.exists():
            print(f"ERROR: no DB at {DB_PATH}")
            return 1
        conn = sqlite3.connect(str(DB_PATH))
        df = load_from_db(conn, since_iso, until_iso)
        conn.close()
    else:
        df = load_from_snapshots(SNAPSHOT_DIR, since_iso, until_iso)

    if df.empty:
        print(f"No settled predictions in [{args.since}, {args.until}] from {args.source}")
        return 1

    print(f"Loaded {len(df)} settled predictions from {args.source}")
    report = build_report(df, args.source, args.since, args.until)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = f"live_period_report_{args.since}_{args.until}"
    json_path = args.out_dir / f"{base}.json"
    md_path = args.out_dir / f"{base}.md"

    with json_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote {json_path}")

    md_path.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {md_path}")

    if not args.no_plots:
        try:
            plots_dir = args.out_dir / "live_period_plots"
            generate_plots(report, plots_dir)
            print(f"Wrote plots -> {plots_dir}/")
        except ImportError:
            print("matplotlib not available — skipping plots (re-run with --no-plots to silence)")
        except Exception as e:
            print(f"Plot generation failed: {e}")

    overall_auc = report["overall"].get("roc_auc")
    print(
        f"\nSummary: n={report['n_settled_predictions']}  "
        f"AUC={overall_auc:.4f}" if overall_auc is not None else "  AUC=n/a"
    )
    print(
        f"         F1={report['overall']['f1']:.4f}  "
        f"Brier={report['overall']['brier']:.4f}  "
        f"ECE={report['calibration']['ece']:.4f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

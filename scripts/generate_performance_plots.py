"""Generate model performance plots (ROC, Calibration, Brier) from live_data.db.

Usage:
    .venv/bin/python scripts/generate_performance_plots.py
"""
import os
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

# Artifact timestamps (matching eval_live.py)
MODEL_VERSIONS = [
    ("v7_recal", "2000-01-01 00:00"),
    ("v9", "2026-05-15 01:43"),
    ("v9_recal", "2026-05-16 18:45"),
    ("v9", "2026-05-17 16:23"),
]

DB_PATH = Path("live_data.db")
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)


def label_model(ts: str) -> str:
    ts_norm = ts.replace("T", " ")[:16]
    model = MODEL_VERSIONS[0][0]
    for name, start in MODEL_VERSIONS:
        if ts_norm >= start:
            model = name
    return model


def load_data() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT p.fa_flight_id,
               p.proba_delay,
               p.predicted_at_utc,
               a.arr_delay_min,
               CASE WHEN a.arr_delay_min > 15 THEN 1 ELSE 0 END AS delayed
        FROM predictions p
        JOIN actuals a ON p.fa_flight_id = a.fa_flight_id
        WHERE a.arr_delay_min IS NOT NULL
          AND a.cancelled = 0
        ORDER BY p.predicted_at_utc
        """,
        con,
    )
    con.close()

    # Deduplicate: Keep latest prediction per flight
    df = (
        df.sort_values("predicted_at_utc")
        .groupby("fa_flight_id", as_index=False)
        .last()
    )
    df["model"] = df["predicted_at_utc"].apply(label_model)
    df["date"] = df["predicted_at_utc"].str[:10]
    return df


def main():
    print("Loading data...")
    df = load_data()
    print(f"Total resolved flights with predictions: {len(df)}")
    print(df["model"].value_counts())

    # Filter out v9_recal since it was only live for 1 day and is not representative
    v7_df = df[df["model"] == "v7_recal"]
    v9_df = df[df["model"] == "v9"]

    if len(v7_df) < 50 or len(v9_df) < 50:
        print("Not enough data to generate comparative plots.")
        return

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 16,
            "legend.fontsize": 11,
        }
    )

    # ----------------------------------------------------
    # Plot 1: ROC Curves
    # ----------------------------------------------------
    print("Generating ROC Curves...")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    for sub_df, name, color in [(v7_df, "v7_recal", "#3182bd"), (v9_df, "v9", "#31a354")]:
        y = sub_df["delayed"].to_numpy()
        p = sub_df["proba_delay"].to_numpy()
        auc = roc_auc_score(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, lw=2.5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve Comparison (Live Data)", pad=15)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_path = OUTPUT_DIR / "model_comparison_roc.png"
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    print(f"ROC plot saved to {roc_path}")

    # ----------------------------------------------------
    # Plot 2: Calibration / Reliability Curves
    # ----------------------------------------------------
    print("Generating Calibration Curves...")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    for sub_df, name, color in [(v7_df, "v7_recal", "#3182bd"), (v9_df, "v9", "#31a354")]:
        y = sub_df["delayed"].to_numpy()
        p = sub_df["proba_delay"].to_numpy()
        brier = brier_score_loss(y, p)
        prob_true, prob_pred = calibration_curve(y, p, n_bins=10)
        ax.plot(
            prob_pred,
            prob_true,
            "s-",
            label=f"{name} (Brier = {brier:.4f})",
            color=color,
            lw=2.5,
            markersize=6,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual Delay Rate)")
    ax.set_title("Calibration Curve Comparison (Reliability Diagram)", pad=15)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    cal_path = OUTPUT_DIR / "model_comparison_calibration.png"
    plt.savefig(cal_path, bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved to {cal_path}")

    # ----------------------------------------------------
    # Plot 3: Daily Brier and AUC Trend
    # ----------------------------------------------------
    print("Generating Daily Performance Trends...")
    daily_stats = []
    for (date, model), g in df.groupby(["date", "model"]):
        if len(g) >= 30 and g["delayed"].nunique() >= 2:
            y = g["delayed"].to_numpy()
            p = g["proba_delay"].to_numpy()
            daily_stats.append(
                {
                    "date": pd.to_datetime(date),
                    "model": model,
                    "AUC": roc_auc_score(y, p),
                    "Brier": brier_score_loss(y, p),
                    "count": len(y),
                }
            )

    trend_df = pd.DataFrame(daily_stats).sort_values("date")

    # Plot AUC and Brier trends side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    for name, color in [("v7_recal", "#3182bd"), ("v9", "#31a354")]:
        sub_trend = trend_df[trend_df["model"] == name]
        if not sub_trend.empty:
            ax1.plot(
                sub_trend["date"],
                sub_trend["AUC"],
                "o-",
                label=name,
                color=color,
                lw=2,
            )
            ax2.plot(
                sub_trend["date"],
                sub_trend["Brier"],
                "o-",
                label=name,
                color=color,
                lw=2,
            )

    ax1.set_title("Daily Live AUC Trend (Higher is Better)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Area Under ROC (AUC)")
    ax1.legend()
    # Rotate x labels
    ax1.tick_params(axis="x", rotation=30)

    ax2.set_title("Daily Live Brier Score Trend (Lower is Better)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Brier Score Loss")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=30)

    plt.suptitle("Model Performance Trends Over Time", y=0.98)
    plt.tight_layout()
    trend_path = OUTPUT_DIR / "model_comparison_trends.png"
    plt.savefig(trend_path, bbox_inches="tight")
    plt.close()
    print(f"Trends plot saved to {trend_path}")


if __name__ == "__main__":
    main()

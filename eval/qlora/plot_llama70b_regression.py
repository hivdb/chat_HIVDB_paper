#!/usr/bin/env python3
"""Plot Llama3.1-70B recall, precision, and accuracy regression curves by rank."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot recall, precision, and accuracy GEE logistic regression curves."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("regression/llama70b_regression_plot.png"),
        help="Output figure path.",
    )
    return parser.parse_args()


def logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def load_numeric_fit(path: Path) -> tuple[float, float, float]:
    df = pd.read_csv(path)
    fit_df = df[df["model_fit"] == "numeric_rank_trend"].copy()
    intercept = float(fit_df.loc[fit_df["term"] == "Intercept", "coef"].iloc[0])
    slope = float(fit_df.loc[fit_df["term"] == "rank", "coef"].iloc[0])
    p_value = float(fit_df.loc[fit_df["term"] == "rank", "p_value"].iloc[0])
    return intercept, slope, p_value


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return f"p = {p_value:.2e}"
    return f"p = {p_value:.3f}"


def plot_panel(
    ax,
    summary: pd.DataFrame,
    odds_ratio_path: Path,
    metric: str,
    numerator_col: str,
    denominator_col: str,
    count_note: str,
    color: str,
) -> None:
    intercept, slope, p_value = load_numeric_fit(odds_ratio_path)

    summary = summary.sort_values("rank").copy()
    ranks = summary["rank"].astype(float)
    observed = summary[metric].astype(float)

    x_grid = pd.Series([8 + i * (32 - 8) / 200 for i in range(201)])
    y_grid = x_grid.map(lambda rank: logistic(intercept + slope * rank))

    ax.plot(x_grid, y_grid, color=color, linewidth=2.5, label="GEE logistic fit")
    ax.scatter(ranks, observed, color=color, edgecolor="black", s=85, zorder=3, label="Observed")

    for _, row in summary.iterrows():
        label = f"{int(row[numerator_col])}/{int(row[denominator_col])}"
        ax.annotate(
            label,
            (row["rank"], row[metric]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    p_box = f"Rank trend\n{format_p_value(p_value)}"
    ax.text(
        0.98,
        1.02,
        p_box,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        clip_on=False,
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#404040",
            "alpha": 0.95,
        },
    )

    ax.text(
        0.02,
        1.02,
        count_note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        clip_on=False,
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#f7f7f7",
            "edgecolor": "#808080",
            "alpha": 0.95,
        },
    )

    ax.set_title(f"Llama3.1-70B {metric.capitalize()} by QLoRA Rank", pad=32)
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel(metric.capitalize())
    ax.set_xticks([8, 16, 25, 32])
    ax.set_xticklabels(["R8", "R16", "FT(R25)", "R32"])
    ax.set_ylim(0.00, 1.00)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    recall_summary = pd.read_csv("regression/llama70b_recall_logistic.recall_summary.csv")
    precision_summary = pd.read_csv("regression/llama70b_precision_logistic.precision_summary.csv")
    accuracy_summary = pd.read_csv("regression/llama70b_accuracy_logistic.accuracy_summary.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "figure.dpi": 150,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    plot_panel(
        axes[0],
        accuracy_summary,
        Path("regression/llama70b_accuracy_logistic.odds_ratios.csv"),
        metric="accuracy",
        numerator_col="correct_cases",
        denominator_col="total_cases",
        count_note="Point label:\nCorrect / total",
        color="#4c956c",
    )
    plot_panel(
        axes[1],
        precision_summary,
        Path("regression/llama70b_precision_logistic.odds_ratios.csv"),
        metric="precision",
        numerator_col="true_positives",
        denominator_col="predicted_positive_cases",
        count_note="Point label:\nTP / predicted positives",
        color="#b85c38",
    )
    plot_panel(
        axes[2],
        recall_summary,
        Path("regression/llama70b_recall_logistic.odds_ratios.csv"),
        metric="recall",
        numerator_col="true_positives",
        denominator_col="positive_cases",
        count_note="Point label:\nTP / ref positives",
        color="#1f6f8b",
    )

    fig.suptitle("GEE Logistic Regression of Rank Effects", fontsize=15, fontweight="bold")
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot Llama3.1-70B Cochran-Armitage trend test summaries by rank."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot recall, precision, and accuracy summaries with Cochran-Armitage p-values."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("regression/llama70b_cochran_armitage_plot.png"),
        help="Output figure path.",
    )
    return parser.parse_args()


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return f"p = {p_value:.2e}"
    return f"p = {p_value:.3f}"


def load_p_value(path: Path) -> float:
    df = pd.read_csv(path)
    return float(df["p_value"].iloc[0])


def plot_panel(
    ax,
    summary_path: Path,
    result_path: Path,
    metric: str,
    numerator_col: str,
    denominator_col: str,
    count_note: str,
    color: str,
) -> None:
    summary = pd.read_csv(summary_path).sort_values("rank").copy()
    p_value = load_p_value(result_path)

    ax.plot(summary["rank"], summary[metric], color=color, linewidth=2.5, zorder=2)
    ax.scatter(
        summary["rank"],
        summary[metric],
        color=color,
        edgecolor="black",
        s=85,
        zorder=3,
    )

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

    ax.text(
        0.98,
        1.02,
        f"Cochran-Armitage\n{format_p_value(p_value)}",
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
        Path("regression/llama70b_cochran_armitage.accuracy_summary.csv"),
        Path("regression/llama70b_cochran_armitage.accuracy_result.csv"),
        metric="accuracy",
        numerator_col="success_cases",
        denominator_col="total_cases",
        count_note="Point label:\nCorrect / total",
        color="#4c956c",
    )
    plot_panel(
        axes[1],
        Path("regression/llama70b_cochran_armitage.precision_summary.csv"),
        Path("regression/llama70b_cochran_armitage.precision_result.csv"),
        metric="precision",
        numerator_col="success_cases",
        denominator_col="total_cases",
        count_note="Point label:\nTP / predicted positives",
        color="#b85c38",
    )
    plot_panel(
        axes[2],
        Path("regression/llama70b_cochran_armitage.recall_summary.csv"),
        Path("regression/llama70b_cochran_armitage.recall_result.csv"),
        metric="recall",
        numerator_col="success_cases",
        denominator_col="total_cases",
        count_note="Point label:\nTP / ref positives",
        color="#1f6f8b",
    )

    fig.suptitle("Cochran-Armitage Trend Test by Rank", fontsize=15, fontweight="bold")
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

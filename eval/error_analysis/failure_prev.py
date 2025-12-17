#!/usr/bin/env python3
"""Plot per-QID precision and recall for key models using evaluation metrics."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "results" / "evaluation_metrics_by_qid_full150.csv"
DATA_PATH = DEFAULT_DATA_PATH
SCENARIO_NAME = "Partial Match"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

MODEL_FAMILIES = [
    ("GPT-4o", "GPT-4o base", "GPT-4o FT"),
    ("Llama3.1-70B", "Llama3.1-70B base", "Llama3.1-70B FT"),
    ("Llama3.1-8B", "Llama3.1-8B base", "Llama3.1-8B FT"),
]

TYPE_COLORS = {
    "boolean": "#4C72B0",
    "list": "#FF7F0E",
    "number": "#55A868",
}

QID_TOPICS = {
    1: "Patient Sequences?",
    2: "In vitro Drug Susceptibility?",
    3: "Open Access?",
    4: "GenBank IDs",
    5: "# Patients Sequenced",
    6: "Countries",
    7: "Sampling Years",
    8: "Were Samples Cloned?",
    9: "HIV Genes",
    10: "Sequencing Methods",
    11: "Sample Types",
    12: "VF on Therapy?",
    13: "Clinical Study?",
    14: "Prior ARV Use?",
    15: "Drug Classes",
    16: "Drugs",
}


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Per-QID metrics file missing: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["QID"] = df["QID"].astype(int)
    df["Type"] = df["Type"].astype(str).str.lower()
    df["Question"] = df.get("Question", "").fillna("")
    return df


def build_rates(df: pd.DataFrame, metric: str) -> dict[str, dict[int, float]]:
    valid_models = {model for family in MODEL_FAMILIES for model in family[1:]}
    rates: dict[str, dict[int, float]] = {model: {} for model in valid_models}
    for _, row in df.iterrows():
        model = row.get("model")
        if model not in rates:
            continue
        qid = int(row["QID"])
        rates[model][qid] = float(row.get(metric, 0.0) or 0.0)
    return rates


def _topic_label(qid: int, question_map: dict[int, str]) -> str:
    if qid in QID_TOPICS:
        return QID_TOPICS[qid]
    question = question_map.get(qid, "")
    tokens = question.split()
    return " ".join(tokens[:4]).title()


def _shade(color: str, target: str, amount: float) -> tuple[float, float, float]:
    base = np.array(mcolors.to_rgb(color))
    tgt = np.array(mcolors.to_rgb(target))
    return tuple((1 - amount) * base + amount * tgt)


def plot_rate_grid(
    qids: list[int],
    type_map: pd.Series,
    question_map: dict[int, str],
    rates: dict[str, dict[int, float]],
    title: str,
    output_name: str,
) -> None:
    labels = [f"Q{qid}\n{_topic_label(qid, question_map)}" for qid in qids]
    base_colors = [TYPE_COLORS.get(type_map.get(qid, ""), "#999999") for qid in qids]
    max_value = max(
        (rates.get(model, {}).get(qid, 0.0) for family in MODEL_FAMILIES for model in family[1:] for qid in qids),
        default=0.0,
    )
    y_limit = min(1.0, max_value * 1.15 if max_value else 0.1)
    width = 0.3
    x_positions = np.arange(len(qids))

    fig, axes = plt.subplots(len(MODEL_FAMILIES), 1, figsize=(18, 12), sharex=True)
    for idx, (family, base_model, ft_model) in enumerate(MODEL_FAMILIES):
        ax = axes[idx]
        base_vals = [rates.get(base_model, {}).get(qid, 0.0) for qid in qids]
        ft_vals = [rates.get(ft_model, {}).get(qid, 0.0) for qid in qids]
        base_shades = [_shade(color, "white", 0.25) for color in base_colors]
        ft_shades = [_shade(color, "black", 0.25) for color in base_colors]
        base_bars = ax.bar(
            x_positions - width / 2, base_vals, width=width, color=base_shades, edgecolor="black", label="Base"
        )
        ft_bars = ax.bar(
            x_positions + width / 2, ft_vals, width=width, color=ft_shades, edgecolor="black", label="FT"
        )
        for bars in (base_bars, ft_bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_ylabel(family)
        ax.set_ylim(0, y_limit)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if idx == len(MODEL_FAMILIES) - 1:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels, rotation=25, ha="right")
        else:
            ax.set_xticks(x_positions)
            ax.set_xticklabels([])

    type_handles = [Patch(facecolor=TYPE_COLORS[label], label=label.title()) for label in TYPE_COLORS]
    style_handles = [
        Patch(facecolor=_shade("#888888", "white", 0.25), edgecolor="black", label="Base"),
        Patch(facecolor=_shade("#888888", "black", 0.25), edgecolor="black", label="FT"),
    ]
    legend1 = axes[0].legend(handles=type_handles, loc="upper left", bbox_to_anchor=(0, 1.25), ncol=len(TYPE_COLORS), title="Question Type")
    axes[0].add_artist(legend1)
    axes[0].legend(handles=style_handles, loc="upper right", bbox_to_anchor=(1, 1.25), ncol=2, title="Model")

    fig.suptitle(title, fontsize=16, y=0.9)
    fig.tight_layout(rect=[0, 0, 0.98, 0.94])
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_overall_histogram(df: pd.DataFrame, qids: list[int], type_map: pd.Series, question_map: dict[int, str], output_name: str) -> None:
    """Plot horizontal stacked bar chart showing FN/FP counts across all base and FT models per QID."""
    # Aggregate FN and FP counts across all base and FT models
    base_ft_models = [model for family in MODEL_FAMILIES for model in family[1:]]

    fn_counts = {}
    fp_counts = {}

    for qid in qids:
        qid_df = df[df["QID"] == qid]
        total_fn = 0
        total_fp = 0
        for model in base_ft_models:
            model_rows = qid_df[qid_df["model"] == model]
            if not model_rows.empty:
                total_fn += model_rows["fn"].sum()
                total_fp += model_rows["fp"].sum()
        fn_counts[qid] = total_fn
        fp_counts[qid] = total_fp

    # Sort QIDs by total incorrect answers (descending)
    qids_sorted = sorted(qids, key=lambda q: fn_counts[q] + fp_counts[q], reverse=True)

    # Prepare data for plotting - use abbreviated topic labels instead of full question text
    labels = [f"QID {qid}: {QID_TOPICS.get(qid, question_map.get(qid, ''))}" for qid in qids_sorted]
    fn_vals = [fn_counts[qid] for qid in qids_sorted]
    fp_vals = [fp_counts[qid] for qid in qids_sorted]

    # Color bars by question type
    colors = [TYPE_COLORS.get(type_map.get(qid, ""), "#999999") for qid in qids_sorted]

    fig, ax = plt.subplots(figsize=(12, 10))
    y_positions = np.arange(len(qids_sorted))

    # Create stacked horizontal bars
    ax.barh(y_positions, fn_vals, color=[mcolors.to_rgba(c, alpha=0.5) for c in colors], label="False Negatives")
    ax.barh(y_positions, fp_vals, left=fn_vals, color=[mcolors.to_rgba(c, alpha=0.8) for c in colors], label="False Positives")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Incorrect Answers (count)", fontsize=12)
    ax.set_title("Incorrect Answer Frequency across all base and FT models", fontsize=14)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Create legends
    error_handles = [
        Patch(facecolor="gray", alpha=0.5, label="False Negatives"),
        Patch(facecolor="gray", alpha=0.8, label="False Positives"),
    ]
    type_handles = [Patch(facecolor=TYPE_COLORS[t], label=t.title()) for t in sorted(TYPE_COLORS.keys())]

    legend1 = ax.legend(handles=error_handles, loc="upper right", title="")
    ax.add_artist(legend1)
    ax.legend(handles=type_handles, loc="lower right", title="Question Type")

    fig.tight_layout()
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_models_histogram(df: pd.DataFrame, qids: list[int], type_map: pd.Series, output_name: str) -> None:
    """Plot stacked bar charts showing FN/FP counts per QID for each base and FT model."""
    base_ft_models = [model for family in MODEL_FAMILIES for model in family[1:]]

    fig, axes = plt.subplots(len(base_ft_models), 1, figsize=(14, 12), sharex=True)
    if len(base_ft_models) == 1:
        axes = [axes]

    for idx, model in enumerate(base_ft_models):
        ax = axes[idx]
        model_df = df[df["model"] == model]

        fn_vals = []
        fp_vals = []
        colors = []

        for qid in qids:
            qid_df = model_df[model_df["QID"] == qid]
            if not qid_df.empty:
                fn_vals.append(qid_df["fn"].sum())
                fp_vals.append(qid_df["fp"].sum())
            else:
                fn_vals.append(0)
                fp_vals.append(0)
            colors.append(TYPE_COLORS.get(type_map.get(qid, ""), "#999999"))

        x_positions = np.arange(len(qids))
        ax.bar(x_positions, fn_vals, color=[mcolors.to_rgba(c, alpha=0.5) for c in colors], label="False Negatives")
        ax.bar(x_positions, fp_vals, bottom=fn_vals, color=[mcolors.to_rgba(c, alpha=0.8) for c in colors], label="False Positives")

        ax.set_ylabel("Incorrect Answers", fontsize=10)
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx == 0:
            error_handles = [
                Patch(facecolor="gray", alpha=0.5, label="False Negatives"),
                Patch(facecolor="gray", alpha=0.8, label="False Positives"),
            ]
            type_handles = [Patch(facecolor=TYPE_COLORS[t], label=t.title()) for t in sorted(TYPE_COLORS.keys())]
            legend1 = ax.legend(handles=error_handles, loc="upper left", ncol=2, fontsize=9)
            ax.add_artist(legend1)
            ax.legend(handles=type_handles, loc="upper right", ncol=3, fontsize=9, title="Question Type")

    axes[-1].set_xlabel("QID", fontsize=12)
    axes[-1].set_xticklabels([str(qid) for qid in qids])

    fig.tight_layout()
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_PATH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(path)
    if df.empty:
        raise SystemExit(f"No rows found in {path} for scenario '{SCENARIO_NAME}'.")
    type_map = df.groupby("QID")["Type"].first()
    question_map = df.groupby("QID")["Question"].first().to_dict()
    qids = sorted(type_map.index.tolist())
    metric_specs = [
        ("precision", "Precision = TP / (TP + FP)", "precision.png"),
        ("recall", "Recall = TP / (TP + FN)", "recall.png"),
        ("accuracy", "Accuracy = (TP + TN) / (TP + TN + FP + FN)", "accuracy.png"),
        ("f1", "F1 = 2 * Precision * Recall / (Precision + Recall)", "f1.png"),
    ]
    for metric, title, filename in metric_specs:
        rates = build_rates(df, metric)
        plot_rate_grid(qids, type_map, question_map, rates, title, filename)

    # Generate overall histogram
    plot_overall_histogram(df, qids, type_map, question_map, "overall_hist.png")

    # Generate per-model histogram
    plot_models_histogram(df, qids, type_map, "models_hist.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

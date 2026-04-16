#!/usr/bin/env python3
"""Generate QLoRA error-analysis figures without importing the shared error-analysis module."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch


QLORA_DIR = Path(__file__).resolve().parent
DATA_PATH = QLORA_DIR / "evaluation_metrics_by_qid_full150.csv"

MODEL_FAMILIES = [
    ("Llama3.1-70B", "Llama3.1-70B FT", "Llama3.1-70B R16", "Llama3.1-70B R32"),
    ("Llama3.1-8B", "Llama3.1-8B FT", "Llama3.1-8B R16", "Llama3.1-8B R32"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=QLORA_DIR / "error_analysis_figures",
        help="Directory for generated figures.",
    )
    return parser.parse_args()


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
        rates[model][int(row["QID"])] = float(row.get(metric, 0.0) or 0.0)
    return rates


def build_error_share_rates(df: pd.DataFrame, error_type: str) -> dict[str, dict[int, float]]:
    if error_type not in {"fn", "fp"}:
        raise ValueError(f"Unsupported error type: {error_type}")
    valid_models = {model for family in MODEL_FAMILIES for model in family[1:]}
    rates: dict[str, dict[int, float]] = {model: {} for model in valid_models}
    totals = df.groupby("model")[error_type].sum().to_dict()
    for _, row in df.iterrows():
        model = row.get("model")
        if model not in rates:
            continue
        qid = int(row["QID"])
        total = float(totals.get(model, 0) or 0)
        value = float(row.get(error_type, 0) or 0)
        rates[model][qid] = value / total if total else 0.0
    return rates


def write_rate_table(rates: dict[str, dict[int, float]], qids: list[int], output_path: Path) -> None:
    rows: List[Dict[str, float]] = []
    for model, qid_map in rates.items():
        for qid in qids:
            rows.append(
                {
                    "model": model,
                    "QID": qid,
                    "proportion": float(qid_map.get(qid, 0.0) or 0.0),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _topic_label(qid: int, question_map: dict[int, str]) -> str:
    return QID_TOPICS.get(qid, question_map.get(qid, ""))


def plot_rate_grid(
    qids: list[int],
    type_map: pd.Series,
    question_map: dict[int, str],
    rates: dict[str, dict[int, float]],
    title: str,
    output_name: str,
    output_dir: Path,
    as_percent: bool = False,
) -> None:
    labels = [f"Q{qid}\n{_topic_label(qid, question_map)}" for qid in qids]
    base_colors = [TYPE_COLORS.get(type_map.get(qid, ""), "#999999") for qid in qids]
    max_value = max(
        (rates.get(model, {}).get(qid, 0.0) for family in MODEL_FAMILIES for model in family[1:] for qid in qids),
        default=0.0,
    )
    scale = 100 if as_percent else 1
    y_limit = min(1.0, max_value * 1.15 if max_value else 0.1) * scale
    bar_width = 5.5
    bar_spacing = bar_width * 1.05
    cluster_gap = 3.5
    step = 3 * bar_spacing + cluster_gap
    x_positions = np.arange(len(qids)) * step

    fig, axes = plt.subplots(len(MODEL_FAMILIES), 1, figsize=(18, 9), sharex=True)
    if len(MODEL_FAMILIES) == 1:
        axes = [axes]

    alpha_map = {"FT": 0.4, "R16": 0.65, "R32": 0.9}
    for idx, family_info in enumerate(MODEL_FAMILIES):
        family = family_info[0]
        models = list(family_info[1:])
        ax = axes[idx]
        offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * bar_spacing
        bars = []
        for offset, model in zip(offsets, models):
            vals = [rates.get(model, {}).get(qid, 0.0) * scale for qid in qids]
            variant = model.split()[-1]
            model_shades = [mcolors.to_rgba(color, alpha=alpha_map[variant]) for color in base_colors]
            bars.append(
                ax.bar(
                    x_positions + offset,
                    vals,
                    width=bar_width,
                    color=model_shades,
                    label=variant,
                )
            )
        label_offset = max(y_limit * 0.02, 0.3 if as_percent else 0.003)
        for bar_group in bars:
            for bar in bar_group:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + label_offset,
                    f"{height:.1f}" if as_percent else f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_ylabel(f"{family}")
        ax.set_ylim(0, y_limit)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=25, ha="right")

    type_handles = [Patch(facecolor=TYPE_COLORS[label], label=label.title()) for label in TYPE_COLORS]
    model_handles = [
        Patch(facecolor=mcolors.to_rgba("#4C4C4C", alpha=alpha_map[label]), edgecolor="black", label=label)
        for label in ["FT", "R16", "R32"]
    ]
    legend1 = axes[0].legend(
        handles=type_handles,
        loc="upper left",
        bbox_to_anchor=(0, 1.25),
        ncol=len(TYPE_COLORS),
        title="Question Type",
    )
    axes[0].add_artist(legend1)
    axes[0].legend(handles=model_handles, loc="upper right", bbox_to_anchor=(1, 1.25), ncol=3, title="Model")
    fig.suptitle(title, fontsize=16, y=0.92)
    fig.tight_layout(rect=[0, 0, 0.98, 0.94])
    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overall_histogram(
    df: pd.DataFrame,
    qids: list[int],
    type_map: pd.Series,
    question_map: dict[int, str],
    output_name: str,
    output_dir: Path,
) -> None:
    models = [model for family in MODEL_FAMILIES for model in family[1:]]
    fn_counts = {}
    fp_counts = {}
    for qid in qids:
        qid_df = df[df["QID"] == qid]
        fn_counts[qid] = sum(qid_df[qid_df["model"] == model]["fn"].sum() for model in models)
        fp_counts[qid] = sum(qid_df[qid_df["model"] == model]["fp"].sum() for model in models)

    qids_sorted = sorted(qids, key=lambda q: fn_counts[q] + fp_counts[q], reverse=True)
    labels = [f"QID {qid}: {_topic_label(qid, question_map)}" for qid in qids_sorted]
    fn_vals = [fn_counts[qid] for qid in qids_sorted]
    fp_vals = [fp_counts[qid] for qid in qids_sorted]
    colors = [TYPE_COLORS.get(type_map.get(qid, ""), "#999999") for qid in qids_sorted]

    fig, ax = plt.subplots(figsize=(12, 10))
    y_positions = np.arange(len(qids_sorted))
    ax.barh(y_positions, fn_vals, color=[mcolors.to_rgba(c, alpha=0.5) for c in colors], label="False Negatives")
    ax.barh(y_positions, fp_vals, left=fn_vals, color=[mcolors.to_rgba(c, alpha=0.8) for c in colors], label="False Positives")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Incorrect Answers (count)", fontsize=12)
    ax.set_title("Incorrect Answer Frequency across all QLoRA models", fontsize=14)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    error_handles = [
        Patch(facecolor="gray", alpha=0.5, label="False Negatives"),
        Patch(facecolor="gray", alpha=0.8, label="False Positives"),
    ]
    type_handles = [Patch(facecolor=TYPE_COLORS[t], label=t.title()) for t in sorted(TYPE_COLORS.keys())]
    legend1 = ax.legend(handles=error_handles, loc="upper right")
    ax.add_artist(legend1)
    ax.legend(handles=type_handles, loc="lower right", title="Question Type")

    fig.tight_layout()
    fig.savefig(output_dir / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_models_histogram(df: pd.DataFrame, qids: list[int], type_map: pd.Series, output_name: str, output_dir: Path) -> None:
    models = [model for family in MODEL_FAMILIES for model in family[1:]]
    fig, axes = plt.subplots(len(models), 1, figsize=(14, 12), sharex=True)
    if len(models) == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = df[df["model"] == model]
        fn_vals = []
        fp_vals = []
        colors = []
        for qid in qids:
            qid_df = model_df[model_df["QID"] == qid]
            fn_vals.append(qid_df["fn"].sum() if not qid_df.empty else 0)
            fp_vals.append(qid_df["fp"].sum() if not qid_df.empty else 0)
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
    fig.savefig(output_dir / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_PATH)
    if df.empty:
        raise SystemExit(f"No rows found in {DATA_PATH}.")

    type_map = df.groupby("QID")["Type"].first()
    question_map = df.groupby("QID")["Question"].first().to_dict()
    qids = sorted(type_map.index.tolist())

    metric_specs = [
        ("precision", "Precision = TP / (TP + FP)", "precision.png", False),
        ("recall", "Recall = TP / (TP + FN)", "recall.png", False),
        ("accuracy", "Accuracy = (TP + TN) / (TP + TN + FP + FN)", "accuracy.png", False),
        ("f1", "F1 = 2 * Precision * Recall / (Precision + Recall)", "f1.png", False),
    ]
    for metric, title, filename, as_percent in metric_specs:
        rates = build_rates(df, metric)
        plot_rate_grid(qids, type_map, question_map, rates, title, filename, args.output_dir, as_percent=as_percent)

    share_specs = [
        ("fn", "Proportion of False Negatives per QID = FNq / Σq(FN)", "fn_by_qid.png", "fn_by_qid.csv"),
        ("fp", "Proportion of False Positives per QID = FPq / Σq(FP)", "fp_by_qid.png", "fp_by_qid.csv"),
    ]
    for error_type, title, filename, csv_name in share_specs:
        rates = build_error_share_rates(df, error_type)
        plot_rate_grid(qids, type_map, question_map, rates, title, filename, args.output_dir, as_percent=True)
        write_rate_table(rates, qids, args.output_dir / csv_name)

    plot_overall_histogram(df, qids, type_map, question_map, "overall_hist.png", args.output_dir)
    plot_models_histogram(df, qids, type_map, "models_hist.png", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

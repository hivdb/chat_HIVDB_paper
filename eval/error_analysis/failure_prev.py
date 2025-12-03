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

from eval.config import OUTPUT_METRICS_BY_QID  # type: ignore


DATA_PATH = OUTPUT_METRICS_BY_QID
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


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Per-QID metrics file missing: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df[df["scenario"] == SCENARIO_NAME].copy()
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


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    if df.empty:
        raise SystemExit(f"No rows found in {DATA_PATH} for scenario '{SCENARIO_NAME}'.")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plot per-QID precision and recall for key models using evaluation metrics."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from eval.config import OUTPUT_METRICS_BY_QID  # type: ignore


DATA_PATH = OUTPUT_METRICS_BY_QID
SCENARIO_NAME = "Overall - partial match"
OUTPUT_DIR = Path(__file__).resolve().parent

MODEL_ORDER = [
    "GPT-4o base",
    "GPT-4o FT",
    "Llama3.1-70B base",
    "Llama3.1-70B FT",
    "Llama3.1-8B base",
    "Llama3.1-8B FT",
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
    rates: dict[str, dict[int, float]] = {model: {} for model in MODEL_ORDER}
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


def plot_rate_grid(
    qids: list[int],
    type_map: pd.Series,
    question_map: dict[int, str],
    rates: dict[str, dict[int, float]],
    title: str,
    output_name: str,
) -> None:
    labels = [f"Q{qid}\n{_topic_label(qid, question_map)}" for qid in qids]
    colors = [TYPE_COLORS.get(type_map.get(qid, ""), "#999999") for qid in qids]
    max_value = max((rates.get(model, {}).get(qid, 0.0) for model in MODEL_ORDER for qid in qids), default=0.0)
    y_limit = min(1.0, max_value * 1.15 if max_value else 0.1)

    fig, axes = plt.subplots(len(MODEL_ORDER), 1, figsize=(18, 16), sharex=True)
    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]
        values = [rates.get(model, {}).get(qid, 0.0) for qid in qids]
        ax.bar(range(len(qids)), values, color=colors, edgecolor="black")
        ax.set_ylabel(model)
        ax.set_ylim(0, y_limit)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if idx == len(MODEL_ORDER) - 1:
            ax.set_xticks(range(len(qids)))
            ax.set_xticklabels(labels, rotation=25, ha="right")
        else:
            ax.set_xticks(range(len(qids)))
            ax.set_xticklabels([])

    handles = [Patch(facecolor=color, label=label.title()) for label, color in TYPE_COLORS.items()]
    axes[0].legend(handles=handles, loc="upper right")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.98, 0.98])
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    if df.empty:
        raise SystemExit(f"No rows found in {DATA_PATH} for scenario '{SCENARIO_NAME}'.")
    type_map = df.groupby("QID")["Type"].first()
    question_map = df.groupby("QID")["Question"].first().to_dict()
    qids = sorted(type_map.index.tolist())
    precision_rates = build_rates(df, "precision")
    recall_rates = build_rates(df, "recall")
    plot_rate_grid(qids, type_map, question_map, precision_rates, "Precision = TP / (TP + FP)", "precision.png")
    plot_rate_grid(qids, type_map, question_map, recall_rates, "Recall = TP / (TP + FN)", "recall.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

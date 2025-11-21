#!/usr/bin/env python3
"""Generate failure histograms for key models."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import sys

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from eval.normalize import canonicalize_answer
from eval.scoring import human_answer_counts


DATA_PATH = Path(__file__).resolve().parents[1] / "detailed_evaluation.csv"
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
    df = pd.read_csv(DATA_PATH)
    scenario_lower = df.get("Scenario", "").str.lower()
    df["ScenarioLower"] = scenario_lower
    df["QID"] = df["QID"].astype(int)
    df["TypeLower"] = df["Type"].str.lower()

    exact = df[df["ScenarioLower"] == "exact"].copy()
    list_partial = df[(df["ScenarioLower"] == "partial") & (df["TypeLower"] == "list")].copy()

    exact_non_list = exact[exact["TypeLower"] != "list"]
    combined = pd.concat([exact_non_list, list_partial], ignore_index=True)
    combined.drop(columns=["ScenarioLower", "TypeLower"], inplace=True, errors="ignore")
    return combined


def compute_model_error_types(df: pd.DataFrame) -> dict[str, dict[int, dict[str, int]]]:
    per_model: dict[str, defaultdict[int, dict[str, int]]] = {
        model: defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0}) for model in MODEL_ORDER
    }
    for _, row in df.iterrows():
        question_type = row.get("Type", "")
        question_text = row.get("Question", "")
        ref_raw = row.get("Human Answer", "")
        ref_norm = canonicalize_answer(ref_raw, convert_special_no=True)
        qid = int(row["QID"])
        allow_partial = (question_type or "").strip().lower() == "list"
        for model in MODEL_ORDER:
            correct_col = f"{model} Correct"
            answer_col = f"{model} Answer"
            if correct_col not in row or answer_col not in row:
                continue
            pred_raw = row.get(answer_col, "")
            pred_norm = canonicalize_answer(pred_raw, convert_special_no=True)
            counts, _ = human_answer_counts(
                question_type,
                pred_norm,
                ref_norm,
                question_text=question_text,
                ref_raw=ref_raw,
                pred_raw=pred_raw,
                allow_partial_list=allow_partial,
            )
            entry = per_model[model][qid]
            for key, value in counts.items():
                entry[key] += value
    return {model: {qid: dict(values) for qid, values in qid_map.items()} for model, qid_map in per_model.items()}


def compute_precision_recall(
    model_errors: dict[str, dict[int, dict[str, int]]]
) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]]]:
    precision: dict[str, dict[int, float]] = {}
    recall: dict[str, dict[int, float]] = {}
    for model, qid_map in model_errors.items():
        precision[model] = {}
        recall[model] = {}
        for qid, counts in qid_map.items():
            tp = counts.get("tp", 0)
            tn = counts.get("tn", 0)
            fp = counts.get("fp", 0)
            fn = counts.get("fn", 0)
            prec_denom = tp + fp
            rec_denom = tp + fn
            precision[model][qid] = (tp / prec_denom) if prec_denom else 0.0
            recall[model][qid] = (tp / rec_denom) if rec_denom else 0.0
    return precision, recall


def _topic_label(question: str, words: int = 4) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", question or "")
    if not tokens:
        return ""
    snippet = " ".join(tokens[:words])
    return snippet.title()


def plot_rate_grid(
    qids: list[int],
    type_map: pd.Series,
    question_map: dict[int, str],
    rates: dict[str, dict[int, float]],
    title: str,
    output_name: str,
) -> None:
    labels = []
    for qid in qids:
        topic = QID_TOPICS.get(qid)
        if not topic:
            topic = _topic_label(question_map.get(qid, ""))
        labels.append(f"Q{qid}\n{topic}")
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
        raise SystemExit("No evaluation rows found in detailed_evaluation.csv")
    type_map = df.groupby("QID")["Type"].first().str.lower()
    question_map = df.groupby("QID")["Question"].first().to_dict()
    qids = sorted(type_map.index.tolist())
    model_errors = compute_model_error_types(df)
    precision, recall = compute_precision_recall(model_errors)
    plot_rate_grid(qids, type_map, question_map, precision, "Precision = TP / (TP + FP)", "precision.png")
    plot_rate_grid(qids, type_map, question_map, recall, "Recall = TP / (TP + FN)", "recall.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

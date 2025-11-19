#!/usr/bin/env python3
"""Generate failure histograms for key models."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
import textwrap

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df.get("Scenario", "").str.lower() == "exact"].copy()
    df["QID"] = df["QID"].astype(int)
    return df


def compute_model_error_types(df: pd.DataFrame) -> dict[str, dict[int, dict[str, int]]]:
    per_model: dict[str, defaultdict[int, dict[str, int]]] = {
        model: defaultdict(lambda: {"fp": 0, "fn": 0}) for model in MODEL_ORDER
    }
    for _, row in df.iterrows():
        question_type = row.get("Type", "")
        question_text = row.get("Question", "")
        ref_raw = row.get("Human Answer", "")
        ref_norm = canonicalize_answer(ref_raw, convert_special_no=True)
        qid = int(row["QID"])
        for model in MODEL_ORDER:
            correct_col = f"{model} Correct"
            answer_col = f"{model} Answer"
            if correct_col not in row or answer_col not in row:
                continue
            correct_val = row[correct_col]
            if pd.isna(correct_val) or int(correct_val) == 1:
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
                allow_partial_list=False,
            )
            entry = per_model[model][qid]
            if counts["fp"]:
                entry["fp"] += counts["fp"]
            if counts["fn"]:
                entry["fn"] += counts["fn"]
    return {model: {qid: dict(values) for qid, values in qid_map.items()} for model, qid_map in per_model.items()}


def aggregate_errors(model_errors: dict[str, dict[int, dict[str, int]]]) -> tuple[dict[int, int], dict[int, int]]:
    false_pos: dict[int, int] = defaultdict(int)
    false_neg: dict[int, int] = defaultdict(int)
    for qid_map in model_errors.values():
        for qid, counts in qid_map.items():
            false_pos[qid] += counts.get("fp", 0)
            false_neg[qid] += counts.get("fn", 0)
    return false_pos, false_neg


def plot_models_hist(qids: list[int], type_map: pd.Series, model_errors: dict[str, dict[int, dict[str, int]]]) -> None:
    colors = [TYPE_COLORS.get(type_map.get(qid, ""), "#999999") for qid in qids]
    fn_colors: list[tuple[float, float, float]] = []
    fp_colors: list[tuple[float, float, float]] = []
    for color in colors:
        fn_color, fp_color = _shade_pair(color)
        fn_colors.append(fn_color)
        fp_colors.append(fp_color)

    fig, axes = plt.subplots(len(MODEL_ORDER), 1, figsize=(16, 14), sharex=True)
    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]
        counts = model_errors.get(model, {})
        fn_vals = [int(counts.get(qid, {}).get("fn", 0)) for qid in qids]
        fp_vals = [int(counts.get(qid, {}).get("fp", 0)) for qid in qids]
        ax.bar(qids, fn_vals, color=fn_colors, edgecolor="black", linewidth=0.5)
        ax.bar(qids, fp_vals, bottom=fn_vals, color=fp_colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Incorrect Answers")
        ax.set_title(model)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if idx == len(MODEL_ORDER) - 1:
            ax.set_xlabel("QID")
            ax.set_xticks(qids)
            ax.set_xticklabels([str(qid) for qid in qids])
        else:
            ax.set_xticks(qids)
            ax.set_xticklabels([])

    legend_handles = [Patch(color=color, label=label.title()) for label, color in TYPE_COLORS.items()]
    fig.legend(handles=legend_handles, loc="upper right", title="Question Type")
    shade_handles = [
        Patch(facecolor=_mix("#888888", "white", 0.35), edgecolor="black", label="False Negatives"),
        Patch(facecolor=_mix("#888888", "black", 0.25), edgecolor="black", label="False Positives"),
    ]
    axes[0].legend(handles=shade_handles, loc="upper left", bbox_to_anchor=(0, 1.2), ncol=2)
    fig.tight_layout(rect=[0, 0, 0.97, 1])
    output_path = OUTPUT_DIR / "models_hist.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _mix(color: str, target: str, amount: float) -> tuple[float, float, float]:
    base = mcolors.to_rgb(color)
    tgt = mcolors.to_rgb(target)
    return tuple((1 - amount) * c + amount * t for c, t in zip(base, tgt))


def _shade_pair(color: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    fn_color = _mix(color, "white", 0.35)
    fp_color = _mix(color, "black", 0.25)
    return fn_color, fp_color


def plot_overall_hist(type_map: pd.Series, question_map: dict[int, str], false_pos: dict[int, int], false_neg: dict[int, int]) -> None:
    qids = sorted(type_map.index)
    data = []
    for qid in qids:
        fp = false_pos.get(qid, 0)
        fn = false_neg.get(qid, 0)
        total = fp + fn
        data.append((qid, fp, fn, total))
    overall = sorted(data, key=lambda item: item[3], reverse=True)
    labels = []
    for qid, *_ in overall:
        question = question_map.get(qid, "")
        wrapped = textwrap.fill(str(question), 60)
        labels.append(f"QID {qid}: {wrapped}")
    fps = [item[1] for item in overall]
    fns = [item[2] for item in overall]
    base_colors = [TYPE_COLORS.get(type_map.get(qid, ""), "#999999") for qid, *_ in overall]
    fn_colors = []
    fp_colors = []
    for base in base_colors:
        fn_color, fp_color = _shade_pair(base)
        fn_colors.append(fn_color)
        fp_colors.append(fp_color)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(labels, fns, color=fn_colors, edgecolor="black", label="False Negatives")
    ax.barh(labels, fps, left=fns, color=fp_colors, edgecolor="black", label="False Positives")
    ax.set_xlabel("Incorrect Answers (count)")
    ax.set_title("Incorrect Answer Frequency across all base and FT models")
    type_handles = [Patch(facecolor=color, edgecolor="black", label=label.title()) for label, color in TYPE_COLORS.items()]
    pattern_handles = [
        Patch(facecolor=_mix("#888888", "white", 0.35), edgecolor="black", label="False Negatives"),
        Patch(facecolor=_mix("#888888", "black", 0.25), edgecolor="black", label="False Positives"),
    ]
    legend1 = ax.legend(handles=pattern_handles, loc="upper right")
    ax.add_artist(legend1)
    ax.legend(handles=type_handles, loc="upper right", bbox_to_anchor=(1, 0.92), title="Question Type")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "overall_hist.png"
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
    false_pos, false_neg = aggregate_errors(model_errors)
    plot_models_hist(qids, type_map, model_errors)
    plot_overall_hist(type_map, question_map, false_pos, false_neg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

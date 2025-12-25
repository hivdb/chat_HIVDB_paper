#!/usr/bin/env python3
"""Error analysis utilities and plots."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch

EVAL_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EVAL_DIR / "results"
ERROR_ANALYSIS_DIR = EVAL_DIR / "error_analysis"


# === analysis_by_qid ===
DEFAULT_SUFFIX = "full150"

COLUMN_ORDER = [
    "PMID",
    "Human Answer",
    "GPT-4o base Answer",
    "GPT-4o base Correct",
    "GPT-4o FT Answer",
    "GPT-4o FT Correct",
    "GPT-4o QSP Answer",
    "GPT-4o QSP Correct",
    "Llama3.1-70B base Answer",
    "Llama3.1-70B base Correct",
    "Llama3.1-70B FT Answer",
    "Llama3.1-70B FT Correct",
    "Llama3.1-70B QSP Answer",
    "Llama3.1-70B QSP Correct",
    "Llama3.1-8B base Answer",
    "Llama3.1-8B base Correct",
    "Llama3.1-8B FT Answer",
    "Llama3.1-8B FT Correct",
    "Llama3.1-8B QSP Answer",
    "Llama3.1-8B QSP Correct",
]
COLUMN_RENAMES = {col: col for col in COLUMN_ORDER}

SHEET_CONFIG = {
    "Q1": {"qid": 1},
    "Q9": {"qid": 9},
    "Q16": {"qid": 16},
}


def _load_frames(details_path: Path) -> Dict[str, pd.DataFrame]:
    frames = {}
    book = pd.read_excel(details_path, sheet_name=None)
    if not book:
        return {"all": pd.DataFrame()}
    if "All" in book:
        frames["all"] = book["All"]
    else:
        first_name = next(iter(book))
        frames["all"] = book[first_name]
    return frames


def _prepare_sheet(df: pd.DataFrame, qid: int) -> pd.DataFrame:
    if "QID" not in df.columns:
        return pd.DataFrame()
    subset = df[df["QID"] == qid].copy()
    if subset.empty:
        return subset
    missing_cols = [col for col in COLUMN_ORDER if col not in subset.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns for QID {qid}: {missing_cols}")
    subset.sort_values("PMID", inplace=True)
    subset = subset[COLUMN_ORDER].rename(columns=COLUMN_RENAMES).reset_index(drop=True)
    return subset


def build_workbook(details_path: Path, output_path: Path) -> None:
    frames = _load_frames(details_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, config in SHEET_CONFIG.items():
            source_df = frames.get("all", pd.DataFrame())
            sheet_df = _prepare_sheet(source_df, config["qid"])
            if sheet_df.empty:
                logging.warning("Skipping sheet %s (no rows for QID %s)", sheet_name, config["qid"])
                continue
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            logging.info("Wrote sheet %s with %d rows", sheet_name, len(sheet_df))
        if not writer.sheets:
            empty_df = pd.DataFrame({"info": ["no data"]})
            empty_df.to_excel(writer, sheet_name="empty", index=False)


def run_analysis_by_qid(suffix: str) -> int:
    details_path = RESULTS_DIR / f"detailed_evaluation_{suffix}.xlsx"
    output_path = ERROR_ANALYSIS_DIR / "results" / f"analysis_by_qid_{suffix}.xlsx"
    build_workbook(details_path, output_path)
    logging.info("Saved workbook to %s", output_path)
    return 0


# === failure_prev (rate plots) ===
DEFAULT_DATA_PATH = RESULTS_DIR / "evaluation_metrics_by_qid_full150.csv"
OUTPUT_DIR = ERROR_ANALYSIS_DIR / "figures"

MODEL_FAMILIES = [
    ("GPT-4o", "GPT-4o base", "GPT-4o FT", "GPT-4o QSP"),
    ("Llama3.1-70B", "Llama3.1-70B base", "Llama3.1-70B FT", "Llama3.1-70B QSP"),
    ("Llama3.1-8B", "Llama3.1-8B base", "Llama3.1-8B FT", "Llama3.1-8B QSP"),
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


def _topic_label(qid: int, question_map: dict[int, str]) -> str:
    return QID_TOPICS.get(qid, question_map.get(qid, ""))


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
    as_percent: bool = False,
    model_families: list[tuple[str, ...]] | None = None,
    include_base: bool = True,
) -> None:
    families = model_families or MODEL_FAMILIES
    labels = [f"Q{qid}\n{_topic_label(qid, question_map)}" for qid in qids]
    base_colors = [TYPE_COLORS.get(type_map.get(qid, ""), "#999999") for qid in qids]
    max_value = max(
        (rates.get(model, {}).get(qid, 0.0) for family in MODEL_FAMILIES for model in family[1:] for qid in qids),
        default=0.0,
    )
    scale = 100 if as_percent else 1
    y_limit = min(1.0, max_value * 1.15 if max_value else 0.1) * scale
    bar_width = 6.0
    bar_spacing = bar_width * 1.05
    cluster_gap = 3.0
    step = (len(families[0]) - 1) * bar_spacing + cluster_gap
    x_positions = np.arange(len(qids)) * step

    fig, axes = plt.subplots(len(families), 1, figsize=(18, 12), sharex=True)
    if len(families) == 1:
        axes = [axes]
    for idx, family_info in enumerate(families):
        family = family_info[0]
        models = list(family_info[1:])
        if not include_base:
            models = [model for model in models if "base" not in model]
        ax = axes[idx]
        num_models = len(models)
        offsets = (np.arange(num_models) - (num_models - 1) / 2) * bar_spacing
        bars = []
        alpha_map = {"Base": 0.35, "FT": 0.6, "QSP": 0.85}
        for offset, model in zip(offsets, models):
            vals = [rates.get(model, {}).get(qid, 0.0) * scale for qid in qids]
            label = "Base" if "base" in model else ("QSP" if "QSP" in model else "FT")
            alpha = alpha_map[label]
            model_shades = [mcolors.to_rgba(color, alpha=alpha) for color in base_colors]
            bars.append(
                ax.bar(
                    x_positions + offset,
                    vals,
                    width=bar_width,
                    color=model_shades,
                    label=label,
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
        ax.set_ylabel(f"{family} (%)")
        ax.set_ylim(0, y_limit)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == len(MODEL_FAMILIES) - 1:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels, rotation=25, ha="right")
        else:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels, rotation=25, ha="right")

    type_handles = [Patch(facecolor=TYPE_COLORS[label], label=label.title()) for label in TYPE_COLORS]
    style_handles = [
        Patch(facecolor=mcolors.to_rgba("#4C4C4C", alpha=0.35), edgecolor="black", label="Base"),
        Patch(facecolor=mcolors.to_rgba("#4C4C4C", alpha=0.6), edgecolor="black", label="FT"),
        Patch(facecolor=mcolors.to_rgba("#4C4C4C", alpha=0.85), edgecolor="black", label="QSP"),
    ]
    legend1 = axes[0].legend(
        handles=type_handles,
        loc="upper left",
        bbox_to_anchor=(0, 1.25),
        ncol=len(TYPE_COLORS),
        title="Question Type",
    )
    axes[0].add_artist(legend1)
    if include_base:
        model_handles = style_handles
    else:
        model_handles = [h for h in style_handles if h.get_label() != "Base"]
    axes[0].legend(handles=model_handles, loc="upper right", bbox_to_anchor=(1, 1.25), ncol=len(model_handles), title="Model")

    fig.suptitle(title, fontsize=16, y=0.9)
    fig.tight_layout(rect=[0, 0, 0.98, 0.94])
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_overall_histogram(df: pd.DataFrame, qids: list[int], type_map: pd.Series, question_map: dict[int, str], output_name: str) -> None:
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

    qids_sorted = sorted(qids, key=lambda q: fn_counts[q] + fp_counts[q], reverse=True)
    labels = [f"QID {qid}: {QID_TOPICS.get(qid, question_map.get(qid, ''))}" for qid in qids_sorted]
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
    ax.set_title("Incorrect Answer Frequency across all base and FT models", fontsize=14)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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


def write_rate_table(rates: dict[str, dict[int, float]], qids: list[int], output_path: Path) -> None:
    rows: List[Dict[str, float]] = []
    for model, qid_map in rates.items():
        for qid in qids:
            rows.append({
                "model": model,
                "QID": qid,
                "proportion": float(qid_map.get(qid, 0.0) or 0.0),
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def run_failure_prev(data_path: Path | None) -> int:
    path = data_path or DEFAULT_DATA_PATH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(path)
    if df.empty:
        raise SystemExit(f"No rows found in {path}.")
    type_map = df.groupby("QID")["Type"].first()
    question_map = df.groupby("QID")["Question"].first().to_dict()
    qids = sorted(type_map.index.tolist())
    metric_specs = [
        ("precision", "Precision = TP / (TP + FP)", "precision.png", None, True, False),
        ("recall", "Recall = TP / (TP + FN)", "recall.png", None, True, False),
        ("accuracy", "Accuracy = (TP + TN) / (TP + TN + FP + FN)", "accuracy.png", None, True, False),
        ("f1", "F1 = 2 * Precision * Recall / (Precision + Recall)", "f1.png", MODEL_FAMILIES[:2], False, True),
    ]
    for metric, title, filename, families, include_base, as_percent in metric_specs:
        rates = build_rates(df, metric)
        plot_rate_grid(
            qids,
            type_map,
            question_map,
            rates,
            title,
            filename,
            model_families=families,
            include_base=include_base,
            as_percent=as_percent,
        )

    share_specs = [
        ("fn", "Proportion of False Negatives per QID = FNq / Σq(FN)", "fn_by_qid.png", "fn_by_qid.csv"),
        ("fp", "Proportion of False Positives per QID = FPq / Σq(FP)", "fp_by_qid.png", "fp_by_qid.csv"),
    ]
    for error_type, title, filename, csv_name in share_specs:
        rates = build_error_share_rates(df, error_type)
        plot_rate_grid(qids, type_map, question_map, rates, title, filename, as_percent=True)
        write_rate_table(rates, qids, ERROR_ANALYSIS_DIR / "results" / csv_name)

    plot_overall_histogram(df, qids, type_map, question_map, "overall_hist.png")
    plot_models_histogram(df, qids, type_map, "models_hist.png")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analysis_parser = subparsers.add_parser("analysis_by_qid", help="Build QID-specific workbook.")
    analysis_parser.add_argument("--suffix", default=DEFAULT_SUFFIX, help="Evaluation suffix (e.g., full150).")

    failure_parser = subparsers.add_parser("failure_prev", help="Generate error analysis plots.")
    failure_parser.add_argument("--data-path", type=Path, default=None, help="Per-QID metrics CSV.")

    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.command == "analysis_by_qid":
        return run_analysis_by_qid(args.suffix)
    if args.command == "failure_prev":
        return run_failure_prev(args.data_path)
    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())

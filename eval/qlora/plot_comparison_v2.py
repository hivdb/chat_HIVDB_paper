#!/usr/bin/env python3
"""Plot QLoRA per-QID metric comparisons for Llama3.1-70B and Llama3.1-8B."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


QLORA_DIR = Path(__file__).resolve().parent
INPUT_CSV = QLORA_DIR / "evaluation_metrics_by_qid_full150.csv"

FAMILIES = [
    {
        "name": "Llama3.1-70B",
        "base_model": "Llama3.1-70B FT",
        "targets": [
            ("Llama3.1-70B R16", "Llama3.1-70B R16", "#F8766D", "o"),
            ("Llama3.1-70B R32", "Llama3.1-70B R32", "#619CFF", "s"),
        ],
    },
    {
        "name": "Llama3.1-8B",
        "base_model": "Llama3.1-8B FT",
        "targets": [
            ("Llama3.1-8B R16", "Llama3.1-8B R16", "#F8766D", "o"),
            ("Llama3.1-8B R32", "Llama3.1-8B R32", "#619CFF", "s"),
        ],
    },
]

METRICS = [
    ("accuracy", "Accuracy", "accuracy_comparison.svg"),
    ("precision", "Precision", "precision_comparison.svg"),
    ("recall", "Recall", "recall_comparison.svg"),
    ("f1", "F1", "f1_comparison.svg"),
]


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as fd:
        for record in csv.DictReader(fd):
            rows.append(record)
    return rows


def build_qid_lookup(
    rows: list[dict[str, str]],
    metric: str,
    base_model: str,
    target_models: list[tuple[str, str, str, str]],
) -> tuple[dict[int, float], dict[str, dict[int, float]]]:
    base_by_qid: dict[int, float] = {}
    targets_by_model: dict[str, dict[int, float]] = {model: {} for model, _, _, _ in target_models}

    for row in rows:
        model = row["model"]
        qid = int(row["QID"])
        value = float(row[metric]) * 100
        if model == base_model:
            base_by_qid[qid] = value
        if model in targets_by_model:
            targets_by_model[model][qid] = value

    return base_by_qid, targets_by_model


def draw_metric(
    rows: list[dict[str, str]],
    family_name: str,
    base_model: str,
    target_models: list[tuple[str, str, str, str]],
    metric_key: str,
    metric_label: str,
    output_name: str,
) -> None:
    base_by_qid, targets_by_model = build_qid_lookup(rows, metric_key, base_model, target_models)
    output_dir = QLORA_DIR / family_name

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for model, legend_label, color, marker in target_models:
        qids = sorted(set(base_by_qid).intersection(targets_by_model[model]))
        x_vals = [base_by_qid[qid] for qid in qids]
        y_vals = [targets_by_model[model][qid] for qid in qids]
        ax.scatter(x_vals, y_vals, color=color, label=legend_label, s=34, marker=marker)

        for qid, x_val, y_val in zip(qids, x_vals, y_vals):
            ax.text(x_val + 0.7, y_val + 0.7, str(qid), color=color, fontsize=8, alpha=0.85)

    ax.plot([0, 110], [0, 110], "--", color="gray", linewidth=1)
    ax.set_xlabel(f"{base_model} {metric_label} (%)", fontweight="bold", fontsize=13)
    ax.set_ylabel(f"Variant {metric_label} (%)", fontweight="bold", fontsize=13)
    ax.set_title(f"{family_name} {metric_label}: FT vs R16/R32", fontsize=15, pad=10)
    ax.legend(title="Model")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle="-", alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / output_name)
    plt.close()


def main() -> int:
    rows = load_rows(INPUT_CSV)
    for family in FAMILIES:
        for metric_key, metric_label, output_name in METRICS:
            draw_metric(
                rows,
                family["name"],
                family["base_model"],
                family["targets"],
                metric_key,
                metric_label,
                output_name,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

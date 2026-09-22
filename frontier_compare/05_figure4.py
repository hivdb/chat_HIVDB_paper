#!/usr/bin/env python3
"""Updated Figure 4: accuracy, precision, recall, F1 for frontier models vs cached GPT-4o.

Bars are the mean of the 16 per-QID values (--aggregation macro, default) or pooled over all
rows (--aggregation pooled, as in eval/figures/full150-bar-chart.png). Error bars are 95%
paper-level bootstrap CIs (macro only). A star marks a BH-adjusted Wilcoxon p < 0.05 versus
config.PRIMARY_COMPARATOR (the best cached GPT-4o condition).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier_compare import config  # noqa: E402

METRICS = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
FRONTIER_COLORS = ["#2a78d6", "#eb6834"]            # categorical slots 1-2
COMPARATOR_COLORS = ["#4d4d4a", "#8a8983", "#bdbcb4"]  # neutral steps: cached baselines recede


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregation", choices=["macro", "pooled"], default="macro")
    args = parser.parse_args()

    summary = pd.read_csv(config.RESULTS_DIR / "metrics_summary.csv")
    tests = pd.read_csv(config.RESULTS_DIR / "pairwise_tests.csv")
    comparators = [c for c in config.COMPARATORS if c in set(summary["model"])]
    frontier = [m for m in summary["model"].unique() if m not in comparators]
    models = frontier + comparators
    colors = dict(zip(frontier, FRONTIER_COLORS)) | dict(zip(comparators, COMPARATOR_COLORS))

    fig, ax = plt.subplots(figsize=(10, 5.2))
    width = 0.8 / len(models)
    x = np.arange(len(METRICS))
    for i, model in enumerate(models):
        rows = summary[summary["model"] == model].set_index("metric").loc[[m for m, _ in METRICS]]
        vals = rows[args.aggregation].to_numpy() * 100
        pos = x - 0.4 + width * (i + 0.5)
        yerr = None
        if args.aggregation == "macro":
            yerr = np.vstack([vals - rows["macro_ci_low"] * 100, rows["macro_ci_high"] * 100 - vals])
        ax.bar(pos, vals, width * 0.92, color=colors[model], label=model, yerr=yerr,
               error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#3d3d3a"})
        ink = "#1a1a19" if colors[model] == COMPARATOR_COLORS[-1] else "white"
        tops = vals if yerr is None else vals + yerr[1]
        for xp, v in zip(pos, vals):
            ax.text(xp, 2, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=ink, rotation=90)
        if model in frontier:
            sig = tests[(tests["frontier"] == model) & (tests["comparator"] == config.PRIMARY_COMPARATOR)
                        & (tests["test"] == "wilcoxon_qid")].set_index("metric")
            for (metric, _), xp, top in zip(METRICS, pos, tops):
                if metric in sig.index and sig.loc[metric, "p_bh"] < 0.05:
                    ax.text(xp, min(top + 0.5, 101), "*", ha="center", fontsize=12, color="#1a1a19")

    ax.set_xticks(x, [label for _, label in METRICS], fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%" + (" (mean of 16 per-question values)" if args.aggregation == "macro" else " (pooled)"))
    ax.yaxis.grid(True, color="#e5e4de", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(ncol=3, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.set_title(f"Frontier QSP vs cached GPT-4o (* BH p<0.05 vs {config.PRIMARY_COMPARATOR}, Wilcoxon over QIDs)",
                 fontsize=10, loc="left")
    fig.tight_layout()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = config.FIGURES_DIR / f"figure4_frontier_{args.aggregation}"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".tiff"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    print(f"Wrote {stem}.png/.tiff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

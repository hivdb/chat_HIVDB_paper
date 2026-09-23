#!/usr/bin/env python3
"""Updated Figure 4, in the style of the paper's eval/figures/full150-bar-chart.png.

Same layout and styling constants as eval/plots.py: stacked accuracy / precision / recall / F1
panels, bars pooled over all PMID x QID rows, value labels on bars, models grouped by family
with bold family names below. The base comparator is GPT-4o QSP (config.PRIMARY_COMPARATOR),
the same prompt the frontier models get. Brackets show BH-adjusted Wilcoxon p < 0.05 over the 16
per-QID values against it (comparison_set "figure4" in results/statistical_tests.csv), as the
paper's brackets do against each family's base model. Brackets are drawn here rather than by
eval/plots.py, whose stacking is tuned for at most three brackets per panel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import plots  # noqa: E402
from frontier_compare import config  # noqa: E402

GPT4O = "GPT-4o"
# The paper's GPT-4o orange, plus one hue per frontier model
FAMILY_COLORS = {GPT4O: "#ff7f0e", "GPT-6 Astra": "#1f77b4", "Kimi K3": "#9467bd"}
# Paper convention: light = prompted without fine-tuning, dark = fine-tuned
GPT4O_TINTS = {"QSP": 0.55, "FT": 0.0, "FT+QSP": 0.0}
ORDER = [f"{GPT4O} QSP", f"{GPT4O} FT", f"{GPT4O} FT+QSP"]
FAMILY_GAP = 0.7
BRACKET_START, BRACKET_STEP, BRACKET_HEIGHT = 115.0, 17.0, 3.0
Y_MAX = 185.0


def family(model: str) -> str:
    return next(f for f in FAMILY_COLORS if model.startswith(f))


def p_label(p: float) -> str:
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.2f}" if p > 0.009 else f"p={p:.3f}"


def main() -> int:
    summary = pd.read_csv(config.RESULTS_DIR / "metrics_summary.csv")
    tests = pd.read_csv(config.RESULTS_DIR / "statistical_tests.csv")
    frontier = [spec.label for spec in config.MODELS.values() if spec.label in set(summary["model"])]
    models = [m for m in ORDER if m in set(summary["model"])] + frontier
    values = summary.pivot(index="model", columns="metric", values="value") * 100
    base = config.PRIMARY_COMPARATOR
    fig4 = tests[(tests["comparison_set"] == "figure4") & (tests["test"] == "wilcoxon_qid")]

    x, positions = 0.0, []
    for i, model in enumerate(models):
        if i and family(model) != family(models[i - 1]):
            x += FAMILY_GAP
        positions.append(x)
        x += plots.MODEL_SPACING
    pos = dict(zip(models, positions))
    colors = [plots._tint_color(FAMILY_COLORS[GPT4O], GPT4O_TINTS[m.split()[-1]]) if family(m) == GPT4O
              else FAMILY_COLORS[family(m)] for m in models]

    fig, axes = plt.subplots(len(plots.METRIC_COLUMNS), 1, figsize=(14, 18), sharex=True,
                             gridspec_kw={"hspace": 0.45})
    for ax, (metric, label) in zip(axes, plots.METRIC_COLUMNS):
        vals = [values.loc[m, metric] for m in models]
        bars = ax.bar(positions, vals, color=colors, width=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 4, plots._round_half_up(v, 0),
                    ha="center", va="bottom", fontsize=plots.BAR_LABEL_SIZE)
        sig = fig4[(fig4["metric"] == metric) & (fig4["p_bh"] < 0.05)].copy()
        sig["distance"] = sig["model"].map(lambda m: abs(pos[m] - pos[base]))
        for level, r in enumerate(sig.sort_values("distance").itertuples()):
            y = BRACKET_START + level * BRACKET_STEP
            x0, x1 = pos[base], pos[r.model]
            ax.plot([x0, x0, x1, x1], [y, y + BRACKET_HEIGHT, y + BRACKET_HEIGHT, y], color="black", linewidth=1)
            ax.text((x0 + x1) / 2, y + BRACKET_HEIGHT + 1.5, p_label(r.p_bh), ha="center", va="bottom",
                    fontsize=plots.ANNOTATION_FONT_SIZE)
        ax.set_ylim(0, Y_MAX)
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel(f"{label} (%)", fontsize=plots.AXIS_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=plots.AXIS_TICK_SIZE)
        ax.tick_params(axis="x", pad=plots.X_TICK_PAD)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels([m.split()[-1] for m in models], rotation=plots.LABEL_ROTATION, ha="right",
                             fontsize=plots.AXIS_TICK_SIZE)
    for fam in dict.fromkeys(family(m) for m in models):
        xs = [pos[m] for m in models if family(m) == fam]
        axes[-1].text((min(xs) + max(xs)) / 2, -0.62, fam, ha="center", va="top", fontweight="bold",
                      fontsize=plots.FAMILY_LABEL_SIZE, transform=axes[-1].get_xaxis_transform())
    papers = int(summary["papers"].min())
    fig.suptitle(f"Frontier models vs GPT-4o ({papers} papers)", fontsize=plots.TITLE_FONT_SIZE * 0.8)
    fig.subplots_adjust(bottom=0.12, top=0.93)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / "figure4_frontier.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Generate bar charts comparing model performance across configurations.

Figures:
1) GPT-4o/Llama3.1-70B/Llama3.1-8B (base)
2) GPT-4o/Llama3.1-70B/Llama3.1-8B (FT)
3) GPT-4o/Llama3.1-70B/Llama3.1-8B (QSP)

Each figure plots Accuracy, Precision, Recall, and F1 together on a
single axis (grouped bars). Pairwise Wilcoxon p-values are
Benjamini–Hochberg corrected and every adjusted p < 0.05 is annotated.
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS: Sequence[str] = ["Accuracy", "Precision", "Recall", "F1"]
METRIC_COLUMNS: Sequence[str] = [f"Q{i}" for i in range(1, 17)]
DATA_FILE = Path("metrics_by_model_and_metric.csv")
STATS_FILE = Path("Inter_model_Stat_results.xlsx")
BH_TABLE_OUT = Path("bh_corrected_pvalues.csv")
SUMMARY_METRICS_FILE = Path("evaluation_metrics_full150.csv")

FIGURE_CONFIGS = [
    {
        "title": "Base models",
        "variant": "base",
        "models": [
            {"label": "GPT-4o base", "pvalue_name": "GPT-4o"},
            {"label": "Llama3.1-70B base", "pvalue_name": "Llama3.1-70B"},
            {"label": "Llama3.1-8B base", "pvalue_name": "Llama3.1-8B"},
        ],
        "output": Path("figure1_base_models.png"),
    },
    {
        "title": "Fine-tuned models",
        "variant": "FT",
        "models": [
            {"label": "GPT-4o FT", "pvalue_name": "GPT-4o"},
            {"label": "Llama3.1-70B FT", "pvalue_name": "Llama3.1-70B"},
            {"label": "Llama3.1-8B FT", "pvalue_name": "Llama3.1-8B"},
        ],
        "output": Path("figure2_ft_models.png"),
    },
    {
        "title": "QSP models",
        "variant": "QSP",
        "models": [
            {"label": "GPT-4o QSP", "pvalue_name": "GPT-4o"},
            {"label": "Llama3.1-70B QSP", "pvalue_name": "Llama3.1-70B"},
            {"label": "Llama3.1-8B QSP", "pvalue_name": "Llama3.1-8B"},
        ],
        "output": Path("figure3_qsp_models.png"),
    },
]


def benjamini_hochberg(pvalues: Sequence[float]) -> List[float]:
    """Benjamini–Hochberg correction (FDR) for a list of p-values."""
    m = len(pvalues)
    if m == 0:
        return []

    pvals = np.asarray(pvalues, dtype=float)
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]

    adjusted = np.empty(m, dtype=float)
    for rank, p in enumerate(sorted_pvals, start=1):
        adjusted[rank - 1] = p * m / rank

    # Enforce monotonicity of adjusted p-values
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    corrected = np.empty(m, dtype=float)
    corrected[sorted_indices] = adjusted
    return corrected.tolist()


def load_metric_means() -> Dict[Tuple[str, str], float]:
    """
    Return metric values for each (model, metric) using precomputed summary metrics.

    Values are converted to percentages.
    """
    df = pd.read_csv(SUMMARY_METRICS_FILE)
    metric_map = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}

    required = set(metric_map.keys()) | {"model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {SUMMARY_METRICS_FILE}: {sorted(missing)}")

    metrics = {}
    for _, row in df.iterrows():
        for col, name in metric_map.items():
            metrics[(row["model"], name)] = float(row[col]) * 100.0
    return metrics


def build_pvalue_lookup(df_stats: pd.DataFrame) -> Dict[Tuple[Tuple[str, str], str, str], float]:
    """
    Build lookup for Wilcoxon p-values keyed by sorted model pair, variant, metric.

    Expected Set format: "<model A> vs <model B> (<variant>, <Metric>)"
    Only cross-model comparisons are used for annotation.
    """
    lookup: Dict[Tuple[Tuple[str, str], str, str], float] = {}
    pattern = re.compile(r"(.+?) vs (.+?) \((.+?),\s*(.+)\)")

    for _, row in df_stats.iterrows():
        match = pattern.match(str(row["Set"]))
        if not match:
            continue
        model_a, model_b, variant, metric = match.groups()
        key = (tuple(sorted([model_a.strip(), model_b.strip()])), variant.strip(), metric.strip())
        lookup[key] = float(row["Wilcoxon p-value"])

    return lookup


def get_pairwise_pvalues(
    models: Sequence[Dict[str, str]], variant: str, metric: str, lookup: Dict[Tuple[Tuple[str, str], str, str], float]
) -> List[Tuple[Tuple[str, str], float]]:
    """Collect Wilcoxon p-values for all model pairs for a variant/metric."""
    pairs = []
    for model_a, model_b in itertools.combinations(models, 2):
        key = (tuple(sorted([model_a["pvalue_name"], model_b["pvalue_name"]])), variant, metric)
        if key not in lookup:
            raise KeyError(
                f"P-value missing for {model_a['pvalue_name']} vs {model_b['pvalue_name']} ({variant}, {metric})"
            )
        pairs.append(((model_a["label"], model_b["label"]), lookup[key]))
    return pairs


def annotate_significance(
    ax,
    positions: Dict[str, float],
    heights: Dict[str, float],
    pairs: Iterable[Tuple[Tuple[str, str], float]],
    y_min: float = 0.0,
    alpha: float = 0.05,
):
    """Annotate every comparison whose BH-corrected p-value is below the threshold."""
    pairs = list(pairs)
    if not pairs:
        return

    raw_pvalues = [p for _, p in pairs]
    corrected = benjamini_hochberg(raw_pvalues)

    sig_entries = [(pair, p_corr) for (pair, _), p_corr in zip(pairs, corrected) if p_corr < alpha]
    if not sig_entries:
        return

    # Plot tighter (shorter-span) comparisons closer to the bars; break ties by p-value.
    sig_entries.sort(key=lambda t: (abs(positions[t[0][0]] - positions[t[0][1]]), t[1]))

    # Group by span so pairs of the same distance share a horizontal line.
    span_groups: Dict[float, List[Tuple[Tuple[str, str], float]]] = {}
    for entry in sig_entries:
        span = abs(positions[entry[0][0]] - positions[entry[0][1]])
        span_groups.setdefault(span, []).append(entry)

    max_height = max(heights.values())
    step = max_height * 0.1 + 1
    current_y = max_height + step

    for span in sorted(span_groups.keys()):
        entries = span_groups[span]
        # Ensure this band clears the tallest bar in the span.
        tallest_in_span = max(max(heights[a], heights[b]) for (a, b), _ in entries)
        y = max(current_y, tallest_in_span + step)

        for pair, p_corr in entries:
            a, b = pair
            x1, x2 = positions[a], positions[b]
            ax.plot([x1, x1, x2, x2], [y, y + step * 0.3, y + step * 0.3, y], color="k", linewidth=0.8)
            label = "p < 0.001" if p_corr < 0.001 else f"p={format(p_corr, '.1g')}"
            ax.text((x1 + x2) / 2, y + step * 0.4, label, ha="center", va="bottom", fontsize=8)

        current_y = y + step

    ylim_upper = current_y + step
    ax.set_ylim(y_min, max(ylim_upper, y_min + 1))


def plot_figure(
    title: str,
    models: Sequence[Dict[str, str]],
    variant: str,
    metric_values: Dict[Tuple[str, str], float],
    pvalue_lookup: Dict[Tuple[Tuple[str, str], str, str], float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title)

    colors = ["#ff7f0e", "#2ca02c", "#d62728"]
    group_width = len(models) + 1  # spacing between metric groups

    x_ticks = []
    x_tick_labels = []
    bar_ticks = []
    bar_tick_labels = []
    bar_entries = []

    for m_idx, metric in enumerate(METRICS):
        base_x = m_idx * group_width
        heights = {model["label"]: metric_values[(model["label"], metric)] for model in models}
        x_positions = {model["label"]: base_x + i for i, model in enumerate(models)}
        bars = ax.bar(list(x_positions.values()), list(heights.values()), color=colors, width=0.6)

        pairwise = get_pairwise_pvalues(models, variant, metric, pvalue_lookup)
        annotate_significance(ax, x_positions, heights, pairwise, y_min=0)

        center = base_x + (len(models) - 1) / 2
        x_ticks.append(center)
        x_tick_labels.append(metric)
        bar_ticks.extend([x_positions[model["label"]] for model in models])
        bar_tick_labels.extend([model["label"] for model in models])

        bar_entries.extend((bar, model, heights) for bar, model in zip(bars, models))

    ax.set_yticks(np.arange(0, 101, 10))
    current_top = ax.get_ylim()[1]
    ax.set_ylim(bottom=0, top=max(current_top, 120))
    ax.set_xticks(bar_ticks)
    ax.set_xticklabels(bar_tick_labels, rotation=45, ha="right")
    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(x_ticks)
    top_ax.set_xticklabels(x_tick_labels)
    top_ax.tick_params(axis="x", pad=2)
    ax.set_xlabel("Model")
    ax.set_ylabel("Percentage")

    for bar, model, heights in bar_entries:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{int(round(heights[model['label']]))}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # legend removed per request

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def dump_bh_table(
    pvalue_lookup: Dict[Tuple[Tuple[str, str], str, str], float],
    output_path: Path,
) -> None:
    """Write a CSV of raw and BH-corrected p-values for all plotted comparisons."""
    rows = []
    for config in FIGURE_CONFIGS:
        for metric in METRICS:
            pairs = get_pairwise_pvalues(config["models"], config["variant"], metric, pvalue_lookup)
            raw_pvals = [p for _, p in pairs]
            corrected = benjamini_hochberg(raw_pvals)
            for (pair, raw), adj in zip(pairs, corrected):
                rows.append(
                    {
                        "variant": config["variant"],
                        "metric": metric,
                        "model_a": pair[0],
                        "model_b": pair[1],
                        "raw_pvalue": raw,
                        "bh_corrected_pvalue": adj,
                    }
                )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    metric_values = load_metric_means()
    stats_df = pd.read_excel(STATS_FILE)
    pvalue_lookup = build_pvalue_lookup(stats_df)

    for config in FIGURE_CONFIGS:
        plot_figure(
            title=config["title"],
            models=config["models"],
            variant=config["variant"],
            metric_values=metric_values,
            pvalue_lookup=pvalue_lookup,
            output_path=config["output"],
        )
        print(f"Wrote {config['output']}")

    dump_bh_table(pvalue_lookup=pvalue_lookup, output_path=BH_TABLE_OUT)
    print(f"Wrote {BH_TABLE_OUT}")


if __name__ == "__main__":
    main()

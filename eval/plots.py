from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from .constants import MODEL_BASE_COLORS, VARIANT_TINTS
from .normalize import slugify

METRIC_COLUMNS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
]

TITLE_FONT_SIZE = 32
AXIS_LABEL_SIZE = 24
AXIS_TICK_SIZE = 18
BAR_LABEL_SIZE = 20
ANNOTATION_FONT_SIZE = 12
FAMILY_LABEL_SIZE = 18

Y_LIM = 1.8

METRIC_PALETTE = {
    "accuracy": "#2a9d8f",
    "precision": "#e76f51",
    "recall": "#f4a261",
}

FAMILY_COMPARISONS = {
    "GPT-4o": {
        "base": "GPT-4o base",
        "targets": ["GPT-4o FT", "GPT-4o QSP"],
    },
    "Llama3.1-70B": {
        "base": "Llama3.1-70B base",
        "targets": ["Llama3.1-70B FT", "Llama3.1-70B QSP"],
    },
    "Llama3.1-8B": {
        "base": "Llama3.1-8B base",
        "targets": ["Llama3.1-8B FT", "Llama3.1-8B QSP"],
    },
}


def _variant_from_label(label: str) -> str:
    lowered = label.lower()
    if "lc" in lowered:
        return "LC"
    if "rag" in lowered:
        return "RAG"
    if "bm25" in lowered:
        return "BM25"
    if "question-specific" in lowered or "pv1" in lowered:
        return "QSP"
    if "ap" in lowered and "base" not in lowered:
        return "AP"
    if " ft" in lowered or lowered.endswith("ft") or " ft " in lowered or lowered.startswith("ft") or lowered.startswith("ft"):
        return "FT"
    if "ft" in lowered:
        return "FT"
    return "base"


def _family_from_label(label: str) -> str:
    for family in MODEL_BASE_COLORS:
        if label.startswith(family):
            return family
    return ""


def _tint_color(hex_color: str, tint: float) -> tuple[float, float, float]:
    base = mcolors.to_rgb(hex_color)
    tint = max(0.0, min(1.0, tint))
    return tuple(channel + (1 - channel) * tint for channel in base)


def _color_for_model(label: str) -> tuple[float, float, float]:
    family = _family_from_label(label)
    base_color = MODEL_BASE_COLORS.get(family, "#4c72b0")
    variant = _variant_from_label(label)
    tint = VARIANT_TINTS.get(variant, 0.0)
    return _tint_color(base_color, tint)


def _variant_handles() -> list[Patch]:
    handles: list[Patch] = []
    base_gray = "#888888"
    for variant, tint in VARIANT_TINTS.items():
        color = _tint_color(base_gray, tint)
        handles.append(Patch(facecolor=color, label=f"{variant} scenario".title()))
    return handles


def _family_handles() -> list[Patch]:
    return [Patch(facecolor=color, label=family) for family, color in MODEL_BASE_COLORS.items()]


def _variant_label(label: str) -> str:
    for family in MODEL_BASE_COLORS:
        if label.startswith(family):
            suffix = label[len(family):].strip()
            return suffix if suffix else "base"
    return label


def _group_positions(models: list[str]) -> tuple[list[float], dict[str, tuple[float, float]]]:
    positions: list[float] = []
    family_ranges: dict[str, list[float]] = {}
    gap = 1.5
    x = 0.0
    previous_family = None
    for model in models:
        family = _family_from_label(model)
        if previous_family is not None and family != previous_family:
            x += gap
        positions.append(x)
        family_ranges.setdefault(family, []).append(x)
        x += 1.0
        previous_family = family
    family_bounds = {fam: (min(pos_list), max(pos_list)) for fam, pos_list in family_ranges.items()}
    return positions, family_bounds


def _annotate_families(ax, family_bounds: dict[str, tuple[float, float]]) -> None:
    for family, (start, end) in family_bounds.items():
        center = (start + end) / 2
        ax.text(
            center,
            -0.24,
            family,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=FAMILY_LABEL_SIZE,
            fontweight="bold",
        )


def _variant_label(label: str) -> str:
    for family in MODEL_BASE_COLORS:
        if label.startswith(family):
            suffix = label[len(family):].strip()
            return suffix if suffix else "base"
    return label


def _position_lookup(models: list[str]) -> dict[str, float]:
    positions, _ = _group_positions(models)
    return dict(zip(models, positions))


def _annotate_significance(
    ax,
    models: list[str],
    pos_lookup: dict[str, float],
    significance: dict | None,
    metric: str,
    comparisons: dict | None = None,
    offset_shift: float = 0.0,
    metric_values: dict[str, float] | None = None,
) -> None:
    if not significance:
        return
    metric_map = significance.get(metric)
    if not metric_map:
        return
    y_max = ax.get_ylim()[1]
    base_margin = 0.6
    bracket_height = 0.05
    text_padding = 0.014
    offset_step = 0.15
    family_offsets: dict[str, float] = {}
    comparison_map = comparisons or FAMILY_COMPARISONS
    for family, mapping in comparison_map.items():
        base_label = mapping["base"]
        base_x = pos_lookup.get(base_label)
        base_val = None if metric_values is None else metric_values.get(base_label)
        if base_x is None:
            continue
        target_rows: list[tuple[float, str, float, str]] = []
        for target in mapping["targets"]:
            target_x = pos_lookup.get(target)
            if target_x is None:
                continue
            target_suffix = target.replace(f"{family} ", "")
            p_value = metric_map.get((family, target_suffix))
            if p_value is None:
                continue
            if metric_values is not None:
                target_val = metric_values.get(target)
                if base_val is not None and target_val is not None and target_val < base_val:
                    continue
            distance = abs(target_x - base_x)
            target_rows.append((distance, target, target_x, target_suffix))
        if not target_rows:
            continue
        target_rows.sort(key=lambda item: item[0])
        offset = family_offsets.get(family, y_max - base_margin - offset_shift)
        for _, target, target_x, target_suffix in target_rows:
            p_value = metric_map.get((family, target_suffix))
            if p_value is None or p_value > 0.05:
                continue
            label = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
            ax.plot(
                [base_x, base_x, target_x, target_x],
                [offset, offset + bracket_height, offset + bracket_height, offset],
                color="black",
                linewidth=1,
            )
            ax.text(
                (base_x + target_x) / 2,
                offset + bracket_height + text_padding,
                label,
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_FONT_SIZE,
            )
            offset += offset_step
        family_offsets[family] = offset


def plot_metric_panels(
    df,
    qid_df,
    title: str,
    output_path: Path,
    significance: dict | None = None,
    comparisons: dict | None = None,
    layout: str = "vertical",
) -> None:
    if df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = df["model"].tolist()
    positions, family_bounds = _group_positions(models)
    pos_lookup = _position_lookup(models)
    colors = [_color_for_model(model) for model in models]
    variant_labels = [_variant_label(model) for model in models]
    if layout == "combined":
        width = max(18, 0.7 * len(models) * len(METRIC_COLUMNS))
        height = 10
        fig, ax = plt.subplots(figsize=(width, height))
        total_width = 0.8
        bar_width = total_width / len(METRIC_COLUMNS)
        handles: list[Patch] = []
        for idx, (metric, label) in enumerate(METRIC_COLUMNS):
            metric_color = METRIC_PALETTE.get(metric, "#4c72b0")
            offsets = [pos - total_width / 2 + bar_width * (idx + 0.5) for pos in positions]
            values = df[metric].tolist()
            bars = ax.bar(offsets, values, color=metric_color, width=bar_width * 0.9)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=BAR_LABEL_SIZE,
            )
            handles.append(Patch(facecolor=metric_color, label=label))
            value_map = dict(zip(models, values))
            _annotate_significance(
                ax,
                models,
                pos_lookup,
                significance,
                metric,
                comparisons,
                offset_shift=idx * 0.35,
                metric_values=value_map,
            )
        ax.set_ylim(0, Y_LIM)
        ax.set_ylabel("Value", fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=AXIS_TICK_SIZE)
        ax.set_xticks(positions)
        ax.set_xticklabels(variant_labels, rotation=25, ha="right", fontsize=AXIS_TICK_SIZE)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=handles, fontsize=AXIS_TICK_SIZE, frameon=False)
        axes = [ax]
    else:
        width = max(14, 0.7 * len(models))
        height = 13
        fig, axes = plt.subplots(len(METRIC_COLUMNS), 1, figsize=(width, height), sharex=True)
        bar_width = 0.8
        if len(METRIC_COLUMNS) == 1:
            axes = [axes]
        for ax, (metric, label) in zip(axes, METRIC_COLUMNS):
            values = df[metric].tolist()
            bar_colors = colors
            bars = ax.bar(positions, values, color=bar_colors, width=bar_width)
            ax.set_ylim(0, Y_LIM)
            ax.set_ylabel(label, fontsize=AXIS_LABEL_SIZE)
            ax.tick_params(axis="both", labelsize=AXIS_TICK_SIZE)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=BAR_LABEL_SIZE,
                )
            value_map = dict(zip(models, values))
            _annotate_significance(ax, models, pos_lookup, significance, metric, comparisons, metric_values=value_map)
        axes[-1].set_xticks(positions)
        axes[-1].set_xticklabels(variant_labels, rotation=25, ha="right", fontsize=AXIS_TICK_SIZE)
    _annotate_families(axes[-1], family_bounds)
    fig.suptitle(title, fontsize=TITLE_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_table(df, title: str, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = 1.0 + 0.6 * len(df)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    data = df[["model", "accuracy", "precision", "recall", "f1"]].round(3).values
    data[:, 0] = df["model"].str.replace("Llama3.1", "L3.1", regex=False)
    table = ax.table(
        cellText=data,
        colLabels=["Model", "Accuracy", "Precision", "Recall", "F1"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(22)
    table.scale(1.4, 2.2)
    ax.set_title(title, fontweight="bold", pad=10, fontsize=TITLE_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def generate_figures(
    subset: pd.DataFrame,
    scenario: str,
    output_dir: Path,
    significance=None,
    comparisons=None,
    base_name: str | None = None,
) -> None:
    """Render accuracy bar chart and metrics table per scenario."""
    if subset.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = base_name or slugify(scenario)
    plot_metric_panels(
        subset,
        qid_df=None,
        title=scenario,
        output_path=output_dir / f"{slug}-bar-chart.png",
        significance=significance,
        comparisons=comparisons or FAMILY_COMPARISONS,
        layout="vertical",
    )
    save_table(subset, f"{scenario} Metrics", output_dir / f"{slug}-table.png")

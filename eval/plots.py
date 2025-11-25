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

FAMILY_COMPARISONS = {
    "GPT-4o": {
        "base": "GPT-4o base",
        "targets": [
            "GPT-4o FT",
            "GPT-4o QSP",
            "GPT-4o RAG",
        ],
    },
    "Llama3.1-70B": {
        "base": "Llama3.1-70B base",
        "targets": [
            "Llama3.1-70B FT",
            "Llama3.1-70B QSP",
            "Llama3.1-70B RAG",
        ],
    },
    "Llama3.1-8B": {
        "base": "Llama3.1-8B base",
        "targets": [
            "Llama3.1-8B FT",
            "Llama3.1-8B QSP",
            "Llama3.1-8B RAG",
        ],
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
            fontsize=10,
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
) -> None:
    if not significance:
        return
    metric_map = significance.get(metric)
    if not metric_map:
        return
    y_max = ax.get_ylim()[1]
    base_margin = 0.35
    bracket_height = 0.04
    text_padding = 0.01
    offset_step = 0.11
    family_offsets: dict[str, float] = {}
    for family, mapping in FAMILY_COMPARISONS.items():
        base_label = mapping["base"]
        base_x = pos_lookup.get(base_label)
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
            distance = abs(target_x - base_x)
            target_rows.append((distance, target, target_x, target_suffix))
        if not target_rows:
            continue
        target_rows.sort(key=lambda item: item[0])
        offset = family_offsets.get(family, y_max - base_margin)
        for _, target, target_x, target_suffix in target_rows:
            p_value = metric_map.get((family, target_suffix))
            if p_value is None:
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
                fontsize=9,
            )
            offset += offset_step
        family_offsets[family] = offset


def plot_metric_panels(
    df,
    qid_df,
    title: str,
    path: Path,
    significance: dict | None = None,
) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    models = df["model"].tolist()
    positions, family_bounds = _group_positions(models)
    pos_lookup = _position_lookup(models)
    colors = [_color_for_model(model) for model in models]
    variant_labels = [_variant_label(model) for model in models]
    width = max(14, 0.7 * len(models))
    fig, axes = plt.subplots(len(METRIC_COLUMNS), 1, figsize=(width, 11), sharex=True)
    if len(METRIC_COLUMNS) == 1:
        axes = [axes]
    for ax, (metric, label) in zip(axes, METRIC_COLUMNS):
        values = df[metric].tolist()
        bars = ax.bar(positions, values, color=colors, width=0.6)
        ax.set_ylim(0, 1.5)
        ax.set_ylabel(label)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        _annotate_significance(ax, models, pos_lookup, significance, metric)
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(variant_labels, rotation=25, ha="right", fontsize=9)
    _annotate_families(axes[-1], family_bounds)
    fig.suptitle(title, fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_table(df, title: str, path: Path) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = 0.7 + 0.4 * len(df)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=df[["model", "accuracy", "precision", "recall", "f1"]].round(3).values,
        colLabels=["Model", "Accuracy", "Precision", "Recall", "F1"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title(title, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def generate_figures(subset, title: str, output_dir: Path, qid_df=None, significance=None) -> None:
    slug = slugify(title)
    plot_metric_panels(subset, qid_df, title, output_dir / f"{slug}_accuracy.png", significance)
    save_table(subset, f"{title} Metrics", output_dir / f"{slug}_table.png")

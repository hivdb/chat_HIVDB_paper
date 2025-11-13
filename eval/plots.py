from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from .constants import MODEL_BASE_COLORS, VARIANT_TINTS
from .normalize import slugify


def _variant_from_label(label: str) -> str:
    lowered = label.lower()
    if "bm25" in lowered:
        return "BM25"
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


def plot_bar_chart(df, title: str, path: Path, footnote: str | None = None) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(14, 0.65 * len(df))
    fig, ax = plt.subplots(figsize=(width, 6))
    models = df["model"].tolist()
    positions, family_bounds = _group_positions(models)
    colors = [_color_for_model(model) for model in df["model"]]
    bars = ax.bar(positions, df["accuracy"], color=colors, width=0.6)
    variant_labels = [_variant_label(model) for model in models]
    ax.set_xticks(positions)
    ax.set_xticklabels(variant_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, df["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")

    _annotate_families(ax, family_bounds)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35, top=0.95, left=0.05, right=0.98)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_table(df, title: str, path: Path, footnote: str | None = None) -> None:
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


def generate_figures(subset, title: str, footnote: str | None, output_dir: Path) -> None:
    slug = slugify(title)
    plot_bar_chart(subset, f"{title} Accuracy", output_dir / f"{slug}_accuracy.png")
    save_table(subset, f"{title} Metrics", output_dir / f"{slug}_table.png")

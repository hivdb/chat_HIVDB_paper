from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .normalize import slugify


def plot_bar_chart(df, title: str, path: Path, footnote: str | None = None) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(df)), df["accuracy"], color="#4c72b0")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    for bar, value in zip(bars, df["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    if footnote:
        fig.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=8, wrap=True)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        fig.tight_layout()
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
    if footnote:
        fig.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=8, wrap=True)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def generate_figures(subset, title: str, footnote: str | None, output_dir: Path) -> None:
    slug = slugify(title)
    plot_bar_chart(subset, f"{title} Accuracy", output_dir / f"{slug}_accuracy.png", footnote)
    save_table(subset, f"{title} Metrics", output_dir / f"{slug}_table.png", footnote)

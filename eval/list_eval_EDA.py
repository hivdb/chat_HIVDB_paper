#!/usr/bin/env python3
"""EDA utility for list-type answers.

Loads eval/detailed_evaluation.csv, canonicalizes every answer column, and
generates two violin plots:
  1. Raw list lengths for all list-type rows (including “No/Not reported”).
  2. Raw list lengths restricted to rows where the human answer contains
     at least one token.

Both plots annotate the mean overlap fraction between each model and the
human list using the same lenient matching logic as evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import seaborn as sns


def project_root() -> Path:
    """Find the repo root by walking upward until eval/__init__.py exists."""
    start = Path(__file__).resolve()
    for candidate in [start.parent, *start.parents]:
        if (candidate / "eval" / "__init__.py").exists():
            return candidate
    return start.parent


ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from eval.normalize import (  # noqa: E402  (deferred import)
    canonicalize_answer,
    human_tokens,
    list_match_stats,
)


def compute_distributions(
    df: pd.DataFrame,
    answer_cols: Iterable[str],
    *,
    skip_empty_human: bool,
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = df.reset_index(drop=True)
    tall_records: list[dict[str, object]] = []
    overlap_values: dict[str, list[float]] = {col.replace(" Answer", ""): [] for col in answer_cols}

    normalized_cols: dict[str, pd.Series] = {
        col: df[col].apply(lambda value: canonicalize_answer(value, convert_special_no=False))
        for col in answer_cols
    }
    token_lists: dict[str, pd.Series] = {
        col: series.apply(lambda value: human_tokens(value) if value not in {"", "no"} else [])
        for col, series in normalized_cols.items()
    }

    for idx, row in df.iterrows():
        ref_tokens = token_lists["Human Answer"].iloc[idx]
        if skip_empty_human and not ref_tokens:
            continue
        ref_norm = normalized_cols["Human Answer"].iloc[idx]
        ref_total = len(ref_tokens)

        for col in answer_cols:
            source = col.replace(" Answer", "")
            tokens = token_lists[col].iloc[idx]
            tall_records.append({"Answer Source": source, "Item Count": len(tokens)})

            if ref_total == 0:
                overlap_values[source].append(float("nan"))
                continue
            if source == "Human":
                overlap_values[source].append(1.0)
                continue

            pred_norm = normalized_cols[col].iloc[idx]
            pred_raw = row[col]
            matches, _ = list_match_stats(ref_norm, pred_norm, pred_raw)
            overlap_values[source].append(matches / ref_total if ref_total else float("nan"))

    overlap_mean = {
        source: (pd.Series(values).mean(skipna=True) if values else float("nan"))
        for source, values in overlap_values.items()
    }
    return pd.DataFrame(tall_records), overlap_mean


def plot_violin(plot_df: pd.DataFrame, overlaps: dict[str, float], output_path: Path, title: str) -> None:
    if plot_df.empty:
        print(f"No list-type rows for '{title}'; skipping plot.")
        return

    sns.set_theme(style="whitegrid", context="talk")
    sources = list(plot_df["Answer Source"].unique())
    priority_order = [
        "Human",
        "GPT-5 base",
        "GPT-4o base",
        "GPT-4o FT",
        "GPT-4o AP Before",
        "GPT-4o AP",
        "GPT-4o AP After",
        "Llama3.1-70B base",
        "Llama3.1-70B FT",
        "Llama3.1-70B AP Before",
        "Llama3.1-70B AP",
        "Llama3.1-70B AP After",
        "Llama3.1-8B base",
        "Llama3.1-8B FT",
        "Llama3.1-8B AP Before",
        "Llama3.1-8B AP",
        "Llama3.1-8B AP After",
    ]
    order = [name for name in priority_order if name in sources] + [
        name for name in sources if name not in priority_order
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    palette = ["#c73832" if source == "Human" else "#4c72b0" for source in order]
    vp = sns.violinplot(
        data=plot_df,
        x="Answer Source",
        y="Item Count",
        order=order,
        inner="quartile",
        scale="width",
        cut=0,
        bw_adjust=0.6,
        palette=palette,
        linewidth=1.2,
        alpha=0.5,
        ax=ax,
    )
    for violin, color in zip(vp.collections[::2], palette):
        violin.set_facecolor(color)
        violin.set_edgecolor(color)
        violin.set_alpha(0.5)

    human_mean = plot_df.loc[plot_df["Answer Source"] == "Human", "Item Count"].mean()
    if pd.notna(human_mean):
        ax.axhline(
            human_mean,
            color="#c73832",
            linestyle="--",
            linewidth=1.5,
            label=f"Average Number of Items in Human Answer = {human_mean:.2f}",
        )

    max_count = plot_df["Item Count"].max()
    ax.set_ylim(0, max_count + 3)
    ax.text(
        0.5,
        1.05,
        "Overlap fraction = <(#Items in Human Answer) ∩ (#Items in Model Answer)| / (#Items in Human Answer)>",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#1f77b4",
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#1f77b4"),
    )
    for idx, source in enumerate(order):
        overlap = overlaps.get(source)
        if overlap is None or pd.isna(overlap):
            continue
        ax.text(
            idx,
            max_count + 1.0,
            f"{overlap:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
            color="#1f77b4",
        )

    # ax.set_title(title)
    ax.set_xlabel("Answer Source")
    ax.set_ylabel("Number of Items in 'List' Type Questions")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.xticks(rotation=25, ha="right")
    if pd.notna(human_mean):
        ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.8))
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved '{title}' violin plot with {len(plot_df):,} points across {len(order)} sources to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detail-path",
        type=Path,
        default=ROOT / "eval" / "detailed_evaluation.csv",
        help="CSV containing detailed evaluation rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "eval" / "figures",
        help="Directory for saving plots (two files will be written).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.detail_path)
    list_df = df[df["Type"].fillna("").str.lower() == "list"].copy()
    print(f"Loaded {len(df):,} total rows; {len(list_df):,} list-type questions.")
    if list_df.empty:
        print("No list rows found. Exiting.")
        return 0

    answer_cols = [col for col in list_df.columns if col == "Human Answer" or col.endswith(" Answer")]

    cases = [
        ("All List Answers", False, args.output_dir / "list-answer-lengths_all.png"),
        ("Non-empty Human List Answers", True, args.output_dir / "list-answer-lengths_nonempty.png"),
    ]
    for title, skip_empty, path in cases:
        plot_df, overlaps = compute_distributions(list_df, answer_cols, skip_empty_human=skip_empty)
        plot_violin(plot_df, overlaps, path, title)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

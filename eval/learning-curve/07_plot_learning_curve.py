#!/usr/bin/env python3
"""Generate learning-curve figures combining GPT-4o base/FT with LC subsets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

LC_DIR = Path(__file__).resolve().parent
ROOT = LC_DIR.parents[1]
ROOT_PARENT = ROOT.parent
for path in (ROOT, ROOT_PARENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval.plots import (  # type: ignore
    AXIS_LABEL_SIZE,
    AXIS_TICK_SIZE,
    BAR_LABEL_SIZE,
    MAX_Y_LIM,
    METRIC_COLUMNS,
    _color_for_model,
    generate_figures,
)


LC_RESULTS = LC_DIR / "results/learning_curve_metrics_full150.csv"
BASE_RESULTS = ROOT / "eval/results/evaluation_metrics_full150.csv"
OUTPUT_DIR = LC_DIR / "figures"
SIGNIFICANCE_JSON = LC_DIR / "results/learning_curve_significance_full150.json"

DISPLAY_SLOTS = [
    ("base", 0, ["GPT-4o base"]),
    ("FT-50", 50, ["GPT-4o FT-50", "GPT-4o LC size050"]),
    ("FT-100", 100, ["GPT-4o FT-100", "GPT-4o LC size100", "GPT-4o LC (100)"]),
    ("FT-150", 150, ["GPT-4o FT-150", "GPT-4o LC size150"]),
    ("FT-200", 200, ["GPT-4o FT-200", "GPT-4o FT (200)"]),
    ("FT-250", 250, ["GPT-4o FT", "GPT-4o FT (250)"]),
]

DISPLAY_SLOTS_LLAMA = [
    ("base", 0, ["Llama3.1-70B base"]),
    ("FT-50", 50, ["Llama3.1-70B FT-50"]),
    ("FT-100", 100, ["Llama3.1-70B FT-100"]),
    ("FT-150", 150, ["Llama3.1-70B FT-150"]),
    ("FT-200", 200, ["Llama3.1-70B FT-200"]),
    ("FT-250", 250, ["Llama3.1-70B FT"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=None, help="Learning-curve metrics CSV.")
    parser.add_argument("--base-results", type=Path, default=None, help="Base evaluation metrics CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for figure outputs.")
    parser.add_argument("--significance", type=Path, default=None, help="Significance JSON path.")
    parser.add_argument("--suffix", type=str, default="", help="Suffix appended to output filenames (e.g., new30).")
    parser.add_argument("--title", type=str, default="Learning Curve Analysis", help="Plot title.")
    return parser.parse_args()


def load_metrics(path: Path, scenarios: List[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file missing: {path}")
    df = pd.read_csv(path)
    if scenarios and "scenario" in df.columns:
        df = df[df["scenario"].isin(scenarios)].copy()
    return df


def select_models(df: pd.DataFrame, slots: List[tuple[str, int, List[str]]]) -> pd.DataFrame:
    rows: List[pd.Series] = []
    for display_label, size, candidates in slots:
        selected: pd.Series | None = None
        for candidate in candidates:
            match = df[df["model"] == candidate]
            if not match.empty:
                selected = match.iloc[0].copy()
                break
        if selected is None:
            continue
        selected["display_label"] = display_label
        selected["training_size"] = size
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_combined(lc_path: Path, base_path: Path, slots: List[tuple[str, int, List[str]]], scenarios: List[str] | None = None) -> pd.DataFrame:
    lc_df = load_metrics(lc_path, scenarios)
    base_df = load_metrics(base_path, scenarios)
    all_df = pd.concat([lc_df, base_df], ignore_index=True)
    combined = select_models(all_df, slots)
    if combined.empty:
        return combined
    # Preserve display order for plotting
    display_order = {label: idx for idx, (label, _, _) in enumerate(slots)}
    combined["display_order"] = combined["display_label"].map(display_order).fillna(len(DISPLAY_SLOTS) + 1)
    # Force ordering by training size (base first, size progression, FT)
    if "training_size" in combined.columns:
        combined.sort_values(["family" if "family" in combined.columns else "model", "training_size"], inplace=True)
    return combined


def build_side_by_side(gpt_df: pd.DataFrame, llama_df: pd.DataFrame) -> pd.DataFrame:
    """Combine GPT-4o and Llama3.1-70B learning-curve rows for a single plot."""
    if gpt_df.empty or llama_df.empty:
        return pd.DataFrame()
    gpt = gpt_df.copy()
    llama = llama_df.copy()
    # Offset display order so GPT-4o bars come first, followed by Llama.
    offset = gpt["display_order"].max() + 2
    llama["display_order"] = llama["display_order"] + offset
    return pd.concat([gpt, llama], ignore_index=True)


def plot_dual_learning_curve(
    gpt_df: pd.DataFrame,
    llama_df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Render side-by-side panels: A) GPT-4o, B) Llama3.1-70B."""
    if gpt_df.empty or llama_df.empty:
        return

    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "display_order" in df.columns:
            df.sort_values("display_order", inplace=True)
        for metric, _ in METRIC_COLUMNS:
            if metric in df.columns and df[metric].max() <= 1.0:
                df[metric] = df[metric] * 100.0
        return df

    def _plot_family(axs, df: pd.DataFrame, panel_title: str) -> None:
        if df.empty:
            return
        models = df["model"].tolist()
        labels = df.get("display_label", df["model"]).tolist()
        positions = list(range(len(models)))
        # Use the same family/variant palette as full150-bar-chart (base color + variant tint).
        colors = [_color_for_model(m) for m in models]
        axs[0].set_title(panel_title, fontsize=18, fontweight="bold", pad=18)
        for ax, (metric, label) in zip(axs, METRIC_COLUMNS):
            values = df[metric].tolist()
            bars = ax.bar(positions, values, color=colors, width=0.8)
            ax.set_ylim(0, MAX_Y_LIM)
            ax.set_ylabel(f"{label} (%)", fontsize=AXIS_LABEL_SIZE)
            ax.tick_params(axis="both", labelsize=AXIS_TICK_SIZE)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for bar, value in zip(bars, values):
                label_y = min(MAX_Y_LIM - 1.0, value + 4.0)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=BAR_LABEL_SIZE,
                )
        axs[-1].set_xticks(positions)
        axs[-1].set_xticklabels(labels, rotation=25, ha="right", fontsize=AXIS_TICK_SIZE)
        for ax in axs[:-1]:
            ax.set_xticks([])

    gpt_df = _prepare(gpt_df)
    llama_df = _prepare(llama_df)
    fig, axes = plt.subplots(
        len(METRIC_COLUMNS),
        2,
        figsize=(20, 16),
        sharey=True,
        gridspec_kw={"wspace": 0.25, "hspace": 0.4},
    )
    _plot_family(axes[:, 0], gpt_df, "A) GPT-4o")
    _plot_family(axes[:, 1], llama_df, "B) Llama3.1-70B")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def load_significance(path: Path) -> Tuple[Dict[str, Dict[Tuple[str, str], float]] | None, Dict[str, dict] | None]:
    if not path.exists():
        return None, None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    raw_map = data.get("ttest") or data.get("wilcoxon", {})
    comparisons = data.get("comparisons")
    tuple_map: Dict[str, Dict[Tuple[str, str], float]] = {}
    for metric, fam_map in raw_map.items():
        metric_entries: Dict[Tuple[str, str], float] = {}
        for family, comp_map in fam_map.items():
            for comparison, value in comp_map.items():
                metric_entries[(family, comparison)] = float(value)
        tuple_map[metric] = metric_entries
    return tuple_map or None, comparisons


def main() -> int:
    args = parse_args()
    suffix = args.suffix.strip()
    suffix = f"_{suffix}" if suffix else ""
    lc_results = args.metrics or LC_RESULTS
    base_results = args.base_results or BASE_RESULTS
    output_dir = args.output_dir or OUTPUT_DIR
    # P-value annotations are intentionally disabled for learning-curve plots.
    # Keep the significance path wiring in case we want to re-enable later, but do not load/use it here.
    significance_json = args.significance or SIGNIFICANCE_JSON.with_name(f"{SIGNIFICANCE_JSON.stem}{suffix}{SIGNIFICANCE_JSON.suffix}")

    combined_ft = build_combined(lc_results, base_results, DISPLAY_SLOTS)
    combined_llama_ft = build_combined(lc_results, base_results, DISPLAY_SLOTS_LLAMA)
    dual_available = not combined_ft.empty and not combined_llama_ft.empty
    if combined_ft.empty and combined_llama_ft.empty:
        print("No learning-curve metrics available.")
        return 1
    significance, comparisons = None, None
    # FT-only chart
    if not combined_ft.empty:
        generate_figures(
            combined_ft,
            f"{args.title}",
            output_dir,
            significance=significance,
            comparisons=comparisons,
            base_name=f"learning-curve_ft{suffix}" if suffix else "learning-curve_ft",
        )
    # Llama3.1-70B FT chart
    if not combined_llama_ft.empty:
        generate_figures(
            combined_llama_ft,
            f"Llama3.1-70B: Learning Curve Analysis",
            output_dir,
            significance=significance,
            comparisons=comparisons,
            base_name=f"learning-curve_llama{suffix}" if suffix else "learning-curve_llama",
        )
    # Combined GPT-4o and Llama3.1-70B FT learning curves side by side
    if dual_available:
        combined_path = output_dir / (f"learning-curve_combined{suffix}-bar-chart.png" if suffix else "learning-curve_combined-bar-chart.png")
        plot_dual_learning_curve(
            combined_ft,
            combined_llama_ft,
            combined_path,
            "Learning Curve: GPT-4o vs Llama3.1-70B",
        )
    print(f"Figures saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

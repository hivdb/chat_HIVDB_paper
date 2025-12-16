#!/usr/bin/env python3
"""Generate learning-curve figures combining GPT-4o base/FT with LC subsets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

LC_DIR = Path(__file__).resolve().parent
ROOT = LC_DIR.parents[1]
ROOT_PARENT = ROOT.parent
for path in (ROOT, ROOT_PARENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval.plots import generate_figures  # type: ignore


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

DISPLAY_SLOTS_QSP = [
    ("base", 0, ["GPT-4o base"]),
    ("FT-50+QSP", 50, ["GPT-4o FT-50+QSP"]),
    ("FT-100+QSP", 100, ["GPT-4o FT-100+QSP"]),
    ("FT-150+QSP", 150, ["GPT-4o FT-150+QSP"]),
    ("FT-200+QSP", 200, ["GPT-4o FT-200+QSP"]),
    ("FT-250+QSP", 250, ["GPT-4o FT+QSP"]),
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
    # Add a style hint for shading groups (base vs size-QSP progression vs final FT)
    def _style(row: pd.Series) -> str:
        label = row["model"].lower()
        if "ft+qsp" in label or "+qsp" in label:
            return "ft_qsp"
        if "ft" in label:
            return "ft"
        return "base"
    combined["style_group"] = combined.apply(_style, axis=1)
    # Force ordering by training size (base first, size QSP progression, FT, FT+QSP)
    if "training_size" in combined.columns:
        combined.sort_values(["family" if "family" in combined.columns else "model", "training_size"], inplace=True)
    # Color overrides to keep the learning-curve cluster visually coherent
    base_color = "#f8c291"   # light orange
    size_qsp_color = "#f5a623"  # mid orange for FT-XX+QSP
    ft_color = "#d35400"     # darker orange for final FT
    ftqsp_color = "#f39c12"  # slightly darker for FT+QSP
    colors = []
    for _, row in combined.iterrows():
        style = row["style_group"]
        if style == "base":
            colors.append(base_color)
        elif style == "ft_qsp":
            # Use a distinct color for the size-QSP progression and FT+QSP
            colors.append(size_qsp_color if "FT-" in row["model"] else ftqsp_color)
        else:
            colors.append(ft_color)
    combined["color_override"] = colors
    return combined


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
    combined_qsp = build_combined(lc_results, base_results, DISPLAY_SLOTS_QSP)
    combined_llama_ft = build_combined(lc_results, base_results, DISPLAY_SLOTS_LLAMA)
    if combined_ft.empty and combined_qsp.empty and combined_llama_ft.empty:
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
    # FT+QSP chart
    if not combined_qsp.empty:
        generate_figures(
            combined_qsp,
            f"{args.title}",
            output_dir,
            significance=significance,
            comparisons=comparisons,
            base_name=f"learning-curve_ftqsp{suffix}" if suffix else "learning-curve_ftqsp",
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
    print(f"Figures saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

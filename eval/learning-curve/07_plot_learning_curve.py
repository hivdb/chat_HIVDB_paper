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
    ("GPT-4o base", 0, ["GPT-4o base"]),
    ("GPT-4o FT-50", 50, ["GPT-4o FT-50", "GPT-4o LC size050"]),
    ("GPT-4o FT-100", 100, ["GPT-4o FT-100", "GPT-4o LC size100", "GPT-4o LC (100)"]),
    ("GPT-4o FT-150", 150, ["GPT-4o FT-150", "GPT-4o LC size150"]),
    ("GPT-4o FT-200", 200, ["GPT-4o FT-200", "GPT-4o FT (200)"]),
    ("GPT-4o FT", 250, ["GPT-4o FT", "GPT-4o FT (250)"]),
    ("Llama3.1-70B base", 0, ["Llama3.1-70B base"]),
    ("Llama3.1-70B FT-50", 50, ["Llama3.1-70B FT-50"]),
    ("Llama3.1-70B FT-100", 100, ["Llama3.1-70B FT-100"]),
    ("Llama3.1-70B FT-150", 150, ["Llama3.1-70B FT-150"]),
    ("Llama3.1-70B FT-200", 200, ["Llama3.1-70B FT-200"]),
    ("Llama3.1-70B FT", 250, ["Llama3.1-70B FT"]),
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
    scenarios = scenarios or ["Partial Match"]
    df = df[df["scenario"].isin(scenarios)].copy()
    return df


def select_models(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.Series] = []
    for display_label, size, candidates in DISPLAY_SLOTS:
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


def build_combined(lc_path: Path, base_path: Path, scenarios: List[str] | None = None) -> pd.DataFrame:
    target_scenarios = scenarios or ["Partial Match"]
    lc_df = load_metrics(lc_path, target_scenarios)
    base_df = load_metrics(base_path, target_scenarios)
    all_df = pd.concat([lc_df, base_df], ignore_index=True)
    combined = select_models(all_df)
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
    significance_json = args.significance or SIGNIFICANCE_JSON.with_name(f"{SIGNIFICANCE_JSON.stem}{suffix}{SIGNIFICANCE_JSON.suffix}")

    combined = build_combined(lc_results, base_results)
    if combined.empty:
        print("No learning-curve metrics available.")
        return 1
    significance, comparisons = load_significance(significance_json)
    base_name = f"learning-curve{suffix}" if suffix else "learning-curve"
    generate_figures(
        combined,
        args.title,
        output_dir,
        significance=significance,
        comparisons=comparisons,
        base_name=base_name,
    )
    print(f"Figures saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate learning-curve figures combining GPT-4o base/FT with LC subsets."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

LC_DIR = Path(__file__).resolve().parent
ROOT = LC_DIR.parents[1]
ROOT_PARENT = ROOT.parent
if str(ROOT_PARENT) not in sys.path:
    sys.path.insert(0, str(ROOT_PARENT))

from eval.plots import generate_figures  # type: ignore


LC_RESULTS = LC_DIR / "results/learning_curve_metrics.csv"
BASE_RESULTS = ROOT / "eval/results/evaluation_metrics.csv"
OUTPUT_DIR = LC_DIR / "figures"
COMBINED_CSV = LC_DIR / "results/learning_curve_overall_combined.csv"
SIGNIFICANCE_JSON = LC_DIR / "results/learning_curve_significance.json"

DISPLAY_SLOTS = [
    ("base", 0, ["GPT-4o base"]),
    ("FT-50", 50, ["GPT-4o FT-50", "GPT-4o LC size050"]),
    ("FT-100", 100, ["GPT-4o FT-100", "GPT-4o LC size100", "GPT-4o LC (100)"]),
    ("FT-150", 150, ["GPT-4o FT-150", "GPT-4o LC size150"]),
    ("FT-200", 200, ["GPT-4o FT-200", "GPT-4o FT (200)"]),
    ("FT", 250, ["GPT-4o FT", "GPT-4o FT (250)"]),
]


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


def build_combined() -> pd.DataFrame:
    target_scenarios = ["Partial Match"]
    lc_df = load_metrics(LC_RESULTS, target_scenarios)
    base_df = load_metrics(BASE_RESULTS, target_scenarios)
    all_df = pd.concat([lc_df, base_df], ignore_index=True)
    combined = select_models(all_df)
    if combined.empty:
        return combined
    combined.sort_values("training_size", inplace=True)
    combined.reset_index(drop=True, inplace=True)
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
    combined = build_combined()
    if combined.empty:
        print("No learning-curve metrics available.")
        return 1
    significance, comparisons = load_significance(SIGNIFICANCE_JSON)
    COMBINED_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED_CSV, index=False)
    title = "GPT-4o Learning Curve"
    generate_figures(combined, title, OUTPUT_DIR, significance=significance, comparisons=comparisons)
    print(f"Wrote combined metrics to {COMBINED_CSV}")
    print(f"Figures saved to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

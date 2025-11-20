#!/usr/bin/env python3
"""Generate learning-curve figures combining GPT-4o base/FT with LC subsets."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

from eval.plots import generate_figures  # type: ignore


LC_RESULTS = Path("eval/learning-curve/results/learning_curve_metrics.csv")
BASE_RESULTS = Path("eval/evaluation_metrics.csv")
OUTPUT_DIR = Path("eval/learning-curve/figures")
COMBINED_CSV = Path("eval/learning-curve/results/learning_curve_overall_combined.csv")

DISPLAY_SLOTS = [
    ("GPT-4o base (0)", 0, ["GPT-4o base"]),
    ("GPT-4o FT (50)", 50, ["GPT-4o LC size050", "GPT-4o FT (50)"]),
    ("GPT-4o FT (100)", 100, ["GPT-4o LC size100", "GPT-4o LC (100)", "GPT-4o FT (100)"]),
    ("GPT-4o FT (150)", 150, ["GPT-4o LC size150", "GPT-4o FT (150)"]),
    ("GPT-4o FT (200)", 200, ["GPT-4o FT", "GPT-4o FT (200)"]),
]


def load_metrics(path: Path, scenarios: List[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file missing: {path}")
    df = pd.read_csv(path)
    scenarios = scenarios or ["Overall (partial match)"]
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
        selected["model"] = display_label
        selected["training_size"] = size
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_combined() -> pd.DataFrame:
    target_scenarios = ["Overall (partial match)"]
    lc_df = load_metrics(LC_RESULTS, target_scenarios)
    base_df = load_metrics(BASE_RESULTS, target_scenarios)
    all_df = pd.concat([lc_df, base_df], ignore_index=True)
    combined = select_models(all_df)
    if combined.empty:
        return combined
    combined.sort_values("training_size", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def main() -> int:
    combined = build_combined()
    if combined.empty:
        print("No learning-curve metrics available.")
        return 1
    COMBINED_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED_CSV, index=False)
    title = "GPT-4o Learning Curve"
    footnote = "*Overall accuracy/precision/recall/F1 for GPT-4o with 0/50/100/150/200 supervised examples."
    generate_figures(combined, title, footnote, OUTPUT_DIR)
    print(f"Wrote combined metrics to {COMBINED_CSV}")
    print(f"Figures saved to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

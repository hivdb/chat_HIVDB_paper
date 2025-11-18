#!/usr/bin/env python3
"""Generate learning-curve figures combining GPT-4o base/FT with LC subsets."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

from eval.plots import generate_figures  # type: ignore


LC_RESULTS = Path("eval/learning-curve/results/learning_curve_metrics.csv")
BASE_RESULTS = Path("eval/evaluation_metrics.csv")
OUTPUT_DIR = Path("eval/learning-curve/figures")
COMBINED_CSV = Path("eval/learning-curve/results/learning_curve_overall_combined.csv")

MODEL_SIZE_MAP: Dict[str, int] = {
    "GPT-4o base": 0,
    "GPT-4o FT (50)": 50,
    "GPT-4o LC size050": 50,
    "GPT-4o FT (100)": 100,
    "GPT-4o LC size100": 100,
    "GPT-4o LC (100)": 100,
    "GPT-4o FT (150)": 150,
    "GPT-4o LC size150": 150,
    "GPT-4o FT": 200,
    "GPT-4o FT (200)": 200,
}


def load_metrics(path: Path, scenario: str = "Overall") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file missing: {path}")
    df = pd.read_csv(path)
    df = df[df["scenario"] == scenario].copy()
    return df


def select_models(df: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    rows: List[pd.Series] = []
    for label in labels:
        match = df[df["model"] == label]
        if match.empty:
            continue
        rows.append(match.iloc[0])
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).copy()
    return result


def build_combined() -> pd.DataFrame:
    lc_df = load_metrics(LC_RESULTS)
    base_df = load_metrics(BASE_RESULTS)
    all_df = pd.concat([lc_df, base_df], ignore_index=True)
    combined = select_models(all_df, list(MODEL_SIZE_MAP.keys()))
    if combined.empty:
        return combined
    combined["training_size"] = combined["model"].map(MODEL_SIZE_MAP)
    combined.sort_values("training_size", inplace=True)
    combined["model"] = combined.apply(_format_label, axis=1)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _format_label(row: pd.Series) -> str:
    size = int(row.get("training_size", 0) or 0)
    if size == 0:
        return "GPT-4o base (0)"
    if size == 50:
        return "GPT-4o FT (50)"
    if size == 100:
        return "GPT-4o FT (100)"
    if size == 150:
        return "GPT-4o FT (150)"
    if size == 200:
        return "GPT-4o FT (200)"
    return f"GPT-4o ({size})"


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

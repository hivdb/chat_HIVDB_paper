#!/usr/bin/env python3
"""
Summarize mismatch cases per model to inspect evaluation failure modes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from eval import config  # type: ignore

DETAIL_PATH = config.DETAIL_METRICS_HUMAN
OUTPUT_PATH = Path("eval/archive/mismatch_summary.csv")


@dataclass
class ModelReport:
    model: str
    scenario: str
    mismatches: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detailed-path",
        type=Path,
        default=DETAIL_PATH,
        help="Path to eval/detailed_evaluation.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to write the mismatch summary CSV",
    )
    return parser.parse_args()


def gather_models(df: pd.DataFrame) -> list[str]:
    models: list[str] = []
    for column in df.columns:
        if column.endswith(" Correct"):
            models.append(column.replace(" Correct", ""))
    return sorted(models)


def build_report(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    records: list[dict] = []
    key_columns = ["Scenario", "PMID", "QID", "Question", "Type", "Human Answer"]
    for model in models:
        correct_col = f"{model} Correct"
        answer_col = f"{model} Answer"
        mismatches = df[df[correct_col] == 0].copy()
        if mismatches.empty:
            continue
        mismatches["Model"] = model
        mismatches["Model Answer"] = mismatches[answer_col]
        records.append(
            mismatches[key_columns + ["Model", "Model Answer", correct_col]]
        )
    if not records:
        return pd.DataFrame(columns=key_columns + ["Model", "Model Answer"])
    return pd.concat(records, ignore_index=True)


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.detailed_path)
    models = gather_models(df)
    report = build_report(df, models)
    report.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote mismatch summary for {len(models)} models to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

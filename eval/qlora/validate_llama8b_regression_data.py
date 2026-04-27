#!/usr/bin/env python3
"""Validate llama8b_regression_data.csv against merged_answers_with_correct.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_COLUMNS = {
    "Llama3.1-8B FT Correct": ("FT", 25),
    "Llama3.1-8B R8 Correct": ("R8", 8),
    "Llama3.1-8B R16 Correct": ("R16", 16),
    "Llama3.1-8B R32 Correct": ("R32", 32),
}

KEY_COLUMNS = ["PMID", "QID"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that llama8b_regression_data.csv matches merged_answers_with_correct.csv."
    )
    parser.add_argument(
        "--merged",
        type=Path,
        default=Path("merged_answers_with_correct.csv"),
        help="Source merged answers file.",
    )
    parser.add_argument(
        "--recall",
        type=Path,
        default=Path("llama8b_regression_data.csv"),
        help="Regression-ready long file to validate.",
    )
    return parser.parse_args()


def normalize_correct(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    mapped = normalized.map(
        {"1": 1, "0": 0, "True": 1, "False": 0, "true": 1, "false": 0}
    )
    invalid = normalized[mapped.isna()]
    if not invalid.empty:
        examples = ", ".join(sorted(invalid.astype(str).unique())[:5])
        raise ValueError(f"Unexpected correctness values found: {examples}")
    return mapped.astype("int8")


def load_expected(merged_path: Path) -> pd.DataFrame:
    merged_df = pd.read_csv(merged_path, encoding="utf-8-sig")
    required = KEY_COLUMNS + list(MODEL_COLUMNS)
    missing = [column for column in required if column not in merged_df.columns]
    if missing:
        raise KeyError(f"Missing required columns in merged file: {missing}")

    working = merged_df[required].copy()
    working["PMID"] = working["PMID"].astype("string").str.strip()
    working["QID"] = pd.to_numeric(working["QID"], errors="raise").astype("int64")
    working["item_id"] = working["PMID"] + "_" + working["QID"].astype(str)

    for column in MODEL_COLUMNS:
        working[column] = normalize_correct(working[column])

    expected = working.melt(
        id_vars=["item_id", *KEY_COLUMNS],
        value_vars=list(MODEL_COLUMNS),
        var_name="source_column",
        value_name="expected_correct",
    )
    expected["model"] = expected["source_column"].map(lambda column: MODEL_COLUMNS[column][0])
    expected["rank"] = expected["source_column"].map(lambda column: MODEL_COLUMNS[column][1]).astype("int64")
    expected = expected[["item_id", "PMID", "QID", "model", "rank", "expected_correct"]]
    expected = expected.sort_values(["QID", "PMID", "rank"], kind="stable").reset_index(drop=True)
    return expected


def load_observed(recall_path: Path) -> pd.DataFrame:
    recall_df = pd.read_csv(recall_path, encoding="utf-8-sig")
    required = ["item_id", "PMID", "QID", "model", "rank", "correct"]
    missing = [column for column in required if column not in recall_df.columns]
    if missing:
        raise KeyError(f"Missing required columns in recall file: {missing}")

    observed = recall_df[required].copy()
    observed["PMID"] = observed["PMID"].astype("string").str.strip()
    observed["QID"] = pd.to_numeric(observed["QID"], errors="raise").astype("int64")
    observed["rank"] = pd.to_numeric(observed["rank"], errors="raise").astype("int64")
    observed["correct"] = pd.to_numeric(observed["correct"], errors="raise").astype("int8")
    observed = observed.rename(columns={"correct": "observed_correct"})
    observed = observed.sort_values(["QID", "PMID", "rank"], kind="stable").reset_index(drop=True)
    return observed


def main() -> None:
    args = parse_args()

    expected = load_expected(args.merged)
    observed = load_observed(args.recall)

    if len(expected) != len(observed):
        raise ValueError(
            f"Row count mismatch: expected {len(expected)} rows from merged file, "
            f"observed {len(observed)} rows in recall file."
        )

    merged = expected.merge(
        observed,
        on=["item_id", "PMID", "QID", "model", "rank"],
        how="outer",
        indicator=True,
    )

    missing_in_recall = merged[merged["_merge"] == "left_only"]
    missing_in_merged = merged[merged["_merge"] == "right_only"]
    mismatches = merged[
        (merged["_merge"] == "both")
        & (merged["expected_correct"].astype("int8") != merged["observed_correct"].astype("int8"))
    ]

    print(f"Expected rows: {len(expected)}")
    print(f"Observed rows: {len(observed)}")
    print(f"Missing in recall file: {len(missing_in_recall)}")
    print(f"Extra in recall file: {len(missing_in_merged)}")
    print(f"Correctness mismatches: {len(mismatches)}")

    if not missing_in_recall.empty:
        print("\nExamples missing in recall file:")
        print(missing_in_recall.head(10).to_string(index=False))
    if not missing_in_merged.empty:
        print("\nExamples extra in recall file:")
        print(missing_in_merged.head(10).to_string(index=False))
    if not mismatches.empty:
        print("\nCorrectness mismatches:")
        print(
            mismatches[
                ["item_id", "PMID", "QID", "model", "rank", "expected_correct", "observed_correct"]
            ].head(20).to_string(index=False)
        )
        raise SystemExit(1)

    if not missing_in_recall.empty or not missing_in_merged.empty:
        raise SystemExit(1)

    print("\nValidation passed: llama8b_regression_data.csv matches merged_answers_with_correct.csv.")


if __name__ == "__main__":
    main()

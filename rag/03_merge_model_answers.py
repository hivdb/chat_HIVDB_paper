#!/usr/bin/env python3
"""Merge one parsed model-answer file into the master merged_answers workbook."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MERGE_KEYS = ["PMID", "QID"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Parsed CSV/XLSX with PMID/QID/Answer.")
    parser.add_argument("--column-name", required=True, help="Destination column name in merged workbook.")
    parser.add_argument(
        "--target",
        type=Path,
        default=ROOT.parent / "advanced-prompting" / "csv" / "merged_answers.xlsx",
        help="Merged workbook to update (default: advanced-prompting/csv/merged_answers.xlsx).",
    )
    parser.add_argument(
        "--value-column",
        default="Answer",
        help="Column from --source containing the predicted answer (default: Answer).",
    )
    return parser.parse_args()


def normalize_identifier(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def load_frame(path: Path) -> pd.DataFrame:
    loader = pd.read_excel if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv
    df = loader(path, dtype=str, keep_default_na=False)
    df["PMID"] = df["PMID"].apply(normalize_identifier)
    df["QID"] = df["QID"].astype(int)
    return df


def main() -> int:
    args = parse_args()
    merged = load_frame(args.target)
    source = load_frame(args.source)

    if args.value_column not in source.columns:
        raise ValueError(f"Source file missing value column '{args.value_column}': {args.source}")

    source = (
        source[MERGE_KEYS + [args.value_column]]
        .rename(columns={args.value_column: args.column_name})
        .drop_duplicates(subset=MERGE_KEYS, keep="last")
    )

    if args.column_name not in merged.columns:
        merged[args.column_name] = ""

    updated = merged.merge(source, on=MERGE_KEYS, how="left", suffixes=("", "__new"))
    new_col = f"{args.column_name}__new"
    has_new_value = updated[new_col].notna() & (updated[new_col].astype(str).str.strip() != "")
    updated[args.column_name] = updated[new_col].where(has_new_value, updated[args.column_name])
    updated.drop(columns=[new_col], inplace=True)

    args.target.parent.mkdir(parents=True, exist_ok=True)
    updated.to_excel(args.target, index=False)
    print(f"Merged {len(source)} rows into {args.target} as '{args.column_name}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

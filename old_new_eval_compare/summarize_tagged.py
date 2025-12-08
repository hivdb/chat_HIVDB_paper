"""
Summarize counts by change_category and change_direction for each model sheet.

Usage:
    python summarize_tagged.py [input_excel] [output_excel]

Defaults:
    input_excel = 20251203_diffs_tagged.xlsx
    output_excel = 20251203_diffs_tagged_summary.xlsx

The input workbook should have `change_category` and `change_direction` columns
on each sheet (as produced by tag_diffs.py). The output workbook contains a
single sheet `summary` with rows:
    sheet | change_category | change_direction | count
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("20251208_diffs_tagged.xlsx")
DEFAULT_OUTPUT = Path("20251208_diffs_tagged_summary.xlsx")
CATEGORY_COLUMN = "change_category"
DIR_COLUMN = "change_direction"


def summarize_sheet(sheet: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return counts grouped by category and direction for one sheet."""
    if CATEGORY_COLUMN not in df.columns or DIR_COLUMN not in df.columns:
        raise ValueError(f"Sheet '{sheet}' missing required columns")
    grouped = (
        df[[CATEGORY_COLUMN, DIR_COLUMN]]
        .fillna("")
        .astype(str)
        .value_counts()
        .reset_index(name="count")
    )
    grouped.insert(0, "sheet", sheet)
    return grouped


def process_workbook(input_path: Path, output_path: Path) -> None:
    xls = pd.ExcelFile(input_path)
    frames = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, dtype=str)
        frames.append(summarize_sheet(sheet, df))
    summary = pd.concat(frames, ignore_index=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"Wrote tagged summary to: {output_path}")
    for sheet in summary["sheet"].unique():
        sub = summary[summary["sheet"] == sheet]
        total = sub["count"].sum()
        print(f"  {sheet}: {total} rows summarized")


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    process_workbook(input_path, output_path)


if __name__ == "__main__":
    main()

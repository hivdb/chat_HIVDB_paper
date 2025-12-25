"""
Add per-row change categories to each sheet in a diffs workbook.

Rules per row:
- truth_updated human answer? is True and old_correct != new_correct -> updated by ground truth
- truth_updated human answer? is True and old_correct == new_correct -> no change by ground truth
- truth_updated human answer? is False and old_correct != new_correct -> updated by algorithm
- truth_updated human answer? is False and old_correct == new_correct -> no change by algorithm

Output:
- A new Excel file with the same sheets as the input, each with an added
  `change_category` column and a `change_direction` column. No summary sheet.

Usage:
    python tag_diffs.py [input_excel] [output_excel]
Defaults:
    input_excel = 20251203_diffs.xlsx
    output_excel = 20251203_diffs_tagged.xlsx
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("20251208_diffs.xlsx")
DEFAULT_OUTPUT = Path("20251208_diffs_tagged.xlsx")
FLAG_COLUMN = "truth_updated human answer?"
CATEGORY_COLUMN = "change_category"
DIR_COLUMN = "change_direction"


def bool_like(series: pd.Series) -> pd.Series:
    """Interpret common truthy/falsey strings; non-empty non-falsey => True."""
    falsey = {"false", "no", "0", "none", "null"}
    s = series.fillna("").astype(str).str.strip()
    return ~s.str.lower().isin(falsey) & (s != "")


def detect_correct_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Find the old/new correct columns in a sheet."""
    old_cols = [c for c in df.columns if c.startswith("old_") and c.endswith(" correct")]
    if not old_cols:
        raise ValueError("No old_* correct column found")
    for old_col in old_cols:
        suffix = old_col[len("old_") :]
        new_col = f"new_{suffix}"
        if new_col in df.columns:
            return old_col, new_col
    raise ValueError("No matching new_* correct column found for detected old_* column(s)")


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    """Add change_category column per rules."""
    old_col, new_col = detect_correct_columns(df)
    flag = bool_like(df[FLAG_COLUMN]) if FLAG_COLUMN in df.columns else pd.Series(False, index=df.index)
    old_vals = df[old_col].fillna("").astype(str)
    new_vals = df[new_col].fillna("").astype(str)
    updated = old_vals != new_vals
    old_bool = bool_like(df[old_col])
    new_bool = bool_like(df[new_col])

    categories = pd.Series("", index=df.index, dtype=str)
    categories.loc[flag & updated] = "updated by ground truth"
    categories.loc[flag & ~updated] = "no change by ground truth"
    categories.loc[~flag & updated] = "updated by algorithm"
    categories.loc[~flag & ~updated] = "no change by algorithm"

    directions = pd.Series("no change", index=df.index, dtype=str)
    directions.loc[(~old_bool) & new_bool] = "false to true"
    directions.loc[old_bool & (~new_bool)] = "true to false"

    out = df.copy()
    out[CATEGORY_COLUMN] = categories
    out[DIR_COLUMN] = directions
    return out


def process_workbook(input_path: Path, output_path: Path) -> None:
    xls = pd.ExcelFile(input_path)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet in xls.sheet_names:
            tagged = categorize(pd.read_excel(xls, sheet_name=sheet, dtype=str))
            tagged.to_excel(writer, sheet_name=sheet[:31], index=False)
    print(f"Wrote tagged workbook to: {output_path}")


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    process_workbook(input_path, output_path)


if __name__ == "__main__":
    main()

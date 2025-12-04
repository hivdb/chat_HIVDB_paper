"""Summarize negative vs non-negative Human-Answer values grouped by QID.

Usage:
    python summarize_human_answers.py [input.xlsx] [output.xlsx]

Defaults:
    input.xlsx  -> Ground-Truth-150 Dec 4.xlsx
    output.xlsx -> <input-stem>_summary.xlsx
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


NEGATIVE_PREFIXES = (
    "not reported",
    "not applicable",
    "not stated",
    "none",
    "not provided",
    "not know",
    "0"
)


def is_negative(value: str) -> bool:
    """Return True when the value is considered negative."""
    text = (value or "").strip().lower()
    if not text:
        return True
    if text == "no":
        return True
    return text.startswith(NEGATIVE_PREFIXES)


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Ground-Truth-150 Dec 4.xlsx")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_name(f"{input_path.stem}_summary.xlsx")

    df = pd.read_excel(input_path, dtype=str, keep_default_na=False)
    if "QID" not in df.columns or "Human-Answer" not in df.columns:
        raise KeyError("Input file must contain 'QID' and 'Human-Answer' columns.")

    df["is_negative"] = df["Human-Answer"].apply(is_negative)
    summary = df.groupby("QID").agg(
        total=("Human-Answer", "size"),
        negative=("is_negative", "sum"),
    )
    summary["not_negative"] = summary["total"] - summary["negative"]
    summary["not_negative_ratio"] = (
        summary["not_negative"] / summary["total"] * 100
    ).map(lambda ratio: f"{ratio:.1f}%")

    summary.reset_index().to_excel(output_path, index=False)
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()

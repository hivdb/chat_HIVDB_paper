"""Summarize negative vs non-negative Human-Answer values grouped by QID.

Usage:
    python summarize_human_answers.py [input.xlsx] [output.xlsx]

Defaults:
    input.xlsx  -> Ground-Truth-150 Dec 18.xlsx
    output.xlsx -> <input-stem>_summary.xlsx
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PMIDS_OF_INTEREST = {
    "40391923",
    "40400229",
    "40431710",
    "40431742",
    "40490983",
    "40573423",
    "40596906",
    "40713598",
    "40763144",
    "40779404",
    "40839365",
    "40857111",
    "40872801",
    "40938120",
    "40938996",
    "40965168",
    "40974886",
    "41004608",
    "41012586",
    "41023944",
    "41029850",
    "41056006",
    "41057785",
    "41088231",
    "41091504",
    "41093949",
    "41112839",
    "41129268",
    "41130593",
    "41140464",
}


NEGATIVE_PREFIXES = (
    "not reported",
    "not applicable",
    "not applicable.",
    "not stated",
    "none",
    "not provided",
    "not know",
    'not known',
    "0"
)


def strip_trailing_parenthetical(value: str) -> str:
    """
    Remove any trailing parenthetical note starting with '('.

    Example: "positive (per note)" -> "positive"
    """
    text = (value or "").strip()
    if "(" in text:
        return text.rsplit("(", 1)[0].rstrip()
    return text


def is_negative(value: str) -> bool:
    """Return True when the value is considered negative."""
    text = strip_trailing_parenthetical(value).lower()
    if not text:
        return True
    if text == "no":
        return True
    return text.startswith(NEGATIVE_PREFIXES)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize negative vs non-negative counts for a dataframe."""
    working = df.copy()
    working["is_negative"] = working["Human-Answer"].apply(is_negative)
    summary = working.groupby("QID").agg(
        total=("Human-Answer", "size"),
        negative=("is_negative", "sum"),
    )
    summary["positive"] = summary["total"] - summary["negative"]
    summary["positive_ratio"] = (
        summary["positive"] / summary["total"] * 100
    ).map(lambda ratio: f"{ratio:.1f}%")
    return summary.reset_index()


def main() -> None:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Ground-Truth-150 Dec 18.xlsx")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_name(f"{input_path.stem}_summary.xlsx")

    df = pd.read_excel(input_path, dtype=str, keep_default_na=False)
    required_columns = {"QID", "Human-Answer", "PMID"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"Input file must contain columns: {', '.join(sorted(missing))}.")
    df["Human-Answer"] = df["Human-Answer"].apply(strip_trailing_parenthetical)

    summary_all = summarize(df)
    summary_all.to_excel(output_path, index=False)
    print(f"Wrote summary to {output_path}")

    pmid_mask = df["PMID"].astype(str).isin(PMIDS_OF_INTEREST)

    summary_included = summarize(df[pmid_mask])
    summary_included_path = output_path.with_name(f"{output_path.stem}_new30.xlsx")
    summary_included.to_excel(summary_included_path, index=False)
    print(f"Wrote PMIDs-only summary to {summary_included_path}")

    summary_excluded = summarize(df[~pmid_mask])
    summary_excluded_path = output_path.with_name(f"{output_path.stem}_120.xlsx")
    summary_excluded.to_excel(summary_excluded_path, index=False)
    print(f"Wrote summary without PMIDs to {summary_excluded_path}")


if __name__ == "__main__":
    main()

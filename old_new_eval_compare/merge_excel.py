"""
Merge the three sheets in 20251203.xlsx into a single Excel file.

- Treat all cell content as text.
- Join rows on the key columns PMID + QID.
- Prefix non-key columns with their sheet provenance to avoid collisions.
"""

from pathlib import Path
import pandas as pd

INPUT_FILE = Path("20251203.xlsx")
MERGED_OUTPUT = Path("20251203_merged.xlsx")
DIFFS_OUTPUT = Path("20251203_diffs.xlsx")
KEY_COLUMNS = ["PMID", "QID"]
SHEETS = [
    ("Ground truth change", "truth"),
    ("Old eval", "old"),
    ("New eval", "new"),
]
QUESTION_COL_PRIORITY = ["truth_Question", "old_Question", "new_Question"]
UPDATED_FLAG_COL = "truth_updated human answer?"


def load_sheet(sheet_name: str, prefix: str) -> pd.DataFrame:
    """Read a sheet as text and prefix non-key columns."""
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name, dtype=str)

    missing_keys = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing_keys:
        raise ValueError(f"Sheet '{sheet_name}' is missing key columns: {missing_keys}")

    rename_map = {col: f"{prefix}_{col}" for col in df.columns if col not in KEY_COLUMNS}
    return df.rename(columns=rename_map)


def merge_sheets() -> pd.DataFrame:
    """Merge the configured sheets on the key columns."""
    dfs = [load_sheet(name, prefix) for name, prefix in SHEETS]
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=KEY_COLUMNS, how="outer")
    return merged.fillna("")


def _bool_like(series: pd.Series) -> pd.Series:
    """Interpret any non-empty, non-falsey string as True."""
    falsey = {"false", "no", "0"}
    return ~series.fillna("").astype(str).str.strip().str.lower().isin(falsey) & (
        series.fillna("").astype(str).str.strip() != ""
    )


def _pick_question(merged: pd.DataFrame) -> pd.Series:
    """Choose the first available non-empty question column."""
    if not len(merged):
        return pd.Series([], dtype=str)
    question = pd.Series([""] * len(merged), index=merged.index, dtype=str)
    for col in QUESTION_COL_PRIORITY:
        if col in merged:
            candidate = merged[col].fillna("").astype(str)
            question = question.where(question != "", candidate)
    return question


def extract_diffs(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create per-model diff DataFrames keyed by suffix."""
    # Find paired correct columns
    old_correct_cols = [c for c in merged.columns if c.startswith("old_") and c.endswith(" correct")]
    pairs = []
    for old_col in old_correct_cols:
        suffix = old_col[len("old_") :]
        new_col = f"new_{suffix}"
        if new_col in merged.columns:
            pairs.append((suffix, old_col, new_col))

    updated_flag = (
        _bool_like(merged[UPDATED_FLAG_COL]) if UPDATED_FLAG_COL in merged else pd.Series([False] * len(merged))
    )
    question = _pick_question(merged)

    diffs: dict[str, pd.DataFrame] = {}
    for suffix, old_col, new_col in pairs:
        changed = merged[old_col].fillna("").astype(str) != merged[new_col].fillna("").astype(str)
        mask = changed | updated_flag
        base_suffix = suffix.replace(" correct", "")
        old_model_col = f"old_{base_suffix}"
        new_model_col = f"new_{base_suffix}"

        columns = KEY_COLUMNS + []
        # Keep model output columns if present
        if old_model_col in merged:
            columns.append(old_model_col)
        if new_model_col in merged:
            columns.append(new_model_col)
        columns += [old_col, new_col]

        subset = merged.loc[mask, columns].copy()
        subset.insert(2, "Question", question.loc[subset.index].values)
        if "truth_Human-Answer corrected" in merged:
            subset["truth_Human-Answer corrected"] = (
                merged.loc[subset.index, "truth_Human-Answer corrected"].fillna("")
            )
        if UPDATED_FLAG_COL in merged:
            subset[UPDATED_FLAG_COL] = merged.loc[subset.index, UPDATED_FLAG_COL].fillna("")
        diffs[suffix] = subset
    return diffs


def main() -> None:
    merged = merge_sheets()
    with pd.ExcelWriter(MERGED_OUTPUT, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False)
    print(f"Wrote merged file to: {MERGED_OUTPUT}")
    print(f"Rows: {len(merged)}, Columns: {len(merged.columns)}")

    # Build and write per-suffix diffs
    diffs = extract_diffs(merged)
    with pd.ExcelWriter(DIFFS_OUTPUT, engine="openpyxl") as writer:
        for suffix, df in diffs.items():
            sheet_name = suffix[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Wrote diffs file to: {DIFFS_OUTPUT}")
    for suffix, df in diffs.items():
        print(f"  Sheet '{suffix}': {len(df)} rows")


if __name__ == "__main__":
    main()

"""Generate ground-truth statistics from the survey answers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_PATH = Path("csv/all answers.xlsx")
OUTPUT_PATH = Path("csv/ground_truth_stat.csv")
DETAILS_OUTPUT_PATH = Path("csv/ground_truth_stat_details.csv")
EMPTY_TOKENS = {"No", "Not reported", "Not applicable", "0", 0}
STRING_EMPTY_TOKENS = {str(token).lower() for token in EMPTY_TOKENS if isinstance(token, str)}


def _clean_answer(value):
    """Return an empty string for tokens that count as "missing"."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in STRING_EMPTY_TOKENS:
            return ""
        return stripped

    if value in EMPTY_TOKENS:
        return ""

    return value


def _is_not_empty(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, float) and pd.isna(value):
        return False
    return True


def main() -> None:
    df = pd.read_excel(INPUT_PATH)

    if "Human Answer" not in df.columns or "QID" not in df.columns:
        missing = {"Human Answer", "QID"} - set(df.columns)
        raise ValueError(f"Missing expected columns: {missing}")

    df["Human Answer"] = df["Human Answer"].apply(_clean_answer)
    df["is_not_empty"] = df["Human Answer"].apply(_is_not_empty)

    DETAILS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DETAILS_OUTPUT_PATH, index=False)

    stats = (
        df.groupby("QID", dropna=False)["is_not_empty"].mean().reset_index()
    )
    stats = stats.rename(columns={"is_not_empty": "not_empty_rate"})
    stats["not_empty_rate"] = (stats["not_empty_rate"] * 100).round(2)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()

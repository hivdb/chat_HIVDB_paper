"""Merge advanced prompting and model answer columns into a single table.

This script reads the base answer sheet along with advanced prompting outputs
and merges them on PMID/QID. The result is saved as `merged_answers.xlsx`.
"""

from __future__ import annotations

import pandas as pd


SOURCE_BASE = "all answers.xlsx"
SOURCE_ADV = "gpt-4o-mini-2024-07-18_parsed.xlsx"
SOURCE_LLAMA_8B = "llama-3.1-8B_parsed.csv"
SOURCE_LLAMA_8B_before = "llama-3.1-8B_before_parsed.csv"
SOURCE_LLAMA_8B_after = "llama-3.1-8B_after_parsed.csv"
SOURCE_LLAMA_70B = "llama-3.1-70B_parsed.csv"
SOURCE_LLAMA_70B_before = "llama-3.1-70B_before_parsed.csv"
SOURCE_LLAMA_70B_after = "llama-3.1-70B_after_parsed.csv"
OUTPUT_PATH = "merged_answers.xlsx"


MERGE_KEYS = ["PMID", "QID"]


def _load_unique(path: str, usecols: list[str] | None) -> pd.DataFrame:
    """Load the given file and drop duplicate PMID/QID combinations."""

    loader = pd.read_excel if path.endswith(".xlsx") else pd.read_csv
    df = loader(path, usecols=usecols)
    # Keeping the last occurrence keeps the most recent revision if duplicates exist.
    return df.drop_duplicates(subset=MERGE_KEYS, keep="last")


def main() -> None:
    base = _load_unique(SOURCE_BASE, usecols=None)

    adv = _load_unique(SOURCE_ADV, usecols=MERGE_KEYS + ["GPT-4o AP"])
    llama_8b = _load_unique(SOURCE_LLAMA_8B, usecols=MERGE_KEYS + ["answer"]).rename(columns={"answer": "llama-3.1-8B AP"})
    llama_8b_before = _load_unique(SOURCE_LLAMA_8B_before, usecols=MERGE_KEYS + ["answer"]).rename(columns={"answer": "llama-3.1-8B AP before"})
    llama_8b_after = _load_unique(SOURCE_LLAMA_8B_after, usecols=MERGE_KEYS + ["answer"]).rename(columns={"answer": "llama-3.1-8B AP after"})

    llama_70b = _load_unique(SOURCE_LLAMA_70B, usecols=MERGE_KEYS + ["answer"]).rename(columns={"answer": "llama-3.1-70B AP"})
    llama_70b_before = _load_unique(SOURCE_LLAMA_70B_before, usecols=MERGE_KEYS + ["answer"]).rename(columns={"answer": "llama-3.1-70B AP before"})
    llama_70b_after = _load_unique(SOURCE_LLAMA_70B_after, usecols=MERGE_KEYS + ["answer"]).rename(columns={"answer": "llama-3.1-70B AP after"})

    merged = base.merge(adv, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_8b, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_8b_before, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_8b_after, on=MERGE_KEYS, how="left")

    merged = merged.merge(llama_70b, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_before, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_after, on=MERGE_KEYS, how="left")

    merged = merged.sort_values(MERGE_KEYS)
    merged.to_excel(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()

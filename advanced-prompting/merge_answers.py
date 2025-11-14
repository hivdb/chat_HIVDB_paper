"""Merge advanced prompting and model answer columns into a single table.

This script reads the base answer sheet along with advanced prompting outputs
and merges them on PMID/QID. The result is saved as `merged_answers.xlsx`.
"""

from __future__ import annotations

import pandas as pd


SOURCE_BASE = "./csv/all answers.xlsx"
SOURCE_ADV = "./csv/gpt-4o-mini-2024-07-18_parsed.xlsx"
SOURCE_ADV_BEFORE = "./csv/gpt-4o-mini-2024-07-18_before_parsed.xlsx"
SOURCE_ADV_AFTER = "./csv/gpt-4o-mini-2024-07-18_after_parsed.xlsx"
SOURCE_ADV_5shot = "./csv/gpt-4o-mini-2024-07-18_bm25_5-shot_parsed.xlsx"
SOURCE_ADV_10shot = "./csv/gpt-4o-mini-2024-07-18_bm25_10-shot_parsed.xlsx"
SOURCE_ADV_RAG = "./csv/gpt-4o-mini-2024-07-18_semantic_5-shot_parsed.xlsx"

SOURCE_LLAMA_8B = "./csv/llama-3.1-8B_parsed.csv"
SOURCE_LLAMA_8B_before = "./csv/llama-3.1-8B_before_parsed.csv"
SOURCE_LLAMA_8B_after = "./csv/llama-3.1-8B_after_parsed.csv"
SOURCE_LLAMA_8B_RAG = "./csv/llama-3.1-8B RAG_parsed.csv"
SOURCE_LLAMA_70B = "./csv/llama-3.1-70B_parsed.csv"
SOURCE_LLAMA_70B_before = "./csv/llama-3.1-70B_before_parsed.csv"
SOURCE_LLAMA_70B_after = "./csv/llama-3.1-70B_after_parsed.csv"
SOURCE_LLAMA_70B_RAG = "./csv/llama-3.1-70B RAG_parsed.csv"

SOURCE_LLAMA_8B_5shot = "./csv/llama-3.1-8B_bm25_5-shot_parsed.csv"
SOURCE_LLAMA_8B_10shot = "./csv/llama-3.1-8B_bm25_10-shot_parsed.csv"

SOURCE_LLAMA_70B_5shot = "./csv/llama-3.1-70B_bm25_5-shot_parsed.csv"
SOURCE_LLAMA_70B_10shot = "./csv/llama-3.1-70B_bm25_10-shot_parsed.csv"


OUTPUT_PATH = "./csv/merged_answers.xlsx"


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
    adv_before = _load_unique(
        SOURCE_ADV_BEFORE, usecols=MERGE_KEYS + ["GPT-4o AP Before"]
    )
    adv_after = _load_unique(SOURCE_ADV_AFTER, usecols=MERGE_KEYS + ["GPT-4o AP After"])
    adv_5shot = _load_unique(SOURCE_ADV_5shot, usecols=MERGE_KEYS + ["GPT-4o BM25 5-shot"])
    adv_10shot = _load_unique(SOURCE_ADV_10shot, usecols=MERGE_KEYS + ["GPT-4o BM25 10-shot"])
    adv_rag = _load_unique(SOURCE_ADV_RAG, usecols=MERGE_KEYS + ["GPT-4o Semantic 5-shot"]).rename(
        columns={"GPT-4o Semantic 5-shot": "GPT-4o RAG"}
    )

    llama_8b = _load_unique(SOURCE_LLAMA_8B, usecols=MERGE_KEYS + ["answer"]).rename(
        columns={"answer": "llama-3.1-8B AP"}
    )
    llama_8b_before = _load_unique(
        SOURCE_LLAMA_8B_before, usecols=MERGE_KEYS + ["answer"]
    ).rename(columns={"answer": "llama-3.1-8B AP before"})
    llama_8b_after = _load_unique(
        SOURCE_LLAMA_8B_after, usecols=MERGE_KEYS + ["answer"]
    ).rename(columns={"answer": "llama-3.1-8B AP after"})

    llama_8b_rag = _load_unique(
        SOURCE_LLAMA_8B_RAG, usecols=MERGE_KEYS + ["answer"]
    ).rename(columns={"answer": "llama-3.1-8B RAG"})

    llama_70b = _load_unique(SOURCE_LLAMA_70B, usecols=MERGE_KEYS + ["answer"]).rename(
        columns={"answer": "llama-3.1-70B AP"}
    )
    llama_70b_before = _load_unique(
        SOURCE_LLAMA_70B_before, usecols=MERGE_KEYS + ["answer"]
    ).rename(columns={"answer": "llama-3.1-70B AP before"})
    llama_70b_after = _load_unique(
        SOURCE_LLAMA_70B_after, usecols=MERGE_KEYS + ["answer"]
    ).rename(columns={"answer": "llama-3.1-70B AP after"})
    llama_70b_rag = _load_unique(
        SOURCE_LLAMA_70B_RAG, usecols=MERGE_KEYS + ["answer"]
    ).rename(columns={"answer": "llama-3.1-70B RAG"})

    llama_8b_5shot = _load_unique(
        SOURCE_LLAMA_8B_5shot, usecols=MERGE_KEYS + ["answer"]
    ).rename(columns={"answer": "llama-3.1-8B 5shot"})
    llama_8b_10shot = _load_unique(
        SOURCE_LLAMA_8B_10shot, usecols=MERGE_KEYS + ["answer"]
    ).rename(columns={"answer": "llama-3.1-8B 10shot"})
    llama_70b_5shot = _load_unique(
        SOURCE_LLAMA_70B_5shot, usecols=MERGE_KEYS + ["answer"]
    ).rename(columns={"answer": "llama-3.1-70B 5shot"})
    llama_70b_10shot = _load_unique(
        SOURCE_LLAMA_70B_10shot, usecols=MERGE_KEYS + ["answer"]
    ).rename(columns={"answer": "llama-3.1-70B 10shot"})

    merged = base.merge(adv, on=MERGE_KEYS, how="left")
    merged = merged.merge(adv_before, on=MERGE_KEYS, how="left")
    merged = merged.merge(adv_after, on=MERGE_KEYS, how="left")

    merged = merged.merge(adv_5shot, on=MERGE_KEYS, how="left")
    merged = merged.merge(adv_10shot, on=MERGE_KEYS, how="left")
    merged = merged.merge(adv_rag, on=MERGE_KEYS, how="left")

    merged = merged.merge(llama_8b, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_8b_before, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_8b_after, on=MERGE_KEYS, how="left")

    merged = merged.merge(llama_8b_rag, on=MERGE_KEYS, how="left")

    merged = merged.merge(llama_70b, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_before, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_after, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_rag, on=MERGE_KEYS, how="left")

    merged = merged.merge(llama_8b_5shot, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_8b_10shot, on=MERGE_KEYS, how="left")

    merged = merged.merge(llama_70b_5shot, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_10shot, on=MERGE_KEYS, how="left")

    merged = merged.sort_values(MERGE_KEYS)
    merged.to_excel(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()

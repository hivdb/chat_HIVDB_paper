"""Merge advanced prompting and model answer columns into a single table.

This script reads the base answer sheet along with advanced prompting outputs
and merges them on PMID/QID. The result is saved as `merged_answers.xlsx`.
"""

from __future__ import annotations

import pandas as pd


SOURCE_BASE = "./csv/ground_truth.xlsx"
# SOURCE_ADV = "./csv/gpt-4o-mini-2024-07-18_parsed.xlsx"
# SOURCE_ADV_BEFORE = "./csv/gpt-4o-mini-2024-07-18_before_parsed.xlsx"
# SOURCE_ADV_AFTER = "./csv/gpt-4o-mini-2024-07-18_after_parsed.xlsx"
# SOURCE_ADV_5shot = "./csv/gpt-4o-mini-2024-07-18_bm25_5-shot_parsed.xlsx"
# SOURCE_ADV_10shot = "./csv/gpt-4o-mini-2024-07-18_bm25_10-shot_parsed.xlsx"
# SOURCE_ADV_RAG = "./csv/gpt-4o-mini-2024-07-18_semantic_5-shot_parsed.xlsx"
# SOURCE_ADV_PV1 = "./csv/gpt-4o-mini-2024-07-18_PV1.xlsx"

# SOURCE_LLAMA_8B_before = "./csv/llama-3.1-8B_before_parsed.csv"
# SOURCE_LLAMA_8B_after = "./csv/llama-3.1-8B_after_parsed.csv"
# SOURCE_LLAMA_8B_RAG = "./csv/llama-3.1-8B RAG_parsed.csv"

SOURCE_GPT4O_BASE = "./csv/gpt-4o/gpt-4o-mini-base_parsed.csv"
SOURCE_GPT4O_FT = "./csv/gpt-4o/gpt-4o-mini-FT_parsed.csv"
SOURCE_GPT4O_FT_50 = "./csv/gpt-4o/gpt-4o-mini-FT 50_parsed.csv"
SOURCE_GPT4O_FT_100 = "./csv/gpt-4o/gpt-4o-mini-FT 100_parsed.csv"
SOURCE_GPT4O_FT_150 = "./csv/gpt-4o/gpt-4o-mini-FT 150_parsed.csv"
SOURCE_GPT4O_FT_200 = "./csv/gpt-4o/gpt-4o-mini-FT 200_parsed.csv"
SOURCE_GPT4O_PV1 = "./csv/gpt-4o/gpt-4o-mini-PV1_parsed.csv"
SOURCE_GPT4O_FT_PV1 = "./csv/gpt-4o/gpt-4o-mini-2024-07-18_FT_PV1.csv"

SOURCE_LLAMA_70B = "./csv/llama-70b/llama-3.1-70B-base_parsed.csv"
SOURCE_LLAMA_70B_FT_50 = "./csv/llama-70b/llama-3.1-70B-FT 50_parsed.csv"
SOURCE_LLAMA_70B_FT_100 = "./csv/llama-70b/llama-3.1-70B-FT 100_parsed.csv"
SOURCE_LLAMA_70B_FT_150 = "./csv/llama-70b/llama-3.1-70B-FT 150_parsed.csv"
SOURCE_LLAMA_70B_FT_200 = "./csv/llama-70b/llama-3.1-70B-FT 200_parsed.csv"
SOURCE_LLAMA_70B_FT = "./csv/llama-70b/llama-3.1-70B-FT_parsed.csv"
SOURCE_LLAMA_70B_FT_PV1 = "./csv/llama-70b/llama-3.1-70B-FT-PV1_parsed.csv"
SOURCE_LLAMA_70B_R16 = "./csv/llama-70b/llama-3.1-70B-FT_R16_parsed.csv"
SOURCE_LLAMA_70B_R32 = "./csv/llama-70b/llama-3.1-70B-FT_R32_parsed.csv"

# SOURCE_LLAMA_70B_before = "./csv/llama-3.1-70B_before_parsed.csv"
# SOURCE_LLAMA_70B_after = "./csv/llama-3.1-70B_after_parsed.csv"
# SOURCE_LLAMA_70B_RAG = "./csv/llama-3.1-70B RAG_parsed.csv"
SOURCE_LLAMA_70B_PV1 = "./csv/llama-70b/llama-3.1-70B-PV1_parsed.csv"

SOURCE_LLAMA_8B = "./csv/llama-8b/llama-3.1-8B-base_parsed.csv"
SOURCE_LLAMA_8B_FT = "./csv/llama-8b/llama-3.1-8B-FT_parsed.csv"
SOURCE_LLAMA_8B_PV1 = "./csv/llama-8b/llama-3.1-8B-PV1_parsed.csv"
SOURCE_LLAMA_8B_FT_PV1 = "./csv/llama-8b/llama-3.1-8B-FT-PV1_parsed.csv"
SOURCE_LLAMA_8B_R16 = "./csv/llama-8b/llama-3.1-8B-FT_R16_parsed.csv"
SOURCE_LLAMA_8B_R32 = "./csv/llama-8b/llama-3.1-8B-FT_R32_parsed.csv"

# SOURCE_LLAMA_8B_5shot = "./csv/llama-3.1-8B_bm25_5-shot_parsed.csv"
# SOURCE_LLAMA_8B_10shot = "./csv/llama-3.1-8B_bm25_10-shot_parsed.csv"

# SOURCE_LLAMA_70B_5shot = "./csv/llama-3.1-70B_bm25_5-shot_parsed.csv"
# SOURCE_LLAMA_70B_10shot = "./csv/llama-3.1-70B_bm25_10-shot_parsed.csv"


OUTPUT_PATH = "./csv/merged_answers.xlsx"
OUTPUT_CSV_PATH = "./csv/merged_answers.csv"


MERGE_KEYS = ["PMID", "QID"]


def _load_unique(path: str, usecols: list[str] | None) -> pd.DataFrame:
    """Load the given file and drop duplicate PMID/QID combinations."""

    loader = pd.read_excel if path.endswith(".xlsx") else pd.read_csv
    load_kwargs = {"usecols": usecols, "dtype": str, "keep_default_na": False}

    if loader is pd.read_csv:
        try:
            df = loader(path, **load_kwargs)
        except UnicodeDecodeError:
            df = loader(path, encoding="latin-1", **load_kwargs)
    else:
        df = loader(path, **load_kwargs)
    df = df.astype(str)
    # Keeping the last occurrence keeps the most recent revision if duplicates exist.
    return df.drop_duplicates(subset=MERGE_KEYS, keep="last")


def main() -> None:
    base = _load_unique(SOURCE_BASE, usecols=None)

    # adv = _load_unique(SOURCE_ADV, usecols=MERGE_KEYS + ["GPT-4o AP"])
    # adv_before = _load_unique(
    #     SOURCE_ADV_BEFORE, usecols=MERGE_KEYS + ["GPT-4o AP Before"]
    # )
    # adv_after = _load_unique(SOURCE_ADV_AFTER, usecols=MERGE_KEYS + ["GPT-4o AP After"])
    # adv_5shot = _load_unique(
    #     SOURCE_ADV_5shot, usecols=MERGE_KEYS + ["GPT-4o BM25 5-shot"]
    # )
    # adv_10shot = _load_unique(
    #     SOURCE_ADV_10shot, usecols=MERGE_KEYS + ["GPT-4o BM25 10-shot"]
    # )
    # adv_rag = _load_unique(
    #     SOURCE_ADV_RAG, usecols=MERGE_KEYS + ["GPT-4o Semantic 5-shot"]
    # ).rename(columns={"GPT-4o Semantic 5-shot": "GPT-4o RAG"})
    # adv_pv1 = _load_unique(SOURCE_ADV_PV1, usecols=MERGE_KEYS + ["GPT-4o PV1"])

    gpt4o_base = _load_unique(
        SOURCE_GPT4O_BASE, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "gpt-4o-mini base"})
    gpt4o_ft_source = _load_unique(
        SOURCE_GPT4O_FT, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "gpt-4o-mini-FT"})
    gpt4o_ft_50 = _load_unique(
        SOURCE_GPT4O_FT_50, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "gpt-4o-mini-FT 50"})
    gpt4o_ft_100 = _load_unique(
        SOURCE_GPT4O_FT_100, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "gpt-4o-mini-FT 100"})
    gpt4o_ft_150 = _load_unique(
        SOURCE_GPT4O_FT_150, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "gpt-4o-mini-FT 150"})
    gpt4o_ft_200 = _load_unique(
        SOURCE_GPT4O_FT_200, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "gpt-4o-mini-FT 200"})
    gpt4o_pv1 = _load_unique(
        SOURCE_GPT4O_PV1, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "gpt-4o-mini PV1"})
    gpt4o_ft_pv1 = _load_unique(
        SOURCE_GPT4O_FT_PV1, usecols=MERGE_KEYS + ["GPT-4o FT_PV1"]
    ).rename(columns={"GPT-4o FT_PV1": "gpt-4o-mini FT PV1"})

    llama_8b = _load_unique(SOURCE_LLAMA_8B, usecols=MERGE_KEYS + ["Answer"]).rename(
        columns={"Answer": "llama-3.1-8B base"}
    )
    llama_8b_ft_source = _load_unique(
        SOURCE_LLAMA_8B_FT, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "llama-3.1-8B-FT"})
    # llama_8b_before = _load_unique(
    #     SOURCE_LLAMA_8B_before, usecols=MERGE_KEYS + ["Answer"]
    # ).rename(columns={"Answer": "llama-3.1-8B AP before"})
    # llama_8b_after = _load_unique(
    #     SOURCE_LLAMA_8B_after, usecols=MERGE_KEYS + ["Answer"]
    # ).rename(columns={"Answer": "llama-3.1-8B AP after"})
    # llama_8b_rag = _load_unique(
    #     SOURCE_LLAMA_8B_RAG, usecols=MERGE_KEYS + ["Answer"]
    # ).rename(columns={"Answer": "llama-3.1-8B RAG"})
    llama_8b_pv1 = _load_unique(
        SOURCE_LLAMA_8B_PV1, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "llama-3.1-8B PV1"})
    llama_8b_ft_pv1 = _load_unique(
        SOURCE_LLAMA_8B_FT_PV1, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "llama-3.1-8B-FT PV1"})
    llama_8b_r16 = _load_unique(
        SOURCE_LLAMA_8B_R16, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "llama_8B_R16"})
    llama_8b_r32 = _load_unique(
        SOURCE_LLAMA_8B_R32, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "llama_8B_R32"})

    llama_70b = _load_unique(SOURCE_LLAMA_70B, usecols=MERGE_KEYS + ["Answer"]).rename(
        columns={"Answer": "llama-3.1-70B base"}
    )
    # llama_70b_before = _load_unique(
    #     SOURCE_LLAMA_70B_before, usecols=MERGE_KEYS + ["Answer"]
    # ).rename(columns={"Answer": "llama-3.1-70B AP before"})
    # llama_70b_after = _load_unique(
    #     SOURCE_LLAMA_70B_after, usecols=MERGE_KEYS + ["Answer"]
    # ).rename(columns={"Answer": "llama-3.1-70B AP after"})
    # llama_70b_rag = _load_unique(
    #     SOURCE_LLAMA_70B_RAG, usecols=MERGE_KEYS + ["Answer"]
    # ).rename(columns={"Answer": "llama-3.1-70B RAG"})
    llama_70b_pv1 = _load_unique(
        SOURCE_LLAMA_70B_PV1, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "llama-3.1-70B PV1"})
    llama_70b_ft_source = _load_unique(
        SOURCE_LLAMA_70B_FT, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "llama-3.1-70B-FT"})
    llama_70b_ft_pv1 = _load_unique(
        SOURCE_LLAMA_70B_FT_PV1, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "llama-3.1-70B-FT PV1"})
    llama_70b_r16 = _load_unique(
        SOURCE_LLAMA_70B_R16, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "llama_70B_R16"})
    llama_70b_r32 = _load_unique(
        SOURCE_LLAMA_70B_R32, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "llama_70B_R32"})

    llama_70b_ft_50 = _load_unique(
        SOURCE_LLAMA_70B_FT_50, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "llama-3.1-70B-FT 50"})
    llama_70b_ft_100 = _load_unique(
        SOURCE_LLAMA_70B_FT_100, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "llama-3.1-70B-FT 100"})
    llama_70b_ft_150 = _load_unique(
        SOURCE_LLAMA_70B_FT_150, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "llama-3.1-70B-FT 150"})
    llama_70b_ft_200 = _load_unique(
        SOURCE_LLAMA_70B_FT_200, usecols=MERGE_KEYS + ["Answer"]
    ).rename(columns={"Answer": "llama-3.1-70B-FT 200"})

    # llama_8b_5shot = _load_unique(
    #     SOURCE_LLAMA_8B_5shot, usecols=MERGE_KEYS + ["Answer"]
    # ).rename(columns={"Answer": "llama-3.1-8B 5shot"})
    # llama_8b_10shot = _load_unique(
    #     SOURCE_LLAMA_8B_10shot, usecols=MERGE_KEYS + ["Answer"]
    # ).rename(columns={"Answer": "llama-3.1-8B 10shot"})
    # llama_70b_5shot = _load_unique(
    #     SOURCE_LLAMA_70B_5shot, usecols=MERGE_KEYS + ["Answer"]
    # ).rename(columns={"Answer": "llama-3.1-70B 5shot"})
    # llama_70b_10shot = _load_unique(
    #     SOURCE_LLAMA_70B_10shot, usecols=MERGE_KEYS + ["Answer"]
    # ).rename(columns={"Answer": "llama-3.1-70B 10shot"})

    merged = base.merge(gpt4o_base, on=MERGE_KEYS, how="left")
    merged = merged.merge(gpt4o_ft_source, on=MERGE_KEYS, how="left")
    merged = merged.merge(gpt4o_ft_50, on=MERGE_KEYS, how="left")
    merged = merged.merge(gpt4o_ft_100, on=MERGE_KEYS, how="left")
    merged = merged.merge(gpt4o_ft_150, on=MERGE_KEYS, how="left")
    merged = merged.merge(gpt4o_ft_200, on=MERGE_KEYS, how="left")
    merged = merged.merge(gpt4o_pv1, on=MERGE_KEYS, how="left")
    merged = merged.merge(gpt4o_ft_pv1, on=MERGE_KEYS, how="left")

    merged = merged.merge(llama_8b, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_8b_ft_source, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_8b_pv1, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_8b_ft_pv1, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_8b_r16, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_8b_r32, on=MERGE_KEYS, how="left")

    merged = merged.merge(llama_70b, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_ft_50, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_ft_100, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_ft_150, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_ft_200, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_ft_source, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_pv1, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_ft_pv1, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_r16, on=MERGE_KEYS, how="left")
    merged = merged.merge(llama_70b_r32, on=MERGE_KEYS, how="left")

    # merged = base.merge(adv, on=MERGE_KEYS, how="left")
    # merged = merged.merge(adv_before, on=MERGE_KEYS, how="left")
    # merged = merged.merge(adv_after, on=MERGE_KEYS, how="left")
    # merged = merged.merge(adv_5shot, on=MERGE_KEYS, how="left")
    # merged = merged.merge(adv_10shot, on=MERGE_KEYS, how="left")
    # merged = merged.merge(adv_rag, on=MERGE_KEYS, how="left")
    # merged = merged.merge(adv_pv1, on=MERGE_KEYS, how="left")

    # merged = merged.merge(llama_8b_before, on=MERGE_KEYS, how="left")
    # merged = merged.merge(llama_8b_after, on=MERGE_KEYS, how="left")
    # merged = merged.merge(llama_8b_rag, on=MERGE_KEYS, how="left")

    # merged = merged.merge(llama_70b_before, on=MERGE_KEYS, how="left")
    # merged = merged.merge(llama_70b_after, on=MERGE_KEYS, how="left")
    # merged = merged.merge(llama_70b_rag, on=MERGE_KEYS, how="left")

    # merged = merged.merge(llama_8b_5shot, on=MERGE_KEYS, how="left")
    # merged = merged.merge(llama_8b_10shot, on=MERGE_KEYS, how="left")
    # merged = merged.merge(llama_70b_5shot, on=MERGE_KEYS, how="left")
    # merged = merged.merge(llama_70b_10shot, on=MERGE_KEYS, how="left")

    merged = merged.sort_values(MERGE_KEYS)
    merged.to_excel(OUTPUT_PATH, index=False)
    merged.to_csv(OUTPUT_CSV_PATH, index=False)


if __name__ == "__main__":
    main()

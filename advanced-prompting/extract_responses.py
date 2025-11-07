#!/usr/bin/env python3
"""Extract answers from pmid_responses.jsonl into an Excel sheet."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROMPTS_EXCEL = Path("S4Table.xlsx")
RESPONSES_JSONL = Path("pmid_responses.jsonl")
OUTPUT_EXCEL = Path("gpt-4o-mini-2024-07-18_parsed.xlsx")


def extract_answers(text: str) -> list[str]:
    answers: list[str] = []
    pattern = re.compile(r"Answer:\s*(.+)")
    for block in re.split(r'\"\"\"', text):
        for match in pattern.finditer(block):
            answers.append(match.group(1).strip())
    if not answers:
        for match in pattern.finditer(text):
            answers.append(match.group(1).strip())
    return answers


def main() -> int:
    df = pd.read_excel(PROMPTS_EXCEL, usecols=["PMID", "QID", "Question", "Type", "Category"])
    responses: dict[str, list[str]] = {}

    if RESPONSES_JSONL.exists():
        with RESPONSES_JSONL.open("r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                pmid = str(record["pmid"])
                responses[pmid] = extract_answers(record.get("response", ""))

    df["GPT-4o AP"] = ""

    for pmid, group in df.groupby("PMID", sort=False):
        answers = responses.get(str(pmid), [])
        if not answers:
            continue
        group_indices = group.sort_values("QID").index.tolist()
        for idx, row_index in enumerate(group_indices):
            if idx < len(answers):
                df.at[row_index, "GPT-4o AP"] = answers[idx]

    df["GPT-4o AP"] = df["GPT-4o AP"].apply(lambda x: "" if pd.isna(x) else str(x))

    df[["PMID", "QID", "Question", "Type", "Category", "GPT-4o AP"]].to_excel(
        OUTPUT_EXCEL, index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

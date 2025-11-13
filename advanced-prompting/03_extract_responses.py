#!/usr/bin/env python3
"""Extract answers from pmid_responses.jsonl into an Excel sheet."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROMPTS_EXCEL = Path("advanced-prompting/csv/S4Table.xlsx")
DATASETS = (
    (
        Path("advanced-prompting/jsonl/dynamic_responses_bm25_5-shot.jsonl"),
        Path("advanced-prompting/csv/gpt-4o-mini-2024-07-18_bm25_5-shot_parsed.xlsx"),
        "GPT-4o BM25 5-shot",
    ),
    (
        Path("advanced-prompting/jsonl/dynamic_responses_bm25_10-shot.jsonl"),
        Path("advanced-prompting/csv/gpt-4o-mini-2024-07-18_bm25_10-shot_parsed.xlsx"),
        "GPT-4o BM25 10-shot",
    ),
)


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


def process_dataset(responses_jsonl: Path, output_excel: Path, column_name: str) -> None:
    df = pd.read_excel(PROMPTS_EXCEL, usecols=["PMID", "QID", "Question", "Type", "Category"])
    responses: dict[str, list[str]] = {}

    if responses_jsonl.exists():
        with responses_jsonl.open("r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                pmid = str(record["pmid"])
                responses[pmid] = extract_answers(record.get("response", ""))

    df[column_name] = ""

    for pmid, group in df.groupby("PMID", sort=False):
        answers = responses.get(str(pmid), [])
        if not answers:
            continue
        group_indices = group.sort_values("QID").index.tolist()
        for idx, row_index in enumerate(group_indices):
            if idx < len(answers):
                df.at[row_index, column_name] = answers[idx]

    df[column_name] = df[column_name].apply(lambda x: "" if pd.isna(x) else str(x))

    output_columns = ["PMID", "QID", "Question", "Type", "Category", column_name]
    df[output_columns].to_excel(output_excel, index=False)


def main() -> int:
    for responses_jsonl, output_excel, column_name in DATASETS:
        process_dataset(responses_jsonl, output_excel, column_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

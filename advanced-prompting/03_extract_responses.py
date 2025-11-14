#!/usr/bin/env python3
"""Extract answers from pmid_responses.jsonl into an Excel sheet."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PROMPTS_EXCEL = Path("advanced-prompting/csv/S4Table.xlsx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="Path to the model responses JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output Excel file.",
    )
    parser.add_argument(
        "--column-name",
        type=str,
        default="Model Answer",
        help="Column name to store extracted answers.",
    )
    return parser.parse_args()


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
    args = parse_args()
    process_dataset(args.responses, args.output, args.column_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

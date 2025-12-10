#!/usr/bin/env python3
"""Extract answers from pmid_responses*.jsonl into per-model Excel sheets."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


RESPONSES_DIR = Path("advanced-prompting/jsonl")
OUTPUT_DIR = Path("advanced-prompting/csv")
TEMPLATE_PATHS = {
    "original120": Path("advanced-prompting/csv/gpt-4o-mini-2024-07-18_PV1.xlsx"),
    "new30": Path("advanced-prompting/csv/gpt-4o-mini-2024-07-18_PV1_new30.xlsx"),
}
COLUMN_NAME = "GPT-4o FT+PV1"
SUFFIXES = ["FT", "FT-50", "FT-100", "FT-150", "FT-200"]


@dataclass(frozen=True)
class ResponseJob:
    suffix: str
    dataset: str
    responses_path: Path
    template_path: Path
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suffix",
        action="append",
        choices=SUFFIXES,
        help="Limit to specific fine-tune suffixes (default: all).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(TEMPLATE_PATHS.keys()),
        help="Limit to specific datasets (default: both).",
    )
    parser.add_argument(
        "--column-name",
        default=COLUMN_NAME,
        help=f"Column name for extracted answers (default: {COLUMN_NAME}).",
    )
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=RESPONSES_DIR,
        help=f"Directory containing pmid_responses JSONL files (default: {RESPONSES_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory to write Excel outputs (default: {OUTPUT_DIR}).",
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


def load_template(template_path: Path, column_name: str) -> pd.DataFrame:
    df = pd.read_excel(template_path)
    if "GPT-4o PV1" in df.columns:
        df = df.rename(columns={"GPT-4o PV1": column_name})
    if column_name not in df.columns:
        # Insert after Question if present, else append.
        cols = list(df.columns)
        if "Question" in cols:
            insert_idx = cols.index("Question") + 1
            cols.insert(insert_idx, column_name)
            df[column_name] = ""
            df = df[cols]
        else:
            df[column_name] = ""
    df[column_name] = ""
    return df


def process_dataset(responses_jsonl: Path, template_path: Path, output_excel: Path, column_name: str) -> None:
    df = load_template(template_path, column_name)
    responses: dict[str, list[str]] = {}

    with responses_jsonl.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pmid = str(record["pmid"])
            responses[pmid] = extract_answers(record.get("response", ""))

    for pmid, group in df.groupby("PMID", sort=False):
        answers = responses.get(str(pmid), [])
        if not answers:
            continue
        ordered = group.sort_values("QID")
        for idx, row_index in enumerate(ordered.index):
            if idx < len(answers):
                df.at[row_index, column_name] = answers[idx]

    df[column_name] = df[column_name].apply(lambda x: "" if pd.isna(x) else str(x))
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_excel, index=False)


def build_jobs(args: argparse.Namespace) -> list[ResponseJob]:
    suffixes: Iterable[str] = args.suffix or SUFFIXES
    datasets: Iterable[str] = args.dataset or TEMPLATE_PATHS.keys()
    jobs: list[ResponseJob] = []

    for dataset in datasets:
        template = TEMPLATE_PATHS[dataset]
        if dataset == "original120":
            pattern = "pmid_responses_Nov17_Version1_{suffix}.jsonl"
        else:
            pattern = "pmid_responses_Nov17_Version1_2025_30_{suffix}.jsonl"
        for suffix in suffixes:
            responses_path = args.responses_dir / pattern.format(suffix=suffix)
            if not responses_path.exists():
                print(f"Skipping missing responses file: {responses_path}")
                continue
            output_name = f"gpt-4o-mini-2024-07-18_{suffix}_PV1_{dataset}.xlsx"
            output_path = args.output_dir / output_name
            jobs.append(
                ResponseJob(
                    suffix=suffix,
                    dataset=dataset,
                    responses_path=responses_path,
                    template_path=template,
                    output_path=output_path,
                )
            )
    return jobs


def main() -> int:
    args = parse_args()
    jobs = build_jobs(args)
    if not jobs:
        print("No response files found for the requested suffix/dataset selection.")
        return 1

    for job in jobs:
        print(
            f"Processing {job.responses_path.name} "
            f"-> {job.output_path.name} (dataset={job.dataset}, suffix={job.suffix})"
        )
        process_dataset(job.responses_path, job.template_path, job.output_path, args.column_name)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

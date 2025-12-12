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
    "full150": Path("advanced-prompting/csv/gpt-4o-mini-2024-07-18_PV1.xlsx"),
    "original120": Path("advanced-prompting/csv/gpt-4o-mini-2024-07-18_PV1.xlsx"),
    "new30": Path("advanced-prompting/csv/gpt-4o-mini-2024-07-18_PV1_new30.xlsx"),
}
COLUMN_NAME = "GPT-4o FT+PV1"
SUFFIXES = ["FT", "FT-50", "FT-100", "FT-150", "FT-200"]


def normalize_question(text: str) -> str:
    """Lowercase/whitespace-normalize for matching question text across files."""
    return " ".join(str(text or "").strip().lower().split())


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
    """
    Extract answers in order; falls back to positional matching when question text is unavailable.
    """
    answers: list[str] = []
    pattern = re.compile(r"Answer:\s*(.+)")
    for block in re.split(r'\"\"\"', text):
        for match in pattern.finditer(block):
            answers.append(match.group(1).strip())
    if not answers:
        for match in pattern.finditer(text):
            answers.append(match.group(1).strip())
    return answers


def extract_qa_pairs(text: str) -> list[tuple[str, str]]:
    """
    Extract (question, answer) pairs from a response block.

    Expected format:
    Question: ...
    ...
    Answer: ...
    """
    pattern = re.compile(r"Question:\s*(.+?)\n.*?Answer:\s*(.+?)(?=\nQuestion:|\Z)", re.DOTALL | re.IGNORECASE)
    pairs: list[tuple[str, str]] = []
    for q_raw, a_raw in pattern.findall(text):
        q_norm = normalize_question(q_raw)
        a_val = a_raw.strip()
        if q_norm or a_val:
            pairs.append((q_norm, a_val))
    return pairs


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
    responses_positional: dict[str, list[str]] = {}
    responses_by_question: dict[str, dict[str, list[str]]] = {}

    # Precompute normalized questions from template for faster lookup
    df["__question_norm"] = df["Question"].apply(normalize_question)

    with responses_jsonl.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pmid = str(record["pmid"])
            text = record.get("response", "")
            responses_positional[pmid] = extract_answers(text)
            qa_pairs = extract_qa_pairs(text)
            if qa_pairs:
                qmap: dict[str, list[str]] = {}
                for q_norm, ans in qa_pairs:
                    qmap.setdefault(q_norm, []).append(ans)
                responses_by_question[pmid] = qmap

    for pmid, group in df.groupby("PMID", sort=False):
        pmid_str = str(pmid)
        answers = responses_positional.get(pmid_str, [])
        qmap = responses_by_question.get(pmid_str, {})
        if not answers and not qmap:
            continue
        ordered = group.sort_values("QID")
        # First, try to fill by question text if available
        for row_index, row in ordered.iterrows():
            q_norm = row["__question_norm"]
            candidates = qmap.get(q_norm, [])
            if candidates:
                df.at[row_index, column_name] = candidates.pop(0)
        # Then fill any remaining blanks positionally
        remaining = ordered[df.loc[ordered.index, column_name] == ""]
        for idx, row_index in enumerate(remaining.index):
            if idx < len(answers):
                df.at[row_index, column_name] = answers[idx]

    df[column_name] = df[column_name].apply(lambda x: "" if pd.isna(x) else str(x))
    df.drop(columns="__question_norm", inplace=True, errors="ignore")
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_excel, index=False)


def build_jobs(args: argparse.Namespace) -> list[ResponseJob]:
    suffixes: Iterable[str] = args.suffix or SUFFIXES
    datasets: Iterable[str] = args.dataset or TEMPLATE_PATHS.keys()
    jobs: list[ResponseJob] = []

    for dataset in datasets:
        template = TEMPLATE_PATHS[dataset]
        if dataset == "new30":
            pattern = "pmid_responses_Nov17_Version1_2025_30_{suffix}.jsonl"
        else:
            pattern = "pmid_responses_Nov17_Version1_{suffix}.jsonl"
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

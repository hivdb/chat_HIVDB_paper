#!/usr/bin/env python3
"""Parse the `Multiple Answer` column in llama-3.1-8B data.

The script expands each "Question" block into its own row containing the
PMID, QID, question text, evidence, rationale, and answer.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
from typing import Dict, List, Set, Tuple

import pandas as pd

# Regex that captures each question block in the bolded format together with its
# trailing content until the next question header.
QUESTION_BLOCK_RE = re.compile(
    r"\*\*Question(?:\s+(?P<id>\d+))?:\s*(?P<question>.*?)\*\*\s*(?P<body>.*?)(?=\n\*\*Question|\Z)",
    re.DOTALL,
)

# Some model responses format each answer inside triple-quoted blocks instead of
# bold headers. Those look like:
# """
# Question: ...
#
# Evidence: ...
#
# ...
# """
TRIPLE_QUOTED_BLOCK_SPLIT_RE = re.compile(r'"""', re.DOTALL)

# Matches one of the known field labels so we can delimit sections in the body.
FIELD_LABELS = "Evidence|Rationale|Answer"
FIELD_RE_TEMPLATE = r"{label}:\s*(.*?)(?=\n\n(?:" + FIELD_LABELS + r"):\s|\Z)"


def clean_text(value: str) -> str:
    """Normalise whitespace and escape sequences within parsed text."""
    if value is None:
        return ""
    # Replace doubled quotes that appear due to CSV escaping and trim whitespace.
    return value.replace('""', '"').strip()


def extract_field(body: str, label: str) -> str:
    pattern = re.compile(FIELD_RE_TEMPLATE.format(label=re.escape(label)), re.DOTALL)
    match = pattern.search(body)
    if not match:
        return ""
    return clean_text(match.group(1))


def parse_multiple_answer(cell: str) -> List[Dict[str, str]]:
    """Yield one dictionary per question extracted from the cell text."""
    if not cell or not cell.strip():
        return []
    results: List[Dict[str, str]] = []

    matches = list(QUESTION_BLOCK_RE.finditer(cell))
    if matches:
        for match in matches:
            body = match.group("body") or ""
            results.append(
                {
                    "question": clean_text(match.group("question")),
                    "evidence": extract_field(body, "Evidence"),
                    "rationale": extract_field(body, "Rationale"),
                    "answer": extract_field(body, "Answer"),
                }
            )
        return results

    # Fallback for triple-quoted blocks that lack the bold question headers.
    for segment in TRIPLE_QUOTED_BLOCK_SPLIT_RE.split(cell):
        segment = segment.strip()
        if not segment:
            continue

        # Ignore any leading narration that precedes the quoted blocks.
        if not segment.lower().startswith("question"):
            continue

        question_line, _, remainder = segment.partition("\n\n")
        if not remainder:
            # If double newline is missing, fall back to single newline separation.
            question_line, _, remainder = segment.partition("\n")

        if ":" in question_line:
            _, _, question_text = question_line.partition(":")
        else:
            question_text = question_line

        body = remainder.strip()
        results.append(
            {
                "question": clean_text(question_text),
                "evidence": extract_field(body, "Evidence"),
                "rationale": extract_field(body, "Rationale"),
                "answer": extract_field(body, "Answer"),
            }
        )

    return results


def load_qid_lookup(s2_table_path: pathlib.Path) -> Dict[str, Dict[str, str]]:
    """Build a lookup from question text to the associated PMID/QID pair."""
    if not s2_table_path.exists():
        raise FileNotFoundError(f"Missing S2 table: {s2_table_path}")

    dataframe = pd.read_excel(s2_table_path)
    lookup: Dict[str, Dict[str, str]] = {}

    for _, row in dataframe.iterrows():
        if pd.isna(row.get("Question")) or pd.isna(row.get("QID")):
            continue

        question = clean_text(str(row["Question"]))
        qid = clean_text(str(row["QID"]))
        pmid_value = row.get("PMID", "")
        pmid = "" if pd.isna(pmid_value) else clean_text(str(pmid_value))

        if question in lookup and lookup[question]["QID"] != qid:
            raise ValueError(
                f"Conflicting QID assignments for question {question!r}: {lookup[question]['QID']} vs {qid}"
            )
        lookup[question] = {"PMID": pmid, "QID": qid, "question": question}

    return lookup


def parse_file(
    input_path: pathlib.Path, output_path: pathlib.Path, s2_table_path: pathlib.Path
) -> None:
    rows: List[Dict[str, str]] = []
    qid_lookup = load_qid_lookup(s2_table_path)

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pmid = clean_text(row.get("PMID", ""))

            matched_pairs: Set[Tuple[str, str]] = set()

            parsed_questions = parse_multiple_answer(row.get("Multiple Answer", ""))
            print(f"PMID {pmid or '[unknown]'}: parsed {len(parsed_questions)} question(s)")
            for question in parsed_questions:
                qid_entry = qid_lookup.get(question["question"])
                if qid_entry is None:
                    raise KeyError(
                        "Could not find QID for question {question!r}".format(
                            question=question["question"]
                        )
                    )

                matched_pairs.add((qid_entry["QID"], pmid))
                rows.append(
                    {
                        "PMID": pmid,
                        "QID": qid_entry["QID"],
                        **question,
                    }
                )

            for qid_entry in qid_lookup.values():
                key = (qid_entry["QID"], pmid)
                if key in matched_pairs:
                    continue
                rows.append(
                    {
                        "PMID": pmid,
                        "QID": qid_entry["QID"],
                        "question": qid_entry["question"],
                        "evidence": "",
                        "rationale": "",
                        "answer": "No",
                    }
                )
    fieldnames = ["PMID", "QID", "question", "evidence", "rationale", "answer"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path("llama-3.1-8B.csv"),
        help="Path to the input CSV file (default: llama-3.1-8B.csv)",
    )
    parser.add_argument(
        "output",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path("llama-3.1-8B_parsed.csv"),
        help="Path to the output CSV file (default: llama-3.1-8B_parsed.csv)",
    )
    parser.add_argument(
        "--s2-table",
        type=pathlib.Path,
        default=pathlib.Path("S2Table.xlsx"),
        help="Path to the S2Table.xlsx file that contains PMID/QID/question mappings",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    parse_file(args.input, args.output, args.s2_table)


if __name__ == "__main__":
    main()

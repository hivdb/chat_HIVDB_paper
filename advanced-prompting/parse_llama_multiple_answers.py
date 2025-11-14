#!/usr/bin/env python3
"""Expand Llama multiple-answer cells into per-question CSV rows.

The 70B RAG runs record all answers inside the `Multiple Answer` column
where each question is written as a stand-alone block (usually wrapped
with triple quotes).  This script splits those blocks, extracts the
Question/Evidence/Rationale/Answer fields, and matches each entry to the
canonical QIDs listed in `S2Table.xlsx`.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

QUESTION_FIELDS = ("question", "evidence", "rationale", "answer")
DELIMITER_LINES = {
    '"""',
    '""',
    "'''",
    '```',
}
FIELD_LINE_RE = re.compile(
    r"""
    ^
    (?P<prefix>[\s\#\*\>\-`]*?)
    (?P<label>Question|Evidence|Rationale|Answer)
    (?:\s*(?:[:\-]\s*|\s+))
    (?P<value>.*)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
QUESTION_HEADER_RE = re.compile(
    r"""
    ^
    \s*\#{2,}\s*
    q(?:uestion)?(?:\s*id)?
    \s*(?P<digits>\d+)
    (?P<rest>.*)
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)
QUESTION_LINE_HEADER_RE = re.compile(
    r"""
    ^
    \s*question
    (?:\s*(?P<digits>\d+))?
    \s*[:\-]\s*
    (?P<value>.*)
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

QuestionBlock = Dict[str, str]


def clean_text(value: object) -> str:
    """Return the CSV-safe representation of any cell-like value."""

    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.replace('""', '"').strip()


def normalise_question_id(value: object) -> str:
    if value is None:
        return ""
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else ""


def normalise_question_text(value: str) -> str:
    if not value:
        return ""
    text = re.sub(r"\s+", " ", value).strip().lower()
    return text.rstrip(" ?.")


def normalise_cell(cell: object) -> str:
    if cell is None:
        return ""
    text = clean_text(cell)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\n", "\n")
    return text.strip()


def _is_delimiter(line: str) -> bool:
    stripped = line.strip()
    return stripped in DELIMITER_LINES


def _detect_header(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if QUESTION_HEADER_RE.match(stripped):
        return stripped
    if QUESTION_LINE_HEADER_RE.match(stripped):
        return stripped
    return None


def extract_question_blocks(text: str) -> List[Tuple[str, str]]:
    """Split a cell into (header, body) tuples."""

    blocks: List[Tuple[str, str]] = []
    header: str | None = None
    body_lines: List[str] = []

    for raw_line in text.splitlines():
        if _is_delimiter(raw_line):
            continue
        candidate = _detect_header(raw_line)
        if candidate:
            if header is not None or body_lines:
                blocks.append((header or "", "\n".join(body_lines).strip()))
                body_lines = []
            header = candidate
            continue
        body_lines.append(raw_line.rstrip())

    if header is not None or body_lines:
        blocks.append((header or "", "\n".join(body_lines).strip()))

    # Filter out empty shells
    return [(hdr, body) for hdr, body in blocks if hdr or body]


def parse_sections(body: str) -> Dict[str, str]:
    collected: Dict[str, List[str]] = {label: [] for label in QUESTION_FIELDS}
    current_label: str | None = None

    for raw_line in body.splitlines():
        if _is_delimiter(raw_line):
            current_label = None
            continue
        detection = FIELD_LINE_RE.match(raw_line.strip())
        if detection:
            current_label = detection.group("label").lower()
            value = detection.group("value") or ""
            if value:
                collected[current_label].append(value.strip())
            continue
        if current_label:
            collected[current_label].append(raw_line.rstrip())

    return {label: clean_text("\n".join(lines).strip()) for label, lines in collected.items()}


def parse_question_header(header_line: str) -> Tuple[str, str]:
    cleaned = header_line.strip().lstrip("#").strip()
    match = re.match(
        r"(?i)q(?:uestion)?(?:\s*id)?\s*(\d+)(?:\s*[:\-]\s*)?(.*)",
        cleaned,
    )
    if match:
        qid = normalise_question_id(match.group(1))
        question_text = clean_text(match.group(2))
        return qid, question_text
    fallback = re.sub(r"(?i)^question\s*[:\-]\s*", "", cleaned)
    return "", clean_text(fallback)


def parse_block(header: str, body: str) -> QuestionBlock | None:
    header_qid, header_question = parse_question_header(header)
    sections = parse_sections(body)

    if not any(sections.values()) and not header_question:
        return None

    question_text = sections.get("question") or header_question
    return {
        "question_id": header_qid,
        "question": question_text or "",
        "evidence": sections.get("evidence", ""),
        "rationale": sections.get("rationale", ""),
        "answer": sections.get("answer", ""),
    }


def parse_multiple_answer(cell: object) -> List[QuestionBlock]:
    text = normalise_cell(cell)
    if not text:
        return []

    parsed: List[QuestionBlock] = []
    blocks = extract_question_blocks(text)
    if not blocks:
        fallback = parse_block("", text)
        if fallback:
            parsed.append(fallback)
        return parsed

    for header, body in blocks:
        block = parse_block(header, body)
        if block:
            parsed.append(block)
    return parsed


def load_question_lookup(
    s2_table_path: pathlib.Path,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    if not s2_table_path.exists():
        raise FileNotFoundError(f"Missing S2 table: {s2_table_path}")

    dataframe = pd.read_excel(s2_table_path)
    question_lookup: Dict[str, Dict[str, str]] = {}
    qid_lookup: Dict[str, Dict[str, str]] = {}

    for _, row in dataframe.iterrows():
        question = clean_text(row.get("Question"))
        qid = normalise_question_id(row.get("QID"))
        if not question or not qid:
            continue

        entry = {"QID": qid, "question": question}
        qid_lookup.setdefault(qid, entry)
        norm_text = normalise_question_text(question)
        if norm_text:
            question_lookup.setdefault(norm_text, entry)

    return question_lookup, qid_lookup


def match_question(
    question_lookup: Dict[str, Dict[str, str]],
    qid_lookup: Dict[str, Dict[str, str]],
    block: QuestionBlock,
) -> Tuple[str, str]:
    qid = normalise_question_id(block.get("question_id"))
    if qid and qid in qid_lookup:
        entry = qid_lookup[qid]
        return entry["QID"], entry["question"]

    question_text = clean_text(block.get("question", ""))
    norm_text = normalise_question_text(question_text)
    if norm_text and norm_text in question_lookup:
        entry = question_lookup[norm_text]
        return entry["QID"], entry["question"]

    return qid, question_text


def _row_sort_key(row: Dict[str, str]) -> Tuple[str, int, str]:
    pmid = row.get("PMID", "")
    qid = row.get("QID", "")
    try:
        return pmid, 0, f"{int(qid):04d}"
    except (TypeError, ValueError):
        return pmid, 1, qid or ""


def parse_file(
    input_path: pathlib.Path, output_path: pathlib.Path, s2_table_path: pathlib.Path
) -> None:
    rows: List[Dict[str, str]] = []
    question_lookup, qid_lookup = load_question_lookup(s2_table_path)

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pmid = clean_text(row.get("PMID", ""))
            parsed_blocks = parse_multiple_answer(row.get("Multiple Answer", ""))
            print(f"PMID {pmid or '[unknown]'}: parsed {len(parsed_blocks)} question(s)")
            matched: set[Tuple[str, str]] = set()

            for block in parsed_blocks:
                qid, canonical_question = match_question(question_lookup, qid_lookup, block)
                if not qid:
                    print(
                        f"  Warning: PMID {pmid or '[unknown]'} missing QID for question {block.get('question', '')!r}"
                    )
                    continue

                key = (pmid, qid)
                if key in matched:
                    continue
                matched.add(key)
                rows.append(
                    {
                        "PMID": pmid,
                        "QID": qid,
                        "question": canonical_question or block.get("question", ""),
                        "evidence": block.get("evidence", ""),
                        "rationale": block.get("rationale", ""),
                        "answer": block.get("answer", ""),
                    }
                )

    rows.sort(key=_row_sort_key)

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
        default=pathlib.Path("./csv/llama-3.1-70B RAG.csv"),
        help="Path to the input CSV file (default: ./csv/llama-3.1-70B RAG.csv)",
    )
    parser.add_argument(
        "output",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path("./csv/llama-3.1-70B RAG_parsed.csv"),
        help="Path to the output CSV file (default: ./csv/llama-3.1-70B RAG_parsed.csv)",
    )
    parser.add_argument(
        "--s2-table",
        type=pathlib.Path,
        default=pathlib.Path("./csv/S2Table.xlsx"),
        help="Path to the S2Table.xlsx file",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    parse_file(args.input, args.output, args.s2_table)


if __name__ == "__main__":
    main()

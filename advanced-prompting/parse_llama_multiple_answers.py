#!/usr/bin/env python3
"""Parse the Llama PV1 multiple-answer column into question-level rows."""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
from typing import Dict, List, Tuple

import pandas as pd

QUESTION_FIELDS = ("question", "evidence", "rationale", "answer")
QUESTION_SYNONYMS = {
    "what are the genbank accession numbers for sequenced hiv isolates": "what were the genbank accession numbers for sequenced hiv isolates",
}
DELIMITER_LINES = {
    '"""',
    '""',
    "'''",
    '```',
}

# Captures section labels like "Evidence:" and "Rationale -".
SECTION_LINE_RE = re.compile(
    r"""
    ^
    \s*
    (?P<label>Question|Evidence|Rationale|Answer)
    (?:\s*(?:[:\-\u2013]\s*|\s+))?
    (?P<value>.*?)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

QuestionBlock = Dict[str, str]


def clean_text(value: object) -> str:
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
    return line.strip() in DELIMITER_LINES


def detect_question_header(line: str, allow_plain_without_id: bool = False) -> Tuple[str, str] | None:
    trimmed = line.strip()
    if not trimmed:
        return None

    # Remove simple Markdown bullets and emphasis markers.
    trimmed = trimmed.lstrip("#>- ")
    emphasised = trimmed.startswith("**") and trimmed.endswith("**") and len(trimmed) >= 4
    if emphasised:
        trimmed = trimmed[2:-2].strip()
    trimmed = trimmed.lstrip("*")

    lowered = trimmed.lower()
    if not lowered.startswith("question"):
        return None

    remainder = trimmed[len("question") :].lstrip(" .:-\u2013")
    while remainder.lower().startswith("question"):
        remainder = remainder[len("question") :].lstrip(" .:-\u2013")
    if remainder.lower().startswith("id"):
        remainder = remainder[2:].lstrip(" .:-\u2013")

    digits_match = re.match(r"(?P<digits>\d+)(?P<rest>.*)", remainder)
    qid = ""
    rest = remainder
    if digits_match:
        qid = digits_match.group("digits")
        rest = digits_match.group("rest")

    rest = rest.lstrip(" .:-\u2013")
    question_text = clean_text(rest)
    qid = normalise_question_id(qid)

    if not qid and not (emphasised or allow_plain_without_id):
        return None

    if not qid and not question_text:
        return None
    return qid, question_text


def parse_section_body(body: str) -> Dict[str, str]:
    collected: Dict[str, List[str]] = {label: [] for label in QUESTION_FIELDS}
    current_label: str | None = None

    for raw_line in body.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if current_label:
                collected[current_label].append("")
            continue
        match = SECTION_LINE_RE.match(stripped)
        if match:
            current_label = match.group("label").lower()
            value = match.group("value") or ""
            value = value.strip()
            if value:
                collected[current_label].append(value)
            continue
        if current_label:
            collected[current_label].append(raw_line.rstrip())

    return {label: clean_text("\n".join(lines).strip()) for label, lines in collected.items()}


def split_question_sections(text: str) -> List[Tuple[str, str, str]]:
    sections: List[Tuple[str, str, str]] = []
    current_header: Tuple[str, str] | None = None
    body_lines: List[str] = []

    def flush() -> None:
        nonlocal current_header, body_lines
        if current_header is None and not body_lines:
            return
        qid, question = current_header if current_header else ("", "")
        body = "\n".join(body_lines).strip()
        if question or body:
            sections.append((qid, question, body))
        current_header = None
        body_lines = []

    for raw_line in text.splitlines():
        if _is_delimiter(raw_line):
            if current_header is None and not body_lines:
                continue
            flush()
            continue

        allow_plain = current_header is None and not body_lines
        detection = detect_question_header(raw_line, allow_plain_without_id=allow_plain)
        if detection:
            if current_header is None and body_lines:
                body_lines = []  # drop intro text preceding the first question
            else:
                flush()
            current_header = detection
            continue

        body_lines.append(raw_line.rstrip())

    flush()

    if not sections and text.strip():
        sections.append(("", "", text.strip()))

    return sections


def parse_block(header_qid: str, header_question: str, body: str) -> QuestionBlock | None:
    sections = parse_section_body(body)
    question_text = sections.get("question") or header_question

    if not (question_text or sections.get("evidence") or sections.get("rationale") or sections.get("answer")):
        return None

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
    sections = split_question_sections(text)
    for header_qid, header_question, body in sections:
        block = parse_block(header_qid, header_question, body)
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

    for alias_text, canonical_text in QUESTION_SYNONYMS.items():
        alias_norm = normalise_question_text(alias_text)
        canonical_norm = normalise_question_text(canonical_text)
        if not alias_norm or not canonical_norm:
            continue
        if canonical_norm in question_lookup:
            question_lookup.setdefault(alias_norm, question_lookup[canonical_norm])

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


def _extract_answer_cell(row: Dict[str, str]) -> str:
    for key in ("Multiple Answers", "Multiple Answer"):
        if key in row:
            return row.get(key, "")
    raise KeyError("Input CSV is missing a 'Multiple Answers' column")


def parse_file(
    input_path: pathlib.Path, output_path: pathlib.Path, s2_table_path: pathlib.Path
) -> None:
    rows: List[Dict[str, str]] = []
    question_lookup, qid_lookup = load_question_lookup(s2_table_path)

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pmid = clean_text(row.get("PMID", ""))
            multiple_answer = _extract_answer_cell(row)
            parsed_blocks = parse_multiple_answer(multiple_answer)
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
        default=pathlib.Path("./csv/llama-3.1-70B-PV1.csv"),
        help="Path to the input CSV file (default: ./csv/llama-3.1-70B-PV1.csv)",
    )
    parser.add_argument(
        "output",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path("./csv/llama-3.1-70B-PV1_parsed.csv"),
        help="Path to the output CSV file (default: ./csv/llama-3.1-70B-PV1_parsed.csv)",
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

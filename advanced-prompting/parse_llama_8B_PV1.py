#!/usr/bin/env python3
"""Parse PV1 multiple answers in csv/llama-3.1-8B-PV1.csv into per-question rows.

The script reads the `Multiple Answer` column, extracts per-question answers,
maps questions/QIDs to the canonical wording from Table S4.xlsx, and writes a
CSV with columns: PMID, QID, Question, Answer. Every PMID in the input gets one
row per expected QID (1-16); missing answers are left blank.
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Dict, Iterable, List, Tuple

import pandas as pd

DEFAULT_INPUT = pathlib.Path("./csv/llama-3.1-8B-PV1.csv")
DEFAULT_S4 = pathlib.Path("./csv/S4Table.xlsx")

QUESTION_SYNONYMS = {
    "what are the genbank accession numbers for sequenced hiv isolates": "what were the genbank accession numbers for sequenced hiv isolates",
}
DELIMITER_LINES = {'"""', "'''", "```", '""'}

# Captures section labels like "Evidence:" and "Answer -".
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


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def normalise_cell(cell: object) -> str:
    text = clean_text(cell)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\n", "\n")
    return text


def strip_question_label(text: str) -> str:
    """Remove leading 'Question', IDs, and separators from a question line."""
    cleaned = clean_text(text).strip("*_`> \t")
    cleaned = re.sub(r"(?i)^question\s*[:\-\u2013]?\s*", "", cleaned)
    cleaned = re.sub(r"(?i)^id\s*[:\-\u2013]?\s*", "", cleaned)
    cleaned = re.sub(r"^\s*\d+\s*[).:\-\u2013]?\s*", "", cleaned)
    cleaned = re.sub(r"(?i)^question\s*", "", cleaned)  # handles "Question: Question 1"
    return cleaned.strip(" .:-\u2013")


def normalise_question_text(question: str) -> str:
    cleaned = strip_question_label(question)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned.rstrip(" ?.")


def normalise_question_id(value: object) -> str:
    if value is None:
        return ""
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else ""


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

    match = re.match(r"(?i)^(?:question|q)\s*(?P<rest>.*)", trimmed)
    if not match:
        return None

    remainder = match.group("rest").lstrip(" .:-\u2013")
    while True:
        lowered = remainder.lower()
        if lowered.startswith("question"):
            remainder = remainder[len("question") :].lstrip(" .:-\u2013")
            continue
        if lowered.startswith("qid"):
            remainder = remainder[3:].lstrip(" .:-\u2013")
            continue
        if lowered.startswith("id"):
            remainder = remainder[2:].lstrip(" .:-\u2013")
            continue
        if lowered.startswith("q"):
            remainder = remainder[1:].lstrip(" .:-\u2013")
            continue
        break

    digits_match = re.match(r"(?P<digits>\d+)(?P<rest>.*)", remainder)
    qid = ""
    rest = remainder
    if digits_match:
        qid = digits_match.group("digits")
        rest = digits_match.group("rest")

    rest = rest.lstrip(" .:-\u2013")
    question_text = strip_question_label(rest)
    qid = normalise_question_id(qid)

    if not qid and not (emphasised or allow_plain_without_id):
        return None

    if not qid and not question_text:
        return None
    return qid, question_text


def parse_section_body(body: str) -> Dict[str, str]:
    collected: Dict[str, List[str]] = {label: [] for label in ("question", "evidence", "rationale", "answer")}
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


def parse_multiple_answer(cell: object) -> List[Dict[str, str]]:
    text = normalise_cell(cell)
    if not text:
        return []

    parsed: List[Dict[str, str]] = []
    sections = split_question_sections(text)
    for header_qid, header_question, body in sections:
        parts = parse_section_body(body)
        question_text = parts.get("question") or header_question
        answer_text = parts.get("answer", "")
        if not (question_text or answer_text):
            continue
        parsed.append(
            {
                "question_id": header_qid,
                "question": question_text or "",
                "answer": answer_text or "",
            }
        )
    return parsed


def load_s4_questions(s4_path: pathlib.Path) -> tuple[Dict[str, str], Dict[str, str], List[str]]:
    dataframe = pd.read_excel(s4_path, dtype={"QID": str})
    qid_to_question: Dict[str, str] = {}
    question_lookup: Dict[str, str] = {}

    for _, row in dataframe.iterrows():
        qid = normalise_question_id(row.get("QID"))
        question = clean_text(row.get("Question"))
        if not qid or not question:
            continue
        qid_to_question[qid] = question
        norm = normalise_question_text(question)
        if norm and norm not in question_lookup:
            question_lookup[norm] = qid

    for alias, canonical in QUESTION_SYNONYMS.items():
        alias_norm = normalise_question_text(alias)
        canonical_norm = normalise_question_text(canonical)
        if alias_norm and canonical_norm and canonical_norm in question_lookup:
            question_lookup.setdefault(alias_norm, question_lookup[canonical_norm])

    expected_qids = sorted(qid_to_question.keys(), key=lambda value: int(value))
    return qid_to_question, question_lookup, expected_qids


def match_question(
    question_lookup: Dict[str, str],
    qid_to_question: Dict[str, str],
    entry: Dict[str, str],
) -> tuple[str, str]:
    qid = normalise_question_id(entry.get("question_id"))
    question_text = strip_question_label(entry.get("question", ""))

    if question_text:
        norm = normalise_question_text(question_text)
        mapped = question_lookup.get(norm)
        if mapped:
            qid = mapped

    if qid and qid in qid_to_question:
        return qid, qid_to_question[qid]

    return qid, question_text


def parse_file(input_path: pathlib.Path, output_path: pathlib.Path, s4_path: pathlib.Path) -> None:
    qid_to_question, question_lookup, expected_qids = load_s4_questions(s4_path)
    dataframe = pd.read_csv(input_path)

    pmid_order: List[str] = []
    answers: Dict[str, Dict[str, Dict[str, str]]] = {}

    def ensure_pmid_order(pmid_value: str) -> None:
        if pmid_value not in pmid_order:
            pmid_order.append(pmid_value)

    for _, row in dataframe.iterrows():
        pmid = clean_text(row.get("PMID"))
        if not pmid:
            continue
        ensure_pmid_order(pmid)

        multiple_answer = row.get("Multiple Answer") or row.get("Multiple Answers")
        parsed_blocks = parse_multiple_answer(multiple_answer)
        for block in parsed_blocks:
            qid, canonical_question = match_question(question_lookup, qid_to_question, block)
            if not qid:
                continue

            pmid_answers = answers.setdefault(pmid, {})
            if qid in pmid_answers:
                continue  # keep the first occurrence
            pmid_answers[qid] = {
                "Question": canonical_question or qid_to_question.get(qid, ""),
                "Answer": clean_text(block.get("answer", "")),
            }

    rows = []
    for pmid in pmid_order:
        pmid_answers = answers.get(pmid, {})
        for qid in expected_qids:
            data = pmid_answers.get(qid, {})
            rows.append(
                {
                    "PMID": pmid,
                    "QID": qid,
                    "Question": qid_to_question.get(qid, data.get("Question", "")),
                    "Answer": data.get("Answer", ""),
                }
            )

    out_df = pd.DataFrame(rows, columns=["PMID", "QID", "Question", "Answer"]).fillna("")
    out_df.to_csv(output_path, index=False)

    counts = out_df.groupby("PMID").size().sort_index()
    for pmid, count in counts.items():
        print(f"{pmid}: {count} rows")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=DEFAULT_INPUT,
        help="Input CSV path (default: ./csv/llama-3.1-8B-PV1.csv)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Output CSV path (default: add _parsed before extension)",
    )
    parser.add_argument(
        "--s4",
        type=pathlib.Path,
        default=DEFAULT_S4,
        help="Path to Table S4.xlsx (default: ./csv/S4Table.xlsx)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path: pathlib.Path = args.input
    output_path = args.output or input_path.with_name(f"{input_path.stem}_parsed{input_path.suffix}")
    print(f"Parsing {input_path} -> {output_path}")
    parse_file(input_path, output_path, args.s4)


if __name__ == "__main__":
    main()

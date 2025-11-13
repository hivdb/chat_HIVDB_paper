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
from typing import Dict, List, Sequence, Tuple

import pandas as pd


QUESTION_LABELS = ("question", "evidence", "rationale", "answer")
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
QUESTION_ID_RE = re.compile(
    r"(?i)^(?:question|q)(?:id)?\s*(?:[:\-]?\s*)?(?:q\s*)?(?P<digits>\d+)",
)
QUESTION_PREFIX_RE = re.compile(
    r"""
    ^
    (?:question|q)(?:id)?      # core label
    (?=\s|[:\-]|$)            # next char must be space/punct/EOL
    \s*(?:[:\-]?\s*)?        # optional punctuation
    (?:q\s*)?                  # allow repeated Q
    (?:\d+[\.):]?)?          # optional numeric indicator
    \s*
    """,
    re.IGNORECASE | re.VERBOSE,
)
TRIPLE_QUOTE_SPLIT_RE = re.compile(r'"{3,}')


class QuestionBlock(Dict[str, str]):
    """TypedDict-like alias for parsed question blocks."""


def normalise_question_id(value: object) -> str:
    """Return only the numeric portion of a Question/QID identifier."""
    if value is None:
        return ""
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else ""


def clean_text(value: str) -> str:
    """Normalise whitespace and escape sequences within parsed text."""
    if value is None:
        return ""
    return value.replace('""', '"').strip()


def normalise_question_text(value: str) -> str:
    if not value:
        return ""
    text = re.sub(r"\s+", " ", value).strip().lower()
    return text.rstrip(" ?.")


def normalise_cell(cell: str) -> str:
    if not cell:
        return ""
    text = cell.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\n", "\n")
    return text.strip()


def strip_question_prefix(value: str) -> str:
    if not value:
        return ""
    text = value.strip()
    match = QUESTION_PREFIX_RE.match(text)
    if match and match.end() < len(text):
        return text[match.end() :].strip()
    if match:
        return ""
    return text


def is_question_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    starts_with_hash = stripped.startswith("#")
    starts_with_star = stripped.startswith("*")
    if not (starts_with_hash or starts_with_star):
        return False
    core = stripped.lstrip("#* ").strip()
    if not core:
        return False
    if QUESTION_ID_RE.match(core):
        return True
    if starts_with_hash:
        prefix = QUESTION_PREFIX_RE.match(core)
        return bool(prefix and prefix.end() < len(core))
    return False


def extract_blocks(text: str) -> List[Tuple[str, str]]:
    blocks: List[Tuple[str, str]] = []
    header: str | None = None
    body_lines: List[str] = []

    for line in text.splitlines():
        if is_question_header(line):
            if header is not None:
                blocks.append((header, "\n".join(body_lines).strip()))
            header = line.strip()
            body_lines = []
        elif header is not None:
            body_lines.append(line.rstrip())
    if header is not None:
        blocks.append((header, "\n".join(body_lines).strip()))
    return blocks


def fallback_blocks(text: str) -> List[Tuple[str, str]]:
    blocks: List[Tuple[str, str]] = []
    header: str | None = None
    body_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        normalized = stripped.lstrip("* ")
        if QUESTION_PREFIX_RE.match(normalized):
            if header is not None:
                blocks.append((header, "\n".join(body_lines).strip()))
            header = stripped
            body_lines = []
        elif header is not None:
            body_lines.append(line.rstrip())
    if header is not None:
        blocks.append((header, "\n".join(body_lines).strip()))
    return blocks


def parse_triple_quoted_segments(text: str) -> List[QuestionBlock]:
    results: List[QuestionBlock] = []
    for segment in TRIPLE_QUOTE_SPLIT_RE.split(text):
        segment = segment.strip()
        if not segment:
            continue
        if not segment.lower().startswith("question"):
            continue
        sections = parse_sections(segment)
        section_qid, section_question = parse_question_section(sections.get("question", ""))
        question_text = strip_question_prefix(section_question or sections.get("question", ""))
        results.append(
            {
                "question_id": section_qid,
                "question": clean_text(question_text),
                "evidence": sections.get("evidence", ""),
                "rationale": sections.get("rationale", ""),
                "answer": sections.get("answer", ""),
            }
        )
    return results


def parse_question_header(header_line: str) -> Tuple[str, str]:
    cleaned = header_line.strip().strip("*")
    cleaned = cleaned.lstrip("#").strip()
    match = QUESTION_ID_RE.match(cleaned)
    question_id = normalise_question_id(match.group("digits")) if match else ""
    question_text = cleaned[match.end():].lstrip(" -:\t").strip() if match else cleaned
    question_text = strip_question_prefix(question_text)
    return question_id, clean_text(question_text)


def detect_field_label(line: str) -> Tuple[str, str] | None:
    match = FIELD_LINE_RE.match(line.strip())
    if not match:
        return None
    label = match.group("label").lower()
    value = match.group("value") or ""
    return label, value.strip()


def join_section(lines: Sequence[str]) -> str:
    text = "\n".join(part.rstrip() for part in lines if part is not None)
    return clean_text(text)


def parse_sections(body: str) -> Dict[str, str]:
    collected: Dict[str, List[str]] = {label: [] for label in QUESTION_LABELS}
    current_label: str | None = None

    for raw_line in body.splitlines():
        detection = detect_field_label(raw_line)
        if detection:
            label, initial_value = detection
            current_label = label
            if initial_value:
                collected[label].append(initial_value)
            continue
        if current_label:
            collected[current_label].append(raw_line)

    return {label: join_section(lines) for label, lines in collected.items()}


def parse_question_section(value: str) -> Tuple[str, str]:
    cleaned = clean_text(value)
    if not cleaned:
        return "", ""
    if re.fullmatch(r"(?i)q(?:uestion)?\s*\d+", cleaned):
        return normalise_question_id(cleaned), ""
    match = re.search(r"(?i)q(?:uestion)?\s*(\d+)", cleaned)
    question_id = normalise_question_id(match.group(1)) if match else ""
    return question_id, strip_question_prefix(cleaned)


def parse_multiple_answer(cell: str) -> List[QuestionBlock]:
    text = normalise_cell(cell)
    if not text:
        return []

    blocks = extract_blocks(text)
    if not blocks:
        blocks = fallback_blocks(text)
    if blocks:
        parsed: List[QuestionBlock] = []
        for header, body in blocks:
            header_qid, header_question = parse_question_header(header)
            sections = parse_sections(body)
            section_qid, section_question = parse_question_section(sections.get("question", ""))
            question_id = header_qid or section_qid
            question_text = header_question or section_question or sections.get("question", "")
            question_text = strip_question_prefix(question_text)
            parsed.append(
                {
                    "question_id": question_id,
                    "question": clean_text(question_text),
                    "evidence": sections.get("evidence", ""),
                    "rationale": sections.get("rationale", ""),
                    "answer": sections.get("answer", ""),
                }
            )
        return parsed

    return parse_triple_quoted_segments(text)


def load_qid_lookup(
    s2_table_path: pathlib.Path,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """Build lookups keyed by question text and by question ID."""
    if not s2_table_path.exists():
        raise FileNotFoundError(f"Missing S2 table: {s2_table_path}")

    dataframe = pd.read_excel(s2_table_path)
    question_lookup: Dict[str, Dict[str, str]] = {}
    qid_lookup: Dict[str, Dict[str, str]] = {}

    for _, row in dataframe.iterrows():
        if pd.isna(row.get("Question")) or pd.isna(row.get("QID")):
            continue

        question = clean_text(str(row["Question"]))
        qid = normalise_question_id(row.get("QID"))
        if not question or not qid:
            continue

        entry = {"QID": qid, "question": question}
        qid_lookup.setdefault(qid, entry)
        norm_text = normalise_question_text(question)
        question_lookup.setdefault(norm_text, entry)

    return question_lookup, qid_lookup


def match_question(
    question_lookup: Dict[str, Dict[str, str]],
    qid_lookup: Dict[str, Dict[str, str]],
    question: QuestionBlock,
) -> Tuple[str, str]:
    question_id = normalise_question_id(question.get("question_id"))
    if question_id and question_id in qid_lookup:
        entry = qid_lookup[question_id]
        return entry["QID"], entry["question"]

    question_text = strip_question_prefix(question.get("question", ""))
    norm_question = normalise_question_text(question_text)
    if norm_question and norm_question in question_lookup:
        entry = question_lookup[norm_question]
        return entry["QID"], entry["question"]

    return "", question_text or question.get("question", "")


def parse_file(
    input_path: pathlib.Path, output_path: pathlib.Path, s2_table_path: pathlib.Path
) -> None:
    rows: List[Dict[str, str]] = []
    question_lookup, qid_lookup = load_qid_lookup(s2_table_path)

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pmid = clean_text(row.get("PMID", ""))
            matched_pairs: set[Tuple[str, str]] = set()

            parsed_questions = parse_multiple_answer(row.get("Multiple Answer", ""))
            print(f"PMID {pmid or '[unknown]'}: parsed {len(parsed_questions)} question(s)")
            for question in parsed_questions:
                qid, canonical_question = match_question(question_lookup, qid_lookup, question)
                if not qid:
                    print(
                        "PMID {pmid}: could not find QID for question {question!r} (ID: {question_id})".format(
                            pmid=pmid or "[unknown]",
                            question=question.get("question", ""),
                            question_id=question.get("question_id", "") or "unknown",
                        )
                    )
                    continue

                key = (qid, pmid)
                if key in matched_pairs:
                    continue
                matched_pairs.add(key)
                rows.append(
                    {
                        "PMID": pmid,
                        "QID": qid,
                        "question": canonical_question or question.get("question", ""),
                        "evidence": question.get("evidence", ""),
                        "rationale": question.get("rationale", ""),
                        "answer": question.get("answer", ""),
                    }
                )

            for qid, entry in qid_lookup.items():
                key = (qid, pmid)
                if key in matched_pairs:
                    continue
                rows.append(
                    {
                        "PMID": pmid,
                        "QID": entry["QID"],
                        "question": entry["question"],
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
        default=pathlib.Path("./csv/llama-3.1-70B_bm25_10-shot.csv"),
        help="Path to the input CSV file (default: llama-3.1-70B_bm25_10-shot.csv)",
    )
    parser.add_argument(
        "output",
        type=pathlib.Path,
        nargs="?",
        default=pathlib.Path("./csv/llama-3.1-70B_bm25_10-shot_parsed.csv"),
        help="Path to the output CSV file (default: llama-3.1-70B_bm25_10-shot_parsed.csv)",
    )
    parser.add_argument(
        "--s2-table",
        type=pathlib.Path,
        default=pathlib.Path("./csv/S2Table.xlsx"),
        help="Path to the S2Table.xlsx file that contains PMID/QID/question mappings",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    parse_file(args.input, args.output, args.s2_table)


if __name__ == "__main__":
    main()

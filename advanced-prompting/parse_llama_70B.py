#!/usr/bin/env python3
"""Parse the FT answers in one or more Llama 3.1 70B FT CSVs into per-question rows."""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Dict, List, Tuple

import pandas as pd

DELIMITER_LINES = {'"""', "'''", "```", '""'}
SECTION_RE = re.compile(r"^(?P<label>Evidence|Rationale|Answer)\s*[:\-\u2013]?\s*(?P<rest>.*)", re.IGNORECASE)
QUESTION_RE = re.compile(r"^Question\s*[:\-\u2013]?\s*(?P<qid>\d+)\s*(?P<question>.*)", re.IGNORECASE)
QUESTION_NO_ID_RE = re.compile(r"^Question\s*[:\-\u2013]?\s*(?P<question>.+)", re.IGNORECASE)
DEFAULT_INPUTS = [
    pathlib.Path("./csv/llama-3.1-70B-FT 50.csv"),
    pathlib.Path("./csv/llama-3.1-70B-FT 100.csv"),
    pathlib.Path("./csv/llama-3.1-70B-FT 150.csv"),
    pathlib.Path("./csv/llama-3.1-70B-FT 200.csv"),
    pathlib.Path("./csv/llama-3.1-70B-base_new30.csv"),
    pathlib.Path("./csv/llama-3.1-70B-FT 50_new30.csv"),
    pathlib.Path("./csv/llama-3.1-70B-FT 100_new30.csv"),
    pathlib.Path("./csv/llama-3.1-70B-FT 150_new30.csv"),
    pathlib.Path("./csv/llama-3.1-70B-FT 200_new30.csv"),
    pathlib.Path("./csv/llama-3.1-70B-FT_new30.csv"),
]

QID_MAP = {
    '1': '1',
    '2': '2',
    '3': '4',
    '5': '9',
    '6': '6',
    '7': '7',
    '8': '10',
    '9': '8',
    '11': '11',
    '12': '12',
    '14': '13',
    '15': '5',
    '16': '14',
    '17': '15',
    '18': '16',
    '19': '3',
}
EXPECTED_QIDS = sorted(set(QID_MAP.values()), key=lambda q: int(q))


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def normalise_cell(cell: object) -> str:
    text = clean_text(cell)
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")


def normalise_question_text(question: str) -> str:
    cleaned = clean_text(question)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned.rstrip("?:.")


def parse_ft_answer(text: str) -> List[Dict[str, str]]:
    """Return a list of extracted sections, including question text and optional qid."""
    parsed: List[Dict[str, List[str] | str]] = []
    current: Dict[str, List[str] | str] | None = None
    current_section: str | None = None

    def start_entry(qid: str | None, question_text: str) -> None:
        nonlocal current, current_section
        if current:
            parsed.append(current)
        current = {
            "qid": str(qid).strip() if qid else "",
            "question": question_text.strip(),
            "evidence": [],
            "rationale": [],
            "answer": [],
        }
        current_section = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        cleaned_line = line.strip("*_ ").lstrip("> \t*-")
        if not cleaned_line:
            if current_section and current:
                current[current_section].append("")
            continue
        if cleaned_line in DELIMITER_LINES:
            continue

        question_match = QUESTION_RE.match(cleaned_line)
        if question_match:
            inline_question = question_match.group("question").lstrip(".\u2013- \t")
            start_entry(question_match.group("qid"), inline_question)
            continue

        question_no_id_match = QUESTION_NO_ID_RE.match(cleaned_line)
        if question_no_id_match:
            inline_question = question_no_id_match.group("question").lstrip(".\u2013- \t")
            start_entry(None, inline_question)
            continue

        section_match = SECTION_RE.match(cleaned_line)
        if section_match:
            current_section = section_match.group("label").lower()
            rest = section_match.group("rest").strip()
            if rest and current:
                current[current_section].append(rest)
            continue

        if current_section and current:
            current[current_section].append(raw_line.rstrip())

    if current:
        parsed.append(current)

    normalised: List[Dict[str, str]] = []
    for sections in parsed:
        normalised.append(
            {
                "qid": str(sections.get("qid", "")).strip(),
                "question": str(sections.get("question", "")).strip(),
                "evidence": "\n".join(sections.get("evidence", [])).strip(),
                "rationale": "\n".join(sections.get("rationale", [])).strip(),
                "answer": "\n".join(sections.get("answer", [])).strip(),
            }
        )

    return normalised


def build_question_lookup(qid_questions: Dict[str, str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for qid, question_text in qid_questions.items():
        key = normalise_question_text(question_text)
        if key and key not in lookup:
            lookup[key] = qid
    return lookup


def load_qid_questions(table_path: pathlib.Path) -> Dict[str, str]:
    """Map QID -> Question text from S4 table."""
    df = pd.read_excel(table_path)
    qid_to_question: Dict[str, str] = {}
    for _, row in df.iterrows():
        qid = clean_text(row.get("QID"))
        question = clean_text(row.get("Question"))
        if not qid or not question:
            continue
        qid_to_question.setdefault(str(qid), question)
    return qid_to_question


def parse_file(input_path: pathlib.Path, output_path: pathlib.Path, s4_table_path: pathlib.Path) -> None:
    qid_questions = load_qid_questions(s4_table_path)
    question_lookup = build_question_lookup(qid_questions)
    expected_qids = sorted(set(QID_MAP.values()) | set(qid_questions.keys()), key=lambda q: int(q))

    df = pd.read_csv(input_path)
    pmid_order: List[str] = []
    pmid_to_qid_rows: Dict[str, Dict[str, Dict[str, str]]] = {}

    for _, row in df.iterrows():
        pmid = clean_text(row.get("PMID"))
        if not pmid:
            continue
        if pmid not in pmid_order:
            pmid_order.append(pmid)
        answer_cell = row.get("FT Answer")
        if clean_text(answer_cell) == "":
            answer_cell = row.get("Multiple Answer")
        ft_answer_text = normalise_cell(answer_cell)
        parsed_entries = parse_ft_answer(ft_answer_text) if ft_answer_text else []

        for sections in parsed_entries:
            raw_qid = sections.get("qid", "")
            mapped_qid = QID_MAP.get(raw_qid) if raw_qid else None
            if not mapped_qid and sections.get("question"):
                mapped_qid = question_lookup.get(normalise_question_text(sections.get("question", "")))
            if not mapped_qid:
                continue
            question_text = qid_questions.get(mapped_qid, "") or sections.get("question", "")
            pmid_to_qid_rows.setdefault(pmid, {})[mapped_qid] = {
                "question": question_text,
                "evidence": sections.get("evidence", ""),
                "rationale": sections.get("rationale", ""),
                "answer": sections.get("answer", ""),
            }

    rows: List[Dict[str, str]] = []
    for pmid in pmid_order:
        qid_rows = pmid_to_qid_rows.get(pmid, {})
        for qid in expected_qids:
            parsed_entry = qid_rows.get(qid, {})
            question_text = qid_questions.get(qid, "") or parsed_entry.get("question", "")
            rows.append(
                {
                    "PMID": pmid,
                    "QID": qid,
                    "Question": question_text,
                    # "Evidence": parsed_entry.get("evidence", ""),
                    # "Rationale": parsed_entry.get("rationale", ""),
                    "Answer": parsed_entry.get("answer", ""),
                }
            )
    out_df = pd.DataFrame(rows, columns=[
        "PMID", "QID", "Question",
        # "Evidence", "Rationale",
        "Answer"]).fillna("")
    out_df.to_csv(output_path, index=False)
    counts = out_df.groupby("PMID").size().sort_index()
    for pmid, count in counts.items():
        print(f"{pmid}: {count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=pathlib.Path,
        nargs="*",
        default=DEFAULT_INPUTS,
        help="Input CSV files to parse (default: the Llama 70B FT/base CSVs in ./csv/)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Output CSV path when parsing a single input; defaults to <input> with _parsed suffix",
    )
    parser.add_argument(
        "--s4-table",
        type=pathlib.Path,
        default=pathlib.Path("./csv/S4Table.xlsx"),
        help="Path to the S4Table.xlsx file",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    inputs: List[pathlib.Path] = list(args.inputs) if args.inputs else DEFAULT_INPUTS

    if args.output and len(inputs) != 1:
        parser.error("--output can only be used when parsing a single input file")

    for input_path in inputs:
        output_path = args.output or input_path.with_name(f"{input_path.stem}_parsed{input_path.suffix}")
        print(f"Parsing {input_path} -> {output_path}")
        parse_file(input_path, output_path, args.s4_table)


if __name__ == "__main__":
    main()

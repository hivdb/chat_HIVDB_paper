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
DEFAULT_INPUTS = [
    pathlib.Path("./csv/llama-3.1-70B-FT 50.csv"),
    pathlib.Path("./csv/llama-3.1-70B-FT 100.csv"),
    pathlib.Path("./csv/llama-3.1-70B-FT 150.csv"),
    pathlib.Path("./csv/llama-3.1-70B-FT 200.csv"),
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


def parse_ft_answer(text: str) -> Dict[str, Dict[str, str]]:
    """Return a mapping of qid -> extracted sections (and inline question text, if any)."""
    parsed: Dict[str, Dict[str, List[str] | str]] = {}
    current_qid: str | None = None
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_section:
                parsed[current_qid][current_section].append("") if current_qid else None
            continue
        if line in DELIMITER_LINES:
            continue

        question_match = QUESTION_RE.match(line)
        if question_match:
            current_qid = question_match.group("qid")
            inline_question = question_match.group("question").lstrip(".\u2013- \t")
            parsed.setdefault(current_qid, {"question": "", "evidence": [], "rationale": [], "answer": []})
            if inline_question:
                parsed[current_qid]["question"] = inline_question
            current_section = None
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            current_section = section_match.group("label").lower()
            rest = section_match.group("rest").strip()
            if rest:
                if current_qid:
                    parsed[current_qid][current_section].append(rest)
            continue

        if current_section and current_qid:
            parsed[current_qid][current_section].append(raw_line.rstrip())

    normalised: Dict[str, Dict[str, str]] = {}
    for qid, sections in parsed.items():
        normalised[qid] = {
            "question": str(sections.get("question", "")).strip(),
            "evidence": "\n".join(sections.get("evidence", [])).strip(),
            "rationale": "\n".join(sections.get("rationale", [])).strip(),
            "answer": "\n".join(sections.get("answer", [])).strip(),
        }

    return normalised


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

    df = pd.read_csv(input_path)
    pmid_order: List[str] = []
    pmid_to_qid_rows: Dict[str, Dict[str, Dict[str, str]]] = {}

    for _, row in df.iterrows():
        pmid = clean_text(row.get("PMID"))
        if not pmid:
            continue
        if pmid not in pmid_order:
            pmid_order.append(pmid)
        ft_answer_text = normalise_cell(row.get("FT Answer"))
        parsed = parse_ft_answer(ft_answer_text) if ft_answer_text else {}

        for qid, sections in parsed.items():
            mapped_qid = QID_MAP.get(qid)
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
        for qid in EXPECTED_QIDS:
            parsed_entry = qid_rows.get(qid, {})
            question_text = qid_questions.get(qid, "") or parsed_entry.get("question", "")
            rows.append(
                {
                    "PMID": pmid,
                    "QID": qid,
                    "Question": question_text,
                    "Evidence": parsed_entry.get("evidence", ""),
                    "Rationale": parsed_entry.get("rationale", ""),
                    "Answer": parsed_entry.get("answer", ""),
                }
            )
    out_df = pd.DataFrame(rows, columns=["PMID", "QID", "Question", "Evidence", "Rationale", "Answer"]).fillna("")
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
        help="Input CSV files to parse (default: the four FT CSVs in ./csv/)",
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

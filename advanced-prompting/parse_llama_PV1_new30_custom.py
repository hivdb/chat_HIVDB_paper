#!/usr/bin/env python3
"""Parse PV1 multiple answers in csv/llama-3.1-70B-PV1_new30.csv into per-question rows.

Output columns: PMID, QID, Question, Answer.
The script extracts questions (or QIDs), remaps QIDs per the provided map,
looks up the canonical Question/QID pairing from Table S4 (csv/S4Table.xlsx),
and ensures 16 rows (one per expected QID) are present for each PMID.
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Dict, List

import pandas as pd

DEFAULT_INPUT = pathlib.Path("./csv/llama-3.1-70B-PV1_new30.csv")
DEFAULT_S4 = pathlib.Path("./csv/S4Table.xlsx")  # aka Table S4.xlsx

QID_MAP = {
    "1": "1",
    "2": "2",
    "3": "4",
    "5": "9",
    "6": "6",
    "7": "7",
    "8": "10",
    "9": "8",
    "11": "11",
    "12": "12",
    "14": "13",
    "15": "5",
    "16": "14",
    "17": "15",
    "18": "16",
    "19": "3",
}
EXPECTED_QIDS = sorted(set(QID_MAP.values()), key=lambda q: int(q))

DELIMITER_LINES = {'"""', "'''", "```", '""'}
QUESTION_WITH_ID_RE = re.compile(
    r"^Question\s*[:\-\u2013]?\s*(?P<qid>\d+)\s*[).:\-\u2013]?\s*(?P<question>.*)",
    re.IGNORECASE,
)
QUESTION_TEXT_RE = re.compile(r"^Question\s*[:\-\u2013]?\s*(?P<question>.+)", re.IGNORECASE)
ANSWER_RE = re.compile(r"^Answer\s*[:\-\u2013]?\s*(?P<answer>.*)", re.IGNORECASE)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def normalise_question_text(question: str) -> str:
    cleaned = clean_text(question)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned.rstrip("?:.")


def load_s4_questions(s4_path: pathlib.Path) -> tuple[Dict[str, str], Dict[str, str]]:
    """Return (qid -> question, normalized question -> qid)."""
    df = pd.read_excel(s4_path)
    qid_to_question: Dict[str, str] = {}
    question_lookup: Dict[str, str] = {}
    for _, row in df.iterrows():
        qid = clean_text(row.get("QID"))
        question = clean_text(row.get("Question"))
        if not qid or not question:
            continue
        qid_to_question[str(qid)] = question
        key = normalise_question_text(question)
        if key and key not in question_lookup:
            question_lookup[key] = str(qid)
    return qid_to_question, question_lookup


def parse_multiple_answer(text: str) -> List[Dict[str, str]]:
    """Extract a list of question/answer entries from a raw multiple-answer blob."""
    parsed: List[Dict[str, List[str] | str]] = []
    current: Dict[str, List[str] | str] | None = None
    in_answer = False

    def start_entry(qid: str | None, question_text: str) -> None:
        nonlocal current, in_answer
        if current:
            parsed.append(current)
        current = {
            "qid": str(qid).strip() if qid else "",
            "question": question_text.strip(),
            "answer_lines": [],
        }
        in_answer = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        cleaned = line.strip("*_ ").lstrip("> \t*-")
        if not cleaned:
            if in_answer and current:
                current["answer_lines"].append("")
            continue
        if cleaned in DELIMITER_LINES:
            continue

        # Question with explicit ID.
        question_id_match = QUESTION_WITH_ID_RE.match(cleaned)
        if question_id_match:
            start_entry(
                question_id_match.group("qid"),
                question_id_match.group("question").lstrip(".\u2013- \t"),
            )
            continue

        # Question without ID.
        question_text_match = QUESTION_TEXT_RE.match(cleaned)
        if question_text_match:
            start_entry(None, question_text_match.group("question"))
            continue

        answer_match = ANSWER_RE.match(cleaned)
        if answer_match:
            in_answer = True
            if current is None:
                start_entry(None, "")
            answer_inline = answer_match.group("answer").strip()
            if answer_inline and current:
                current["answer_lines"].append(answer_inline)
            continue

        if in_answer and current:
            current["answer_lines"].append(raw_line.rstrip())

    if current:
        parsed.append(current)

    normalised: List[Dict[str, str]] = []
    for item in parsed:
        normalised.append(
            {
                "qid": clean_text(item.get("qid")),
                "question": clean_text(item.get("question")),
                "answer": "\n".join(item.get("answer_lines", [])).strip(),
            }
        )
    return normalised


def parse_file(input_path: pathlib.Path, output_path: pathlib.Path, s4_path: pathlib.Path) -> None:
    qid_to_question, question_lookup = load_s4_questions(s4_path)
    df = pd.read_csv(input_path)

    pmid_order: List[str] = []
    pmid_to_qid_data: Dict[str, Dict[str, Dict[str, str]]] = {}

    for _, row in df.iterrows():
        pmid = clean_text(row.get("PMID"))
        if not pmid:
            continue
        if pmid not in pmid_order:
            pmid_order.append(pmid)

        multi_answer_text = clean_text(row.get("Multiple Answer"))
        if not multi_answer_text:
            continue
        parsed_entries = parse_multiple_answer(
            multi_answer_text.replace("\r\n", "\n").replace("\r", "\n")
        )

        for entry in parsed_entries:
            raw_qid = clean_text(entry.get("qid"))
            question_text = clean_text(entry.get("question"))

            mapped_qid: str | None = None
            if question_text:
                mapped_qid = question_lookup.get(normalise_question_text(question_text))
            if not mapped_qid and raw_qid:
                mapped_qid = QID_MAP.get(raw_qid)
            if not mapped_qid:
                continue

            canonical_question = qid_to_question.get(mapped_qid, question_text)
            pmid_to_qid_data.setdefault(pmid, {})[mapped_qid] = {
                "question": canonical_question,
                "answer": clean_text(entry.get("answer")),
            }

    rows: List[Dict[str, str]] = []
    for pmid in pmid_order:
        qid_data = pmid_to_qid_data.get(pmid, {})
        for qid in EXPECTED_QIDS:
            data = qid_data.get(qid, {})
            rows.append(
                {
                    "PMID": pmid,
                    "QID": qid,
                    "Question": qid_to_question.get(qid, data.get("question", "")),
                    "Answer": data.get("answer", ""),
                }
            )

    out_df = pd.DataFrame(rows, columns=["PMID", "QID", "Question", "Answer"]).fillna("")
    out_df.to_csv(output_path, index=False)

    counts = out_df.groupby("PMID").size().sort_index()
    for pmid, count in counts.items():
        print(f"{pmid}: {count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=DEFAULT_INPUT,
        help="Input CSV path (default: ./csv/llama-3.1-70B-PV1_new30.csv)",
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

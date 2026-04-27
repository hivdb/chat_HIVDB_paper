#!/usr/bin/env python3
"""Parse FT answers in csv/llama-3.1-8B-FT_new30.csv where questions start with 'Question:'."""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Dict, List, Tuple

import pandas as pd

DEFAULT_INPUT = pathlib.Path("./csv/llama-8b/llama-3.1-8B-FT_R16.csv")
DEFAULT_S4 = pathlib.Path("./csv/S4Table.xlsx")  # aka Table S4.xlsx
DELIMITER_LINES = {'"""', "'''", "```", '""'}
SECTION_RE = re.compile(
    r"^(?P<label>Evidence|Rationale|Answer)\s*[:\-\u2013]?\s*(?P<rest>.*)",
    re.IGNORECASE,
)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def strip_formatting(text: str) -> str:
    """Remove markdown fences/backticks and trim whitespace."""
    return clean_text(text.replace("```", ""))


def normalise_cell(value: object) -> str:
    text = clean_text(value)
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")


def normalise_question_text(question: str) -> str:
    cleaned = strip_formatting(question)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned.rstrip("?:.")


QUESTION_LIST: List[str] = [
    "Does the paper report HIV sequences from patient samples?",
    "Does the paper report in vitro drug susceptibility data?",
    "Were sequences from the paper made publicly available?",
    "What were the GenBank accession numbers for sequenced HIV isolates?",
    "How many individuals had samples obtained for HIV sequencing?",
    "From which countries were the sequenced samples obtained?",
    "From what years were the sequenced samples obtained?",
    "Were samples cloned prior to sequencing?",
    "Which HIV genes were reported to have been sequenced?",
    "What method was used for sequencing?",
    "What type of samples were sequenced?",
    "Were any sequences obtained from individuals with virological failure on a treatment regimen?",
    "Were the patients in the study in a clinical trial?",
    "Does the paper report HIV sequences from individuals who had previously received ARV drugs?",
    "Which drug classes were received by individuals in the study before sample sequencing?",
    "Which drugs were received by individuals in the study before sample sequencing?",
]

QUESTION_MAP = {idx: text for idx, text in enumerate(QUESTION_LIST, start=1)}
QUESTION_LOOKUP = {
    normalise_question_text(text): str(idx) for idx, text in QUESTION_MAP.items() if text
}
EXPECTED_QIDS = [str(i) for i in range(1, 17)]


def load_s4_questions(s4_path: pathlib.Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (qid -> question, normalized question -> qid) using the S4 table."""
    df = pd.read_excel(s4_path)
    qid_to_question: Dict[str, str] = {}
    question_lookup: Dict[str, str] = {}
    for _, row in df.iterrows():
        qid = clean_text(row.get("QID"))
        question = clean_text(row.get("Question"))
        if not qid or not question:
            continue
        qid_to_question[str(qid)] = question
        norm_question = normalise_question_text(question)
        if norm_question and norm_question not in question_lookup:
            question_lookup[norm_question] = str(qid)

    # Ensure all expected QIDs map to a question string
    for qid in EXPECTED_QIDS:
        if qid not in qid_to_question and QUESTION_MAP.get(int(qid)):
            qid_to_question[qid] = QUESTION_MAP[int(qid)]
    return qid_to_question, question_lookup


def match_question_to_qid(question_line: str, s4_lookup: Dict[str, str]) -> str | None:
    norm_line = normalise_question_text(question_line)
    if not norm_line:
        return None

    if norm_line in s4_lookup:
        return s4_lookup[norm_line]

    for norm_question, qid in QUESTION_LOOKUP.items():
        if norm_question in norm_line or norm_line in norm_question:
            return qid

    numeric_match = re.match(r"(?P<id>\d+)", norm_line)
    if numeric_match:
        candidate = numeric_match.group("id")
        if candidate in EXPECTED_QIDS:
            return candidate
    return None


def parse_single_entry(
    ft_answer: str,
    fallback_qid: str,
    fallback_question: str,
    s4_lookup: Dict[str, str],
) -> Dict[str, str] | None:
    """Parse a single question block using row metadata as fallback."""
    lines = normalise_cell(ft_answer).splitlines()
    question_line = fallback_question
    answer_lines: List[str] = []
    current_section: str | None = None

    for raw_line in lines:
        stripped = raw_line.strip()
        cleaned = strip_formatting(stripped.strip("*_ ").lstrip("> \t*-"))
        if not cleaned or cleaned in DELIMITER_LINES:
            continue

        if cleaned.lower().startswith("question"):
            inline = re.sub(r"^question\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
            if inline:
                question_line = inline
            current_section = None
            continue

        section_match = SECTION_RE.match(cleaned)
        if section_match:
            current_section = section_match.group("label").lower()
            rest = section_match.group("rest").strip()
            if current_section == "answer" and rest:
                answer_lines.append(rest)
            continue

        if current_section == "answer":
            answer_lines.append(cleaned)

    qid = fallback_qid or match_question_to_qid(question_line, s4_lookup) or ""
    question = strip_formatting(question_line)
    answer = "\n".join(line for line in answer_lines if line).strip()

    if not qid and not question and not answer:
        return None
    return {"qid": qid, "question": question, "answer": answer}


def extract_entries(ft_answer: str, s4_lookup: Dict[str, str]) -> List[Dict[str, str]]:
    """Extract question/answer pairs from a single FT Answer cell."""
    entries: List[Dict[str, str]] = []
    lines = normalise_cell(ft_answer).splitlines()
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        cleaned = strip_formatting(raw_line.strip().strip("*_ ").lstrip("> \t*-"))
        if not cleaned or cleaned in DELIMITER_LINES:
            i += 1
            continue

        if cleaned.lower().startswith("question"):
            # inline or next-line question text
            inline = re.sub(r"^question\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
            question_line = inline
            qid = match_question_to_qid(question_line, s4_lookup)
            if not qid:
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    question_line = strip_formatting(lines[j].strip())

            qid = match_question_to_qid(question_line, s4_lookup)
            answer_lines: List[str] = []

            k = i + 1
            while k < len(lines):
                candidate = strip_formatting(lines[k].strip().strip("*_ ").lstrip("> \t*-"))
                if candidate in DELIMITER_LINES:
                    k += 1
                    continue
                if candidate.lower().startswith("question"):
                    break
                if candidate.lower().startswith("answer"):
                    inline_ans = re.sub(
                        r"^answer\s*[:-\u2013]?\s*",
                        "",
                        candidate,
                        flags=re.IGNORECASE,
                    )
                    if inline_ans:
                        answer_lines.append(inline_ans)
                    k += 1
                    while k < len(lines):
                        follow = strip_formatting(lines[k].strip().strip("*_ ").lstrip("> \t*-"))
                        if follow.lower().startswith("question") or follow.lower().startswith("answer"):
                            break
                        if follow:
                            answer_lines.append(follow)
                        k += 1
                    break
                k += 1

            entries.append(
                {
                    "qid": qid or "",
                    "question": strip_formatting(question_line),
                    "answer": "\n".join([line for line in answer_lines if line]).strip(),
                }
            )
        i += 1
    return entries


def parse_file(input_path: pathlib.Path, output_path: pathlib.Path, s4_path: pathlib.Path) -> None:
    qid_to_question, s4_lookup = load_s4_questions(s4_path)

    df = pd.read_csv(input_path, dtype={"PMID": str})
    pmid_order: List[str] = []
    pmid_to_qid: Dict[str, Dict[str, Dict[str, str]]] = {}

    for _, row in df.iterrows():
        pmid = clean_text(row.get("PMID"))
        if not pmid:
            continue
        if pmid not in pmid_order:
            pmid_order.append(pmid)

        ft_answer_text = normalise_cell(row.get("FT Answer"))
        if not ft_answer_text:
            continue

        fallback_qid = clean_text(row.get("QID"))
        fallback_question = clean_text(row.get("Question"))
        parsed_entries = extract_entries(ft_answer_text, s4_lookup)
        if not parsed_entries:
            single_entry = parse_single_entry(
                ft_answer_text,
                fallback_qid,
                fallback_question,
                s4_lookup,
            )
            parsed_entries = [single_entry] if single_entry else []

        for entry in parsed_entries:
            qid = entry.get("qid") or match_question_to_qid(entry.get("question", ""), s4_lookup)
            if not qid and len(parsed_entries) == 1:
                qid = fallback_qid
            if not qid:
                continue

            pmid_to_qid.setdefault(pmid, {})[qid] = {
                "question": qid_to_question.get(qid, strip_formatting(entry.get("question", ""))),
                "answer": strip_formatting(entry.get("answer", "")),
            }

    rows: List[Dict[str, str]] = []
    for pmid in pmid_order:
        found = pmid_to_qid.get(pmid, {})
        for qid in EXPECTED_QIDS:
            data = found.get(qid, {})
            rows.append(
                {
                    "PMID": pmid,
                    "QID": qid,
                    "Question": qid_to_question.get(qid, data.get("question", "")),
                    "Answer": data.get("answer", ""),
                }
            )

    pd.DataFrame(rows, columns=["PMID", "QID", "Question", "Answer"]).to_csv(
        output_path, index=False
    )

    counts = pd.DataFrame(rows).groupby("PMID").size().sort_index()
    for pmid, count in counts.items():
        print(f"{pmid}: {count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=DEFAULT_INPUT,
        help="Input CSV path (default: ./csv/llama-8b/llama-3.1-8B-FT_R16.csv)",
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

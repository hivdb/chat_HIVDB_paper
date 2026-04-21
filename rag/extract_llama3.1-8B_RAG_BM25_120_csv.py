#!/usr/bin/env python3
"""Extract structured question/answer responses from CSV model outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
ADVANCED_PROMPTING_DIR = ROOT.parent / "advanced-prompting"
DEFAULT_METADATA = ADVANCED_PROMPTING_DIR / "csv" / "S4Table.xlsx"
DEFAULT_PROMPT_TEMPLATE = ROOT.parent / "eval" / "gpt-5" / "gpt-5-mini-prompt.md"
DEFAULT_RESPONSES_CSV = ROOT / "llama3.1" / "8B_RAG_BM25_120.csv"
PROMPT_QUESTION_PATTERN = re.compile(r"Question\s+(\d+):\s*(.+)")
QUESTION_HEADER_PATTERN = re.compile(
    r'^\s*(?:["`>\-]+\s*)?(?:#+\s*)?(?:\*\*)?Question(?:\s+(\d+))?\s*:?\s*(.*?)(?:\*\*)?\s*$',
    re.IGNORECASE | re.MULTILINE,
)
INLINE_QUESTION_PATTERN = re.compile(r"^\s*(?:\*\*)?Question:\s*(.+?)(?:\*\*)?\s*$", re.IGNORECASE | re.MULTILINE)
ANSWER_IN_BLOCK_PATTERN = re.compile(r"(?is)\bAnswer:\s*(.+)")
ANSWER_ONLY_PATTERN = re.compile(
    r"Answer:\s*(.+?)(?=\n(?:---\s*\n)?\s*(?:[#>*`\"\-]+\s*)?(?:\*\*)?Question(?:\s+\d+)?\s*:?\s*|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-xlsx",
        type=Path,
        default=DEFAULT_METADATA,
        help="Workbook containing canonical QID/Question/Type/Category metadata.",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template used to derive canonical question ordering.",
    )
    return parser.parse_args()


def normalize_question(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def normalize_identifier(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def clean_answer(text: str) -> str:
    cleaned = str(text or "").replace("```", " ").replace('"""', " ").strip()
    cleaned = cleaned.strip('"').strip("'")
    return " ".join(cleaned.split())


def extract_pairs(response_text: str, ordered_questions: list[tuple[int, str]]) -> list[tuple[str, str]]:
    cleaned = str(response_text or "")
    cleaned = cleaned.replace('"""', " ")
    cleaned = re.sub(r"\*\*(Question:\s*.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"^\s*---\s*$", "", cleaned, flags=re.MULTILINE)
    answers = [clean_answer(answer) for answer in ANSWER_ONLY_PATTERN.findall(cleaned)]
    if len(answers) == len(ordered_questions):
        return [(normalize_question(question_text), answer) for (_, question_text), answer in zip(ordered_questions, answers)]
    pairs: list[tuple[str, str]] = []
    matches = list(QUESTION_HEADER_PATTERN.finditer(cleaned))
    for index, match in enumerate(matches):
        block_start = match.start()
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        block = cleaned[block_start:block_end]
        header_question = match.group(2).strip().strip('"').strip("'")
        if not header_question:
            inline = INLINE_QUESTION_PATTERN.search(block[match.end() - block_start :])
            if inline:
                header_question = inline.group(1).strip().strip('"').strip("'")
        answer_match = ANSWER_IN_BLOCK_PATTERN.search(block)
        if not header_question or not answer_match:
            continue
        pairs.append((normalize_question(header_question), clean_answer(answer_match.group(1))))
    return pairs


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str, keep_default_na=False)
    required = ["QID", "Question", "Type", "Category"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Metadata workbook missing columns: {missing}")
    df = df[required].copy()
    df["QID"] = df["QID"].astype(int)
    df["__question_norm"] = df["Question"].apply(normalize_question)
    df = df.drop_duplicates(subset=["QID", "__question_norm"], keep="first").sort_values("QID")
    return df


def load_prompt_questions(path: Path) -> dict[str, tuple[int, str]]:
    question_map: dict[str, tuple[int, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PROMPT_QUESTION_PATTERN.match(line.strip())
        if not match:
            continue
        qid = int(match.group(1))
        question = match.group(2).strip()
        question_map[normalize_question(question)] = (qid, question)
    if not question_map:
        raise ValueError(f"No canonical questions found in prompt template: {path}")
    return question_map


def load_responses_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows from {path}")
    required = ["PMID", "FT Answer"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Responses CSV missing columns: {missing}")
    return df


def build_output_csv_path(input_csv: Path) -> Path:
    parsed_name = f"llama3.1-{input_csv.stem}_parsed.csv"
    return ROOT / parsed_name


def main() -> int:
    args = parse_args()
    metadata = load_metadata(args.metadata_xlsx)
    prompt_questions = load_prompt_questions(args.prompt_template)
    responses_df = load_responses_csv(DEFAULT_RESPONSES_CSV)
    output_csv = build_output_csv_path(DEFAULT_RESPONSES_CSV)
    ordered_prompt_questions = sorted(prompt_questions.values(), key=lambda item: item[0])
    metadata_by_qid = {
        int(row["QID"]): {
            "Type": row["Type"],
            "Category": row["Category"],
        }
        for _, row in metadata.drop_duplicates(subset=["QID"], keep="first").iterrows()
    }
    parsed_rows: list[dict[str, str | int]] = []

    for _, record in responses_df.iterrows():
        pmid = normalize_identifier(record.get("PMID", ""))
        pairs = extract_pairs(record.get("FT Answer", ""), ordered_prompt_questions)
        if not pairs:
            continue
        for q_norm, answer in pairs:
            prompt_match = prompt_questions.get(q_norm)
            if not prompt_match:
                continue
            qid, question_text = prompt_match
            meta = metadata_by_qid.get(qid, {})
            parsed_rows.append(
                {
                    "PMID": pmid,
                    "QID": qid,
                    "Question": question_text,
                    "Type": meta.get("Type", ""),
                    "Category": meta.get("Category", ""),
                    "Answer": answer,
                }
            )

    output_df = pd.DataFrame(parsed_rows)
    if output_df.empty:
        output_df = pd.DataFrame(columns=["PMID", "QID", "Question", "Type", "Category", "Answer"])
    else:
        output_df = output_df.sort_values(["PMID", "QID"]).drop_duplicates(subset=["PMID", "QID"], keep="first")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    if output_df.empty:
        print("No parsed rows found for any PMID")
    else:
        pmid_counts = output_df.groupby("PMID").size()
        for pmid, row_count in pmid_counts.items():
            print(f"PMID {pmid}: {row_count} rows")
    print(f"Wrote {len(output_df)} parsed answers to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

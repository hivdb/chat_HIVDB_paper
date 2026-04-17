#!/usr/bin/env python3
"""Extract structured question/answer responses from JSONL model outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
ADVANCED_PROMPTING_DIR = ROOT.parent / "advanced-prompting"
QA_PATTERN = re.compile(
    r"Question:\s*(.+?)\n.*?Answer:\s*(.+?)(?=\nQuestion:|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-jsonl", type=Path, required=True, help="Input JSONL with {pmid,response}.")
    parser.add_argument(
        "--metadata-xlsx",
        type=Path,
        default=ADVANCED_PROMPTING_DIR / "csv" / "ground_truth.xlsx",
        help="Workbook containing PMID/QID/Question/Type/Category metadata.",
    )
    parser.add_argument("--output-csv", type=Path, required=True, help="Output parsed CSV path.")
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


def extract_pairs(response_text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for question, answer in QA_PATTERN.findall(response_text or ""):
        pairs.append((normalize_question(question), clean_answer(answer)))
    return pairs


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str, keep_default_na=False)
    required = ["PMID", "QID", "Question", "Type", "Category"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Metadata workbook missing columns: {missing}")
    df = df[required].copy()
    df["PMID"] = df["PMID"].apply(normalize_identifier)
    df["QID"] = df["QID"].astype(int)
    df["__question_norm"] = df["Question"].apply(normalize_question)
    return df


def main() -> int:
    args = parse_args()
    metadata = load_metadata(args.metadata_xlsx)
    parsed_rows: list[dict[str, str | int]] = []

    with args.responses_jsonl.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pmid = normalize_identifier(record.get("pmid", ""))
            qmap = {}
            for q_norm, answer in extract_pairs(record.get("response", "")):
                qmap.setdefault(q_norm, []).append(answer)

            if not qmap:
                continue

            pmid_rows = metadata[metadata["PMID"] == pmid].sort_values("QID")
            for _, row in pmid_rows.iterrows():
                q_norm = row["__question_norm"]
                answers = qmap.get(q_norm, [])
                if not answers:
                    continue
                parsed_rows.append(
                    {
                        "PMID": pmid,
                        "QID": int(row["QID"]),
                        "Question": row["Question"],
                        "Type": row["Type"],
                        "Category": row["Category"],
                        "Answer": answers.pop(0),
                    }
                )

    output_df = pd.DataFrame(parsed_rows)
    if output_df.empty:
        output_df = pd.DataFrame(columns=["PMID", "QID", "Question", "Type", "Category", "Answer"])
    else:
        output_df = output_df.sort_values(["PMID", "QID"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(output_df)} parsed answers to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

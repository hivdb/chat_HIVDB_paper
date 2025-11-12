#!/usr/bin/env python3
"""Generate specialized prompt files from S2Table.xlsx via GPT-5 Responses API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


DATA_FILE = Path("S2Table.xlsx")
ENV_FILE = Path(".env")
OUTPUT_DIR = Path("prompts")
ENV_KEY = "OPENAI_API_KEY"


def load_records(data_file: Path) -> Dict[str, List[Dict[str, str]]]:
    """Read Excel entries grouped by QID and normalise each record."""
    if not data_file.exists():
        raise FileNotFoundError(f"{data_file} not found.")

    df = pd.read_excel(data_file)
    required = {"QID", "Question", "Evidence", "Rationale", "Answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(sorted(missing))}")

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for qid, rows in df.groupby("QID", sort=True):
        grouped[str(qid)] = [
            {
                "Question": str(row["Question"]).strip(),
                "Evidence": str(row["Evidence"]).strip(),
                "Rationale": str(row["Rationale"]).strip(),
                "Answer": str(row["Answer"]).strip(),
            }
            for _, row in rows.iterrows()
        ]
    return grouped


def compile_dataset(records: Iterable[Dict[str, str]]) -> str:
    """Turn per-row dictionaries into an enumerated block for the LLM call."""
    blocks = []
    for idx, item in enumerate(records, start=1):
        block = (
            f"Data Point {idx}:\n"
            f"- Question: {item['Question']}\n"
            f"- Evidence: {item['Evidence']}\n"
            f"- Rationale: {item['Rationale']}\n"
            f"- Answer: {item['Answer']}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


def craft_prompt_input(qid: str, records: List[Dict[str, str]]) -> str:
    """Build the instruction payload guiding GPT-5 to deliver if-then prompts."""
    canonical_question = records[0]["Question"]
    dataset_block = compile_dataset(records)
    return (
        "You are preparing an instruction prompt for a smaller LLM that must answer the "
        "question by scanning the full text of the associated paper. Synthesize the following "
        "250 data points into ~250 words of precise if-then heuristics. Each rule should explain "
        "which phrases or findings in the paper imply specific answer choices, and should be "
        "expressed in the form 'If <pattern/phrase>, then consider <implication>'. Prefer grouping "
        "related biomedical signals together, and ensure coverage of evidence and rationale for "
        "all answer options. End with a brief reminder to validate findings against the paper.\n\n"
        f"QID: {qid}\n"
        f"Canonical Question: {canonical_question}\n\n"
        f"Source Data (Question, Evidence, Rationale, Answer across 250 entries):\n"
        f"{dataset_block}\n\n"
        "Deliverable: A single ~250 word prompt comprised of if-then style rules that "
        "capture discriminative phrases and concepts enabling GPT-4o, Llama-70B, or Llama-8B "
        "to answer the question accurately."
    )


def main() -> None:
    load_dotenv(ENV_FILE)
    api_key = os.getenv(ENV_KEY)
    if not api_key:
        raise KeyError(f"{ENV_KEY} not found in environment or {ENV_FILE}.")

    grouped_records = load_records(DATA_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    for qid, records in grouped_records.items():
        if not records:
            continue
        input_text = craft_prompt_input(qid, records)
        response = client.responses.create(model="gpt-5", input=input_text)
        prompt_text = response.output_text.strip()
        output_path = OUTPUT_DIR / f"{qid}.txt"
        output_path.write_text(prompt_text + "\n", encoding="utf-8")
        print(f"Wrote prompt for QID {qid} -> {output_path}")


if __name__ == "__main__":
    main()

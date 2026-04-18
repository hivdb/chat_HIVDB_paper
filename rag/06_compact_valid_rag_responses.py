#!/usr/bin/env python3
"""Compact retry-heavy RAG response files down to one preferred row per PMID."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import tiktoken


ROOT = Path(__file__).resolve().parent
DEFAULT_RESPONSE_FILES = [
    ROOT / "jsonl" / "pmid_responses_bm25_rag_gpt4o_original120.jsonl",
    ROOT / "jsonl" / "pmid_responses_bm25_rag_gpt4o_new30.jsonl",
    ROOT / "jsonl" / "pmid_responses_semantic_rag_gpt4o_original120.jsonl",
    ROOT / "jsonl" / "pmid_responses_semantic_rag_gpt4o_new30.jsonl",
]
ANSWER_PATTERN = re.compile(r"Answer:\s*", re.IGNORECASE)
EXPECTED_ANSWER_COUNT = 16
MIN_RESPONSE_TOKENS = 200
MODEL_NAME = "gpt-4o-mini-2024-07-18"
FAILED_RESPONSES = {
    "I'm unable to fulfill that request.",
    "I'm unable to help with that.",
    "I'm unable to process or analyze the paper you provided as a full text.",
    "I’m unable to help with that.",
    "I’m unable to process or analyze the paper you provided as a full text.",
}
FAILURE_SUBSTRINGS = (
    "i'm unable to help with that",
    "i’m unable to help with that",
    "i'm unable to process",
    "i’m unable to process",
    "unable to comply",
    "cannot help with that",
    "cannot process",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses-jsonl",
        action="append",
        type=Path,
        help="Response JSONL path(s) to compact. Defaults to all four RAG GPT-4o response files.",
    )
    parser.add_argument(
        "--suffix",
        default="_compacted",
        help="Suffix to append before the .jsonl extension for compacted outputs.",
    )
    return parser.parse_args()


def estimate_tokens(text: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(MODEL_NAME)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text or ""))


def is_failed_response(text: str) -> tuple[bool, str]:
    stripped = (text or "").strip()
    if not stripped:
        return True, "empty"
    if stripped in FAILED_RESPONSES:
        return True, "known_failure"
    lower = stripped.lower()
    if any(keyword in lower for keyword in FAILURE_SUBSTRINGS):
        return True, "refusal"
    answer_count = len(ANSWER_PATTERN.findall(stripped))
    if answer_count < EXPECTED_ANSWER_COUNT:
        return True, f"answer_count_{answer_count}"
    if estimate_tokens(stripped) < MIN_RESPONSE_TOKENS:
        return True, "too_short"
    return False, "ok"


def preferred_records(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            pmid = str(record.get("pmid", "")).strip()
            response = str(record.get("response", ""))
            failed, reason = is_failed_response(response)
            rows.append(
                {
                    "line_no": line_no,
                    "pmid": pmid,
                    "response": response,
                    "valid": not failed,
                    "reason": reason,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return []

    preferred: list[dict] = []
    for _, group in df.groupby("pmid", sort=False):
        valid_group = group[group["valid"] == True]
        chosen = valid_group.iloc[-1] if not valid_group.empty else group.iloc[-1]
        preferred.append({"pmid": chosen["pmid"], "response": chosen["response"]})
    return preferred


def main() -> int:
    args = parse_args()
    response_files = args.responses_jsonl or DEFAULT_RESPONSE_FILES

    for path in response_files:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        records = preferred_records(path)
        output_path = path.with_name(f"{path.stem}{args.suffix}{path.suffix}")
        with output_path.open("w", encoding="utf-8") as outfile:
            for record in records:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} compacted records to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

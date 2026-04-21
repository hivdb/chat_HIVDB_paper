#!/usr/bin/env python3
"""Validate RAG response JSONLs for complete 16-answer outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import tiktoken


SCRIPT_ROOT = Path(__file__).resolve().parent
RAG_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = RAG_ROOT.parent
DEFAULT_RESPONSE_FILES = [
    RAG_ROOT / "jsonl" / "pmid_responses_bm25_rag_gpt4o_original120.jsonl",
    RAG_ROOT / "jsonl" / "pmid_responses_bm25_rag_gpt4o_new30.jsonl",
    RAG_ROOT / "jsonl" / "pmid_responses_semantic_rag_gpt4o_original120.jsonl",
    RAG_ROOT / "jsonl" / "pmid_responses_semantic_rag_gpt4o_new30.jsonl",
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
        help="Response JSONL path(s) to validate. Defaults to all four RAG GPT-4o response files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=RAG_ROOT / "csv" / "verification" / "response_validation_summary.csv",
        help="Write per-PMID validation results to this CSV.",
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


def main() -> int:
    args = parse_args()
    response_files = args.responses_jsonl or DEFAULT_RESPONSE_FILES
    rows: list[dict[str, object]] = []

    for path in response_files:
        if not path.exists():
            rows.append(
                {
                    "file": str(path.relative_to(REPO_ROOT)),
                    "pmid": "",
                    "valid": False,
                    "reason": "missing_file",
                    "answer_count": 0,
                    "char_count": 0,
                }
            )
            continue

        with path.open("r", encoding="utf-8") as infile:
            for line in infile:
                if not line.strip():
                    continue
                record = json.loads(line)
                pmid = str(record.get("pmid", "")).strip()
                response = str(record.get("response", ""))
                valid, reason = is_failed_response(response)
                rows.append(
                    {
                        "file": str(path.relative_to(REPO_ROOT)),
                        "pmid": pmid,
                        "valid": not valid,
                        "reason": reason,
                        "answer_count": len(ANSWER_PATTERN.findall(response)),
                        "char_count": len(response),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No response rows found.")
        return 1

    # The runner appends retries, so evaluate one effective row per PMID/file by preferring
    # the most recent valid response when present; otherwise fall back to the last row.
    effective_rows: list[pd.Series] = []
    for (_, _), group in df.groupby(["file", "pmid"], dropna=False, sort=False):
        valid_group = group[group["valid"] == True]
        chosen = valid_group.iloc[-1] if not valid_group.empty else group.iloc[-1]
        effective_rows.append(chosen)

    effective = pd.DataFrame(effective_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    effective.to_csv(args.output_csv, index=False)

    summary = (
        effective.groupby("file")
        .agg(
            total_rows=("pmid", "count"),
            valid_rows=("valid", "sum"),
        )
        .reset_index()
    )
    summary["invalid_rows"] = summary["total_rows"] - summary["valid_rows"]
    print(summary.to_string(index=False))
    print(f"\nWrote validation summary to {args.output_csv}")
    return 0 if int(summary["invalid_rows"].sum()) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

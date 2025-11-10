#!/usr/bin/env python3
"""Parallel updater that enforces Nov 5 guidelines on human answers via GPT-5-mini."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


MODEL_NAME = "gpt-5-mini"
QUESTIONS_PATH = Path("advanced-prompting/merged_answers.xlsx")
GUIDELINES_PATH = Path("advanced-prompting/Prompts_Nov5.md")
OUTPUT_CSV = Path("eval/updated_human_answers.csv")
RAW_JSONL = Path("eval/updated_human_answers.jsonl")
RATE_LIMIT_TOKENS_PER_MIN = 180_000_000
TOKEN_BUFFER = 200
CSV_FIELDS = ["PMID", "QID", "Question", "Human Answer", "Updated Human Answer"]
ANSWER_PATTERN = re.compile(r"Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)

SYSTEM_PROMPT = (
    "You are auditing extracted answers for adherence to the Nov 5 curation rules. "
    "Rewrite each answer so every conditional rule in the guidelines is satisfied. "
    "Do not invent evidence; use only guideline-allowed categorical outputs when required."
)


class UpdatedEntry(BaseModel):
    QID: int
    UpdatedHumanAnswer: str = Field(alias="Updated Human Answer")


class UpdateResponse(BaseModel):
    updates: List[UpdatedEntry]


@dataclass(frozen=True)
class QARecord:
    pmid: str
    qid: int
    question: str
    human_answer: str


@dataclass(frozen=True)
class UpdateJob:
    pmid: str
    records: List[QARecord]
    payload: str
    token_estimate: int


class TokenBucket:
    """Tokens-per-minute governor shared across async workers."""

    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self.events: deque[tuple[float, int]] = deque()
        self.current = 0
        self.lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        while self.events and now - self.events[0][0] >= 60:
            _, spent = self.events.popleft()
            self.current -= spent

    async def acquire(self, tokens: int) -> None:
        while True:
            async with self.lock:
                now = time.monotonic()
                self._prune(now)
                if self.current + tokens <= self.limit:
                    self.events.append((now, tokens))
                    self.current += tokens
                    return
                wait_time = (
                    max(60 - (now - self.events[0][0]), 0.1) if self.events else 1.0
                )
            await asyncio.sleep(wait_time)


def configure_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("update_human_answers")


def load_guidelines(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Guidelines file missing: {path}")
    return path.read_text(encoding="utf-8")


def load_questions(path: Path) -> Dict[str, List[QARecord]]:
    df = pd.read_excel(path, usecols=["PMID", "QID", "Question", "Human Answer"])
    df["PMID"] = df["PMID"].astype(str)
    df = df.fillna({"Human Answer": ""})

    grouped: Dict[str, List[QARecord]] = defaultdict(list)
    for record in df.to_dict(orient="records"):
        pmid = str(record["PMID"])
        grouped[pmid].append(
            QARecord(
                pmid=pmid,
                qid=int(record["QID"]),
                question=str(record["Question"]).strip(),
                human_answer=str(record["Human Answer"]).strip(),
            )
        )

    for pmid in grouped:
        grouped[pmid].sort(key=lambda rec: rec.qid)

    return grouped


def load_existing_pairs(path: Path) -> Dict[str, set[int]]:
    pairs: Dict[str, set[int]] = defaultdict(set)
    if not path.exists():
        return pairs

    df = pd.read_csv(path, usecols=["PMID", "QID"])
    for row in df.itertuples(index=False):
        pairs[str(row.PMID)].add(int(row.QID))
    return pairs


def prepare_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()


def build_token_counter(model_name: str) -> Callable[[str], int]:
    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        def counter(text: str) -> int:
            return len(encoding.encode(text))

        return counter

    logging.warning("tiktoken not found; using heuristic token counter.")

    def fallback(text: str) -> int:
        return max(len(text) // 4, len(text.split()))

    return fallback


def build_payload(records: List[QARecord]) -> str:
    payload = [
        {
            "QID": rec.qid,
            "Question": rec.question,
            "Human Answer": rec.human_answer or "Not Reported",
        }
        for rec in records
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def prepare_jobs(
    question_map: Dict[str, List[QARecord]],
    existing: Dict[str, set[int]],
    guidelines: str,
    token_counter: Callable[[str], int],
    limit: int | None,
    logger: logging.Logger,
) -> List[UpdateJob]:
    jobs: List[UpdateJob] = []
    system_tokens = token_counter(SYSTEM_PROMPT)
    guideline_tokens = token_counter(guidelines)

    for pmid, records in question_map.items():
        if limit is not None and len(jobs) >= limit:
            break

        completed = existing.get(pmid, set())
        if len(completed) >= len(records):
            logger.info("Skipping PMID %s; already updated.", pmid)
            continue

        payload = build_payload(records)
        user_tokens = token_counter(payload)
        request_tokens = system_tokens + guideline_tokens + user_tokens + TOKEN_BUFFER
        jobs.append(
            UpdateJob(
                pmid=pmid,
                records=records,
                payload=payload,
                token_estimate=request_tokens,
            )
        )
        logger.info("Prepared PMID %s (~%d tokens).", pmid, request_tokens)

    return jobs


async def call_model_async(client: AsyncOpenAI, guidelines: str, payload: str) -> Dict[int, str]:
    response = await client.responses.parse(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Guidelines:\n{guidelines}"},
            {"role": "user", "content": f"Original answers:\n{payload}"},
            {
                "role": "user",
                "content": (
                    "Return JSON with an 'updates' array. Each entry must contain "
                    "'QID' and 'Updated Human Answer'. Include every QID once."
                ),
            },
        ],
        text_format=UpdateResponse,
    )
    return {entry.QID: entry.UpdatedHumanAnswer.strip() for entry in response.output_parsed.updates}


def append_rows(rows: List[dict]) -> None:
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writerows(rows)


def append_raw(pmid: str, updates: Dict[int, str]) -> None:
    RAW_JSONL.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): v for k, v in updates.items()}
    with RAW_JSONL.open("a", encoding="utf-8") as outfile:
        outfile.write(json.dumps({"pmid": pmid, "updates": serializable}, ensure_ascii=False) + "\n")


def extract_final_answer(text: str) -> str:
    if text is None:
        return ""
    cleaned = str(text).strip()
    if not cleaned:
        return ""

    match = ANSWER_PATTERN.search(cleaned)
    if not match:
        return cleaned

    answer_block = match.group(1).strip()
    first_line = answer_block.splitlines()[0].strip()
    return first_line.strip('" ')


def cleanup_updated_answers(path: Path, logger: logging.Logger) -> None:
    if not path.exists():
        logger.info("No CSV found at %s to clean.", path)
        return

    df = pd.read_csv(path)
    if "Updated Human Answer" not in df.columns:
        logger.warning("Column 'Updated Human Answer' missing in %s.", path)
        return

    changes = 0
    for idx, value in df["Updated Human Answer"].items():
        final_value = extract_final_answer(value)
        if final_value != value:
            df.at[idx, "Updated Human Answer"] = final_value
            changes += 1

    if changes:
        df.to_csv(path, index=False)
        logger.info("Cleaned %d rows in %s.", changes, path)
    else:
        logger.info("No cleanup needed for %s.", path)


async def persist_results(
    job: UpdateJob,
    updates: Dict[int, str],
    existing: Dict[str, set[int]],
    writer_lock: asyncio.Lock,
    logger: logging.Logger,
) -> None:
    rows: List[dict] = []
    for rec in job.records:
        new_answer = updates.get(rec.qid, rec.human_answer)
        rows.append(
            {
                "PMID": rec.pmid,
                "QID": rec.qid,
                "Question": rec.question,
                "Human Answer": rec.human_answer,
                "Updated Human Answer": new_answer,
            }
        )

    async with writer_lock:
        completed = existing.setdefault(job.pmid, set())
        new_rows = [row for row in rows if row["QID"] not in completed]
        if not new_rows:
            logger.info("No new rows to write for PMID %s.", job.pmid)
            return

        await asyncio.to_thread(append_rows, new_rows)
        await asyncio.to_thread(append_raw, job.pmid, updates)
        completed.update(row["QID"] for row in new_rows)
        logger.info("Persisted %d rows for PMID %s.", len(new_rows), job.pmid)


async def process_job(
    job: UpdateJob,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    rate_limiter: TokenBucket,
    writer_lock: asyncio.Lock,
    existing: Dict[str, set[int]],
    guidelines: str,
    logger: logging.Logger,
) -> bool:
    try:
        await rate_limiter.acquire(job.token_estimate)
        async with semaphore:
            updates = await call_model_async(client, guidelines, job.payload)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Model call failed for PMID %s: %s", job.pmid, exc)
        return False

    await persist_results(job, updates, existing, writer_lock, logger)
    return True


async def run_jobs(
    jobs: List[UpdateJob],
    max_concurrency: int,
    existing: Dict[str, set[int]],
    guidelines: str,
    logger: logging.Logger,
) -> None:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    rate_limiter = TokenBucket(RATE_LIMIT_TOKENS_PER_MIN)
    writer_lock = asyncio.Lock()
    client = AsyncOpenAI()

    tasks = [
        asyncio.create_task(
            process_job(job, client, semaphore, rate_limiter, writer_lock, existing, guidelines, logger)
        )
        for job in jobs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = sum(1 for r in results if r is True)
    failures = len(results) - successes
    logger.info("Completed %d jobs with %d failures.", successes, failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce Nov 5 rules on human answers.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many PMIDs (after skipping completed ones).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum concurrent API calls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    logger = configure_logger()

    if not QUESTIONS_PATH.exists():
        logger.error("Questions file missing: %s", QUESTIONS_PATH)
        return 1

    guidelines = load_guidelines(GUIDELINES_PATH)
    question_map = load_questions(QUESTIONS_PATH)
    existing = load_existing_pairs(OUTPUT_CSV)
    token_counter = build_token_counter(MODEL_NAME)
    jobs = prepare_jobs(question_map, existing, guidelines, token_counter, args.limit, logger)

    if not jobs:
        logger.info("No PMIDs to process. Running cleanup pass instead.")
        cleanup_updated_answers(OUTPUT_CSV, logger)
        return 0

    prepare_output(OUTPUT_CSV)
    RAW_JSONL.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_jobs(jobs, args.max_concurrency, existing, guidelines, logger))
    cleanup_updated_answers(OUTPUT_CSV, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Call GPT-4o mini for each PubMed prompt using a rate-limited worker pool."""

from __future__ import annotations

import asyncio
import os
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

from openai import AsyncOpenAI
from dotenv import load_dotenv

import tiktoken


MODEL_NAME = "gpt-4o-mini-2024-07-18"
RATE_LIMIT_TOKENS_PER_MINUTE = 180_000_000
DEFAULT_COMPLETION_BUDGET = 2_000
MAX_WORKERS = int(os.environ.get("PMID_MAX_WORKERS", "8"))
EXPECTED_ANSWER_COUNT = 16
MIN_RESPONSE_TOKENS = int(os.environ.get("PMID_MIN_RESPONSE_TOKENS", "200"))
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
ANSWER_PATTERN = re.compile(r"Answer:\s*", re.IGNORECASE)


@dataclass(frozen=True)
class JobConfig:
    label: str
    prompts_path: Path
    responses_path: Path
    log_path: Path


JOB_CONFIGS: tuple[JobConfig, ...] = (
    JobConfig(
        label="bm25-5shot",
        prompts_path=Path("advanced-prompting/jsonl/dynamic_prompts_bm25_5-shot.jsonl"),
        responses_path=Path("advanced-prompting/jsonl/dynamic_responses_bm25_5-shot.jsonl"),
        log_path=Path("advanced-prompting/log/dynamic_responses_bm25_5-shot.log"),
    ),
    JobConfig(
        label="bm25-10shot",
        prompts_path=Path("advanced-prompting/jsonl/dynamic_prompts_bm25_10-shot.jsonl"),
        responses_path=Path("advanced-prompting/jsonl/dynamic_responses_bm25_10-shot.jsonl"),
        log_path=Path("advanced-prompting/log/dynamic_responses_bm25_10-shot.log"),
    ),
)


@dataclass(frozen=True)
class PromptJob:
    pmid: str
    prompt: str
    tokens: int


@dataclass
class Progress:
    total: int
    completed: int = 0


def setup_logger(log_path: Path, label: str) -> logging.Logger:
    logger_name = f"pmid_runner_{label}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(f"%(asctime)s %(levelname)s [{label}] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_processed(path: Path) -> tuple[set[str], set[str]]:
    processed: set[str] = set()
    retry: set[str] = set()
    failed_seen: set[str] = set()
    if not path.exists():
        return processed, retry
    with path.open("r", encoding="utf-8") as infile:
        lines = infile.readlines()

    kept_lines: list[str] = []
    dropped = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        pmid = data.get("pmid")
        if not pmid:
            continue
        pmid_str = str(pmid)
        response_text = data.get("response")
        failed = not response_text or is_failed_response(response_text)
        if failed:
            retry.add(pmid_str)
            if pmid_str in failed_seen:
                dropped += 1
                continue
            failed_seen.add(pmid_str)
        else:
            processed.add(pmid_str)
        kept_lines.append(line)

    if dropped:
        with path.open("w", encoding="utf-8") as outfile:
            outfile.writelines(kept_lines)

    return processed, retry


def estimate_tokens(text: str, model_name: str = MODEL_NAME) -> int:
    """Return an estimated token count for the provided text."""
    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    # Heuristic fallback: assume 4 characters per token.
    return max(1, len(text) // 4)


def is_failed_response(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if stripped in FAILED_RESPONSES:
        return True
    lower = stripped.lower()
    if any(keyword in lower for keyword in FAILURE_SUBSTRINGS):
        return True
    answer_count = len(ANSWER_PATTERN.findall(stripped))
    if answer_count < EXPECTED_ANSWER_COUNT:
        return True
    if estimate_tokens(stripped) < MIN_RESPONSE_TOKENS:
        return True
    return False


def load_pending_jobs(
    prompts_path: Path, processed_pmids: Iterable[str], logger: logging.Logger
) -> list[PromptJob]:
    processed_set = set(processed_pmids)
    jobs: list[PromptJob] = []
    queued_pmids: set[str] = set()
    with prompts_path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON on line %d", line_no)
                continue
            pmid = str(record.get("pmid"))
            if not pmid:
                logger.warning("Missing PMID on line %d", line_no)
                continue
            if pmid in processed_set:
                continue
            if pmid in queued_pmids:
                logger.warning(
                    "Duplicate prompt for PMID %s on line %d; skipping", pmid, line_no
                )
                continue
            prompt = record.get("prompt")
            if not isinstance(prompt, str):
                logger.warning("Missing prompt text for PMID %s", pmid)
                continue
            prompt_tokens = estimate_tokens(prompt)
            total_budget = prompt_tokens + DEFAULT_COMPLETION_BUDGET
            jobs.append(PromptJob(pmid=pmid, prompt=prompt, tokens=total_budget))
            queued_pmids.add(pmid)
    logger.info(
        "Prepared %d pending PMIDs (estimated %d tokens)",
        len(jobs),
        sum(job.tokens for job in jobs),
    )
    return jobs


class TokenBucket:
    """Simple async-aware token bucket to enforce token-per-minute limits."""

    def __init__(self, tokens_per_minute: int):
        self.capacity = float(tokens_per_minute)
        self.tokens = float(tokens_per_minute)
        self.rate_per_second = float(tokens_per_minute) / 60.0
        self.updated_at = time.monotonic()
        self.lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.updated_at
        if elapsed <= 0:
            return
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_second)
        self.updated_at = now

    async def consume(self, amount: int) -> None:
        requested = float(amount)
        if requested <= 0:
            return
        while True:
            async with self.lock:
                if requested > self.capacity:
                    self.capacity = requested
                self._refill()
                if self.tokens >= requested:
                    self.tokens -= requested
                    return
                deficit = requested - self.tokens
            wait_time = deficit / self.rate_per_second if self.rate_per_second else 0.0
            await asyncio.sleep(wait_time)


async def worker(
    worker_id: int,
    queue: "asyncio.Queue[PromptJob | None]",
    client: AsyncOpenAI,
    responses_file,
    token_bucket: TokenBucket,
    write_lock: asyncio.Lock,
    progress: Progress,
    logger: logging.Logger,
) -> None:
    while True:
        job = await queue.get()
        if job is None:
            queue.task_done()
            logger.info("Worker %d exiting", worker_id)
            return

        await token_bucket.consume(job.tokens)
        logger.info(
            "Worker %d requesting PMID %s (token budget %d)",
            worker_id,
            job.pmid,
            job.tokens,
        )

        try:
            response = await client.responses.create(model=MODEL_NAME, input=job.prompt)
            output_text = response.output_text
        except Exception as exc:  # pragma: no cover - network failures
            logger.exception(
                "Worker %d API call failed for PMID %s: %s", worker_id, job.pmid, exc
            )
            queue.task_done()
            continue

        async with write_lock:
            response_record = {"pmid": job.pmid, "response": output_text}
            responses_file.write(json.dumps(response_record, ensure_ascii=False) + "\n")
            responses_file.flush()
            progress.completed += 1
            current = progress.completed
            total = progress.total
        logger.info(
            "Worker %d stored PMID %s (%d/%d complete)",
            worker_id,
            job.pmid,
            current,
            total,
        )
        queue.task_done()


async def process_jobs(
    jobs: list[PromptJob],
    responses_file,
    logger: logging.Logger,
    api_key: str,
) -> None:
    if not jobs:
        logger.info("No new PMIDs to process.")
        return

    client = AsyncOpenAI(api_key=api_key)
    worker_count = max(1, min(MAX_WORKERS, len(jobs)))
    queue: asyncio.Queue[PromptJob | None] = asyncio.Queue()
    for job in jobs:
        await queue.put(job)
    for _ in range(worker_count):
        await queue.put(None)

    token_bucket = TokenBucket(RATE_LIMIT_TOKENS_PER_MINUTE)
    write_lock = asyncio.Lock()
    progress = Progress(total=len(jobs))

    logger.info(
        "Starting %d workers with rate limit %.1fM tokens/min",
        worker_count,
        RATE_LIMIT_TOKENS_PER_MINUTE / 1_000_000,
    )

    tasks = [
        asyncio.create_task(
            worker(
                worker_id=i + 1,
                queue=queue,
                client=client,
                responses_file=responses_file,
                token_bucket=token_bucket,
                write_lock=write_lock,
                progress=progress,
                logger=logger,
            )
        )
        for i in range(worker_count)
    ]

    await queue.join()
    for task in tasks:
        await task
    logger.info("Completed %d/%d PMIDs", progress.completed, progress.total)


def run_job_config(config: JobConfig, api_key: str) -> int:
    logger = setup_logger(config.log_path, config.label)

    if not config.prompts_path.exists():
        logger.error("Prompts file not found at %s", config.prompts_path)
        return 1

    processed_pmids, retry_pmids = load_processed(config.responses_path)
    logger.info(
        "Loaded %d processed PMIDs (%d flagged for retry)",
        len(processed_pmids),
        len(retry_pmids),
    )

    pending_jobs = load_pending_jobs(config.prompts_path, processed_pmids, logger)
    if retry_pmids:
        pending_pmids = {job.pmid for job in pending_jobs}
        missing_retries = retry_pmids - pending_pmids
        if missing_retries:
            logger.warning(
                "Retry PMIDs missing from prompts and will be skipped: %s",
                ", ".join(sorted(missing_retries)),
            )
        else:
            logger.info(
                "All %d retry PMIDs queued for reprocessing.", len(retry_pmids)
            )
    if not pending_jobs:
        logger.info("No new prompts to process.")
        return 0

    with config.responses_path.open("a", encoding="utf-8") as responses_file:
        asyncio.run(process_jobs(pending_jobs, responses_file, logger, api_key))

    logger.info("All pending PMIDs processed.")
    return 0


def main() -> int:
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    exit_code = 0
    for config in JOB_CONFIGS:
        exit_code = max(exit_code, run_job_config(config, api_key))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

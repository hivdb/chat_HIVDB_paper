#!/usr/bin/env python3
"""Parallel GPT-5-mini evaluation with token-aware rate limiting and resume support."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, create_model


MODEL_NAME = "gpt-5-mini-2025-08-07"
QUESTIONS_PATH = Path("advanced-prompting/merged_answers.xlsx")
PAPERS_DIR = Path("advanced-prompting/papers")
PROMPT_PATH = Path("eval/gpt-5-mini-prompt.md")
OUTPUT_CSV = Path("eval/gpt5_responses.csv")
RAW_JSONL = Path("eval/gpt5_responses.jsonl")
TOTAL_QUESTIONS = 16
TOKEN_BUFFER = 200
RATE_LIMIT_TOKENS_PER_MIN = 180_000_000
CSV_FIELDS = ["PMID", "QID", "Question", "Answer", "Evidence", "Rationale"]


class QAEntry(BaseModel):
    Question: str
    Answer: str
    Evidence: str
    Rationale: str


def build_parsed_response_model(total_questions: int) -> type[BaseModel]:
    """Construct a schema with aliased Question fields without manual repetition."""

    field_definitions = {
        f"question_{idx}": (QAEntry, Field(alias=f"Question {idx}"))
        for idx in range(1, total_questions + 1)
    }
    model = create_model("ParsedResponse", **field_definitions)

    def as_dict(self) -> Dict[str, QAEntry]:
        return {
            f"Question {idx}": getattr(self, f"question_{idx}")
            for idx in range(1, total_questions + 1)
        }

    setattr(model, "as_dict", as_dict)
    return model


ParsedResponse = build_parsed_response_model(TOTAL_QUESTIONS)
_SYSTEM_PROMPT: str | None = None


def normalize_pmid(value: str) -> str:
    stripped = value.strip()
    try:
        as_float = float(stripped)
        as_int = int(as_float)
        if as_float == float(as_int):
            return str(as_int)
    except ValueError:
        pass
    return stripped


@dataclass(frozen=True)
class Question:
    pmid: str
    qid: int
    question_num: int
    text: str


@dataclass(frozen=True)
class PromptJob:
    pmid: str
    questions: List[Question]
    user_message: str
    token_estimate: int


class TokenBucket:
    """Simple token bucket to enforce a tokens-per-minute ceiling."""

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
    return logging.getLogger("gpt5_eval")


def load_system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        if not PROMPT_PATH.exists():
            raise FileNotFoundError(f"Prompt file missing: {PROMPT_PATH}")
        _SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()
    return _SYSTEM_PROMPT


def load_question_table(path: Path) -> Dict[str, List[Question]]:
    df = pd.read_excel(path)
    df = df.dropna(subset=["PMID"])
    df["PMID"] = df["PMID"].astype(str).map(normalize_pmid)
    df = df[df["PMID"].str.lower() != "nan"]
    questions: Dict[str, List[Question]] = {}

    for pmid, group in df.groupby("PMID"):
        ordered = group.sort_values("QID")
        pmid_questions: List[Question] = []
        question_count = 0
        for row in ordered.itertuples(index=False):
            if pd.isna(row.QID):
                continue
            question_count += 1
            pmid_questions.append(
                Question(
                    pmid=pmid,
                    qid=int(row.QID),
                    question_num=question_count,
                    text=str(row.Question).strip(),
                )
            )
        questions[pmid] = pmid_questions

    return questions


def read_paper_text(pmid: str) -> str:
    pmid_dir = PAPERS_DIR / pmid
    paper_path = pmid_dir / f"{pmid}.checked.md"
    if not paper_path.exists():
        raise FileNotFoundError(f"Missing markdown for PMID {pmid}: {paper_path}")
    return paper_path.read_text(encoding="utf-8")


def build_question_block(questions: Iterable[Question]) -> str:
    lines = []
    for question in questions:
        lines.append(
            f"Question {question.question_num} (QID {question.qid}): {question.text}"
        )
    return "\n".join(lines)


def build_token_counter(model_name: str) -> Callable[[str], int]:
    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        def counter(text: str) -> int:
            return len(encoding.encode(text))

        return counter

    logging.warning(
        "tiktoken not found; falling back to approximate token counting."
    )

    def fallback_counter(text: str) -> int:
        # Rough heuristic: assume 4 characters per token on average.
        return max(len(text) // 4, len(text.split()))

    return fallback_counter


def load_existing_answers(path: Path) -> Dict[str, set[int]]:
    answers: Dict[str, set[int]] = defaultdict(set)
    if not path.exists():
        return answers

    df = pd.read_csv(path, usecols=["PMID", "QID"])
    for row in df.itertuples(index=False):
        answers[str(row.PMID)].add(int(row.QID))
    return answers


def prepare_output_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()


def prepare_jobs(
    question_table: Dict[str, List[Question]],
    existing_answers: Dict[str, set[int]],
    token_counter: Callable[[str], int],
    limit: int | None,
    logger: logging.Logger,
) -> List[PromptJob]:
    jobs: List[PromptJob] = []
    system_prompt = load_system_prompt()
    system_tokens = token_counter(system_prompt)

    for pmid_index, (pmid, questions) in enumerate(question_table.items(), start=1):
        if limit is not None and len(jobs) >= limit:
            break

        completed = existing_answers.get(pmid, set())
        if len(completed) >= TOTAL_QUESTIONS:
            logger.info("Skipping PMID %s; already complete.", pmid)
            continue

        if len(questions) != TOTAL_QUESTIONS:
            logger.warning(
                "PMID %s has %d questions; expected %d.",
                pmid,
                len(questions),
                TOTAL_QUESTIONS,
            )

        try:
            paper_text = read_paper_text(pmid)
        except FileNotFoundError as exc:
            logger.error(str(exc))
            continue

        question_block = build_question_block(questions)
        user_message = (
            f"PMID: {pmid}\n\n"
            f"Questions:\n{question_block}\n\n"
            "Paper Content:\n"
            f"{paper_text}"
        )
        user_tokens = token_counter(user_message)
        token_estimate = system_tokens + user_tokens + TOKEN_BUFFER
        jobs.append(
            PromptJob(
                pmid=pmid,
                questions=questions,
                user_message=user_message,
                token_estimate=token_estimate,
            )
        )
        logger.info("Prepared PMID %s (~%d tokens)", pmid, token_estimate)

    return jobs


async def call_model_async(client: AsyncOpenAI, user_message: str) -> Dict[str, QAEntry]:
    system_prompt = load_system_prompt()
    response = await client.responses.parse(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        text_format=ParsedResponse,
    )
    return response.output_parsed.as_dict()


def append_rows(rows: List[dict]) -> None:
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writerows(rows)


def append_raw_response(pmid: str, payload: Dict[str, QAEntry]) -> None:
    RAW_JSONL.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value.model_dump()
        for key, value in payload.items()
    }
    record = {"pmid": pmid, "response": serializable}
    with RAW_JSONL.open("a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


async def persist_results(
    job: PromptJob,
    parsed_answers: Dict[str, QAEntry],
    existing_answers: Dict[str, set[int]],
    writer_lock: asyncio.Lock,
    logger: logging.Logger,
) -> None:
    rows: List[dict] = []
    missing = False

    for question in job.questions:
        key = f"Question {question.question_num}"
        qa_entry = parsed_answers.get(key)
        if qa_entry is None:
            logger.warning("Missing entry '%s' in response for PMID %s", key, job.pmid)
            missing = True
            continue
        rows.append(
            {
                "PMID": question.pmid,
                "QID": question.qid,
                "Question": question.text,
                "Answer": qa_entry.Answer.strip(),
                "Evidence": qa_entry.Evidence.strip(),
                "Rationale": qa_entry.Rationale.strip(),
            }
        )

    await asyncio.to_thread(append_raw_response, job.pmid, parsed_answers)

    if missing:
        logger.warning("PMID %s response was incomplete.", job.pmid)

    async with writer_lock:
        known_qids = existing_answers.setdefault(job.pmid, set())
        new_rows = [row for row in rows if row["QID"] not in known_qids]
        if not new_rows:
            logger.info("No new rows to write for PMID %s.", job.pmid)
            return

        await asyncio.to_thread(append_rows, new_rows)
        known_qids.update(row["QID"] for row in new_rows)
        logger.info("Persisted %d answers for PMID %s.", len(new_rows), job.pmid)


async def process_job(
    job: PromptJob,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    rate_limiter: TokenBucket,
    writer_lock: asyncio.Lock,
    existing_answers: Dict[str, set[int]],
    logger: logging.Logger,
) -> bool:
    try:
        await rate_limiter.acquire(job.token_estimate)
        async with semaphore:
            parsed_answers = await call_model_async(client, job.user_message)
    except Exception as exc:  # noqa: BLE001 - want to log any failure
        logger.exception("Model call failed for PMID %s: %s", job.pmid, exc)
        return False

    await persist_results(job, parsed_answers, existing_answers, writer_lock, logger)
    return True


async def run_jobs(
    jobs: List[PromptJob],
    max_concurrency: int,
    existing_answers: Dict[str, set[int]],
    logger: logging.Logger,
) -> None:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    rate_limiter = TokenBucket(RATE_LIMIT_TOKENS_PER_MIN)
    writer_lock = asyncio.Lock()
    client = AsyncOpenAI()

    tasks = [
        asyncio.create_task(
            process_job(job, client, semaphore, rate_limiter, writer_lock, existing_answers, logger)
        )
        for job in jobs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = 0
    failures = 0
    for result in results:
        if result is True:
            successes += 1
        else:
            failures += 1

    logger.info("Completed %d jobs with %d failures.", successes, failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-5-mini evaluation across PMIDs.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of PMIDs to process (applies to new PMIDs only).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent API calls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    logger = configure_logger()

    if not QUESTIONS_PATH.exists():
        logger.error("Question file missing: %s", QUESTIONS_PATH)
        return 1

    try:
        load_system_prompt()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    question_table = load_question_table(QUESTIONS_PATH)
    logger.info("Loaded %d PMIDs from %s.", len(question_table), QUESTIONS_PATH)

    existing_answers = load_existing_answers(OUTPUT_CSV)
    token_counter = build_token_counter(MODEL_NAME)
    jobs = prepare_jobs(question_table, existing_answers, token_counter, args.limit, logger)

    if not jobs:
        logger.info("No PMIDs to process. Exiting.")
        return 0

    prepare_output_csv(OUTPUT_CSV)
    RAW_JSONL.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_jobs(jobs, args.max_concurrency, existing_answers, logger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

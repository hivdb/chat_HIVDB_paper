#!/usr/bin/env python3
"""Standalone evaluation runner for fine-tuned HIVDB models."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import tiktoken
from dotenv import load_dotenv, dotenv_values
from openai import AsyncOpenAI, OpenAI, OpenAIError
from pydantic import BaseModel, Field, create_model


DEFAULT_OUTPUT_DIR = Path("eval/learning-curve/responses")
QUESTIONS_PATH_DEFAULT = Path("advanced-prompting/csv/merged_answers.xlsx")
PAPERS_DIR_DEFAULT = Path("advanced-prompting/papers")
TRAIN_FILE_DEFAULT = Path("advanced-prompting/train_val/train_set.jsonl")
TOTAL_QUESTIONS = 16
TOKEN_BUFFER = 200
CSV_FIELDS = ["PMID", "QID", "Question", "Answer", "Evidence", "Rationale"]
FAILED_STATUSES = {"failed", "cancelled", "rejected"}
MAX_FIELD_LENGTH = 2000
BREVITY_NOTE = (
    "Keep every answer concise. For each question, provide only the evidence sentences, a short rationale,"
    " and a short answer."
    "Avoid extra explanations or formatting outside the required fields."
)


class QAEntry(BaseModel):
    Question: str
    Answer: str
    Evidence: str
    Rationale: str


def build_parsed_response_model(total_questions: int) -> type[BaseModel]:
    fields = {
        f"question_{idx}": (QAEntry, Field(alias=f"Question {idx}"))
        for idx in range(1, total_questions + 1)
    }
    model = create_model("LearningCurveParsedResponse", **fields)

    def as_dict(self) -> Dict[str, QAEntry]:
        return {
            f"Question {idx}": getattr(self, f"question_{idx}")
            for idx in range(1, total_questions + 1)
        }

    setattr(model, "as_dict", as_dict)
    return model


ParsedResponse = build_parsed_response_model(TOTAL_QUESTIONS)


def load_env() -> None:
    load_dotenv()
    for env_path in (".env", "eval/.env"):
        path = Path(env_path)
        if not path.exists():
            continue
        for key, value in dotenv_values(path).items():
            if value is not None:
                os.environ.setdefault(key, value)


def load_system_prompt_from_training(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Training file missing: {path}")
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            messages = record.get("messages", [])
            for message in messages:
                if message.get("role") == "system":
                    content = message.get("content", "").strip()
                    if content:
                        return content
    raise RuntimeError(f"Could not find a system prompt in {path}")


def build_system_prompt(path: Path) -> str:
    base_prompt = load_system_prompt_from_training(path)
    return f"{base_prompt}\n\n{BREVITY_NOTE}"


def configure_logger(tag: str) -> logging.Logger:
    logger = logging.getLogger(f"learning_curve_{tag}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def normalize_pmid(value: str) -> str:
    stripped = str(value).strip()
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
                now = asyncio.get_running_loop().time()
                self._prune(now)
                if self.current + tokens <= self.limit:
                    self.events.append((now, tokens))
                    self.current += tokens
                    return
                wait_time = (
                    max(60 - (now - self.events[0][0]), 0.1) if self.events else 1.0
                )
            await asyncio.sleep(wait_time)


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


def read_paper_text(pmid: str, papers_dir: Path) -> str:
    pmid_dir = papers_dir / pmid
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


def build_token_counter(model_name: str):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    def counter(text: str) -> int:
        return len(encoding.encode(text))

    return counter


def load_existing_answers(path: Path) -> Dict[str, set[int]]:
    answers: Dict[str, set[int]] = defaultdict(set)
    if not path.exists():
        return answers
    df = pd.read_csv(path, usecols=["PMID", "QID"], dtype={"PMID": str})
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


def append_rows(path: Path, rows: List[dict]) -> None:
    with path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writerows(rows)


def clamp_field(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_FIELD_LENGTH:
        return text
    return text[:MAX_FIELD_LENGTH]


def append_raw_response(path: Path, pmid: str, payload: Dict[str, QAEntry]) -> None:
    serializable = {key: value.model_dump() for key, value in payload.items()}
    record = {"pmid": pmid, "response": serializable}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


def rebuild_csv_from_raw(csv_path: Path, raw_path: Path, question_table: Dict[str, List[Question]]) -> None:
    if not raw_path.exists():
        return
    rows: List[dict] = []
    with raw_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            pmid = str(record.get("pmid", "")).strip()
            responses = record.get("response", {})
            questions = question_table.get(pmid, [])
            for question in questions:
                payload = responses.get(f"Question {question.question_num}")
                if not payload:
                    continue
                rows.append(
                    {
                        "PMID": question.pmid,
                        "QID": question.qid,
                        "Question": question.text,
                        "Answer": payload.get("Answer", "").strip(),
                        "Evidence": payload.get("Evidence", "").strip(),
                        "Rationale": payload.get("Rationale", "").strip(),
                    }
                )
    if not rows:
        return
    rows.sort(key=lambda row: (row["PMID"], row["QID"]))
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_failures(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    failures: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            pmid = record.get("pmid")
            if not pmid:
                continue
            failures[str(pmid)] = record
    return failures


def save_failures(path: Path, failures: Dict[str, dict]) -> None:
    if not failures:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        for pmid, info in sorted(failures.items()):
            record = {"pmid": pmid, **info}
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


async def mark_failure(
    failures: Dict[str, dict],
    lock: asyncio.Lock,
    path: Path,
    pmid: str,
    error: str,
) -> None:
    async with lock:
        failures[pmid] = {"error": error, "timestamp": time.time()}
        await asyncio.to_thread(save_failures, path, failures)


async def clear_failure(
    failures: Dict[str, dict],
    lock: asyncio.Lock,
    path: Path,
    pmid: str,
) -> None:
    async with lock:
        if pmid in failures:
            del failures[pmid]
            await asyncio.to_thread(save_failures, path, failures)


def prepare_jobs(
    question_table: Dict[str, List[Question]],
    existing_answers: Dict[str, set[int]],
    token_counter,
    system_prompt: str,
    papers_dir: Path,
    limit: int | None,
    logger: logging.Logger,
    retry_pmids: set[str] | None = None,
) -> List[PromptJob]:
    jobs: List[PromptJob] = []
    system_tokens = token_counter(system_prompt)

    for _, (pmid, questions) in enumerate(question_table.items(), start=1):
        if limit is not None and len(jobs) >= limit:
            break
        completed = existing_answers.get(pmid, set())
        needs_retry = retry_pmids is not None and pmid in retry_pmids
        if len(completed) >= TOTAL_QUESTIONS and not needs_retry:
            logger.info("Skipping PMID %s; already complete.", pmid)
            continue
        try:
            paper_text = read_paper_text(pmid, papers_dir)
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


async def call_model_async(client: AsyncOpenAI, model: str, system_prompt: str, user_message: str) -> Dict[str, QAEntry]:
    response = await client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        text_format=ParsedResponse,
    )
    if response.output_parsed is None:
        preview = response.output or []
        raise ValueError(f"Model response could not be parsed: {preview!r}")
    return response.output_parsed.as_dict()


async def persist_results(
    job: PromptJob,
    parsed_answers: Dict[str, QAEntry],
    existing_answers: Dict[str, set[int]],
    writer_lock: asyncio.Lock,
    csv_path: Path,
    raw_path: Path,
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
                "Answer": clamp_field(qa_entry.Answer),
                "Evidence": clamp_field(qa_entry.Evidence),
                "Rationale": clamp_field(qa_entry.Rationale),
            }
        )
    await asyncio.to_thread(append_raw_response, raw_path, job.pmid, parsed_answers)
    if missing:
        logger.warning("PMID %s response was incomplete.", job.pmid)
    async with writer_lock:
        known_qids = existing_answers.setdefault(job.pmid, set())
        new_rows = [row for row in rows if row["QID"] not in known_qids]
        if not new_rows:
            logger.info("No new rows to write for PMID %s.", job.pmid)
            return
        await asyncio.to_thread(append_rows, csv_path, new_rows)
        known_qids.update(row["QID"] for row in new_rows)
        logger.info("Persisted %d answers for PMID %s.", len(new_rows), job.pmid)


async def process_job(
    job: PromptJob,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    rate_limiter: TokenBucket,
    writer_lock: asyncio.Lock,
    existing_answers: Dict[str, set[int]],
    csv_path: Path,
    raw_path: Path,
    system_prompt: str,
    model: str,
    logger: logging.Logger,
    failure_records: Dict[str, dict],
    failure_lock: asyncio.Lock,
    failure_path: Path,
) -> bool:
    try:
        await rate_limiter.acquire(job.token_estimate)
        async with semaphore:
            parsed_answers = await call_model_async(client, model, system_prompt, job.user_message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Model call failed for PMID %s: %s", job.pmid, exc)
        await mark_failure(failure_records, failure_lock, failure_path, job.pmid, str(exc))
        return False
    await persist_results(job, parsed_answers, existing_answers, writer_lock, csv_path, raw_path, logger)
    await clear_failure(failure_records, failure_lock, failure_path, job.pmid)
    return True


async def run_jobs(
    jobs: List[PromptJob],
    model: str,
    system_prompt: str,
    existing_answers: Dict[str, set[int]],
    csv_path: Path,
    raw_path: Path,
    max_concurrency: int,
    logger: logging.Logger,
    failure_records: Dict[str, dict],
    failure_path: Path,
) -> None:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    rate_limiter = TokenBucket(180_000_000)
    writer_lock = asyncio.Lock()
    failure_lock = asyncio.Lock()
    client = AsyncOpenAI()
    tasks = [
        asyncio.create_task(
            process_job(
                job,
                client,
                semaphore,
                rate_limiter,
                writer_lock,
                existing_answers,
                csv_path,
                raw_path,
                system_prompt,
                model,
                logger,
                failure_records,
                failure_lock,
                failure_path,
            )
        )
        for job in jobs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    successes = sum(1 for result in results if result is True)
    failures = len(results) - successes
    logger.info("Completed %d jobs with %d failures.", successes, failures)


def read_finetune_jobs(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Jobs file missing: {path}")
    jobs: List[dict] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def is_failed_status(status: str | None) -> bool:
    if not status:
        return False
    return status.lower() in FAILED_STATUSES


def sanitize_tag(value: str) -> str:
    text = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return cleaned or "run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Fine-tuned model name (ft:...) for single-run mode.")
    parser.add_argument("--tag", type=str, default=None, help="Output tag for single-run mode.")
    parser.add_argument(
        "--jobs-file",
        type=Path,
        default=Path("eval/learning-curve/finetune_jobs.jsonl"),
        help="JSONL file containing fine-tune metadata (enables batch mode).",
    )
    parser.add_argument(
        "--tag-template",
        type=str,
        default="size{size:03d}",
        help="Format string for tags when using --jobs-file.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=TRAIN_FILE_DEFAULT,
        help="Training file containing messages with the canonical system prompt.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=QUESTIONS_PATH_DEFAULT,
        help="Questions workbook path.",
    )
    parser.add_argument(
        "--papers",
        type=Path,
        default=PAPERS_DIR_DEFAULT,
        help="Directory containing paper markdown files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where response CSV/JSONL files are stored.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on PMIDs.")
    parser.add_argument("--max-concurrency", type=int, default=10, help="Concurrent API calls.")
    return parser.parse_args()


def run_single(model: str, tag: str, args: argparse.Namespace) -> None:
    logger = configure_logger(tag)
    system_prompt = build_system_prompt(args.train_file)
    question_table = load_question_table(args.questions)
    logger.info("Loaded %d PMIDs from %s", len(question_table), args.questions)
    token_counter = build_token_counter(model)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{tag}_responses.csv"
    raw_jsonl = output_dir / f"{tag}_raw.jsonl"
    failure_path = output_dir / f"{tag}_failures.jsonl"
    rebuild_csv_from_raw(output_csv, raw_jsonl, question_table)
    prepare_output_csv(output_csv)
    existing_answers = load_existing_answers(output_csv)
    failure_records = load_failures(failure_path)
    retry_pmids = set(failure_records.keys()) or None
    jobs = prepare_jobs(
        question_table,
        existing_answers,
        token_counter,
        system_prompt,
        args.papers,
        args.limit,
        logger,
        retry_pmids=retry_pmids,
    )
    if not jobs:
        logger.info("No PMIDs to process for tag %s", tag)
        return
    asyncio.run(
        run_jobs(
            jobs,
            model,
            system_prompt,
            existing_answers,
            output_csv,
            raw_jsonl,
            args.max_concurrency,
            logger,
            failure_records,
            failure_path,
        )
    )
    save_failures(failure_path, failure_records)


def main() -> int:
    args = parse_args()
    load_env()
    if args.model:
        tag = args.tag or sanitize_tag(args.model)
        run_single(args.model, tag, args)
        return 0

    jobs_file = args.jobs_file
    try:
        jobs = read_finetune_jobs(jobs_file)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Jobs file missing: {jobs_file}. Provide --model for single-run mode or --jobs-file to override."
        ) from exc
    if not jobs:
        raise SystemExit(f"No jobs found in {jobs_file}")
    client = OpenAI()
    for job in jobs:
        try:
            job_id = job.get("job_id")
            if not job_id:
                raise RuntimeError("Job id missing.")
            status = job.get("status")
            if is_failed_status(status):
                print(f"Skipping job {job_id}: recorded status={status}")
                continue

            model_name = job.get("result_model")
            if not model_name:
                info = client.fine_tuning.jobs.retrieve(job_id)
                api_status = getattr(info, "status", None)
                if is_failed_status(api_status):
                    print(f"Skipping job {job_id}: API status={api_status}")
                    continue
                model_name = info.fine_tuned_model
            if not model_name:
                raise RuntimeError("Fine-tuned model not ready.")
        except (OpenAIError, RuntimeError) as exc:
            print(f"Skipping job {job_id}: {exc}")
            continue
        tag = args.tag_template.format(size=job.get("size", "run"))
        run_single(model_name, tag, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

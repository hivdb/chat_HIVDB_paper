#!/usr/bin/env python3
"""Parallel GPT-5 judge for GPT-5-mini answers."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, Field


MERGED_PATH = Path("advanced-prompting/merged_answers.xlsx")
UPDATED_PATH = Path("eval/updated_human_answers.csv")
GPT5_MINI_PATH = Path("eval/gpt5_responses.csv")
OUTPUT_CSV = Path("eval/judge_gpt5_results.csv")
SUMMARY_CSV = Path("eval/judge_gpt5_summary.csv")

MODEL_NAME = "gpt-5"
MAX_TOKENS_PER_CALL = 250_000
TOKEN_BUFFER = 2_000
DEFAULT_MAX_RECORDS = 25
DEFAULT_CONCURRENCY = 15
MAX_RETRIES = 5

SYSTEM_PROMPT = (
    "You are an HIV literature QA adjudicator. For each record decide whether the "
    "GPT-5-mini answer matches (a) the original human annotation and (b) the updated "
    "human annotation.\n"
    "- Treat paraphrases, reordered lists, and common HIV drug abbreviations "
    "(AZT↔zidovudine, TFV/TDF↔tenofovir, FTC↔emtricitabine, 3TC↔lamivudine, LPV↔lopinavir, "
    "DTG↔dolutegravir, RAL↔raltegravir, EVG↔elvitegravir, etc.) as equivalent.\n"
    "- Ignore casing, punctuation, and explanatory prose.\n"
    "- If a human answer says 'Not reported / Not applicable / None', only mark the model "
    "correct when it clearly conveys that the information is unavailable.\n"
    "- Flag contradictions, hallucinations, or missing critical details as incorrect.\n"
    "Return JSON with an 'items' list; each item must have 'sample_id', 'human_correct' "
    "(bool), 'updated_correct' (bool), and a short 'explanation'."
)

USER_TEMPLATE = "Records:\n{records}"


class JudgedItem(BaseModel):
    sample_id: str
    human_correct: bool
    updated_correct: bool
    explanation: str


class JudgementResponse(BaseModel):
    items: List[JudgedItem]


@dataclass(frozen=True)
class Record:
    sample_id: str
    pmid: str
    qid: int
    question: str
    human_answer: str
    updated_answer: str
    model_answer: str


def configure_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("judge_gpt5")


def token_counter_factory(model_name: str):
    if tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        def counter(text: str) -> int:
            return len(encoding.encode(text))

        return counter

    logging.warning("tiktoken not installed; using heuristic token counter.")

    def fallback(text: str) -> int:
        return max(len(text) // 4, len(text.split()))

    return fallback


def load_records(limit: int | None = None) -> List[Record]:
    merged = pd.read_excel(MERGED_PATH)
    merged["PMID"] = merged["PMID"].astype(str)

    gpt5 = pd.read_csv(GPT5_MINI_PATH, dtype={"PMID": str}).rename(columns={"Answer": "model_answer"})

    updated = pd.read_csv(UPDATED_PATH, dtype={"PMID": str})
    updated = (
        updated.sort_values(["PMID", "QID"])
        .drop_duplicates(subset=["PMID", "QID"], keep="last")
        .rename(columns={"Updated Human Answer": "updated_answer"})
    )

    df = (
        merged.merge(gpt5[["PMID", "QID", "model_answer"]], on=["PMID", "QID"], how="left")
        .merge(updated[["PMID", "QID", "updated_answer"]], on=["PMID", "QID"], how="left")
    )
    df = df[df["model_answer"].notna()]
    if limit is not None:
        df = df.head(limit)

    records: List[Record] = []
    for row in df.itertuples(index=False):
        sample_id = f"{row.PMID}-{int(row.QID)}"
        records.append(
            Record(
                sample_id=sample_id,
                pmid=row.PMID,
                qid=int(row.QID),
                question=str(row.Question).strip(),
                human_answer=str(row._5).strip() if row._5 is not None else "",
                updated_answer=str(row.updated_answer).strip()
                if row.updated_answer is not None
                else "",
                model_answer=str(row.model_answer).strip(),
            )
        )
    return records


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["sample_id"])
    return set(df["sample_id"].astype(str))


def chunk_records(
    records: Iterable[Record],
    token_counter,
    max_tokens: int,
    base_tokens: int,
    max_records: int,
) -> List[List[Record]]:
    chunks: List[List[Record]] = []
    current: List[Record] = []
    current_tokens = base_tokens

    for record in records:
        record_dict = {
            "sample_id": record.sample_id,
            "question": record.question,
            "human_answer": record.human_answer or "Not provided",
            "updated_answer": record.updated_answer or "Not provided",
            "model_answer": record.model_answer or "",
        }
        record_tokens = token_counter(json.dumps(record_dict, ensure_ascii=False)) + 4

        if current and (
            current_tokens + record_tokens > max_tokens - TOKEN_BUFFER
            or len(current) >= max_records
        ):
            chunks.append(current)
            current = []
            current_tokens = base_tokens

        current.append(record)
        current_tokens += record_tokens

    if current:
        chunks.append(current)

    return chunks


def build_payload(records: List[Record]) -> str:
    serializable = [
        {
            "sample_id": record.sample_id,
            "question": record.question,
            "human_answer": record.human_answer or "Not provided",
            "updated_answer": record.updated_answer or "Not provided",
            "model_answer": record.model_answer or "",
        }
        for record in records
    ]
    return USER_TEMPLATE.format(records=json.dumps(serializable, ensure_ascii=False))


async def call_judge_async(
    client: AsyncOpenAI,
    records: List[Record],
) -> List[JudgedItem]:
    user_content = build_payload(records)
    response = await client.responses.parse(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        text_format=JudgementResponse,
    )
    return response.output_parsed.items


async def judge_with_retry_async(
    client: AsyncOpenAI,
    records: List[Record],
    *,
    logger: logging.Logger,
    max_retries: int = MAX_RETRIES,
) -> List[JudgedItem] | None:
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            return await call_judge_async(client, records)
        except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
            logger.warning(
                "Judge call failed (attempt %d/%d): %s", attempt, max_retries, exc
            )
            if attempt == max_retries:
                return None
            await asyncio.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, 60)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected judge error: %s", exc)
            return None
    return None


FIELDNAMES = [
    "sample_id",
    "PMID",
    "QID",
    "Question",
    "Human Answer",
    "Updated Human Answer",
    "Model Answer",
    "judge_human_correct",
    "judge_updated_correct",
    "judge_explanation",
]


def append_results(rows: List[dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUTPUT_CSV.exists()
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def write_summary() -> None:
    if not OUTPUT_CSV.exists():
        return
    df = pd.read_csv(OUTPUT_CSV)
    summary = {
        "human_correct_accuracy": df["judge_human_correct"].mean(),
        "updated_correct_accuracy": df["judge_updated_correct"].mean(),
        "samples": len(df),
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_CSV, index=False)


async def process_chunk(
    chunk_id: int,
    total_chunks: int,
    chunk: List[Record],
    client: AsyncOpenAI,
    token_counter,
    base_tokens: int,
    writer_lock: asyncio.Lock,
    logger: logging.Logger,
) -> bool:
    est_tokens = base_tokens + sum(
        token_counter(
            json.dumps(
                {
                    "sample_id": r.sample_id,
                    "question": r.question,
                    "human_answer": r.human_answer or "Not provided",
                    "updated_answer": r.updated_answer or "Not provided",
                    "model_answer": r.model_answer or "",
                },
                ensure_ascii=False,
            )
        )
        + 4
        for r in chunk
    )
    logger.info(
        "Judging chunk %d/%d (%d records, ~%d tokens).",
        chunk_id,
        total_chunks,
        len(chunk),
        est_tokens,
    )

    judged = await judge_with_retry_async(client, chunk, logger=logger)
    if judged is None:
        logger.error("Chunk %d failed after retries.", chunk_id)
        return False

    mapping: Dict[str, Record] = {record.sample_id: record for record in chunk}
    rows: List[dict] = []
    for item in judged:
        record = mapping.get(item.sample_id)
        if not record:
            logger.warning("Unknown sample_id %s returned by judge; skipping.", item.sample_id)
            continue
        rows.append(
            {
                "sample_id": record.sample_id,
                "PMID": record.pmid,
                "QID": record.qid,
                "Question": record.question,
                "Human Answer": record.human_answer,
                "Updated Human Answer": record.updated_answer,
                "Model Answer": record.model_answer,
                "judge_human_correct": item.human_correct,
                "judge_updated_correct": item.updated_correct,
                "judge_explanation": item.explanation.strip(),
            }
        )

    async with writer_lock:
        await asyncio.to_thread(append_results, rows)
    logger.info("Chunk %d stored (%d rows).", chunk_id, len(rows))
    return True


async def run_async(args, logger: logging.Logger) -> int:
    if not (MERGED_PATH.exists() and GPT5_MINI_PATH.exists()):
        logger.error("Required data files missing.")
        return 1

    records = load_records(limit=args.limit)
    if not records:
        logger.error("No records to judge.")
        return 1

    completed_ids = load_completed_ids(OUTPUT_CSV)
    records = [record for record in records if record.sample_id not in completed_ids]
    if not records:
        logger.info("All records already judged. Writing summary.")
        write_summary()
        return 0

    logger.info("Remaining records to judge: %d", len(records))

    token_counter = token_counter_factory(MODEL_NAME)
    base_tokens = token_counter(SYSTEM_PROMPT) + token_counter(USER_TEMPLATE.format(records="[]"))
    chunks = chunk_records(
        records,
        token_counter,
        MAX_TOKENS_PER_CALL,
        base_tokens,
        max_records=max(1, args.max_records_per_call),
    )
    logger.info(
        "Prepared %d chunk(s) (<=%d records each).",
        len(chunks),
        args.max_records_per_call,
    )

    client = AsyncOpenAI(timeout=90)
    writer_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))

    async def worker(chunk_id: int, chunk: List[Record]) -> bool:
        async with semaphore:
            return await process_chunk(
                chunk_id,
                len(chunks),
                chunk,
                client,
                token_counter,
                base_tokens,
                writer_lock,
                logger,
            )

    tasks = [
        asyncio.create_task(worker(idx, chunk))
        for idx, chunk in enumerate(chunks, start=1)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    if any(result is False or isinstance(result, Exception) for result in results):
        logger.error("One or more chunks failed. Aborting.")
        return 1

    write_summary()
    logger.info("Judging complete. Results in %s", OUTPUT_CSV)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge GPT-5-mini answers.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records.")
    parser.add_argument(
        "--max-records-per-call",
        type=int,
        default=DEFAULT_MAX_RECORDS,
        help="Maximum records per GPT-5 request.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Maximum parallel judge calls.",
    )
    args = parser.parse_args()

    load_dotenv()
    logger = configure_logger()
    return asyncio.run(run_async(args, logger))


if __name__ == "__main__":
    raise SystemExit(main())

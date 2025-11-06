#!/usr/bin/env python3
"""Call GPT-4o mini for each PubMed prompt and store responses incrementally."""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


MODEL_NAME = "gpt-4o-mini-2024-07-18"
PROMPTS_PATH = Path("pmid_prompts.jsonl")
RESPONSES_PATH = Path("pmid_responses.jsonl")
LOG_PATH = Path("pmid_responses.log")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("pmid_runner")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_processed(path: Path) -> set[str]:
    processed: set[str] = set()
    if not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            pmid = data.get("pmid")
            if pmid:
                processed.add(str(pmid))
    return processed


def main() -> int:
    load_dotenv()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    logger = setup_logger()

    if not PROMPTS_PATH.exists():
        logger.error("Prompts file not found at %s", PROMPTS_PATH)
        return 1

    processed_pmids = load_processed(RESPONSES_PATH)
    logger.info("Loaded %d processed PMIDs", len(processed_pmids))

    with PROMPTS_PATH.open("r", encoding="utf-8") as prompts_file, RESPONSES_PATH.open(
        "a", encoding="utf-8"
    ) as responses_file:
        for line_number, line in enumerate(prompts_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            pmid = str(record["pmid"])

            if pmid in processed_pmids:
                logger.info("Skipping already processed PMID %s", pmid)
                continue

            prompt = record["prompt"]
            logger.info("Requesting response for PMID %s", pmid)

            try:
                response = client.responses.create(model=MODEL_NAME, input=prompt)
            except Exception as exc:
                logger.exception("API call failed for PMID %s: %s", pmid, exc)
                continue

            output_text = response.output_text
            response_record = {"pmid": pmid, "response": output_text}
            responses_file.write(json.dumps(response_record, ensure_ascii=False) + "\n")
            responses_file.flush()
            processed_pmids.add(pmid)
            logger.info("Stored response for PMID %s", pmid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

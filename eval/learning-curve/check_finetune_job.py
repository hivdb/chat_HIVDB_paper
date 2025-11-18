#!/usr/bin/env python3
"""Verify OpenAI access and inspect a fine-tuning job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError


def load_env() -> None:
    """Load root .env plus eval/.env if present."""
    load_dotenv()
    eval_env = Path("eval/.env")
    if eval_env.exists():
        load_dotenv(eval_env)


def format_json(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-id",
        type=str,
        default="ftjob-KcO0ZDfs21Hq688zDsZwvtDN",
        help="Fine-tuning job ID to inspect.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Also print a short listing of recent fine-tune jobs to confirm access.",
    )
    return parser.parse_args()


def print_job(job_id: str, client: OpenAI) -> int:
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
    except OpenAIError as exc:
        print(f"Failed to retrieve job {job_id}: {exc}", file=sys.stderr)
        return 1

    payload = {
        "job_id": job.id,
        "status": job.status,
        "created_at": job.created_at,
        "finished_at": getattr(job, "finished_at", None),
        "model": job.model,
        "result_model": getattr(job, "fine_tuned_model", None),
        "training_file": getattr(job, "training_file", None),
        "validation_file": getattr(job, "validation_file", None),
        "hyperparameters": getattr(job, "hyperparameters", None),
        "trained_tokens": getattr(job, "trained_tokens", None),
        "object": job.object,
    }
    print("=== Fine-tuning job details ===")
    print(format_json(payload))
    return 0


def list_jobs(client: OpenAI, limit: int = 5) -> None:
    response = client.fine_tuning.jobs.list(limit=limit)
    jobs = [
        {
            "job_id": item.id,
            "status": item.status,
            "model": item.model,
            "result_model": getattr(item, "fine_tuned_model", None),
            "created_at": item.created_at,
        }
        for item in response.data
    ]
    print("\n=== Recent fine-tune jobs ===")
    print(format_json(jobs))


def main() -> int:
    args = parse_args()
    load_env()
    client = OpenAI()
    status = print_job(args.job_id, client)
    if args.list:
        list_jobs(client)
    return status


if __name__ == "__main__":
    raise SystemExit(main())

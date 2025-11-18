#!/usr/bin/env python3
"""Print raw fine-tuning job events to inspect loss metrics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI


def load_env(paths: tuple[str, ...] = (".env", "eval/.env")) -> None:
    for env_path in paths:
        path = Path(env_path)
        if not path.exists():
            continue
        for key, value in dotenv_values(path).items():
            if value is not None:
                os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True, help="Fine-tuning job ID (ftjob-...).")
    parser.add_argument("--limit", type=int, default=20, help="Number of events to show.")
    parser.add_argument("--after", type=str, default=None, help="Event ID to start after.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    client = OpenAI()
    response = client.fine_tuning.jobs.list_events(
        fine_tuning_job_id=args.job_id,
        limit=args.limit,
        after=args.after,
    )
    for event in response.data:
        print("=" * 60)
        print(f"event_id   : {event.id}")
        print(f"type       : {event.type}")
        print(f"created_at : {event.created_at}")
        print(f"message    : {getattr(event, 'message', None)}")
        payload = getattr(event, "data", None)
        if payload:
            for key, value in payload.items():
                print(f"data[{key!r}] = {value}")
        else:
            print("data       : None")
    print("=" * 60)
    print(f"has_more   : {response.has_more}")
    if response.has_more and response.data:
        print(f"Next cursor: {response.data[-1].id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

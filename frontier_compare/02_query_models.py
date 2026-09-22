#!/usr/bin/env python3
"""Send each full-text PDF + the paper's question-specific prompt (QSP) to a frontier model.

One request per PMID answers all 16 questions (same unit as the paper's QSP runs). Every
response, including failures, is appended to runs/<model>/run<N>.jsonl with latency, token
usage, and cost so operational metrics can be computed later. Re-running resumes: PMIDs that
already have a successful record in the target run file are skipped.

Examples:
  python frontier_compare/02_query_models.py --model gpt6-astra --run 1 --dry-run --limit 2
  python frontier_compare/02_query_models.py --model qwen3.8 --run 1 --max-concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier_compare import config  # noqa: E402

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6
REQUEST_TIMEOUT_S = 900

OUTPUT_INSTRUCTIONS = """## Output format

The full-text article is attached as a PDF. Use all of it, including tables, figures,
figure legends, and any supplementary material included in the file.

Return ONLY a JSON object (no markdown fences, no text before or after) of the form:
{
  "answers": [
    {
      "qid": 1,
      "question": "<question text>",
      "evidence": "<two or three sentences quoted from the paper that support the answer>",
      "evidence_location": "<one of: main_text, table, figure, supplement, not_found>",
      "rationale": "<how the answer follows from the evidence>",
      "answer": "<the answer, following the rules for that question above>"
    }
  ]
}

Include exactly one entry for each of Questions 1-16, in order, with "qid" equal to the
question number.
"""


def build_system_prompt() -> str:
    """The paper's QSP text, with its free-text output section replaced by a JSON contract."""
    text = config.QSP_PROMPT_PATH.read_text(encoding="utf-8")
    marker = "## For each question:"
    if marker not in text:
        raise ValueError(f"Could not find '{marker}' in {config.QSP_PROMPT_PATH}")
    return text.split(marker)[0].rstrip() + "\n\n" + OUTPUT_INSTRUCTIONS


def load_manifest() -> pd.DataFrame:
    if not config.PDF_MANIFEST.exists():
        raise FileNotFoundError("Run 01_pdf_manifest.py --stage first.")
    return pd.read_csv(config.PDF_MANIFEST, dtype=str)


def pdf_path(pmid: str) -> Path:
    return config.PDF_DIR / f"{pmid}.pdf"


def build_payload(spec: config.ModelSpec, system_prompt: str, pmid: str) -> dict:
    encoded = base64.b64encode(pdf_path(pmid).read_bytes()).decode("ascii")
    user_content = [
        {"type": "text", "text": f"PMID: {pmid}\nAnswer Questions 1-16 for the attached article."},
        {
            "type": "file",
            "file": {"filename": f"{pmid}.pdf", "file_data": f"data:application/pdf;base64,{encoded}"},
        },
    ]
    payload: dict = {
        "model": spec.model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if spec.provider == "openai":
        payload["max_completion_tokens"] = spec.max_output_tokens
        if spec.reasoning_effort:
            payload["reasoning_effort"] = spec.reasoning_effort
    else:
        payload["max_tokens"] = spec.max_output_tokens
        payload["usage"] = {"include": True}  # OpenRouter returns billed cost in usage.cost
        if spec.pdf_engine:
            payload["plugins"] = [{"id": "file-parser", "pdf": {"engine": spec.pdf_engine}}]
        if spec.reasoning_effort:
            payload["reasoning"] = {"effort": spec.reasoning_effort}
    return payload


def estimate_cost(spec: config.ModelSpec, usage: dict) -> float | None:
    if usage.get("cost") is not None:
        return float(usage["cost"])
    if spec.price_in is None or spec.price_out is None:
        return None
    return (
        usage.get("prompt_tokens", 0) * spec.price_in + usage.get("completion_tokens", 0) * spec.price_out
    ) / 1e6


def completed_pmids(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("ok"):
                done.add(rec["pmid"])
    return done


async def call_once(
    client: httpx.AsyncClient, spec: config.ModelSpec, payload: dict, pmid: str, run_id: int
) -> dict:
    url, key_env = config.PROVIDER_ENDPOINTS[spec.provider]
    headers = {"Authorization": f"Bearer {os.environ[key_env]}"}
    record = {
        "pmid": pmid,
        "model_key": spec.key,
        "model_id": spec.model_id,
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ok": False,
    }
    delay = 5.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            resp = await client.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_S)
        except httpx.HTTPError as exc:
            status, body, error = None, None, repr(exc)
        else:
            status, error = resp.status_code, None
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else None
            if status >= 400:
                error = resp.text[:2000]
        latency = time.monotonic() - started
        record.update(attempts=attempt, status_code=status, latency_s=round(latency, 2), error=error)

        if error is None and body is not None:
            choice = body["choices"][0]
            usage = body.get("usage") or {}
            record.update(
                ok=True,
                content=choice["message"].get("content") or "",
                finish_reason=choice.get("finish_reason"),
                usage=usage,
                cost_usd=estimate_cost(spec, usage),
                served_model=body.get("model"),
                provider=body.get("provider"),
            )
            return record
        if status is not None and status not in RETRY_STATUS:
            return record
        await asyncio.sleep(delay)
        delay = min(delay * 2, 120)
    return record


async def run(spec: config.ModelSpec, run_id: int, pmids: list[str], concurrency: int) -> None:
    out_path = config.run_path(spec.key, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    system_prompt = build_system_prompt()
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    log = logging.getLogger("frontier")

    async with httpx.AsyncClient() as client:

        async def worker(pmid: str) -> None:
            async with sem:
                record = await call_once(client, spec, build_payload(spec, system_prompt, pmid), pmid, run_id)
            async with lock:
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.info(
                "%s run%d PMID %s ok=%s %.1fs", spec.key, run_id, pmid, record["ok"], record.get("latency_s", 0)
            )

        await asyncio.gather(*(worker(p) for p in pmids))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(config.MODELS))
    parser.add_argument("--run", type=int, default=1, help="Replicate index (for run-to-run stability).")
    parser.add_argument("--pmids", nargs="*", help="Restrict to these PMIDs.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Build requests and report sizes; no API calls.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(config.ROOT / ".env")
    load_dotenv(config.FC_DIR / ".env", override=True)
    spec = config.MODELS[args.model]

    manifest = load_manifest()
    available = [p for p in manifest["PMID"] if pdf_path(p).exists()]
    missing = sorted(set(manifest["PMID"]) - set(available))
    if missing:
        logging.warning("%d PMIDs have no staged PDF and will be skipped (see pdf_manifest.csv).", len(missing))
    targets = [p for p in available if not args.pmids or p in set(args.pmids)]
    done = completed_pmids(config.run_path(spec.key, args.run))
    targets = [p for p in targets if p not in done][: args.limit]

    if args.dry_run:
        system_prompt = build_system_prompt()
        for pmid in targets:
            payload = build_payload(spec, system_prompt, pmid)
            print(f"{pmid}: request body {len(json.dumps(payload)) / 1e6:.2f} MB -> {spec.model_id}")
        print(f"\n{len(targets)} requests; system prompt {len(system_prompt)} chars; {len(done)} already done.")
        return 0

    key_env = config.PROVIDER_ENDPOINTS[spec.provider][1]
    if not os.environ.get(key_env):
        logging.error("%s is not set (put it in .env or frontier_compare/.env).", key_env)
        return 1
    asyncio.run(run(spec, args.run, targets, args.max_concurrency))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

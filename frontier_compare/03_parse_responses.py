#!/usr/bin/env python3
"""Parse raw run JSONL files into per-(PMID, QID) answer tables plus a per-request ops table.

JSON validity is recorded at two levels so the invalid-JSON rate is not hidden by cleanup:
  strict  - the message content parses as-is with json.loads
  lenient - parses after stripping markdown fences / surrounding prose
Answers are taken from the lenient parse; unparseable responses yield blank answers
(scored as wrong, same as a missing answer in the paper pipeline).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier_compare import config  # noqa: E402

FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_content(content: str) -> tuple[dict | None, str]:
    try:
        return json.loads(content), "strict"
    except (json.JSONDecodeError, TypeError):
        pass
    text = FENCE.sub("", (content or "").strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1]), "lenient"
        except json.JSONDecodeError:
            pass
    return None, "invalid"


def extract_answers(parsed: dict | None) -> dict[int, dict]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("answers"), list):
        return {}
    out: dict[int, dict] = {}
    for idx, entry in enumerate(parsed["answers"], start=1):
        if not isinstance(entry, dict):
            continue
        try:
            qid = int(entry.get("qid", idx))
        except (TypeError, ValueError):
            qid = idx
        if 1 <= qid <= config.TOTAL_QUESTIONS and qid not in out:
            out[qid] = entry
    return out


def latest_records(path: Path) -> dict[str, dict]:
    """Last successful record per PMID; falls back to the last failure if none succeeded."""
    records: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        prev = records.get(rec["pmid"])
        if prev is None or rec.get("ok") or not prev.get("ok"):
            records[rec["pmid"]] = rec
    return records


def main() -> int:
    questions = (
        pd.read_excel(config.MERGED_PATH, dtype=str, usecols=["QID", "Question"])
        .drop_duplicates("QID")
        .assign(QID=lambda d: d["QID"].astype(int))
        .set_index("QID")["Question"]
    )
    ops_rows = []
    for run_file in sorted(config.RUNS_DIR.glob("*/run*.jsonl")):
        model_key, run_id = run_file.parent.name, int(run_file.stem.removeprefix("run"))
        answer_rows = []
        for pmid, rec in latest_records(run_file).items():
            parsed, json_status = parse_content(rec.get("content", "")) if rec.get("ok") else (None, "no_response")
            answers = extract_answers(parsed)
            usage = rec.get("usage") or {}
            ops_rows.append(
                {
                    "model_key": model_key,
                    "run_id": run_id,
                    "PMID": pmid,
                    "ok": rec.get("ok", False),
                    "attempts": rec.get("attempts"),
                    "latency_s": rec.get("latency_s"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    "cost_usd": rec.get("cost_usd"),
                    "finish_reason": rec.get("finish_reason"),
                    "json_status": json_status,
                    "n_answers": len(answers),
                    "served_model": rec.get("served_model"),
                }
            )
            for qid in range(1, config.TOTAL_QUESTIONS + 1):
                entry = answers.get(qid, {})
                answer_rows.append(
                    {
                        "PMID": pmid,
                        "QID": qid,
                        "Question": questions.get(qid, ""),
                        "Answer": str(entry.get("answer", "")).strip(),
                        "Evidence": str(entry.get("evidence", "")).strip(),
                        "EvidenceLocation": str(entry.get("evidence_location", "")).strip(),
                        "Rationale": str(entry.get("rationale", "")).strip(),
                        "json_status": json_status,
                    }
                )
        out = config.answers_path(model_key, run_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(answer_rows).to_csv(out, index=False)
        print(f"{model_key} run{run_id}: {len(answer_rows) // config.TOTAL_QUESTIONS} PMIDs -> {out.name}")

    if ops_rows:
        ops = pd.DataFrame(ops_rows)
        ops.to_csv(config.RESULTS_DIR / "ops_requests.csv", index=False)
        print(ops.groupby(["model_key", "run_id", "json_status"]).size().to_string())
    else:
        print("No run files found under", config.RUNS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

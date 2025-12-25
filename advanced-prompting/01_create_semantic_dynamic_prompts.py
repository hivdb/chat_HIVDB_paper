#!/usr/bin/env python3
"""
Create semantic few-shot prompts by embedding the train/val instruction set and
retrieving the most relevant examples for each paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None

LOG_DIR = Path("advanced-prompting/log")
DEFAULT_EMBED_MODEL = "text-embedding-3-large"
MAX_EMBED_TOKENS = 7000

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instruction-xlsx",
        type=Path,
        default=Path("advanced-prompting/csv/S2Table.xlsx"),
        help="Path to S2Table.xlsx containing Question/Evidence/Rationale/Answer.",
    )
    parser.add_argument(
        "--prompt-md",
        type=Path,
        default=Path("eval/gpt-5/gpt-5-mini-prompt.md"),
        help="Prompt template prepended to each paper.",
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=Path("advanced-prompting/papers"),
        help="Directory containing paper markdown files grouped by PMID.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("advanced-prompting/jsonl"),
        help="Directory to store generated JSONL prompts.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=[5],
        help="Top-k values used to build few-shot prompts (default: 5).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBED_MODEL,
        help="OpenAI embedding model used for semantic search.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("advanced-prompting/train_val/train_set.jsonl"),
        help="Path to the training JSONL file.",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("advanced-prompting/train_val/val_set.jsonl"),
        help="Path to the validation JSONL file.",
    )
    return parser.parse_args()

def configure_logging(run_id: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"create_semantic_dynamic_prompts_{run_id}.log"
    logger = logging.getLogger("semantic_dynamic_prompts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    logger.info("Logging to %s", log_path)
    return logger


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

QA_BLOCK_RE = re.compile(
    r"Question:\s*(.*?)\n+Evidence:\s*(.*?)\n+Rationale:\s*(.*?)\n+Answer:\s*(.*?)(?:```|\Z)",
    re.S,
)
PAPER_RE = re.compile(r"## Paper content:\s*```(.*?)```", re.S)


def normalize_text(text: str | None) -> str:
    return " ".join(str(text or "").split()).lower()


def extract_qas(answer_text: str) -> list[dict[str, str]]:
    cleaned = answer_text.replace("```", "").strip()
    blocks: list[dict[str, str]] = []
    for match in QA_BLOCK_RE.finditer(cleaned):
        blocks.append(
            {
                "question": match.group(1).strip(),
                "evidence": match.group(2).strip(),
                "rationale": match.group(3).strip(),
                "answer": match.group(4).strip(),
            }
        )
    return blocks


def extract_paper_text(user_prompt: str) -> str:
    match = PAPER_RE.search(user_prompt)
    return match.group(1).strip() if match else user_prompt.strip()


def truncate_text(text: str, max_tokens: int = MAX_EMBED_TOKENS) -> str:
    if not text:
        return ""
    fallback = text[: max_tokens * 4]
    if tiktoken is None:
        return fallback
    try:
        encoding = tiktoken.encoding_for_model(DEFAULT_EMBED_MODEL)
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoding.decode(tokens[:max_tokens])
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_s2_table(path: Path) -> tuple[dict[str, list[dict]], dict[tuple[str, str], str]]:
    df = pd.read_excel(path).sort_values(["PMID", "QID"])
    pmid_records: dict[str, list[dict]] = {}
    index: dict[tuple[str, str], str] = {}
    for pmid, group in df.groupby("PMID", sort=False):
        pmid_str = str(pmid)
        records: list[dict] = []
        for _, row in group.iterrows():
            record = {
                "QID": int(row["QID"]),
                "Question": str(row["Question"]),
                "Evidence": str(row["Evidence"]),
                "Rationale": str(row["Rationale"]),
                "Answer": str(row["Answer"]),
            }
            records.append(record)
            key = (normalize_text(record["Question"]), normalize_text(record["Evidence"]))
            index[key] = pmid_str
        pmid_records[pmid_str] = records
    return pmid_records, index


def load_conversation_examples(
    files: Iterable[Path],
    qa_index: dict[tuple[str, str], str],
    logger: logging.Logger,
) -> list[dict]:
    """Return a list of {'pmid': str, 'text': paper_markdown} items."""
    entries: list[dict] = []
    seen_pmids: set[str] = set()
    for file_path in files:
        if not file_path.exists():
            logger.warning("Conversation file missing: %s", file_path)
            continue
        with file_path.open("r", encoding="utf-8") as infile:
            for line_no, line in enumerate(infile, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON (%s line %d)", file_path, line_no)
                    continue
                messages = record.get("messages", [])
                if len(messages) < 2:
                    continue
                user_msg = messages[1]
                assistant_msg = messages[-1]
                pmid = identify_pmid(assistant_msg.get("content", ""), qa_index)
                if not pmid or pmid in seen_pmids:
                    continue
                paper_text = extract_paper_text(user_msg.get("content", ""))
                entries.append({"pmid": pmid, "text": truncate_text(paper_text)})
                seen_pmids.add(pmid)
    logger.info("Loaded %d semantic examples covering %d PMIDs.", len(entries), len(seen_pmids))
    return entries


def identify_pmid(answer_text: str, qa_index: dict[tuple[str, str], str]) -> str | None:
    for block in extract_qas(answer_text):
        key = (normalize_text(block["question"]), normalize_text(block["evidence"]))
        pmid = qa_index.get(key)
        if pmid:
            return pmid
    return None


def collect_pmids(papers_dir: Path) -> list[str]:
    pmids = sorted(
        entry.name for entry in papers_dir.iterdir() if entry.is_dir() and not entry.name.startswith(".")
    )
    if not pmids:
        raise ValueError(f"No PMID directories found in {papers_dir}")
    return pmids


def load_paper_markdown(pmid: str, papers_dir: Path) -> str:
    path = papers_dir / pmid / f"{pmid}.checked.md"
    if not path.exists():
        raise FileNotFoundError(f"Missing paper markdown for PMID {pmid}: {path}")
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Embeddings + retrieval
# ---------------------------------------------------------------------------

def embed_texts(client: OpenAI, model: str, texts: list[str], batch_size: int = 32) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def retrieve_neighbors(
    client: OpenAI,
    model: str,
    corpus_embeddings: np.ndarray,
    corpus_entries: list[dict],
    query_text: str,
    top_k: int,
    exclude: str | None = None,
) -> list[str]:
    query_embedding = embed_texts(client, model, [query_text])
    scores = corpus_embeddings @ query_embedding[0]
    ranked = np.argsort(scores)[::-1]
    neighbors: list[str] = []
    for idx in ranked:
        pmid = corpus_entries[idx]["pmid"]
        if pmid == exclude or pmid in neighbors:
            continue
        neighbors.append(pmid)
        if len(neighbors) >= top_k:
            break
    return neighbors


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def few_shot_block(example_pmids: Iterable[str], pmid_records: dict[str, list[dict]]) -> tuple[str, list[str]]:
    question_map: dict[int, list[dict[str, str]]] = {}
    used_pmids: list[str] = []
    for pmid in example_pmids:
        entries = pmid_records.get(pmid)
        if not entries:
            continue
        used_pmids.append(pmid)
        for record in entries:
            question_map.setdefault(int(record["QID"]), []).append(
                {
                    "Evidence": record["Evidence"],
                    "Rationale": record["Rationale"],
                    "Answer": record["Answer"],
                }
            )
    ordered = [
        {"QID": qid, "Examples": examples}
        for qid, examples in sorted(question_map.items())
    ]
    return json.dumps(ordered, ensure_ascii=False, indent=2), used_pmids


def assemble_prompt(base_prompt: str, few_shot: str, paper_text: str) -> str:
    sections = [base_prompt]
    if few_shot.strip("[] \n"):
        sections.extend(["### FEW SHOT EXAMPLES", few_shot])
    sections.extend(["### PAPER FULL TEXT", paper_text])
    return "\n\n".join(sections)


def write_jsonl(records: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_usage(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["pmid", "k", "example_pmids"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    load_dotenv()
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = configure_logging(run_id)

    pmid_records, qa_index = load_s2_table(args.instruction_xlsx)
    conversations = load_conversation_examples([args.train_file, args.val_file], qa_index, logger)
    if not conversations:
        logger.error("No conversation examples could be mapped to PMIDs.")
        return 1

    client = OpenAI()
    corpus_embeddings = embed_texts(client, args.embedding_model, [entry["text"] for entry in conversations])

    pmids = collect_pmids(args.papers_dir)
    paper_texts = {pmid: load_paper_markdown(pmid, args.papers_dir) for pmid in pmids}
    base_prompt = args.prompt_md.read_text(encoding="utf-8").strip()

    shot_values = sorted({k for k in args.shots if k > 0})
    if not shot_values:
        raise ValueError("At least one positive k value is required.")

    for k in shot_values:
        logger.info("Generating semantic prompts with top-%d neighbors.", k)
        records: list[dict[str, str]] = []
        usage_rows: list[dict[str, str]] = []
        for pmid in pmids:
            query = truncate_text(paper_texts[pmid])
            neighbors = retrieve_neighbors(
                client,
                args.embedding_model,
                corpus_embeddings,
                conversations,
                query,
                top_k=k,
                exclude=pmid,
            )
            few_shot, used_pmids = few_shot_block(neighbors, pmid_records)
            prompt = assemble_prompt(base_prompt, few_shot, paper_texts[pmid])
            records.append({"pmid": pmid, "prompt": prompt})
            usage_rows.append({"pmid": pmid, "k": str(k), "example_pmids": "|".join(used_pmids)})

        output_jsonl = args.output_dir / f"dynamic_prompts_semantic_{k}-shot.jsonl"
        write_jsonl(records, output_jsonl)
        logger.info("Wrote %d prompts to %s", len(records), output_jsonl)

        usage_csv = LOG_DIR / f"few_shot_usage_semantic_{k}-shot_{run_id}.csv"
        write_usage(usage_rows, usage_csv)
        logger.info("Logged usage statistics to %s", usage_csv)

    logger.info("Semantic prompt generation finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

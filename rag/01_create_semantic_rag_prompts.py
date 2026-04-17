#!/usr/bin/env python3
"""Build per-paper semantic-similarity RAG prompts for the HIVDB evaluation papers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
ADVANCED_PROMPTING_DIR = ROOT.parent / "advanced-prompting"
DEFAULT_METADATA = ADVANCED_PROMPTING_DIR / "csv" / "ground_truth.xlsx"
DEFAULT_PROMPT_TEMPLATE = ROOT.parent / "eval" / "gpt-5" / "gpt-5-mini-prompt.md"
DEFAULT_OUTPUT_DIR = ROOT / "jsonl"
DEFAULT_LOG_DIR = ROOT / "log"
DEFAULT_MANIFEST_PATH = ROOT / "run_manifest.json"
DEFAULT_DATASETS = {
    "original120": ADVANCED_PROMPTING_DIR / "papers",
    "new30": ADVANCED_PROMPTING_DIR / "papers_2025_30",
}
DEFAULT_OUTPUTS = {
    "original120": DEFAULT_OUTPUT_DIR / "pmid_prompts_semantic_rag_original120.jsonl",
    "new30": DEFAULT_OUTPUT_DIR / "pmid_prompts_semantic_rag_new30.jsonl",
}
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
REFERENCE_HEADING_RE = re.compile(r"^(references|bibliography)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Question:
    qid: int
    question: str
    qtype: str
    category: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    section_path: str
    text: str


@dataclass(frozen=True)
class EvidencePoolEntry:
    chunk: Chunk
    matched_qids: tuple[int, ...]
    hit_count: int
    best_rank: int
    aggregate_rank_score: float
    max_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-xlsx",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Question metadata workbook (default: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=DEFAULT_PROMPT_TEMPLATE,
        help=f"Prompt template markdown (default: {DEFAULT_PROMPT_TEMPLATE})",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DEFAULT_DATASETS),
        help="Limit generation to one or more datasets (default: both).",
    )
    parser.add_argument(
        "--papers-dir",
        action="append",
        nargs=2,
        metavar=("DATASET", "PATH"),
        help="Override the papers directory for a dataset.",
    )
    parser.add_argument(
        "--output",
        action="append",
        nargs=2,
        metavar=("DATASET", "PATH"),
        help="Override the output JSONL path for a dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k chunks to retrieve per question (default: 5).",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=1800,
        help="Target maximum characters per chunk before splitting (default: 1800).",
    )
    parser.add_argument(
        "--chunk-overlap-paragraphs",
        type=int,
        default=1,
        help="Paragraph overlap between adjacent chunks within a section (default: 1).",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model to use (default: {DEFAULT_EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory for retrieval audit CSVs (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Run manifest JSON path (default: {DEFAULT_MANIFEST_PATH})",
    )
    return parser.parse_args()


def load_env() -> None:
    load_dotenv(override=False)
    advanced_env = ADVANCED_PROMPTING_DIR / ".env"
    if advanced_env.exists():
        load_dotenv(advanced_env, override=True)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY not found. Checked environment and {advanced_env}.")


def normalize_identifier(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_questions(metadata_xlsx: Path) -> list[Question]:
    df = pd.read_excel(metadata_xlsx, dtype=str, keep_default_na=False)
    required = {"QID", "Question", "Type", "Category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata workbook missing required columns: {sorted(missing)}")
    questions: list[Question] = []
    seen_qids: set[int] = set()
    for _, row in df.sort_values(["QID"]).iterrows():
        qid = int(row["QID"])
        if qid in seen_qids:
            continue
        seen_qids.add(qid)
        questions.append(
            Question(
                qid=qid,
                question=str(row["Question"]).strip(),
                qtype=str(row["Type"]).strip(),
                category=str(row["Category"]).strip(),
            )
        )
    return questions


def resolve_markdown_path(pmid_dir: Path) -> Path:
    candidates = [
        pmid_dir / f"{pmid_dir.name}.checked.md",
        pmid_dir / f"{pmid_dir.name}_checked.md",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing markdown for PMID {pmid_dir.name}: tried {candidates[0].name} and {candidates[1].name}"
    )


def collect_papers(papers_dir: Path) -> list[tuple[str, Path]]:
    papers: list[tuple[str, Path]] = []
    for entry in sorted(papers_dir.iterdir(), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        pmid = normalize_identifier(entry.name)
        papers.append((pmid, resolve_markdown_path(entry)))
    return papers


def split_sections(markdown_text: str) -> list[tuple[str, list[str]]]:
    heading_stack: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current_paragraphs: list[str] = []

    def flush_current() -> None:
        nonlocal current_paragraphs
        if not current_paragraphs:
            return
        section = " > ".join(heading_stack) if heading_stack else "Document"
        sections.append((section, current_paragraphs))
        current_paragraphs = []

    paragraph_lines: list[str] = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        heading_match = HEADING_RE.match(line.strip())
        if heading_match:
            if paragraph_lines:
                current_paragraphs.append("\n".join(paragraph_lines).strip())
                paragraph_lines = []
            flush_current()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            if REFERENCE_HEADING_RE.match(heading_text):
                break
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(heading_text)
            continue
        if not line.strip():
            if paragraph_lines:
                current_paragraphs.append("\n".join(paragraph_lines).strip())
                paragraph_lines = []
            continue
        paragraph_lines.append(line)

    if paragraph_lines:
        current_paragraphs.append("\n".join(paragraph_lines).strip())
    flush_current()
    return sections


def build_chunks(markdown_text: str, chunk_chars: int, overlap_paragraphs: int) -> list[Chunk]:
    sections = split_sections(markdown_text)
    chunks: list[Chunk] = []
    chunk_id = 1
    for section_path, paragraphs in sections:
        if not paragraphs:
            continue
        start = 0
        while start < len(paragraphs):
            buffer: list[str] = []
            size = 0
            idx = start
            while idx < len(paragraphs):
                paragraph = paragraphs[idx]
                projected = size + len(paragraph) + (2 if buffer else 0)
                if buffer and projected > chunk_chars:
                    break
                buffer.append(paragraph)
                size = projected
                idx += 1
            chunk_text = "\n\n".join(buffer).strip()
            if chunk_text:
                chunks.append(Chunk(chunk_id=chunk_id, section_path=section_path, text=chunk_text))
                chunk_id += 1
            if idx >= len(paragraphs):
                break
            start = max(start + 1, idx - max(0, overlap_paragraphs))
    return chunks


def build_base_prompt(prompt_text: str, article_text: str) -> str:
    article_body = article_text.strip()
    return (
        f"{prompt_text.strip()}\n\n"
        f"PAPER FULL TEXT\n\n"
        f"{article_body}\n\n"
        f"PAPER ENDED\n\n"
        f"{prompt_text.strip()}"
    )


def aggregate_evidence_pool(
    retrievals: dict[int, list[tuple[Chunk, float]]],
    rrf_k: int = 60,
) -> list[EvidencePoolEntry]:
    pooled: dict[int, dict[str, object]] = {}
    for qid, retrieved in retrievals.items():
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            entry = pooled.setdefault(
                chunk.chunk_id,
                {
                    "chunk": chunk,
                    "matched_qids": [],
                    "hit_count": 0,
                    "best_rank": rank,
                    "aggregate_rank_score": 0.0,
                    "max_score": score,
                },
            )
            entry["matched_qids"].append(qid)
            entry["hit_count"] = int(entry["hit_count"]) + 1
            entry["best_rank"] = min(int(entry["best_rank"]), rank)
            entry["aggregate_rank_score"] = float(entry["aggregate_rank_score"]) + (1.0 / (rrf_k + rank))
            entry["max_score"] = max(float(entry["max_score"]), score)

    return sorted(
        (
            EvidencePoolEntry(
                chunk=data["chunk"],
                matched_qids=tuple(sorted(set(data["matched_qids"]))),
                hit_count=int(data["hit_count"]),
                best_rank=int(data["best_rank"]),
                aggregate_rank_score=float(data["aggregate_rank_score"]),
                max_score=float(data["max_score"]),
            )
            for data in pooled.values()
        ),
        key=lambda entry: (
            entry.hit_count,
            entry.aggregate_rank_score,
            -entry.best_rank,
            -entry.chunk.chunk_id,
        ),
        reverse=True,
    )


def format_prompt(
    template: str,
    markdown_text: str,
    evidence_pool: list[EvidencePoolEntry],
) -> tuple[str, list[EvidencePoolEntry], int]:
    base_prompt_chars = len(build_base_prompt(template, markdown_text))

    def render(entries: list[EvidencePoolEntry]) -> str:
        sections = [
            template.strip(),
            "",
            "## Retrieved Evidence Pool",
            "Use only the deduplicated evidence pool below instead of the full paper text.",
            "The pool was assembled by retrieving passages separately for each question, deduplicating by chunk_id, and ranking passages by how often they were retrieved across questions.",
            "",
        ]
        if not entries:
            sections.append("[No passages retrieved]")
            sections.append("")
        for rank, entry in enumerate(entries, start=1):
            matched = ",".join(f"Q{qid}" for qid in entry.matched_qids)
            sections.append(
                "[Passage "
                f"{rank} | chunk_id={entry.chunk.chunk_id} | matched_questions={matched} "
                f"| hit_count={entry.hit_count} | aggregate_rank_score={entry.aggregate_rank_score:.4f} "
                f"| max_question_score={entry.max_score:.4f} | section={entry.chunk.section_path}]"
            )
            sections.append(entry.chunk.text)
            sections.append("")
        return "\n".join(sections).strip() + "\n"

    trimmed_pool = list(evidence_pool)
    prompt = render(trimmed_pool)
    while len(trimmed_pool) > 1 and len(prompt) >= base_prompt_chars:
        trimmed_pool.pop()
        prompt = render(trimmed_pool)
    return prompt, trimmed_pool, base_prompt_chars


def write_jsonl(records: Iterable[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False))
            outfile.write("\n")


def write_retrieval_audit(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "pmid",
                "qid",
                "question",
                "chunk_ids",
                "sections",
                "scores",
                "embedding_model",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_pool_audit(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "pmid",
                "chunk_ids",
                "sections",
                "matched_qids",
                "hit_counts",
                "aggregate_rank_scores",
                "max_scores",
                "pool_chunk_count",
                "prompt_chars",
                "base_prompt_chars",
                "embedding_model",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def update_run_manifest(manifest_path: Path, entry_key: str, payload: dict[str, object]) -> None:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    manifest[entry_key] = payload
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def embed_texts(client: OpenAI, model: str, texts: list[str], batch_size: int) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def main() -> int:
    args = parse_args()
    load_env()
    client = OpenAI()
    template = args.prompt_template.read_text(encoding="utf-8").strip()
    questions = load_questions(args.metadata_xlsx)
    datasets = args.dataset or list(DEFAULT_DATASETS.keys())

    papers_dirs = dict(DEFAULT_DATASETS)
    if args.papers_dir:
        for dataset, path in args.papers_dir:
            papers_dirs[dataset] = Path(path)

    outputs = dict(DEFAULT_OUTPUTS)
    if args.output:
        for dataset, path in args.output:
            outputs[dataset] = Path(path)

    manifest_datasets: dict[str, dict[str, object]] = {}

    for dataset in datasets:
        papers_dir = papers_dirs[dataset]
        output_path = outputs[dataset]
        papers = collect_papers(papers_dir)
        records: list[dict[str, str]] = []
        retrieval_rows: list[dict[str, str]] = []
        pool_rows: list[dict[str, str]] = []
        chunk_counts: list[int] = []

        for pmid, markdown_path in papers:
            markdown_text = markdown_path.read_text(encoding="utf-8")
            chunks = build_chunks(
                markdown_text,
                chunk_chars=args.chunk_chars,
                overlap_paragraphs=args.chunk_overlap_paragraphs,
            )
            if not chunks:
                raise ValueError(f"No chunks created for PMID {pmid} from {markdown_path}")
            chunk_counts.append(len(chunks))

            chunk_texts = [chunk.text for chunk in chunks]
            question_texts = [question.question for question in questions]
            chunk_embeddings = embed_texts(client, args.embedding_model, chunk_texts, args.batch_size)
            question_embeddings = embed_texts(client, args.embedding_model, question_texts, args.batch_size)

            retrievals: dict[int, list[tuple[Chunk, float]]] = {}
            for question, query_vec in zip(questions, question_embeddings):
                scores = chunk_embeddings @ query_vec
                ranked = np.argsort(scores)[::-1][: args.top_k]
                retrieved = [(chunks[int(idx)], float(scores[int(idx)])) for idx in ranked]
                retrievals[question.qid] = retrieved
                retrieval_rows.append(
                    {
                        "pmid": pmid,
                        "qid": str(question.qid),
                        "question": question.question,
                        "chunk_ids": "|".join(str(chunk.chunk_id) for chunk, _ in retrieved),
                        "sections": "|".join(chunk.section_path for chunk, _ in retrieved),
                        "scores": "|".join(f"{score:.4f}" for _, score in retrieved),
                        "embedding_model": args.embedding_model,
                    }
                )

            evidence_pool = aggregate_evidence_pool(retrievals)
            prompt, trimmed_pool, base_prompt_chars = format_prompt(
                template,
                markdown_text,
                evidence_pool,
            )
            records.append({"pmid": pmid, "prompt": prompt})
            pool_rows.append(
                {
                    "pmid": pmid,
                    "chunk_ids": "|".join(str(entry.chunk.chunk_id) for entry in trimmed_pool),
                    "sections": "|".join(entry.chunk.section_path for entry in trimmed_pool),
                    "matched_qids": "|".join(
                        ",".join(str(qid) for qid in entry.matched_qids) for entry in trimmed_pool
                    ),
                    "hit_counts": "|".join(str(entry.hit_count) for entry in trimmed_pool),
                    "aggregate_rank_scores": "|".join(
                        f"{entry.aggregate_rank_score:.4f}" for entry in trimmed_pool
                    ),
                    "max_scores": "|".join(f"{entry.max_score:.4f}" for entry in trimmed_pool),
                    "pool_chunk_count": str(len(trimmed_pool)),
                    "prompt_chars": str(len(prompt)),
                    "base_prompt_chars": str(base_prompt_chars),
                    "embedding_model": args.embedding_model,
                }
            )

        write_jsonl(records, output_path)
        audit_path = args.log_dir / f"semantic_rag_retrieval_{dataset}.csv"
        write_retrieval_audit(retrieval_rows, audit_path)
        pool_audit_path = args.log_dir / f"semantic_rag_pool_{dataset}.csv"
        write_pool_audit(pool_rows, pool_audit_path)
        manifest_datasets[dataset] = {
            "papers_dir": repo_relative(papers_dir),
            "paper_count": len(papers),
            "questions_per_paper": len(questions),
            "output_jsonl": repo_relative(output_path),
            "retrieval_log_csv": repo_relative(audit_path),
            "pool_log_csv": repo_relative(pool_audit_path),
            "chunk_counts": {
                "total": sum(chunk_counts),
                "min_per_paper": min(chunk_counts),
                "max_per_paper": max(chunk_counts),
                "avg_per_paper": round(sum(chunk_counts) / len(chunk_counts), 2),
            },
        }
        print(f"Wrote {len(records)} prompts to {output_path}")
        print(f"Wrote retrieval audit to {audit_path}")
        print(f"Wrote pool audit to {pool_audit_path}")

    update_run_manifest(
        args.manifest_path,
        "semantic",
        {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "retriever": "semantic",
            "parameters": {
                "top_k": args.top_k,
                "chunk_chars": args.chunk_chars,
                "chunk_overlap_paragraphs": args.chunk_overlap_paragraphs,
                "section_aware_chunking": True,
                "stop_at_reference_headings": True,
                "reference_heading_regex": REFERENCE_HEADING_RE.pattern,
                "shared_evidence_pool": True,
                "pool_ranking": "hit_count_then_rrf",
                "pool_prompt_shorter_than_base": True,
                "embedding_model": args.embedding_model,
                "embedding_batch_size": args.batch_size,
                "similarity": "cosine",
                "prompt_template": repo_relative(args.prompt_template),
                "metadata_xlsx": repo_relative(args.metadata_xlsx),
                "manifest_path": repo_relative(args.manifest_path),
            },
            "datasets": manifest_datasets,
        },
    )
    print(f"Updated run manifest at {args.manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

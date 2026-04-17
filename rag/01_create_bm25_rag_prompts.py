#!/usr/bin/env python3
"""Build per-paper BM25 RAG prompts for the HIVDB evaluation papers.

This script implements the reviewer-aligned retrieval baseline:
each paper is chunked independently, each of the 16 questions retrieves
top-k chunks from that paper only, and the final prompt contains only the
retrieved passages rather than the full article text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


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
    "original120": DEFAULT_OUTPUT_DIR / "pmid_prompts_bm25_rag_original120.jsonl",
    "new30": DEFAULT_OUTPUT_DIR / "pmid_prompts_bm25_rag_new30.jsonl",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")
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


def normalize_identifier(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


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
    """Split markdown into section-aware paragraph lists."""
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


def build_chunks(
    markdown_text: str,
    chunk_chars: int,
    overlap_paragraphs: int,
) -> list[Chunk]:
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
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        section_path=section_path,
                        text=chunk_text,
                    )
                )
                chunk_id += 1

            if idx >= len(paragraphs):
                break
            start = max(start + 1, idx - max(0, overlap_paragraphs))

    return chunks


class PaperBM25Index:
    """Small BM25 implementation without extra retrieval dependencies."""

    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.doc_freqs: Counter[str] = Counter()

        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            counts = Counter(tokens)
            self.term_freqs.append(counts)
            self.doc_lengths.append(sum(counts.values()))
            self.doc_freqs.update(counts.keys())

        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )

    def score(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        if not query_terms or not self.chunks:
            return [0.0 for _ in self.chunks]

        num_docs = len(self.chunks)
        scores = [0.0 for _ in self.chunks]
        unique_terms = Counter(query_terms)

        for term, query_weight in unique_terms.items():
            df = self.doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
            for idx, counts in enumerate(self.term_freqs):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[idx] or 1
                denom = tf + self.k1 * (
                    1 - self.b + self.b * (doc_len / (self.avg_doc_length or 1.0))
                )
                scores[idx] += query_weight * idf * ((tf * (self.k1 + 1)) / denom)
        return scores

    def retrieve(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        scores = self.score(query)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (item[1], -self.chunks[item[0]].chunk_id),
            reverse=True,
        )
        selected: list[tuple[Chunk, float]] = []
        for idx, score in ranked[:top_k]:
            selected.append((self.chunks[idx], score))
        if selected and any(score > 0 for _, score in selected):
            return selected
        return [(chunk, 0.0) for chunk in self.chunks[:top_k]]


def format_prompt(
    template: str,
    questions: list[Question],
    retrievals: dict[int, list[tuple[Chunk, float]]],
) -> str:
    sections = [
        template.strip(),
        "",
        "## Retrieved Evidence By Question",
        "Use the retrieved passages below instead of the full paper text. Each question should be answered using only the passages listed under that question.",
        "",
    ]
    for question in questions:
        sections.append(f"### Question {question.qid}")
        sections.append(f"Question: {question.question}")
        sections.append("")
        sections.append("Retrieved Evidence:")
        retrieved = retrievals.get(question.qid, [])
        if not retrieved:
            sections.append("[No passages retrieved]")
            sections.append("")
            continue
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            sections.append(
                f"[Passage {rank} | chunk_id={chunk.chunk_id} | score={score:.4f} | section={chunk.section_path}]"
            )
            sections.append(chunk.text)
            sections.append("")
    return "\n".join(sections).strip() + "\n"


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


def main() -> int:
    args = parse_args()
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

            index = PaperBM25Index(chunks)
            retrievals: dict[int, list[tuple[Chunk, float]]] = {}
            for question in questions:
                retrieved = index.retrieve(question.question, args.top_k)
                retrievals[question.qid] = retrieved
                retrieval_rows.append(
                    {
                        "pmid": pmid,
                        "qid": str(question.qid),
                        "question": question.question,
                        "chunk_ids": "|".join(str(chunk.chunk_id) for chunk, _ in retrieved),
                        "sections": "|".join(chunk.section_path for chunk, _ in retrieved),
                        "scores": "|".join(f"{score:.4f}" for _, score in retrieved),
                    }
                )

            prompt = format_prompt(template, questions, retrievals)
            records.append({"pmid": pmid, "prompt": prompt})

        write_jsonl(records, output_path)
        audit_path = args.log_dir / f"bm25_rag_retrieval_{dataset}.csv"
        write_retrieval_audit(retrieval_rows, audit_path)
        manifest_datasets[dataset] = {
            "papers_dir": repo_relative(papers_dir),
            "paper_count": len(papers),
            "questions_per_paper": len(questions),
            "output_jsonl": repo_relative(output_path),
            "retrieval_log_csv": repo_relative(audit_path),
            "chunk_counts": {
                "total": sum(chunk_counts),
                "min_per_paper": min(chunk_counts),
                "max_per_paper": max(chunk_counts),
                "avg_per_paper": round(sum(chunk_counts) / len(chunk_counts), 2),
            },
        }
        print(f"Wrote {len(records)} prompts to {output_path}")
        print(f"Wrote retrieval audit to {audit_path}")

    update_run_manifest(
        args.manifest_path,
        "bm25",
        {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "retriever": "bm25",
            "parameters": {
                "top_k": args.top_k,
                "chunk_chars": args.chunk_chars,
                "chunk_overlap_paragraphs": args.chunk_overlap_paragraphs,
                "section_aware_chunking": True,
                "stop_at_reference_headings": True,
                "reference_heading_regex": REFERENCE_HEADING_RE.pattern,
                "tokenizer_regex": TOKEN_RE.pattern,
                "bm25_k1": 1.5,
                "bm25_b": 0.75,
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

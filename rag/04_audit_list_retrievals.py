#!/usr/bin/env python3
"""Audit hard list-question retrieval quality for BM25 vs semantic RAG."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

from eval.normalize import canonicalize_answer, list_match_stats  # type: ignore

ADVANCED_PROMPTING_DIR = ROOT.parent / "advanced-prompting"
DEFAULT_TRUTH = ADVANCED_PROMPTING_DIR / "csv" / "ground_truth.xlsx"
DEFAULT_BM25_POOL_LOGS = [
    ROOT / "log" / "bm25_rag_pool_original120.csv",
    ROOT / "log" / "bm25_rag_pool_new30.csv",
]
DEFAULT_SEMANTIC_POOL_LOGS = [
    ROOT / "log" / "semantic_rag_pool_original120.csv",
    ROOT / "log" / "semantic_rag_pool_new30.csv",
]
PAPER_ROOTS = [
    ADVANCED_PROMPTING_DIR / "papers",
    ADVANCED_PROMPTING_DIR / "papers_2025_30",
]
LIST_QIDS = {9, 15, 16}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
REFERENCE_HEADING_RE = re.compile(r"^(references|bibliography)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    section_path: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-xlsx", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "list_retrieval_audit.csv")
    parser.add_argument("--top-n", type=int, default=20, help="Max rows per QID, hardest first (default: 20).")
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--chunk-overlap-paragraphs", type=int, default=1)
    return parser.parse_args()


def normalize_identifier(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def load_truth(path: Path) -> pd.DataFrame:
    truth = pd.read_excel(path, dtype=str, keep_default_na=False)
    truth["PMID"] = truth["PMID"].apply(normalize_identifier)
    truth["QID"] = truth["QID"].astype(int)
    truth = truth[truth["QID"].isin(LIST_QIDS)].copy()
    truth = truth[~truth["Human-Answer"].isin(["", "Not reported", "Not applicable", "None"])]
    return truth


def load_logs(paths: list[Path], prefix: str) -> pd.DataFrame:
    frames = [pd.read_csv(path, dtype=str) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError(f"No log files found for {prefix}: {paths}")
    df = pd.concat(frames, ignore_index=True)
    rename_map = {
        "chunk_ids": f"{prefix}_pool_chunk_ids",
        "sections": f"{prefix}_pool_sections",
        "matched_qids": f"{prefix}_pool_matched_qids",
        "pool_chunk_count": f"{prefix}_pool_chunk_count",
        "prompt_chars": f"{prefix}_prompt_chars",
        "base_prompt_chars": f"{prefix}_base_prompt_chars",
    }
    keep = ["pmid", *rename_map.keys()]
    return df[keep].rename(columns=rename_map)


def resolve_markdown_path(pmid: str) -> Path:
    for root in PAPER_ROOTS:
        pmid_dir = root / pmid
        if not pmid_dir.is_dir():
            continue
        for candidate in (
            pmid_dir / f"{pmid}.checked.md",
            pmid_dir / f"{pmid}_checked.md",
        ):
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Unable to locate markdown for PMID {pmid}")


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


@lru_cache(maxsize=None)
def chunk_map_for_pmid(pmid: str, chunk_chars: int, overlap_paragraphs: int) -> dict[int, Chunk]:
    markdown_path = resolve_markdown_path(pmid)
    markdown_text = markdown_path.read_text(encoding="utf-8")
    chunks = build_chunks(markdown_text, chunk_chars=chunk_chars, overlap_paragraphs=overlap_paragraphs)
    return {chunk.chunk_id: chunk for chunk in chunks}


def parse_chunk_ids(value: str) -> list[int]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [int(part) for part in str(value).split("|") if part.strip()]


def token_coverage(answer: str, text_blob: str) -> float:
    ref_norm = canonicalize_answer(answer)
    if not ref_norm:
        return 0.0
    matches, total = list_match_stats(ref_norm, "", str(text_blob or ""))
    if not total:
        return 0.0
    return matches / total


def normalize_excerpt(text: str, max_chars: int = 480) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1] + "…"


def retrieved_blob(pmid: str, chunk_ids: str, chunk_chars: int, overlap_paragraphs: int) -> tuple[str, str]:
    cmap = chunk_map_for_pmid(pmid, chunk_chars=chunk_chars, overlap_paragraphs=overlap_paragraphs)
    texts: list[str] = []
    sections: list[str] = []
    for chunk_id in parse_chunk_ids(chunk_ids):
        chunk = cmap.get(chunk_id)
        if not chunk:
            continue
        sections.append(chunk.section_path)
        texts.append(f"[{chunk.section_path}] {chunk.text}")
    return "\n\n".join(texts), " | ".join(sections)


def main() -> int:
    args = parse_args()
    truth = load_truth(args.truth_xlsx)
    bm25 = load_logs(DEFAULT_BM25_POOL_LOGS, "bm25")
    semantic = load_logs(DEFAULT_SEMANTIC_POOL_LOGS, "semantic")

    merged = truth.merge(bm25, left_on="PMID", right_on="pmid", how="left")
    merged = merged.merge(semantic, left_on="PMID", right_on="pmid", how="left", suffixes=("_bm25", "_semantic"))
    merged.drop(columns=["pmid_bm25", "pmid_semantic"], inplace=True, errors="ignore")

    merged[["bm25_pool_text", "bm25_pool_reconstructed_sections"]] = merged.apply(
        lambda row: pd.Series(
            retrieved_blob(
                row["PMID"],
                row.get("bm25_pool_chunk_ids", ""),
                chunk_chars=args.chunk_chars,
                overlap_paragraphs=args.chunk_overlap_paragraphs,
            )
        ),
        axis=1,
    )
    merged[["semantic_pool_text", "semantic_pool_reconstructed_sections"]] = merged.apply(
        lambda row: pd.Series(
            retrieved_blob(
                row["PMID"],
                row.get("semantic_pool_chunk_ids", ""),
                chunk_chars=args.chunk_chars,
                overlap_paragraphs=args.chunk_overlap_paragraphs,
            )
        ),
        axis=1,
    )

    merged["bm25_pool_text_coverage"] = merged.apply(
        lambda row: token_coverage(row["Human-Answer"], row.get("bm25_pool_text", "")),
        axis=1,
    )
    merged["semantic_pool_text_coverage"] = merged.apply(
        lambda row: token_coverage(row["Human-Answer"], row.get("semantic_pool_text", "")),
        axis=1,
    )
    merged["coverage_delta"] = (
        merged["semantic_pool_text_coverage"] - merged["bm25_pool_text_coverage"]
    )
    merged["bm25_pool_excerpt"] = merged["bm25_pool_text"].apply(normalize_excerpt)
    merged["semantic_pool_excerpt"] = merged["semantic_pool_text"].apply(normalize_excerpt)

    rows: list[pd.DataFrame] = []
    for qid in sorted(LIST_QIDS):
        subset = merged[merged["QID"] == qid].copy()
        subset["best_coverage"] = subset[
            ["bm25_pool_text_coverage", "semantic_pool_text_coverage"]
        ].max(axis=1)
        subset = subset.sort_values(
            by=["best_coverage", "coverage_delta", "PMID"],
            ascending=[True, False, True],
        )
        rows.append(subset.head(args.top_n))

    audit = pd.concat(rows, ignore_index=True)
    ordered_cols = [
        "PMID",
        "QID",
        "Question",
        "Human-Answer",
        "bm25_pool_text_coverage",
        "semantic_pool_text_coverage",
        "coverage_delta",
        "bm25_pool_chunk_ids",
        "semantic_pool_chunk_ids",
        "bm25_pool_reconstructed_sections",
        "semantic_pool_reconstructed_sections",
        "bm25_pool_chunk_count",
        "semantic_pool_chunk_count",
        "bm25_prompt_chars",
        "bm25_base_prompt_chars",
        "semantic_prompt_chars",
        "semantic_base_prompt_chars",
        "bm25_pool_excerpt",
        "semantic_pool_excerpt",
    ]
    audit = audit[ordered_cols]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_csv, index=False)
    print(f"Wrote retrieval audit to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

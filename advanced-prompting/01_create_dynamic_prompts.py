#!/usr/bin/env python3
"""
Create retrieval-based dynamic few-shot prompts for each PubMed article.

Steps:
1. Load the S2Table.xlsx instruction set and convert each PMID block into a single LangChain Document.
2. Build a BM25 retriever over those Documents.
3. For every PMID directory under advanced-prompting/papers, build the base query consisting of:
   - The evaluation prompt (eval/gpt-5/gpt-5-mini-prompt.md)
   - The corresponding paper markdown (advanced-prompting/papers/<pmid>/<pmid>.checked.md)
   - A trailing \"### FEW SHOT EXAMPLES\" section marker.
4. Use the BM25 retriever to fetch the top k (k in {5, 10}) Documents that best match the query.
5. Append the retrieved examples to the query and write JSONL files named
   dynamic_prompts_bm25_<k>-shot.jsonl, each containing 120 records.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


LOG_DIR = Path("advanced-prompting/log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate retrieval-based dynamic few-shot prompts."
    )
    parser.add_argument(
        "--instruction-xlsx",
        type=Path,
        default=Path("advanced-prompting/csv/S2Table.xlsx"),
        help="Path to the S2 instruction table.",
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
        help="Directory to store the generated dynamic prompt JSONL files.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Top-k values used to build few-shot prompts.",
    )
    return parser.parse_args()


def configure_logging(log_dir: Path) -> logging.Logger:
    """Configure console and file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"create_dynamic_prompts_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger("dynamic_prompts")
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


def collect_pmids(papers_dir: Path) -> list[str]:
    """Collect PMIDs based on subdirectory names under papers_dir."""
    if not papers_dir.exists():
        raise FileNotFoundError(f"Papers directory not found: {papers_dir}")
    pmids = sorted(
        entry.name for entry in papers_dir.iterdir() if entry.is_dir() and not entry.name.startswith(".")
    )
    if not pmids:
        raise ValueError(f"No PMID directories found in {papers_dir}")
    return pmids


def load_instruction_documents(xlsx_path: Path) -> list[Document]:
    """Load the Excel table and convert each PMID into a Document."""
    df = pd.read_excel(xlsx_path).sort_values(["PMID", "QID"])

    documents: list[Document] = []
    for pmid, group in df.groupby("PMID", sort=False):
        parts: list[str] = []
        for _, row in group.iterrows():
            part = (
                f"Question {int(row['QID'])}: {row['Question']}\n"
                f"Evidence: {row['Evidence']}\n"
                f"Rationale: {row['Rationale']}\n"
                f"Answer: {row['Answer']}"
            )
            parts.append(part.strip())

        page_content = f"PMID: {pmid}\n\n" + "\n\n---\n\n".join(parts)
        documents.append(
            Document(page_content=page_content, metadata={"pmid": pmid, "count": len(parts)})
        )
    return documents


def build_query_base(pmid: str, prompt_text: str, papers_dir: Path) -> str:
    """Build the query string prior to appending few-shot examples."""
    paper_path = papers_dir / pmid / f"{pmid}.checked.md"
    if not paper_path.exists():
        raise FileNotFoundError(f"Missing paper markdown for PMID {pmid}: {paper_path}")
    paper_text = paper_path.read_text(encoding="utf-8").strip()
    sections = [
        prompt_text,
        "### PAPER FULL TEXT",
        paper_text,
        "### FEW SHOT EXAMPLES",
    ]
    return "\n\n".join(sections)


def append_examples(base_query: str, examples: Iterable[Document]) -> str:
    few_shot_blocks = [doc.page_content.strip() for doc in examples if doc.page_content]
    if not few_shot_blocks:
        return base_query
    examples_text = "\n\n".join(few_shot_blocks)
    return f"{base_query}\n\n{examples_text}"


def write_jsonl(records: Sequence[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    logger = configure_logging(LOG_DIR)
    logger.info("Starting dynamic prompt generation.")

    pmids = collect_pmids(args.papers_dir)
    logger.info("Found %d PMIDs in %s", len(pmids), args.papers_dir)
    documents = load_instruction_documents(args.instruction_xlsx)
    logger.info("Loaded %d instruction documents from %s", len(documents), args.instruction_xlsx)

    prompt_text = args.prompt_md.read_text(encoding="utf-8").strip()
    base_queries = {
        pmid: build_query_base(pmid, prompt_text, args.papers_dir) for pmid in pmids
    }
    logger.info("Prepared base queries for %d PMIDs.", len(base_queries))

    shot_values = sorted({int(value) for value in args.shots if value > 0})
    if not shot_values:
        raise ValueError("At least one positive k value must be provided.")
    logger.info("Shot values configured: %s", shot_values)

    for k in shot_values:
        retriever = BM25Retriever.from_documents(documents, k=k)
        logger.info("Built BM25 retriever for k=%d.", k)
        file_name = f"dynamic_prompts_bm25_{k}-shot.jsonl"
        output_path = args.output_dir / file_name
        records: list[dict[str, str]] = []
        logger.info("Generating BM25 prompts with k=%d.", k)

        for pmid in pmids:
            query = base_queries[pmid]
            examples = retriever.invoke(query)
            prompt = append_examples(query, examples)
            records.append({"pmid": pmid, "prompt": prompt})

        write_jsonl(records, output_path)
        logger.info("Wrote %d prompts to %s", len(records), output_path)
    logger.info("Dynamic prompt generation finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

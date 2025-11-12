#!/usr/bin/env python3
"""
Create retrieval-based dynamic few-shot prompts for each PubMed article.

Steps:
1. Load the S2Table.xlsx instruction set and convert each PMID block into a single LangChain Document.
2. Build a BM25 retriever over those Documents.
3. For every PMID directory under advanced-prompting/papers, build the base query.
4. Use the BM25 retriever to fetch the top k (k in {5, 10}) Documents that best match the query.
5. Insert the compact few-shot examples before the paper markdown and write JSONL files named
   dynamic_prompts_bm25_<k>-shot.jsonl, each containing 120 records.
"""
from __future__ import annotations

import argparse
import csv
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


def configure_logging(log_dir: Path, run_id: str) -> logging.Logger:
    """Configure console and file logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"create_dynamic_prompts_{run_id}.log"

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
        qas: list[dict[str, object]] = []
        pmid_str = str(pmid)
        for _, row in group.iterrows():
            question = str(row["Question"])
            evidence = str(row["Evidence"])
            rationale = str(row["Rationale"])
            answer = str(row["Answer"])
            qa_entry = {
                "qid": int(row["QID"]),
                "question": question,
                "evidence": evidence,
                "rationale": rationale,
                "answer": answer,
            }
            qas.append(qa_entry)
            part = (
                f"Question {qa_entry['qid']}: {question}\n"
                f"Evidence: {evidence}\n"
                f"Rationale: {rationale}\n"
                f"Answer: {answer}"
            )
            parts.append(part.strip())

        page_content = "\n\n---\n\n".join(parts)
        documents.append(
            Document(
                page_content=page_content,
                metadata={"pmid": pmid_str, "count": len(parts), "qas": qas},
            )
        )
    return documents


def load_paper_text(pmid: str, papers_dir: Path) -> str:
    """Load the markdown content for a PMID."""
    paper_path = papers_dir / pmid / f"{pmid}.checked.md"
    if not paper_path.exists():
        raise FileNotFoundError(f"Missing paper markdown for PMID {pmid}: {paper_path}")
    return paper_path.read_text(encoding="utf-8").strip()


def build_few_shot_block(examples: Sequence[Document]) -> tuple[str, list[str]]:
    """Create the compact few-shot block and capture example PMIDs."""
    question_map: dict[int, dict[str, object]] = {}
    example_pmids: list[str] = []

    for doc in examples:
        pmid = str(doc.metadata.get("pmid", "")).strip()
        if pmid:
            example_pmids.append(pmid)
        qas = doc.metadata.get("qas") or []
        for entry in qas:
            qid = int(entry["qid"])
            question = str(entry["question"])
            question_map.setdefault(qid, {"question": question, "examples": []})
            question_map[qid]["examples"].append(
                (str(entry["evidence"]), str(entry["rationale"]), str(entry["answer"]))
            )

    lines: list[str] = []
    for qid in sorted(question_map):
        data = question_map[qid]
        tuples = data["examples"]
        tuple_strs = [
            f"[{evidence}, {rationale}, {answer}]" for evidence, rationale, answer in tuples
        ]
        line = f"Question {qid}: {data['question']} - " + ", ".join(tuple_strs)
        lines.append(line)

    deduped_pmids: list[str] = []
    seen: set[str] = set()
    for pmid in example_pmids:
        if pmid and pmid not in seen:
            seen.add(pmid)
            deduped_pmids.append(pmid)

    return ("\n".join(lines), deduped_pmids)


def write_jsonl(records: Sequence[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = configure_logging(LOG_DIR, run_id)
    logger.info("Starting dynamic prompt generation.")

    pmids = collect_pmids(args.papers_dir)
    logger.info("Found %d PMIDs in %s", len(pmids), args.papers_dir)
    documents = load_instruction_documents(args.instruction_xlsx)
    logger.info("Loaded %d instruction documents from %s", len(documents), args.instruction_xlsx)
    paper_texts = {pmid: load_paper_text(pmid, args.papers_dir) for pmid in pmids}
    logger.info("Loaded markdown for %d PMIDs.", len(paper_texts))

    prompt_text = args.prompt_md.read_text(encoding="utf-8").strip()
    logger.info("Base prompt loaded from %s", args.prompt_md)

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
        usage_rows: list[dict[str, str]] = []
        logger.info("Generating BM25 prompts with k=%d.", k)

        for pmid in pmids:
            query = f"{prompt_text}\n\n### PAPER FULL TEXT\n\n{paper_texts[pmid]}"
            examples = retriever.invoke(query)
            few_shot_text, example_pmids = build_few_shot_block(examples)

            sections = [prompt_text]
            if few_shot_text:
                sections.extend(["### FEW SHOT EXAMPLES", few_shot_text])
            sections.extend(["### PAPER FULL TEXT", paper_texts[pmid]])
            prompt = "\n\n".join(sections)
            records.append({"pmid": pmid, "prompt": prompt})

            usage_rows.append(
                {"pmid": pmid, "k": str(k), "example_pmids": "|".join(example_pmids)}
            )

        write_jsonl(records, output_path)
        logger.info("Wrote %d prompts to %s", len(records), output_path)

        usage_path = LOG_DIR / f"few_shot_usage_bm25_{k}-shot_{run_id}.csv"
        with usage_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["pmid", "k", "example_pmids"])
            writer.writeheader()
            writer.writerows(usage_rows)
        logger.info("Logged few-shot usage to %s", usage_path)
    logger.info("Dynamic prompt generation finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

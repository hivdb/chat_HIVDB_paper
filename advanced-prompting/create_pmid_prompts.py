#!/usr/bin/env python3
"""
Generate a JSON Lines file that pairs each PubMed article with a
prompt “sandwich” surrounding the article content.

Each JSON object contains:
    - pmid: the folder name within papers/ (PubMed ID)
    - prompt: Prompts_Nov5.md + article markdown + Prompts_Nov5.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_prompt(prompt_path: Path) -> str:
    text = prompt_path.read_text(encoding="utf-8")
    return text.strip()


def build_prompt(prompt_text: str, article_text: str) -> str:
    article_body = article_text.strip()
    return (
        f"{prompt_text}\n\n"
        f"PAPER FULL TEXT\n\n"
        f"{article_body}\n\n"
        f"PAPER ENDED\n\n"
        f"{prompt_text}"
    )


def collect_articles(papers_dir: Path) -> list[tuple[str, Path]]:
    articles = []
    for entry in papers_dir.iterdir():
        if not entry.is_dir():
            continue
        pmid = entry.name
        md_path = entry / f"{pmid}.checked.md"
        if not md_path.exists():
            raise FileNotFoundError(f"Missing markdown file for PMID {pmid}: {md_path}")
        articles.append((pmid, md_path))
    return sorted(articles, key=lambda item: item[0])


def generate_jsonl(prompt_path: Path, papers_dir: Path, output_path: Path) -> None:
    base_prompt = load_prompt(prompt_path)
    articles = collect_articles(papers_dir)

    with output_path.open("w", encoding="utf-8") as outfile:
        for pmid, md_path in articles:
            article_text = md_path.read_text(encoding="utf-8")
            prompt = build_prompt(base_prompt, article_text)
            record = {"pmid": pmid, "prompt": prompt}
            outfile.write(json.dumps(record, ensure_ascii=False))
            outfile.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create JSONL with pmid and prompt sandwich content."
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("Prompts_Nov7b.md"),
        help="Path to the base prompt Markdown file.",
    )
    parser.add_argument(
        "--papers",
        type=Path,
        default=Path("papers"),
        help="Directory containing PubMed article folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pmid_prompts_Nov7.jsonl"),
        help="Output JSONL file path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.prompt.exists():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt}")
    if not args.papers.exists():
        raise FileNotFoundError(f"Papers directory not found: {args.papers}")

    generate_jsonl(args.prompt, args.papers, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Generate JSON Lines files that pair each PubMed article with different
prompt placements around the article content.

Each JSON object contains:
    - pmid: the folder name within papers/ (PubMed ID)
    - prompt: variants of Prompts_Nov5.md and the article markdown
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable


PromptBuilder = Callable[[str, str], str]


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


def build_prompt_before(prompt_text: str, article_text: str) -> str:
    article_body = article_text.strip()
    return (
        f"{prompt_text}\n\n"
        f"PAPER FULL TEXT\n\n"
        f"{article_body}\n\n"
        f"PAPER ENDED\n\n"
    )


def build_prompt_after(prompt_text: str, article_text: str) -> str:
    article_body = article_text.strip()
    return (
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
            alt_path = entry / f"{pmid}_checked.md"
            if alt_path.exists():
                md_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Missing markdown file for PMID {pmid}: tried {md_path} and {alt_path}"
                )
        articles.append((pmid, md_path))
    return sorted(articles, key=lambda item: item[0])


def generate_jsonl(
    prompt_path: Path,
    papers_dir: Path,
    output_path: Path,
    prompt_builder: PromptBuilder = build_prompt,
) -> None:
    base_prompt = load_prompt(prompt_path)
    articles = collect_articles(papers_dir)

    with output_path.open("w", encoding="utf-8") as outfile:
        for pmid, md_path in articles:
            article_text = md_path.read_text(encoding="utf-8")
            prompt = prompt_builder(base_prompt, article_text)
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
        help="Output JSONL file path. For --before/--after this is the only file produced.",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--all",
        action="store_const",
        dest="mode",
        const="all",
        help="Generate sandwich, before, and after outputs (default).",
    )
    mode_group.add_argument(
        "--sandwich",
        action="store_const",
        dest="mode",
        const="sandwich",
        help="Generate only the prompt sandwich output.",
    )
    mode_group.add_argument(
        "--before",
        action="store_const",
        dest="mode",
        const="before",
        help="Generate only the 'prompt before paper' output.",
    )
    mode_group.add_argument(
        "--after",
        action="store_const",
        dest="mode",
        const="after",
        help="Generate only the 'prompt after paper' output.",
    )
    parser.set_defaults(mode="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.prompt.exists():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt}")
    if not args.papers.exists():
        raise FileNotFoundError(f"Papers directory not found: {args.papers}")

    if args.mode in {"all", "sandwich"}:
        generate_jsonl(args.prompt, args.papers, args.output)

    if args.mode in {"all", "before"}:
        before_target = (
            args.output
            if args.mode == "before"
            else args.output.parent / "pmid_prompts_before_Nov10.jsonl"
        )
        generate_jsonl(
            args.prompt,
            args.papers,
            before_target,
            prompt_builder=build_prompt_before,
        )

    if args.mode in {"all", "after"}:
        after_target = (
            args.output
            if args.mode == "after"
            else args.output.parent / "pmid_prompts_after_Nov10.jsonl"
        )
        generate_jsonl(
            args.prompt,
            args.papers,
            after_target,
            prompt_builder=build_prompt_after,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

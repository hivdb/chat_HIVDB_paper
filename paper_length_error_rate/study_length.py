"""Scan a folder for *checked.md files and export their content lengths to Excel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def collect_lengths(root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for file_path in root.rglob("*checked.md"):
        print(file_path)
        if not file_path.is_file():
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        records.append(
            {
                "folder": file_path.parent.name,
                "file": str(file_path.relative_to(root)),
                "content_length": len(content),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find *checked.md files under a folder and export their lengths."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Folder to scan (defaults to current directory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="study_length.xlsx",
        help="Path to the Excel file to write (defaults to study_length.xlsx).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        parser.error(f"Root folder does not exist: {root}")

    records = collect_lengths(root)
    df = pd.DataFrame(records, columns=["folder", "file", "content_length"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)

    print(f"Wrote {len(records)} entries to {output_path}")


if __name__ == "__main__":
    main()

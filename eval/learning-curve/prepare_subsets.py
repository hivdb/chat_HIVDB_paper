#!/usr/bin/env python3
"""Create nested training subsets for learning-curve experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, List


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_size(size: int) -> str:
    return f"{size:03d}"


def build_manifest_entry(size: int, output_path: Path) -> dict:
    return {
        "size": size,
        "file": str(output_path),
        "sha256": sha256(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("advanced-prompting/train_val/train_set.jsonl"),
        help="Full training file containing 200 JSONL records.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[50, 100, 150],
        help="Subset sizes to materialize (default: 50 100 150).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed used to shuffle the dataset once before taking prefixes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/learning-curve/data"),
        help="Directory where subset files and manifest will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = read_jsonl(args.train_file)
    total = len(data)
    if total == 0:
        raise SystemExit(f"{args.train_file} is empty.")

    unique_sizes = sorted({size for size in args.sizes if size > 0})
    for size in unique_sizes:
        if size > total:
            raise SystemExit(f"Requested subset of {size} exceeds dataset size {total}.")

    rng = random.Random(args.seed)
    indices = list(range(total))
    rng.shuffle(indices)
    permuted = [data[idx] for idx in indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_file": str(args.train_file),
        "total_examples": total,
        "seed": args.seed,
        "permutation": indices,
        "subsets": [],
    }

    for size in unique_sizes:
        subset_records = permuted[:size]
        filename = f"train_subset_{format_size(size)}.jsonl"
        output_path = args.output_dir / filename
        write_jsonl(output_path, subset_records)
        manifest["subsets"].append(build_manifest_entry(size, output_path))
        print(f"Wrote {size} examples to {output_path}")

    manifest_path = args.output_dir / "subset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as outfile:
        json.dump(manifest, outfile, indent=2)
    print(f"Saved manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

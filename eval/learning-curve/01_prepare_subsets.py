#!/usr/bin/env python3
"""Create nested train+val subsets for learning-curve experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple


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
        help="Training file containing 200 JSONL records.",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("advanced-prompting/train_val/val_set.jsonl"),
        help="Validation file containing 50 JSONL records.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200],
        help="Total subset sizes to materialize (default: 50 100 150 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed used to shuffle the combined dataset once before taking prefixes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/learning-curve/data"),
        help="Directory where subset files and manifest will be written.",
    )
    return parser.parse_args()


def split_subset(total_size: int) -> Tuple[int, int]:
    """Calculate 80/20 train/val split for a given total size."""
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size
    return train_size, val_size


def main() -> int:
    args = parse_args()

    # Load both train and val files
    train_data = read_jsonl(args.train_file)
    val_data = read_jsonl(args.val_file)

    if len(train_data) == 0:
        raise SystemExit(f"{args.train_file} is empty.")
    if len(val_data) == 0:
        raise SystemExit(f"{args.val_file} is empty.")

    print(f"Loaded {len(train_data)} train + {len(val_data)} val examples")

    unique_sizes = sorted({size for size in args.sizes if size > 0})

    # Shuffle train and val pools separately with the same seed
    rng = random.Random(args.seed)
    train_indices = list(range(len(train_data)))
    rng.shuffle(train_indices)
    shuffled_train = [train_data[idx] for idx in train_indices]

    val_indices = list(range(len(val_data)))
    rng.shuffle(val_indices)
    shuffled_val = [val_data[idx] for idx in val_indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
        "total_train": len(train_data),
        "total_val": len(val_data),
        "seed": args.seed,
        "train_permutation": train_indices,
        "val_permutation": val_indices,
        "subsets": [],
    }

    # Create subsets by sampling from separate train and val pools
    for size in unique_sizes:
        train_size, val_size = split_subset(size)

        # Validate we have enough examples
        if train_size > len(train_data):
            raise SystemExit(f"Requested {train_size} train examples but only have {len(train_data)}.")
        if val_size > len(val_data):
            raise SystemExit(f"Requested {val_size} val examples but only have {len(val_data)}.")

        # Take first N examples from each shuffled pool (nested subsets)
        subset_train = shuffled_train[:train_size]
        subset_val = shuffled_val[:val_size]

        # Write separate train and val files for OpenAI fine-tuning API
        train_filename = f"train_{format_size(size)}.jsonl"
        val_filename = f"val_{format_size(size)}.jsonl"
        train_path = args.output_dir / train_filename
        val_path = args.output_dir / val_filename

        write_jsonl(train_path, subset_train)
        write_jsonl(val_path, subset_val)

        manifest["subsets"].append({
            "size": size,
            "train_size": train_size,
            "val_size": val_size,
            "train_file": str(train_path),
            "val_file": str(val_path),
            "train_sha256": sha256(train_path),
            "val_sha256": sha256(val_path),
        })

        print(f"Wrote {size} total examples: {train_path} ({train_size}) + {val_path} ({val_size})")

    manifest_path = args.output_dir / "subset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as outfile:
        json.dump(manifest, outfile, indent=2)
    print(f"Saved manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

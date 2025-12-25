#!/usr/bin/env python3
"""Load a CSV file and rewrite it as UTF-8 encoded CSV."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from typing import Optional, Tuple


def detect_encoding(raw_bytes: bytes) -> str:
    """
    Try to determine the source encoding.

    Uses charset-normalizer if available, otherwise falls back to a small set
    of common encodings before defaulting to UTF-8.
    """
    try:
        from charset_normalizer import from_bytes
    except ImportError:
        from_bytes = None

    if from_bytes:
        result = from_bytes(raw_bytes).best()
        if result and result.encoding:
            return result.encoding

    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1", "utf-8"):
        try:
            raw_bytes.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue

    return "utf-8"


def sniff_dialect(sample_text: str) -> csv.Dialect:
    """Infer CSV dialect from a text sample."""
    try:
        return csv.Sniffer().sniff(sample_text)
    except csv.Error:
        return csv.excel


def convert_csv(
    input_path: Path, output_path: Path, source_encoding: Optional[str] = None
) -> Tuple[str, str]:
    """Stream rows from input_path and rewrite them to output_path as UTF-8."""
    with input_path.open("rb") as raw_in:
        sample = raw_in.read(8192)
        encoding = source_encoding or detect_encoding(sample)
        raw_in.seek(0)

        text_stream = io.TextIOWrapper(
            raw_in, encoding=encoding, errors="replace", newline=""
        )
        sample_text = sample.decode(encoding, errors="replace")
        dialect = sniff_dialect(sample_text)
        reader = csv.reader(text_stream, dialect)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as out_file:
            writer = csv.writer(
                out_file,
                dialect,
                quoting=csv.QUOTE_MINIMAL,
                escapechar="\\",
                doublequote=True,
            )
            for row in reader:
                writer.writerow(row)

    return encoding, "utf-8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a CSV file to UTF-8 encoding."
    )
    parser.add_argument("input_csv", type=Path, help="Path to the source CSV file.")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_csv",
        type=Path,
        help="Path for the UTF-8 CSV file. Defaults to <input>_utf8.csv.",
    )
    parser.add_argument(
        "--encoding",
        dest="encoding",
        help="Optional source encoding to skip detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_csv or args.input_csv.with_name(
        f"{args.input_csv.stem}_utf8{args.input_csv.suffix}"
    )
    source_encoding, target_encoding = convert_csv(
        args.input_csv, output_path, args.encoding
    )
    print(
        f"Read {args.input_csv} using {source_encoding} and wrote "
        f"{target_encoding} CSV to {output_path}"
    )


if __name__ == "__main__":
    main()

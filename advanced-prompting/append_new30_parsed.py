#!/usr/bin/env python3
"""Append *_new30_parsed.csv rows to their corresponding *_parsed.csv files.

For each file matching ./csv/*_new30_parsed.csv, the script finds the paired
file with the _new30 suffix removed (i.e., *_parsed.csv) and appends the rows
from the _new30 file to the end of the paired file, writing back in place.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import List, Set

import pandas as pd


def load_csv_strings(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def append_pair(base_path: pathlib.Path, new30_path: pathlib.Path) -> None:
    print(base_path, new30_path)
    base_df = load_csv_strings(base_path)
    new_df = load_csv_strings(new30_path)

    # Remove any rows in the base file that share a (PMID, QID) with the new rows
    # so that new_df values replace existing ones for those keys.
    new_pairs: Set[tuple[str, str]] = set(zip(new_df["PMID"], new_df["QID"]))
    base_df = base_df[~base_df[["PMID", "QID"]].apply(tuple, axis=1).isin(new_pairs)]

    # Align columns: keep base columns order, add any new columns at the end.
    base_cols: List[str] = list(base_df.columns)
    extra_cols: List[str] = [c for c in new_df.columns if c not in base_cols]
    final_cols: List[str] = base_cols + extra_cols

    base_df = base_df.reindex(columns=final_cols, fill_value="")
    new_df = new_df.reindex(columns=final_cols, fill_value="")

    merged = pd.concat([base_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["PMID", "QID"], keep="first")
    merged.to_csv(base_path, index=False)

    print(
        f"Appended {len(new_df)} rows from {new30_path.name} "
        f"into {base_path.name} (new total {len(merged)} after dedup)."
    )


def find_pairs(csv_dir: pathlib.Path) -> List[tuple[pathlib.Path, pathlib.Path]]:
    pairs = []
    for new30_path in csv_dir.glob("*_new30_parsed.csv"):
        base_path = new30_path.with_name(new30_path.name.replace("_new30_parsed.csv", "_parsed.csv"))
        if base_path.exists():
            pairs.append((base_path, new30_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-dir",
        type=pathlib.Path,
        default=pathlib.Path("./csv"),
        help="Directory to scan for *_new30_parsed.csv files (default: ./csv)",
    )
    args = parser.parse_args()

    pairs = find_pairs(args.csv_dir)
    if not pairs:
        print("No *_new30_parsed.csv pairs found.")
        return

    for base_path, new30_path in pairs:
        append_pair(base_path, new30_path)


if __name__ == "__main__":
    main()

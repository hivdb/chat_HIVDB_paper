#!/usr/bin/env python3
"""Build merged answer sheets for the new 30-paper set and the combined 150-paper set."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.scoring import format_identifier

MERGE_KEYS = ["PMID", "QID"]

ADV_CSV = ROOT / "advanced-prompting" / "csv"
ADV_TEST = ROOT / "advanced-prompting" / "test"

BASE_MERGED = ADV_CSV / "merged_answers.xlsx"
OUTPUT_NEW30 = ADV_CSV / "merged_answers_new30.xlsx"
OUTPUT_FULL = ADV_CSV / "merged_answers_full_150.xlsx"
OUTPUT_ORIGINAL120 = ADV_CSV / "merged_answers_original_120.xlsx"
S4TABLE = ADV_CSV / "S4Table.xlsx"

PAPERS_DIR = ADV_CSV.parent / "papers"
NEW30_PAPERS_DIR = ADV_CSV.parent / "papers_2025_30"

# Map collaborator column names to the canonical evaluation names.
COLUMN_RENAMES: Dict[str, str] = {
    "Human-Answer": "Human Answer",
    "gpt-4o-mini base": "GPT-4o base",
    "gpt-4o-mini-FT": "GPT-4o FT",
    "gpt-4o-mini-FT 50": "GPT-4o FT-50",
    "gpt-4o-mini-FT 100": "GPT-4o FT-100",
    "gpt-4o-mini-FT 150": "GPT-4o FT-150",
    "gpt-4o-mini-FT 200": "GPT-4o FT-200",
    "gpt-4o-mini PV1": "GPT-4o QSP",
    "llama-3.1-8B base": "Llama3.1-8B base",
    "llama-3.1-8B-FT": "Llama3.1-8B FT",
    "llama-3.1-8B PV1": "Llama3.1-8B QSP",
    "llama-3.1-70B base": "Llama3.1-70B base",
    "llama-3.1-70B-FT 50": "Llama3.1-70B FT-50",
    "llama-3.1-70B-FT 100": "Llama3.1-70B FT-100",
    "llama-3.1-70B-FT 150": "Llama3.1-70B FT-150",
    "llama-3.1-70B-FT 200": "Llama3.1-70B FT-200",
}

# Excel forbids certain control characters; strip them before writing workbooks
ILLEGAL_CHARS = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")


def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PMID"] = df["PMID"].apply(format_identifier)
    df["QID"] = df["QID"].apply(format_identifier)
    return df


def get_canonical_qid_mapping() -> Dict[str, int]:
    """Get canonical QID mapping from S4Table (question text -> QID)."""
    s4_df = pd.read_excel(S4TABLE)
    # Build question text -> QID map (QID ordering is same across all PMIDs)
    question_map = {}
    for _, row in s4_df.iterrows():
        q_norm = normalize_question(row["Question"])
        qid = int(row["QID"])
        if q_norm not in question_map:
            question_map[q_norm] = qid
    return question_map


def load_model(path: Path, column: str, value_column: str | None = None, remap_by_question: bool = False) -> pd.DataFrame | None:
    if not path.exists():
        logging.warning("Model source missing for %s: %s", column, path)
        return None
    loader = pd.read_excel if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv
    df = loader(path)
    candidates: List[str] = []
    if value_column:
        candidates.append(value_column)
    candidates.extend(["Answer", "answer", column])
    found: str | None = None
    for cand in candidates:
        if cand in df.columns:
            found = cand
            break
    if not found:
        logging.warning("Model source %s missing answer column for %s", path, column)
        return None
    if remap_by_question and "Question" in df.columns:
        # Remap from any QID ordering to canonical S4Table QIDs by matching question text
        question_map = get_canonical_qid_mapping()
        df = df.copy()
        original_qid = df["QID"].copy()
        df["QID"] = df["Question"].apply(lambda q: question_map.get(normalize_question(q), None))
        df = df[df["QID"].notna()]
        remapped = (df["QID"] != original_qid).sum()
        if remapped > 0:
            logging.info("Remapped %d QIDs for %s to match S4Table canonical ordering", remapped, column)

    df = normalize_ids(df)
    df = df.rename(columns={found: column})
    return df[MERGE_KEYS + [column]]


def merge_column(df: pd.DataFrame, model_df: pd.DataFrame, column: str) -> pd.DataFrame:
    merged = df.merge(model_df, on=MERGE_KEYS, how="left", suffixes=("", "__new"))
    new_col = f"{column}__new"
    if column not in merged.columns:
        merged[column] = ""
    merged[column] = merged[new_col].combine_first(merged[column]).fillna("")
    merged = merged.drop(columns=[new_col])
    return merged


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df


def inject_models(df: pd.DataFrame, model_sources: Dict[str, tuple[Path, str | None, bool]]) -> pd.DataFrame:
    df = df.copy()
    for column, (path, value_col, remap_pv1) in model_sources.items():
        model_df = load_model(path, column, value_col, remap_pv1)
        if model_df is None:
            continue
        df = ensure_columns(df, [column])
        df = merge_column(df, model_df, column)
    return df


def normalize_question(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def load_new30_pmids() -> Set[str]:
    pmids: Set[str] = set()
    if NEW30_PAPERS_DIR.exists():
        for entry in NEW30_PAPERS_DIR.iterdir():
            if entry.is_dir():
                pmids.add(format_identifier(entry.name))
    return pmids


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    core = ["PMID", "QID", "Question", "Type", "Category", "Human Answer"]
    rest = [col for col in df.columns if col not in core]
    return df[core + rest]


def build_outputs(base_merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Clean column names and identifiers
    base_merged = base_merged.rename(columns=COLUMN_RENAMES)
    base_merged = normalize_ids(base_merged)
    base_merged["QID"] = base_merged["QID"].astype(int)
    for col in ["Type", "Category", "Human Answer"]:
        if col in base_merged.columns:
            base_merged[col] = base_merged[col].fillna("").astype(str)

    # Normalize question text/type/category to canonical S4Table definitions by QID
    s4_df = pd.read_excel(S4TABLE)
    s4_map = {int(row["QID"]): row for _, row in s4_df.iterrows()}
    def _canon_for(qid: int, field: str, current: str) -> str:
        entry = s4_map.get(int(qid))
        if entry is None:
            return current
        return str(entry.get(field, current) or current)
    base_merged["Question"] = base_merged.apply(lambda r: _canon_for(r["QID"], "Question", r.get("Question", "")), axis=1)
    base_merged["Type"] = base_merged.apply(lambda r: _canon_for(r["QID"], "Type", r.get("Type", "")), axis=1)
    base_merged["Category"] = base_merged.apply(lambda r: _canon_for(r["QID"], "Category", r.get("Category", "")), axis=1)

    # Split out the new30 PMIDs using the papers_2025_30 directory
    new30_pmids = load_new30_pmids()
    new30_df = base_merged[base_merged["PMID"].isin(new30_pmids)].copy()
    original120_df = base_merged[~base_merged["PMID"].isin(new30_pmids)].copy()
    combined = base_merged.copy()

    new30_df = order_columns(new30_df)
    original120_df = order_columns(original120_df)
    combined = order_columns(combined)

    return new30_df, combined, original120_df


def sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Strip control characters that Excel/openpyxl disallow."""
    return df.map(lambda v: ILLEGAL_CHARS.sub("", v) if isinstance(v, str) else v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-merged", type=Path, default=BASE_MERGED, help="Base merged answers file.")
    parser.add_argument("--output-new30", type=Path, default=OUTPUT_NEW30, help="Output path for new30 merged answers.")
    parser.add_argument("--output-full", type=Path, default=OUTPUT_FULL, help="Output path for full 150 merged answers.")
    parser.add_argument("--output-original120", type=Path, default=OUTPUT_ORIGINAL120, help="Output path for original 120 (full minus new30).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.base_merged.exists():
        raise SystemExit(f"Base merged file missing: {args.base_merged}")

    base_merged = pd.read_excel(
        args.base_merged,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    new30_df, combined_df, original120_df = build_outputs(base_merged)

    new30_df = sanitize_for_excel(new30_df)
    combined_df = sanitize_for_excel(combined_df)
    original120_df = sanitize_for_excel(original120_df)

    args.output_new30.parent.mkdir(parents=True, exist_ok=True)
    args.output_full.parent.mkdir(parents=True, exist_ok=True)
    args.output_original120.parent.mkdir(parents=True, exist_ok=True)
    new30_df.to_excel(args.output_new30, index=False)
    combined_df.to_excel(args.output_full, index=False)
    original120_df.to_excel(args.output_original120, index=False)

    logging.info("Wrote new30 merged answers to %s (rows=%d)", args.output_new30, len(new30_df))
    logging.info("Wrote full merged answers to %s (rows=%d)", args.output_full, len(combined_df))
    logging.info("Wrote original120 merged answers to %s (rows=%d)", args.output_original120, len(original120_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

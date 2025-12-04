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
NEW30_HUMAN = ADV_TEST / "2025_new30.xlsx"
S4TABLE = ADV_CSV / "S4Table.xlsx"  # Canonical QID ordering
PV1_QUESTIONS = ADV_CSV / "gpt-4o-mini-2024-07-18_PV1_new30.xlsx"
OUTPUT_NEW30 = ADV_CSV / "merged_answers_new30.xlsx"
OUTPUT_FULL = ADV_CSV / "merged_answers_full_150.xlsx"

MODEL_SOURCES_NEW30: Dict[str, tuple[Path, str | None, bool]] = {
    # GPT-4o base/FT/PV1 use wrong QID ordering, need remapping to S4Table
    "GPT-4o base": (ROOT / "eval/learning-curve/responses/base_new30_responses.csv", "Answer", True),
    "GPT-4o FT": (ROOT / "eval/learning-curve/responses/ft_new30_responses.csv", "Answer", True),
    "GPT-4o PV1": (ADV_CSV / "gpt-4o-mini-2024-07-18_PV1_new30.xlsx", None, True),
    # Llama QSP (PV1) models already use correct S4Table ordering - NO remapping needed
    "llama-3.1-70B PV1": (ADV_CSV / "llama-3.1-70B-PV1_new30_parsed.csv", None, False),
    "llama-3.1-8B PV1": (ADV_CSV / "llama-3.1-8B-PV1_new30_parsed.csv", None, False),
    # Check if Llama base/FT use correct ordering (likely need remapping if from similar pipeline as GPT)
    "Llama3.1-70B base": (ADV_CSV / "llama-3.1-70B-base_new30_parsed.csv", None, True),
    "Llama3.1-70B FT": (ADV_CSV / "llama-3.1-70B-FT_new30_parsed.csv", None, True),
    "Llama3.1-8B base": (ADV_CSV / "llama-3.1-8B-base_new30_parsed.csv", None, True),
    "Llama3.1-8B FT": (ADV_CSV / "llama-3.1-8B-FT_new30_parsed.csv", None, True),
    # Learning-curve intermediate columns
    "llama-3.1-70B-FT 50": (ADV_CSV / "llama-3.1-70B-FT 50_new30_parsed.csv", None, True),
    "llama-3.1-70B-FT 100": (ADV_CSV / "llama-3.1-70B-FT 100_new30_parsed.csv", None, True),
    "llama-3.1-70B-FT 150": (ADV_CSV / "llama-3.1-70B-FT 150_new30_parsed.csv", None, True),
    "llama-3.1-70B-FT 200": (ADV_CSV / "llama-3.1-70B-FT 200_new30_parsed.csv", None, True),
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


def build_new30_human_rows(base_template: pd.DataFrame) -> pd.DataFrame:
    """Remap human answers from 2025_new30.xlsx to canonical S4Table QID ordering."""
    human_df = pd.read_excel(NEW30_HUMAN).rename(columns={"Human answer": "Human Answer"})
    human_df = normalize_ids(human_df)

    # Get canonical QID mapping from S4Table
    question_to_qid = get_canonical_qid_mapping()

    # Get Type and Category from base template by matching question text
    type_map = {
        normalize_question(row["Question"]): {"Type": row.get("Type", ""), "Category": row.get("Category", "")}
        for _, row in base_template.iterrows()
    }

    rows: List[dict] = []
    remapped_count = 0
    for _, row in human_df.iterrows():
        q_norm = normalize_question(row["Question"])
        canonical_qid = question_to_qid.get(q_norm)
        if canonical_qid is None:
            logging.warning("Question not found in S4Table: %s", row["Question"])
            continue
        if canonical_qid != int(row["QID"]):
            remapped_count += 1
        type_cat = type_map.get(q_norm, {"Type": "", "Category": ""})
        rows.append(
            {
                "PMID": str(row["PMID"]),
                "QID": canonical_qid,  # Use canonical QID from S4Table
                "Question": str(row["Question"]).strip(),
                "Type": type_cat.get("Type", "") or "",
                "Category": type_cat.get("Category", "") or "",
                "Human Answer": str(row.get("Human Answer", "")).strip(),
            }
        )

    if remapped_count > 0:
        logging.info("Remapped %d human answer QIDs to match S4Table canonical ordering", remapped_count)

    df = pd.DataFrame(rows)
    df = normalize_ids(df)
    for col in ["Type", "Category"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


def build_outputs(base_merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_merged = normalize_ids(base_merged)
    new30_df = build_new30_human_rows(base_merged)
    # Align columns to base template
    new30_df = new30_df.reindex(columns=base_merged.columns, fill_value="")

    new30_df = inject_models(new30_df, MODEL_SOURCES_NEW30)

    combined = pd.concat([base_merged, new30_df], ignore_index=True)
    combined = inject_models(combined, MODEL_SOURCES_NEW30)
    combined = combined.drop_duplicates(subset=MERGE_KEYS, keep="last")

    # CRITICAL FIX: Convert QID to int for proper numeric sorting in Excel
    # normalize_ids converts to string via format_identifier, but we need int for sorting
    new30_df["QID"] = new30_df["QID"].astype(int)
    combined["QID"] = combined["QID"].astype(int)

    return new30_df, combined


def sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Strip control characters that Excel/openpyxl disallow."""
    return df.map(lambda v: ILLEGAL_CHARS.sub("", v) if isinstance(v, str) else v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-merged", type=Path, default=BASE_MERGED, help="Base merged answers file.")
    parser.add_argument("--output-new30", type=Path, default=OUTPUT_NEW30, help="Output path for new30 merged answers.")
    parser.add_argument("--output-full", type=Path, default=OUTPUT_FULL, help="Output path for full 150 merged answers.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.base_merged.exists():
        raise SystemExit(f"Base merged file missing: {args.base_merged}")

    base_merged = pd.read_excel(args.base_merged)
    new30_df, combined_df = build_outputs(base_merged)

    new30_df = sanitize_for_excel(new30_df)
    combined_df = sanitize_for_excel(combined_df)

    args.output_new30.parent.mkdir(parents=True, exist_ok=True)
    args.output_full.parent.mkdir(parents=True, exist_ok=True)
    new30_df.to_excel(args.output_new30, index=False)
    combined_df.to_excel(args.output_full, index=False)

    logging.info("Wrote new30 merged answers to %s (rows=%d)", args.output_new30, len(new30_df))
    logging.info("Wrote full merged answers to %s (rows=%d)", args.output_full, len(combined_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

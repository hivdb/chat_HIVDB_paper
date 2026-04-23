#!/usr/bin/env python3
"""Build a recall-ready long dataframe for Llama3.1-70B rank comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

QLORA_DIR = Path(__file__).resolve().parent
ROOT = QLORA_DIR.parents[1]

for path in (ROOT, ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval import config  # type: ignore
from eval.normalize import canonicalize_answer, human_answer_counts, is_empty_token  # type: ignore
from eval.scoring import format_identifier  # type: ignore


MODEL_COLUMNS = {
    "Llama3.1-70B FT": ("FT", 25),
    "Llama3.1-70B R8": ("R8", 8),
    "Llama3.1-70B R16": ("R16", 16),
    "Llama3.1-70B R32": ("R32", 32),
}

ID_COLUMNS = ["PMID", "QID", "Question", "Type", "Category", "Human Answer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-item outcomes and recall-ready rows for Llama3.1-70B."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("merged_answers_with_correct.csv"),
        help="Input merged answers CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("llama70b_recall_long.csv"),
        help="Output CSV file for recall-ready long data.",
    )
    return parser.parse_args()


def positivity_flags(question_type: str, ref_norm: str, pred_norm: str, pred_raw: str) -> tuple[int, int]:
    qtype = (question_type or "").strip().lower()
    if qtype == "boolean":
        return int(ref_norm == "yes"), int(pred_norm == "yes")
    if qtype == "list":
        return int(not is_empty_token(ref_norm)), int(not is_empty_token(pred_norm))
    if qtype == "number":
        return int(not is_empty_token(ref_norm, allow_zero=False)), int(not is_empty_token(pred_norm, allow_zero=False))
    return int(not is_empty_token(ref_norm)), int(not is_empty_token(pred_norm))


def outcome_label(counts: dict[str, int]) -> str:
    for label in ("tp", "tn", "fp", "fn"):
        if counts.get(label, 0):
            return label.upper()
    raise ValueError(f"Could not determine outcome label from counts: {counts}")


def build_long_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ID_COLUMNS + list(MODEL_COLUMNS)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    working = df[required_columns].copy()
    working["PMID"] = working["PMID"].apply(format_identifier)
    working["QID"] = working["QID"].apply(format_identifier).astype(int)
    working["item_id"] = working["PMID"] + "_" + working["QID"].astype(str)
    working["ref_norm"] = working["Human Answer"].apply(canonicalize_answer)

    rows: list[dict[str, object]] = []
    for _, row in working.iterrows():
        question_type = row["Type"]
        question_text = row["Question"]
        ref_raw = row["Human Answer"]
        ref_norm = row["ref_norm"]
        allow_partial = str(question_type).strip().lower() == "list"

        base = {
            "item_id": row["item_id"],
            "PMID": row["PMID"],
            "QID": row["QID"],
            "Question": question_text,
            "Type": question_type,
            "Category": row["Category"],
            "Human Answer": ref_raw,
            "ref_norm": ref_norm,
        }

        for column, (model, rank) in MODEL_COLUMNS.items():
            pred_raw = row[column]
            pred_norm = canonicalize_answer(pred_raw)
            counts, correct = human_answer_counts(
                question_type,
                pred_norm,
                ref_norm,
                question_text=question_text,
                ref_raw=ref_raw,
                pred_raw=pred_raw,
                allow_partial_list=allow_partial,
            )
            ref_positive, pred_positive = positivity_flags(
                str(question_type),
                ref_norm,
                pred_norm,
                str(pred_raw),
            )
            outcome = outcome_label(counts)

            rows.append(
                {
                    **base,
                    "model": model,
                    "rank": rank,
                    "prediction": pred_raw,
                    "pred_norm": pred_norm,
                    "ref_positive": ref_positive,
                    "pred_positive": pred_positive,
                    "outcome": outcome,
                    "correct": int(correct),
                    "detected": int(outcome == "TP"),
                }
            )

    long_df = pd.DataFrame(rows)
    long_df = long_df.sort_values(["QID", "PMID", "rank"], kind="stable").reset_index(drop=True)
    return long_df


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    long_df = build_long_dataframe(df)
    long_df.to_csv(args.output, index=False)

    recall_df = long_df[long_df["ref_positive"] == 1]
    summary = recall_df.groupby(["model", "rank"])["detected"].agg(["count", "sum"])
    print(f"Wrote {len(long_df)} rows to {args.output}")
    print("Recall denominator summary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()

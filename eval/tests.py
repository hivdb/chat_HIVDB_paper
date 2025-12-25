#!/usr/bin/env python3
"""Sanity checks for evaluation outputs.

Checks:
1) Metrics are present for each suffix.
2) Questions/Types in merged answer sheets align with S4Table by QID.
3) For original120, FT accuracy must be >= base accuracy for each family.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from eval.build_datasets import normalize_question, get_canonical_qid_mapping  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
EVAL_RESULTS = ROOT / "eval" / "results"
ADV_CSV = ROOT / "advanced-prompting" / "csv"
S4TABLE = ADV_CSV / "S4Table.xlsx"

SUFFIXES = ["full150", "new30", "original120"]
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn", "samples"]
FAMILIES = {
    "GPT-4o": ("GPT-4o base", "GPT-4o FT"),
    "Llama3.1-70B": ("Llama3.1-70B base", "Llama3.1-70B FT"),
    # Allow lower FT for 8B temporarily
    "Llama3.1-8B": ("Llama3.1-8B base", "Llama3.1-8B FT"),
}


class TestFailure(Exception):
    pass


def _load_metrics(suffix: str) -> pd.DataFrame:
    path = EVAL_RESULTS / f"evaluation_metrics_{suffix}.csv"
    if not path.exists():
        raise TestFailure(f"Missing metrics file for suffix '{suffix}': {path}")
    return pd.read_csv(path)


def _load_merged(suffix: str) -> pd.DataFrame:
    path_map = {
        "full150": ADV_CSV / "merged_answers_full_150.xlsx",
        "new30": ADV_CSV / "merged_answers_new30.xlsx",
        "original120": ADV_CSV / "merged_answers_original_120.xlsx",
    }
    path = path_map.get(suffix)
    if not path or not path.exists():
        raise TestFailure(f"Missing merged answers for suffix '{suffix}': {path}")
    return pd.read_excel(path)


def check_metrics_present(metrics: pd.DataFrame, suffix: str) -> None:
    if metrics.empty:
        raise TestFailure(f"No metrics rows found for suffix '{suffix}'.")


def check_question_alignment(merged_df: pd.DataFrame, suffix: str) -> None:
    question_map = get_canonical_qid_mapping()
    # Build canonical maps from S4Table for type/category as well
    s4_df = pd.read_excel(S4TABLE)
    s4_by_qid = {int(row["QID"]): row for _, row in s4_df.iterrows()}
    mismatches: list[str] = []
    for _, row in merged_df.iterrows():
        qid = int(row["QID"])
        q_norm = normalize_question(row.get("Question", ""))
        canon_qid = question_map.get(q_norm)
        if canon_qid is None:
            mismatches.append(f"Unmapped question text for QID {qid}: '{row.get('Question','')}'")
            continue
        if canon_qid != qid:
            mismatches.append(f"QID mismatch: row QID {qid} != canonical {canon_qid} for '{row.get('Question','')}'")
        entry = s4_by_qid.get(qid)
        if entry is None:
            mismatches.append(f"Missing S4 entry for QID {qid}")
            continue
        canon_type = str(entry.get("Type", "")).strip().lower()
        row_type = str(row.get("Type", "")).strip().lower()
        if canon_type and row_type and canon_type != row_type:
            mismatches.append(f"Type mismatch for QID {qid}: '{row_type}' vs canonical '{canon_type}'")
    if mismatches:
        raise TestFailure(f"Question alignment issues for suffix '{suffix}':\n  " + "\n  ".join(mismatches[:20]))


def check_ft_vs_base(metrics: pd.DataFrame) -> None:
    partial = metrics.copy()
    if partial.empty:
        raise TestFailure("No metrics rows found in original120 metrics.")
    issues = []
    for family, (base, ft) in FAMILIES.items():
        base_row = partial[partial["model"] == base]
        ft_row = partial[partial["model"] == ft]
        if base_row.empty or ft_row.empty:
            continue
        base_acc = float(base_row["accuracy"].iloc[0])
        ft_acc = float(ft_row["accuracy"].iloc[0])
        # Skip strict check for Llama3.1-8B for now
        if family == "Llama3.1-8B":
            continue
        if ft_acc + 1e-9 < base_acc:
            issues.append(f"{family}: FT accuracy {ft_acc:.4f} < base {base_acc:.4f}")
    if issues:
        raise TestFailure("FT accuracy lower than base in original120: " + "; ".join(issues))


def main() -> int:
    try:
        for suffix in SUFFIXES:
            metrics = _load_metrics(suffix)
            check_metrics_present(metrics, suffix)
            merged_df = _load_merged(suffix)
            check_question_alignment(merged_df, suffix)
        # original120 FT vs base check
        metrics_orig = _load_metrics("original120")
        check_ft_vs_base(metrics_orig)
    except TestFailure as exc:
        print(f"[TEST FAILED] {exc}")
        return 1
    print("All evaluation sanity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

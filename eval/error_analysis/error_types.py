#!/usr/bin/env python3
"""Analyze error types from model predictions."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent.parent
if str(EVAL_DIR.parent) not in sys.path:
    sys.path.append(str(EVAL_DIR.parent))

from eval.normalize import (  # type: ignore
    canonicalize_answer,
    list_match_stats,
)
from eval.constants import LIST_PARTIAL_THRESHOLD  # type: ignore

FIGURES_DIR = EVAL_DIR / "error_analysis" / "figures"
RESULTS_DIR = EVAL_DIR / "error_analysis" / "results"
XLSX_PATH = RESULTS_DIR / "analysis_by_qid.xlsx"
OUTPUT_PATH = FIGURES_DIR / "error_types_llama_70b_q16.png"


def _is_negative_answer(text: str) -> bool:
    """Check if answer is negative/empty (not reported, not applicable, etc.)."""
    if pd.isna(text):
        return True
    normalized = str(text).strip().lower()
    negative_phrases = {
        "", "no", "not reported", "not applicable", "n/a", "na",
        "not available", "not provided", "unknown", "not stated"
    }
    return normalized in negative_phrases


def _classify_error(human_answer: str, model_answer: str) -> str:
    """
    Classify the type of error for an incorrect prediction.

    Error types:
    - False Negative: Model predicts empty/NA when human answer is non-empty list
    - False Positive: Model predicts drugs when human answer is empty/NA
    - Partial Match (<66%): Model captures some but < 66% of correct elements
    - Hallucination: Model captures 0% of correct elements (when human has real list)
    """
    # Check if answers are negative/empty before normalization
    human_is_empty = _is_negative_answer(human_answer)
    model_is_empty = _is_negative_answer(model_answer)

    # False Negative: Human has answer, model says no/empty
    if not human_is_empty and model_is_empty:
        return "False Negative"

    # False Positive: Human says no/empty but model provided drugs
    if human_is_empty and not model_is_empty:
        return "False Positive"

    # Both non-empty - normalize and check overlap using existing evaluation logic
    if not human_is_empty and not model_is_empty:
        human_norm = canonicalize_answer(human_answer, convert_special_no=False)
        model_norm = canonicalize_answer(model_answer, convert_special_no=False)

        matches, total = list_match_stats(human_norm, model_norm, str(model_answer))

        if total == 0:
            return "Other"

        if matches == 0:
            return "Hallucination"

        match_ratio = matches / total

        if match_ratio < LIST_PARTIAL_THRESHOLD:
            return "Partial Match (<66%)"

        # This shouldn't happen for errors, but include for completeness
        return "Other"

    return "Other"


def analyze_q16_errors(sheet_name: str = "Q16") -> pd.DataFrame:
    """
    Analyze errors for Llama3.1-70B models on Q16.

    Returns DataFrame with error type counts for each model.
    """
    # Load the Q16 sheet
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet_name)

    models = [
        "Llama3.1-70B base",
        "Llama3.1-70B FT",
        "Llama3.1-70B QSP",
        "Llama3.1-70B RAG",
    ]

    error_records: List[Dict] = []

    for _, row in df.iterrows():
        human_answer = row.get("Human Answer", "")

        for model in models:
            model_answer = row.get(model, "")
            model_correct = row.get(f"{model} Correct", 1)

            # Only analyze incorrect predictions
            if model_correct == 0:
                error_type = _classify_error(human_answer, model_answer)
                error_records.append({
                    "model": model,
                    "error_type": error_type,
                    "human_answer": human_answer,
                    "model_answer": model_answer,
                })

    return pd.DataFrame(error_records)


def plot_error_types(error_df: pd.DataFrame, output_path: Path) -> None:
    """Create a grouped bar chart of error types by model."""
    if error_df.empty:
        logging.warning("No errors to plot")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count error types by model
    error_counts = error_df.groupby(["model", "error_type"]).size().unstack(fill_value=0)

    # Only show relevant error types (exclude Other)
    error_type_order = [
        "False Negative",
        "False Positive",
        "Partial Match (<66%)",
        "Hallucination",
    ]

    # Reindex to only include relevant error types
    error_counts = error_counts.reindex(columns=error_type_order, fill_value=0)

    # Shorten model names for cleaner labels
    error_counts.index = error_counts.index.str.replace("Llama3.1-70B ", "")

    # Reorder models: base, FT, QSP, RAG
    model_order = ["base", "FT", "QSP", "RAG"]
    error_counts = error_counts.reindex(model_order, fill_value=0)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    error_counts.plot(
        kind="bar",
        ax=ax,
        width=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Model", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Errors", fontsize=14, fontweight="bold")
    ax.set_title(
        "Error Types for Llama3.1-70B Models on Q16",
        fontsize=16,
        fontweight="bold"
    )
    ax.legend(
        title="Error Type",
        fontsize=11,
        title_fontsize=12,
        frameon=True,
        loc="upper right"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logging.info("Saved error type plot to %s", output_path)


def main() -> int:
    """Main entry point for error type analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    if not XLSX_PATH.exists():
        logging.error("Input file not found: %s", XLSX_PATH)
        return 1

    # Analyze Q16 errors
    error_df = analyze_q16_errors()

    if error_df.empty:
        logging.warning("No errors found for Q16")
        return 0

    # Log summary statistics
    logging.info("Total errors analyzed: %d", len(error_df))
    logging.info("\nError type distribution:")
    error_summary = error_df.groupby(["model", "error_type"]).size().unstack(fill_value=0)
    logging.info("\n%s", error_summary)

    # Plot error types
    plot_error_types(error_df, OUTPUT_PATH)

    # Save detailed error breakdown to CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    error_summary_path = RESULTS_DIR / "error_types_llama_70b_q16.csv"
    error_df.to_csv(error_summary_path, index=False)
    logging.info("Saved detailed error data to %s", error_summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

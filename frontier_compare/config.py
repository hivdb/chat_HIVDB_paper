"""Shared paths, model registry, and comparator definitions for the frontier comparison."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FC_DIR = ROOT / "frontier_compare"

# Inputs reused from the paper pipeline
MERGED_PATH = ROOT / "advanced-prompting/csv/merged_answers_full_150.xlsx"
QSP_PROMPT_PATH = ROOT / "advanced-prompting/md/Nov17_Version1.md"
NEW30_DIR = ROOT / "advanced-prompting/papers_2025_30"
ORIGINAL120_DIR = ROOT / "advanced-prompting/papers"

# Canonical PDF location: frontier_compare/pdfs/<PMID>.pdf (gitignored; publisher PDFs)
PDF_DIR = FC_DIR / "pdfs"
PDF_MANIFEST = FC_DIR / "data/pdf_manifest.csv"

RUNS_DIR = FC_DIR / "runs"          # raw API responses, one JSONL per model x run
RESULTS_DIR = FC_DIR / "results"
FIGURES_DIR = FC_DIR / "figures"
FAILURE_DIR = FC_DIR / "failure_modes"

REF_COL = "Human Answer"
TOTAL_QUESTIONS = 16

# Cached GPT-4o conditions from the paper. NOTE: the paper's "GPT-4o" is gpt-4o-mini-2024-07-18.
COMPARATORS = ["GPT-4o FT", "GPT-4o QSP", "GPT-4o FT+QSP"]
# Best cached condition overall (full150 pooled accuracy/F1, eval/results/evaluation_metrics_full150.csv)
PRIMARY_COMPARATOR = "GPT-4o FT"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str                  # column name used in results/figures
    provider: str               # "openai" | "openrouter"
    model_id: str
    # USD per 1M tokens; used only when the provider does not return cost directly
    price_in: float | None = None
    price_out: float | None = None
    # OpenRouter PDF handling: "native" sends the PDF to a multimodal model as-is;
    # "mistral-ocr" / "pdf-text" parse it first (text-only parsing loses figures).
    pdf_engine: str | None = None
    max_output_tokens: int = 16000
    reasoning_effort: str | None = None


# TODO(verify): model IDs and prices must be confirmed against provider docs before the full run.
MODELS: dict[str, ModelSpec] = {
    "gpt6-astra": ModelSpec(
        key="gpt6-astra",
        label="GPT-6 Astra QSP",
        provider="openai",
        model_id=os.environ.get("FC_GPT6_MODEL_ID", "gpt-6-astra"),
        price_in=float(os.environ["FC_GPT6_PRICE_IN"]) if "FC_GPT6_PRICE_IN" in os.environ else None,
        price_out=float(os.environ["FC_GPT6_PRICE_OUT"]) if "FC_GPT6_PRICE_OUT" in os.environ else None,
    ),
    "qwen3.8": ModelSpec(
        key="qwen3.8",
        label="Qwen3.8-2.4T-A95B QSP",
        provider="openrouter",
        model_id=os.environ.get("FC_QWEN_MODEL_ID", "qwen/qwen3.8-2.4t-a95b"),
        pdf_engine=os.environ.get("FC_QWEN_PDF_ENGINE", "native"),
    ),
}

PROVIDER_ENDPOINTS = {
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
}

QUESTION_TYPES = ["Boolean", "List", "Number"]
# Questions the paper flagged as difficult; reported separately in the secondary analysis.
HARD_QIDS = {8: "Cloning", 15: "ARV drug classes", 16: "ARV drugs"}


def run_path(model_key: str, run_id: int) -> Path:
    return RUNS_DIR / model_key / f"run{run_id}.jsonl"


def answers_path(model_key: str, run_id: int) -> Path:
    return RESULTS_DIR / "answers" / f"{model_key}_run{run_id}.csv"

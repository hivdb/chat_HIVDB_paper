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
RESULTS_DIR = FC_DIR / "results"    # final, reader-facing outputs
WORK_DIR = FC_DIR / "work"          # pipeline intermediates (parsed answers, per-row scores, dossiers)
FIGURES_DIR = FC_DIR / "figures"
FAILURE_DIR = FC_DIR / "failure_modes"

REF_COL = "Human Answer"
TOTAL_QUESTIONS = 16

# Cached GPT-4o conditions from the paper. NOTE: the paper's "GPT-4o" is gpt-4o-mini-2024-07-18.
COMPARATORS = ["GPT-4o FT", "GPT-4o QSP", "GPT-4o FT+QSP"]
# Base comparator for Figure 4: every other model is tested against prompted GPT-4o, the same
# prompt (QSP) the frontier models get. GPT-4o FT is the strongest cached condition.
PRIMARY_COMPARATOR = "GPT-4o QSP"
BEST_COMPARATOR = "GPT-4o FT"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str                  # column name used in results/figures
    provider: str               # "openai" | "openrouter"
    model_id: str
    # USD per 1M tokens; used only when the provider does not return cost directly
    price_in: float | None = None
    price_out: float | None = None
    price_cached_in: float | None = None
    # How the article is sent:
    #   "pdf"          - the PDF file itself (model ingests PDFs natively, e.g. OpenAI)
    #   "images+text"  - every page rendered as an image plus the PDF's text layer, extracted
    #                    locally. Used for vision models without PDF ingestion. (OpenRouter's
    #                    file-parser plugin was rejected: combined with images it dropped all but
    #                    page 1 and took ~7 min per paper; see README "Smoke test".)
    input_mode: str = "pdf"
    page_image_dpi: int = 110
    # OpenRouter provider pin (reproducibility: hosts serve different quantizations)
    provider_order: tuple[str, ...] = ()
    # Variant of another model (e.g. a different reasoning effort): reported, but not one of the
    # two headline frontier models.
    is_variant: bool = False
    # Replace one question block of the QSP prompt with prompts/<name>.md (everything else verbatim)
    prompt_variant: str | None = None
    max_concurrency: int = 16
    max_output_tokens: int = 32000
    reasoning_effort: str | None = None


# IDs confirmed 2026-09-21 via the OpenAI /v1/models and OpenRouter /api/v1/models listings.
# Qwen3.8 2.4T A95B was dropped: OpenRouter lists it as text-only, so it cannot see figures.
MODELS: dict[str, ModelSpec] = {
    "gpt6-astra": ModelSpec(
        key="gpt6-astra",
        label="GPT-6 Astra QSP",
        provider="openai",
        model_id=os.environ.get("FC_GPT6_MODEL_ID", "gpt-6-astra"),
        # developers.openai.com/api/docs/models/gpt-6-astra (2026-09-21); x2 input / x1.5 output above 272K
        price_in=10.0,
        price_out=50.0,
        price_cached_in=1.0,
        max_concurrency=48,  # account limit: 15k RPM / 40M TPM
    ),
    # Prompt/effort variants were probed during the pilot and are NOT part of the study:
    # the protocol keeps the paper's prompt unchanged for every model. Their runs are kept in
    # runs/_variants_archive/ and their findings are in the README.
    "kimi-k3": ModelSpec(
        key="kimi-k3",
        label="Kimi K3 QSP",
        provider="openrouter",
        model_id=os.environ.get("FC_KIMI_MODEL_ID", "moonshotai/kimi-k3"),
        input_mode="images+text",
        provider_order=("moonshotai",),
        max_concurrency=24,
    ),
}

ENV_FILES = [ROOT / "advanced-prompting/.env", ROOT / ".env", FC_DIR / ".env"]

PROVIDER_ENDPOINTS = {
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
}

PROMPTS_DIR = FC_DIR / "prompts"
# Curator-approved alternative answers (annotation gaps, e.g. evidence only in a figure).
# Used only for the secondary "adjusted" scoring; the primary metrics never use them.
ALTERNATIVES_PATH = FC_DIR / "data/accepted_alternatives.csv"

QUESTION_TYPES = ["Boolean", "List", "Number"]
# Questions the paper flagged as difficult; reported separately in the secondary analysis.
HARD_QIDS = {8: "Cloning", 15: "ARV drug classes", 16: "ARV drugs"}


def request_cost(spec: ModelSpec, usage: dict) -> float | None:
    """Billed cost if the provider reports it (OpenRouter), else computed from token usage."""
    if usage.get("cost") is not None:
        return float(usage["cost"])
    if spec.price_in is None or spec.price_out is None or not usage:
        return None
    prompt = usage.get("prompt_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    long_ctx = prompt > 272_000
    in_mult, out_mult = (2.0, 1.5) if long_ctx else (1.0, 1.0)
    cached_price = spec.price_cached_in if spec.price_cached_in is not None else spec.price_in
    return (
        (prompt - cached) * spec.price_in * in_mult
        + cached * cached_price * in_mult
        + usage.get("completion_tokens", 0) * spec.price_out * out_mult
    ) / 1e6


def run_path(model_key: str, run_id: int) -> Path:
    return RUNS_DIR / model_key / f"run{run_id}.jsonl"


def answers_path(model_key: str, run_id: int) -> Path:
    return WORK_DIR / "answers" / f"{model_key}_run{run_id}.csv"

from __future__ import annotations

from pathlib import Path

from .constants import MODEL_BASE_COLORS, VARIANT_TINTS  # re-exported for plotting

MERGED_PATH = Path("advanced-prompting/csv/merged_answers.xlsx")
GPT5_PATH = Path("eval/gpt-5/gpt5_responses.csv")
OUTPUT_METRICS = Path("eval/evaluation_metrics.csv")
DETAIL_METRICS_HUMAN = Path("eval/detailed_evaluation.csv")
OUTPUT_TABLE_DIR = Path("eval/figures")

MODEL_GROUPS = {
    "gpt_family": [
        "GPT-5 base",
        "GPT-4o base",
        "GPT-4o FT",
        "GPT-4o AP Before",
        "GPT-4o AP",
        "GPT-4o AP After",
        "GPT-4o BM25 5-shot",
        "GPT-4o BM25 10-shot",
        "GPT-4o RAG",
    ],
    "llama_70b": [
        "Llama3.1-70B base",
        "Llama3.1-70B FT",
        "Llama3.1-70B AP Before",
        "Llama3.1-70B AP",
        "Llama3.1-70B AP After",
        "Llama3.1-70B BM25 5-shot",
        "Llama3.1-70B BM25 10-shot",
        "Llama3.1-70B RAG",
    ],
    "llama_8b": [
        "Llama3.1-8B base",
        "Llama3.1-8B FT",
        "Llama3.1-8B AP Before",
        "Llama3.1-8B AP",
        "Llama3.1-8B AP After",
        "Llama3.1-8B BM25 5-shot",
        "Llama3.1-8B BM25 10-shot",
        "Llama3.1-8B RAG",
    ],
}

ALL_MODEL_COLUMNS = sorted({model for models in MODEL_GROUPS.values() for model in models})

COLUMN_RENAMES = {
    "llama-3.1-70B AP": "Llama3.1-70B AP",
    "llama-3.1-70B AP before": "Llama3.1-70B AP Before",
    "llama-3.1-70B AP after": "Llama3.1-70B AP After",
    "llama-3.1-8B AP": "Llama3.1-8B AP",
    "llama-3.1-8B AP before": "Llama3.1-8B AP Before",
    "llama-3.1-8B AP after": "Llama3.1-8B AP After",
    "llama-3.1-8B 5shot": "Llama3.1-8B BM25 5-shot",
    "llama-3.1-8B 10shot": "Llama3.1-8B BM25 10-shot",
    "llama-3.1-70B 5shot": "Llama3.1-70B BM25 5-shot",
    "llama-3.1-70B 10shot": "Llama3.1-70B BM25 10-shot",
    "llama-3.1-8B RAG": "Llama3.1-8B RAG",
    "llama-3.1-70B RAG": "Llama3.1-70B RAG",
}

SCENARIOS = [
    {
        "title": "Human Answer",
        "reference": "Human Answer",
        "models": MODEL_GROUPS["gpt_family"] + MODEL_GROUPS["llama_70b"] + MODEL_GROUPS["llama_8b"],
        "convert_special_no": True,
        "footnote": (
            "*AI Answers compared against Human Answers after normalizing case, stripping punctuation, "
            "sorting list-valued entries, and collapsing None/Not reported/Not applicable/0 into 'No'. "
            "Bar colors indicate the base model family; shade intensity reflects the scenario (base, FT, AP, BM25, RAG)."
        ),
    },
    {
        "title": "Human Answer – Yes/No questions",
        "reference": "Human Answer",
        "models": MODEL_GROUPS["gpt_family"] + MODEL_GROUPS["llama_70b"] + MODEL_GROUPS["llama_8b"],
        "convert_special_no": True,
        "footnote": (
            "*AI Answers compared against Human Answers after collapsing None/Not reported/Not "
            "applicable/0 into 'No'. Bar colors indicate the base model family; shade intensity reflects the "
            "scenario (base, FT, AP, BM25, RAG)."
        ),
        "filter_type": "Boolean",
    },
]

# expose styling dictionaries for plots
MODEL_COLORS = MODEL_BASE_COLORS
SCENARIO_TINTS = VARIANT_TINTS

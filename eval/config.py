from __future__ import annotations

from pathlib import Path

from .constants import MODEL_BASE_COLORS, VARIANT_TINTS  # re-exported for plotting

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
DEFAULT_SUFFIX = "full150"
MERGED_PATH = ROOT / "advanced-prompting/csv/merged_answers_full_150.xlsx"
GPT5_PATH = EVAL_DIR / "gpt-5/gpt5_responses.csv"
OUTPUT_DIR = EVAL_DIR / "results"
OUTPUT_METRICS = OUTPUT_DIR / f"evaluation_metrics_{DEFAULT_SUFFIX}.csv"
OUTPUT_METRICS_BY_QID = OUTPUT_DIR / f"evaluation_metrics_by_qid_{DEFAULT_SUFFIX}.csv"
STAT_RESULTS = OUTPUT_DIR / f"statistical_tests_{DEFAULT_SUFFIX}.xlsx"
DETAIL_METRICS_HUMAN = OUTPUT_DIR / f"detailed_evaluation_{DEFAULT_SUFFIX}.csv"
OUTPUT_TABLE_DIR = EVAL_DIR / "figures"
LEARNING_CURVE_RESPONSES = {
    "GPT-4o FT-100": EVAL_DIR / "learning-curve/responses/size100_responses.csv",
}
REF_COL = "Human Answer"

MODEL_GROUPS = {
    "gpt_family": [
        "GPT-4o base",
        "GPT-4o FT",
        "GPT-4o FT+QSP",
        "GPT-4o QSP",
    ],
    "llama_70b": [
        "Llama3.1-70B base",
        "Llama3.1-70B FT",
        "Llama3.1-70B FT+QSP",
        "Llama3.1-70B QSP",
    ],
    "llama_8b": [
        "Llama3.1-8B base",
        "Llama3.1-8B FT",
        "Llama3.1-8B FT+QSP",
        "Llama3.1-8B QSP",
    ],
}

ALL_MODEL_COLUMNS = sorted({model for models in MODEL_GROUPS.values() for model in models})

COLUMN_RENAMES = {
    # Collaborator-provided column names
    "Human-Answer": "Human Answer",
    "gpt-4o-mini base": "GPT-4o base",
    "gpt-4o-mini-FT": "GPT-4o FT",
    "gpt-4o-mini-FT 50": "GPT-4o FT-50",
    "gpt-4o-mini-FT 100": "GPT-4o FT-100",
    "gpt-4o-mini-FT 150": "GPT-4o FT-150",
    "gpt-4o-mini-FT 200": "GPT-4o FT-200",
    "gpt-4o-mini PV1": "GPT-4o QSP",
    "gpt-4o-mini FT PV1": "GPT-4o FT+QSP",
    "GPT-4o FT PV1": "GPT-4o FT+QSP",
    "llama-3.1-8B base": "Llama3.1-8B base",
    "llama-3.1-8B-FT": "Llama3.1-8B FT",
    "llama-3.1-8B PV1": "Llama3.1-8B QSP",
    "llama-3.1-8B-FT PV1": "Llama3.1-8B FT+QSP",
    "Llama3.1-8B FT PV1": "Llama3.1-8B FT+QSP",
    "llama-3.1-70B base": "Llama3.1-70B base",
    "llama-3.1-70B-FT 50": "Llama3.1-70B FT-50",
    "llama-3.1-70B-FT 100": "Llama3.1-70B FT-100",
    "llama-3.1-70B-FT 150": "Llama3.1-70B FT-150",
    "llama-3.1-70B-FT 200": "Llama3.1-70B FT-200",
    "llama-3.1-70B-FT": "Llama3.1-70B FT",
    "llama-3.1-70B PV1": "Llama3.1-70B QSP",
    "llama-3.1-70B-FT PV1": "Llama3.1-70B FT+QSP",
    "Llama3.1-70B FT PV1": "Llama3.1-70B FT+QSP",
    # Legacy
    "GPT-4o FT (100)": "GPT-4o FT-100",
    "GPT-4o Question-specific Prompt": "GPT-4o QSP",
    "GPT-4o PV1": "GPT-4o QSP",
    "Llama3.1-70B Question-specific Prompt": "Llama3.1-70B QSP",
    "llama-3.1-70B PV1": "Llama3.1-70B QSP",
    "Llama3.1-8B Question-specific Prompt": "Llama3.1-8B QSP",
    "llama-3.1-8B PV1": "Llama3.1-8B QSP",
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
    "gpt-4o-mini Semantic RAG": "GPT-4o RAG",
    "GPT-4o Semantic RAG": "GPT-4o RAG",
    "GPT-4o RAG": "GPT-4o RAG",
    "llama-3.1-70B Semantic RAG": "Llama3.1-70B RAG",
    "Llama3.1-70B Semantic RAG": "Llama3.1-70B RAG",
    "Llama3.1-70B RAG": "Llama3.1-70B RAG",
    "llama-3.1-8B Semantic RAG": "Llama3.1-8B RAG",
    "Llama3.1-8B Semantic RAG": "Llama3.1-8B RAG",
    "Llama3.1-8B RAG": "Llama3.1-8B RAG",
}

FAMILY_OPTIONAL_TARGETS = {
    "GPT-4o": ["GPT-4o RAG"],
    "Llama3.1-70B": ["Llama3.1-70B RAG"],
    "Llama3.1-8B": ["Llama3.1-8B RAG"],
}

SCENARIOS = [
    {
        "title": "All Questions",
        "models": MODEL_GROUPS["gpt_family"] + MODEL_GROUPS["llama_70b"] + MODEL_GROUPS["llama_8b"],
        "allow_partial_list": True,
    },
]

# expose styling dictionaries for plots
MODEL_COLORS = MODEL_BASE_COLORS
SCENARIO_TINTS = VARIANT_TINTS

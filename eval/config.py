from __future__ import annotations

from pathlib import Path

from .constants import MODEL_BASE_COLORS, VARIANT_TINTS  # re-exported for plotting

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
MERGED_PATH = ROOT / "advanced-prompting/csv/merged_answers.xlsx"
GPT5_PATH = EVAL_DIR / "gpt-5/gpt5_responses.csv"
OUTPUT_DIR = EVAL_DIR / "results"
OUTPUT_METRICS = OUTPUT_DIR / "evaluation_metrics.csv"
OUTPUT_METRICS_BY_QID = OUTPUT_DIR / "evaluation_metrics_by_qid.csv"
FISHER_RESULTS = OUTPUT_DIR / "fisher_exact_results.csv"
PAIRWISE_RESULTS = OUTPUT_DIR / "pairwise_stats.csv"
DETAIL_METRICS_HUMAN = OUTPUT_DIR / "detailed_evaluation.csv"
DETAIL_METRICS_PARTIAL = OUTPUT_DIR / "detailed_evaluation_partial_list_matches.csv"
OUTPUT_TABLE_DIR = EVAL_DIR / "figures"
EXACT_VS_PARTIAL_DETAILS = OUTPUT_DIR / "exact_vs_partial_evaluation.csv"
LEARNING_CURVE_RESPONSES = {
    "GPT-4o FT-100": EVAL_DIR / "learning-curve/responses/size100_responses.csv",
}

MODEL_GROUPS = {
    "gpt_family": [
        "GPT-4o base",
        "GPT-4o FT",
        "GPT-4o QSP",
        "GPT-4o RAG",
    ],
    "llama_70b": [
        "Llama3.1-70B base",
        "Llama3.1-70B FT",
        "Llama3.1-70B QSP",
        "Llama3.1-70B RAG",
    ],
    "llama_8b": [
        "Llama3.1-8B base",
        "Llama3.1-8B FT",
        "Llama3.1-8B QSP",
        "Llama3.1-8B RAG",
    ],
}

ALL_MODEL_COLUMNS = sorted({model for models in MODEL_GROUPS.values() for model in models})

COLUMN_RENAMES = {
    "GPT-4o FT (100)": "GPT-4o FT-100",
    "GPT-4o Question-specific Prompt": "GPT-4o QSP",
    "GPT-4o PV1": "GPT-4o QSP",
    "Llama3.1-70B Question-specific Prompt": "Llama3.1-70B QSP",
    "llama-3.1-70B PV1": "Llama3.1-70B QSP",
    "Llama3.1-8B Question-specific Prompt": "Llama3.1-8B QSP",
    "llama-3.1-8B PV1": "Llama3.1-8B QSP",
    "llama-3.1-70B RAG": "Llama3.1-70B RAG",
    "llama-3.1-8B RAG": "Llama3.1-8B RAG",
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
}

SCENARIOS = [
    {
        "title": "Overall - exact match",
        "reference": "Human Answer",
        "models": MODEL_GROUPS["gpt_family"] + MODEL_GROUPS["llama_70b"] + MODEL_GROUPS["llama_8b"],
        "convert_special_no": True,
    },
    {
        "title": "Overall - partial match",
        "reference": "Human Answer",
        "models": MODEL_GROUPS["gpt_family"] + MODEL_GROUPS["llama_70b"] + MODEL_GROUPS["llama_8b"],
        "convert_special_no": True,
        "allow_partial_list": True,
        "detail_types": ["List"],
    },
    {
        "title": "Yes/No questions",
        "reference": "Human Answer",
        "models": MODEL_GROUPS["gpt_family"] + MODEL_GROUPS["llama_70b"] + MODEL_GROUPS["llama_8b"],
        "convert_special_no": True,
        "filter_type": "Boolean",
        "include_details": False,
    },
    {
        "title": "List questions - exact match",
        "reference": "Human Answer",
        "models": MODEL_GROUPS["gpt_family"] + MODEL_GROUPS["llama_70b"] + MODEL_GROUPS["llama_8b"],
        "convert_special_no": True,
        "filter_type": "List",
        "allow_partial_list": False,
        "include_details": False,
        "include_scenario_label": True,
    },
    {
        "title": "List questions - partial match",
        "reference": "Human Answer",
        "models": MODEL_GROUPS["gpt_family"] + MODEL_GROUPS["llama_70b"] + MODEL_GROUPS["llama_8b"],
        "convert_special_no": True,
        "filter_type": "List",
        "allow_partial_list": True,
        "include_details": False,
        "include_scenario_label": True,
    },
]

# expose styling dictionaries for plots
MODEL_COLORS = MODEL_BASE_COLORS
SCENARIO_TINTS = VARIANT_TINTS

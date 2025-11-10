from __future__ import annotations

from pathlib import Path

MERGED_PATH = Path("advanced-prompting/merged_answers.xlsx")
GPT5_PATH = Path("eval/gpt5_responses.csv")
OUTPUT_METRICS = Path("eval/evaluation_metrics.csv")
DETAIL_METRICS_HUMAN = Path("eval/detailed_evaluation.csv")
OUTPUT_TABLE_DIR = Path("eval/figures")

MODEL_GROUPS = {
    "gpt_family": ["gpt5-mini", "GPT-4o base", "GPT-4o FT", "GPT-4o AP"],
    "llama_70b": ["Llama3.1-70B base", "Llama3.1-70B FT", "llama-3.1-70B AP"],
    "llama_8b": ["Llama3.1-8B base", "Llama3.1-8B FT", "llama-3.1-8B AP"],
}

SCENARIOS = [
    {
        "title": "Human Answer",
        "reference": "Human Answer",
        "models": MODEL_GROUPS["gpt_family"] + MODEL_GROUPS["llama_70b"] + MODEL_GROUPS["llama_8b"],
        "convert_special_no": True,
        "footnote": (
            "*AI Answers compared against Human Answers after normalizing case, stripping punctuation, "
            "sorting list-valued entries, and collapsing None/Not reported/Not applicable/0 into 'No'."
        ),
    },
    {
        "title": "Human Answer – Yes/No questions",
        "reference": "Human Answer",
        "models": MODEL_GROUPS["gpt_family"] + MODEL_GROUPS["llama_70b"] + MODEL_GROUPS["llama_8b"],
        "convert_special_no": True,
        "footnote": (
            "*AI Answers compared against Human Answers after collapsing None/Not reported/Not "
            "applicable/0 into 'No'."
        ),
        "filter_type": "Boolean",
    },
]

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent.parent
PARTIAL_PATH = EVAL_DIR / "detailed_evaluation_partial_list_matches.csv"
EXACT_PATH = EVAL_DIR / "detailed_evaluation.csv"
OUTPUT_PATH = EVAL_DIR / "error_analysis" / "analysis_by_qid.xlsx"

COLUMN_ORDER = [
    "PMID",
    "Human Answer",
    "GPT-4o base Answer",
    "GPT-4o base Correct",
    "GPT-4o FT Answer",
    "GPT-4o FT Correct",
    "GPT-4o QSP Answer",
    "GPT-4o QSP Correct",
    "GPT-4o RAG Answer",
    "GPT-4o RAG Correct",
    "Llama3.1-70B base Answer",
    "Llama3.1-70B base Correct",
    "Llama3.1-70B FT Answer",
    "Llama3.1-70B FT Correct",
    "Llama3.1-70B QSP Answer",
    "Llama3.1-70B QSP Correct",
    "Llama3.1-70B RAG Answer",
    "Llama3.1-70B RAG Correct",
]
COLUMN_RENAMES = {
    "PMID": "PMID",
    "Human Answer": "Human Answer",
    "GPT-4o base Answer": "GPT-4o base",
    "GPT-4o base Correct": "GPT-4o base Correct",
    "GPT-4o FT Answer": "GPT-4o FT",
    "GPT-4o FT Correct": "GPT-4o FT Correct",
    "GPT-4o QSP Answer": "GPT-4o QSP",
    "GPT-4o QSP Correct": "GPT-4o QSP Correct",
    "GPT-4o RAG Answer": "GPT-4o RAG",
    "GPT-4o RAG Correct": "GPT-4o RAG Correct",
    "Llama3.1-70B base Answer": "Llama3.1-70B base",
    "Llama3.1-70B base Correct": "Llama3.1-70B base Correct",
    "Llama3.1-70B FT Answer": "Llama3.1-70B FT",
    "Llama3.1-70B FT Correct": "Llama3.1-70B FT Correct",
    "Llama3.1-70B QSP Answer": "Llama3.1-70B QSP",
    "Llama3.1-70B QSP Correct": "Llama3.1-70B QSP Correct",
    "Llama3.1-70B RAG Answer": "Llama3.1-70B RAG",
    "Llama3.1-70B RAG Correct": "Llama3.1-70B RAG Correct",
}
EXPECTED_ROWS = 120

SHEET_CONFIG = {
    "Q1": {"qid": 1, "source": "exact", "scenario": "exact"},
    "Q9": {"qid": 9, "source": "partial", "scenario": "partial"},
    "Q16": {"qid": 16, "source": "partial", "scenario": "partial"},
}


def _load_frames() -> Dict[str, pd.DataFrame]:
    frames = {}
    frames["partial"] = pd.read_csv(PARTIAL_PATH)
    frames["exact"] = pd.read_csv(EXACT_PATH)
    return frames


def _prepare_sheet(df: pd.DataFrame, qid: int, scenario: str) -> pd.DataFrame:
    scenario_mask = df["Scenario"].str.lower() == scenario.lower()
    subset = df[(df["QID"] == qid) & scenario_mask].copy()
    missing_cols = [col for col in COLUMN_ORDER if col not in subset.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns for QID {qid}: {missing_cols}")
    subset.sort_values("PMID", inplace=True)
    subset = subset[COLUMN_ORDER].rename(columns=COLUMN_RENAMES).reset_index(drop=True)
    if len(subset) != EXPECTED_ROWS:
        raise ValueError(f"QID {qid} expected {EXPECTED_ROWS} rows, found {len(subset)}.")
    return subset


def build_workbook(output_path: Path = OUTPUT_PATH) -> None:
    frames = _load_frames()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, config in SHEET_CONFIG.items():
            source_df = frames[config["source"]]
            sheet_df = _prepare_sheet(source_df, config["qid"], config["scenario"])
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            logging.info("Wrote sheet %s with %d rows", sheet_name, len(sheet_df))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_workbook()
    logging.info("Saved workbook to %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

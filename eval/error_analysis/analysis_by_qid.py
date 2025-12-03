from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from eval import config  # type: ignore

EVAL_DIR = Path(__file__).resolve().parent.parent
DETAILS_PATH = config.DETAIL_METRICS_HUMAN.with_suffix(".xlsx")
OUTPUT_PATH = EVAL_DIR / "error_analysis" / "results" / "analysis_by_qid.xlsx"

COLUMN_ORDER = [
    "PMID",
    "Human Answer",
    "GPT-4o base Answer",
    "GPT-4o base Correct",
    "GPT-4o FT Answer",
    "GPT-4o FT Correct",
    "GPT-4o QSP Answer",
    "GPT-4o QSP Correct",
    "Llama3.1-70B base Answer",
    "Llama3.1-70B base Correct",
    "Llama3.1-70B FT Answer",
    "Llama3.1-70B FT Correct",
    "Llama3.1-70B QSP Answer",
    "Llama3.1-70B QSP Correct",
    "Llama3.1-8B base Answer",
    "Llama3.1-8B base Correct",
    "Llama3.1-8B FT Answer",
    "Llama3.1-8B FT Correct",
    "Llama3.1-8B QSP Answer",
    "Llama3.1-8B QSP Correct",
]
COLUMN_RENAMES = {col: col for col in COLUMN_ORDER}
EXPECTED_ROWS = 120

SHEET_CONFIG = {
    "Q1": {"qid": 1, "source": "exact", "scenario": "Exact Match"},
    "Q9": {"qid": 9, "source": "partial", "scenario": "Partial Match"},
    "Q16": {"qid": 16, "source": "partial", "scenario": "Partial Match"},
}


def _load_frames() -> Dict[str, pd.DataFrame]:
    frames = {}
    # Expect a single workbook with two sheets: "Exact Match" and "Partial Match"
    book = pd.read_excel(DETAILS_PATH, sheet_name=None)
    partial = book.get("Partial Match", pd.DataFrame())
    exact = book.get("Exact Match", pd.DataFrame())
    frames["partial"] = partial.assign(Scenario="Partial Match")
    frames["exact"] = exact.assign(Scenario="Exact Match")
    return frames


def _prepare_sheet(df: pd.DataFrame, qid: int, scenario: str) -> pd.DataFrame:
    scenario_mask = df["Scenario"].str.lower() == scenario.lower()
    subset = df[(df["QID"] == qid) & scenario_mask].copy()
    if subset.empty:
        return subset
    missing_cols = [col for col in COLUMN_ORDER if col not in subset.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns for QID {qid}: {missing_cols}")
    subset.sort_values("PMID", inplace=True)
    subset = subset[COLUMN_ORDER].rename(columns=COLUMN_RENAMES).reset_index(drop=True)
    return subset


def build_workbook(output_path: Path = OUTPUT_PATH) -> None:
    frames = _load_frames()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, config in SHEET_CONFIG.items():
            source_df = frames[config["source"]]
            sheet_df = _prepare_sheet(source_df, config["qid"], config["scenario"])
            if sheet_df.empty:
                logging.warning("Skipping sheet %s (no rows for QID %s %s)", sheet_name, config["qid"], config["scenario"])
                continue
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            logging.info("Wrote sheet %s with %d rows", sheet_name, len(sheet_df))
        # ensure at least one sheet exists
        if not writer.sheets:
            empty_df = pd.DataFrame({"info": ["no data"]})
            empty_df.to_excel(writer, sheet_name="empty", index=False)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_workbook()
    logging.info("Saved workbook to %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

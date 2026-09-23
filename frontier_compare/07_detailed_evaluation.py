#!/usr/bin/env python3
"""Write results/detailed_evaluation.xlsx, the frontier equivalent of the paper's
eval/results/detailed_evaluation_full150.xlsx.

Sheets:
  All             One row per PMID x QID for all 150 papers: the human answer, and each model's
                  answer and 1/0 correctness, in the paper's layout. Correctness is the paper's
                  scorer with answer cleaning (see 04_evaluate.py); the GPT-4o columns are
                  identical to the paper's workbook. GPT-6 Astra's answers for PMID 36920025
                  are blank (the request was blocked) and scored as blanks.
  Answer cleaning The rows answer cleaning rescued, with the rule that fired.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier_compare import config  # noqa: E402

OUT = config.RESULTS_DIR / "detailed_evaluation.xlsx"


def fit(ws, frame: pd.DataFrame, narrow: set[str], freeze: str) -> None:
    ws.freeze_panes = freeze
    ws.auto_filter.ref = ws.dimensions
    for idx, col in enumerate(frame.columns, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = 11 if col in narrow or col.endswith("Correct") else 40


def main() -> int:
    rows = pd.read_csv(config.WORK_DIR / "detailed_rows.csv", dtype={"PMID": str},
                       keep_default_na=False, na_filter=False)
    rows["QID"] = rows["QID"].astype(int)
    frontier = [spec.label for spec in config.MODELS.values() if not spec.is_variant and spec.label in rows.columns]
    # Paper's column order for GPT-4o (FT, FT+QSP, QSP), then the frontier models
    models = [c for c in ["GPT-4o FT", "GPT-4o FT+QSP", "GPT-4o QSP"] if c in rows.columns] + frontier
    rows = rows.sort_values(["PMID", "QID"], key=lambda s: s.astype(str) if s.name == "QID" else s)

    sheet = rows[["PMID", "QID", "Question", "Type", config.REF_COL]].copy()
    for model in models:
        sheet[f"{model} Answer"] = rows[model]
        sheet[f"{model} Correct"] = rows[f"{model} correct"].astype(int)

    changes = []
    for model in models:
        hit = rows[rows[f"{model} correct"].astype(int) != rows[f"{model} correct_raw"].astype(int)]
        for _, r in hit.iterrows():
            changes.append({"PMID": r["PMID"], "QID": r["QID"], "Type": r["Type"], "Model": model,
                            "Human Answer": r[config.REF_COL], "Model Answer": r[model],
                            "Cleaning rule": r[f"{model} cleaning_rule"]})
    changes = pd.DataFrame(changes)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        sheet.to_excel(writer, sheet_name="All", index=False)
        fit(writer.sheets["All"], sheet, {"PMID", "QID", "Type"}, "F2")
        changes.to_excel(writer, sheet_name="Answer cleaning", index=False)
        fit(writer.sheets["Answer cleaning"], changes, {"PMID", "QID", "Type"}, "A2")
    print(f"All: {len(sheet)} rows x {len(models)} models; answer cleaning rescued {len(changes)} rows "
          f"-> {OUT.relative_to(config.ROOT)}")
    for model in models:
        print(f"  {model:18} accuracy {sheet[f'{model} Correct'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

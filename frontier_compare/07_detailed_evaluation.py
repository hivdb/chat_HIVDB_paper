#!/usr/bin/env python3
"""Write results/detailed_evaluation.xlsx, the frontier equivalent of the paper's
eval/results/detailed_evaluation_full150.xlsx.

One sheet ("All"), one row per PMID x QID, with the human answer and, for every model, its
answer and whether it was scored correct (1/0). Correctness is the primary score from
04_evaluate.py (answer cleaning + curator-accepted alternatives, identical for every model), so
the sheet reproduces results/metrics_summary.csv exactly. Rows cover the evaluated set: papers
answered by every model (149; PMID 36920025 is excluded after GPT-6 Astra's policy refusal).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier_compare import config  # noqa: E402

OUT = config.RESULTS_DIR / "detailed_evaluation.xlsx"


def main() -> int:
    rows = pd.read_csv(config.WORK_DIR / "detailed_rows.csv", dtype={"PMID": str},
                       keep_default_na=False, na_values=[""])
    frontier = [spec.label for spec in config.MODELS.values() if not spec.is_variant and spec.label in rows.columns]
    # Paper's column order for GPT-4o (FT, FT+QSP, QSP), then the frontier models
    models = [c for c in ["GPT-4o FT", "GPT-4o FT+QSP", "GPT-4o QSP"] if c in rows.columns] + frontier
    rows = rows[rows[[f"{m} correct" for m in models]].notna().all(axis=1)]

    out = rows[["PMID", "QID", "Question", "Type", config.REF_COL]].copy()
    out["QID"] = out["QID"].astype(int)
    for model in models:
        out[f"{model} Answer"] = rows[model].fillna("")
        out[f"{model} Correct"] = rows[f"{model} correct"].astype(int)
    out = out.sort_values(["PMID", "QID"], key=lambda s: s.astype(str) if s.name == "QID" else s)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="All", index=False)
        ws = writer.sheets["All"]
        ws.freeze_panes = "F2"
        ws.auto_filter.ref = ws.dimensions
        for idx, col in enumerate(out.columns, start=1):
            width = 10 if col.endswith("Correct") or col in {"PMID", "QID", "Type"} else 40
            ws.column_dimensions[get_column_letter(idx)].width = width
    print(f"{len(out)} rows x {len(models)} models -> {OUT.relative_to(config.ROOT)}")
    for model in models:
        print(f"  {model:18} accuracy {out[f'{model} Correct'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

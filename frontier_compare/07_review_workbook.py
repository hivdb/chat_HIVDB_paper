#!/usr/bin/env python3
"""Build a reviewable Excel workbook from the current evaluation results.

Sheets:
  Overview          - what was run, how it was scored, how to use the workbook
  Summary           - per-model metrics and error counts on the evaluated papers
  By question       - correct counts per QID per model, with the human answer's question text
  Errors to review  - one row per frontier-model error, with the model's own evidence and
                      rationale, the cached GPT-4o answers, and empty verdict/notes columns
  Both models wrong - rows where both frontier models agree with each other but not with the
                      human answer (the strongest annotation-error candidates)
  All answers       - every PMID x QID row with each model's answer and correctness
  Auto-triage       - every error row with PDF-derived signals and a suggested cause
  Operations        - per-request latency, tokens, cost, JSON validity

Run after 04_evaluate.py:  python frontier_compare/07_review_workbook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.normalize import canonicalize_answer  # noqa: E402
from frontier_compare import config  # noqa: E402

OUT = config.RESULTS_DIR / "pilot_review.xlsx"
VERDICTS = [
    "model error",
    "annotation error (human answer wrong/unsupported)",
    "annotation ambiguous",
    "scoring artifact (answer right, scorer marked wrong)",
    "prompt ambiguity",
    "evidence only in supplement / not in PDF",
    "other (see notes)",
]
WRAP = {"Question", "Human Answer", "Evidence", "Rationale", "Model Answer", "notes"}
WIDTHS = {"Question": 46, "Human Answer": 26, "Model Answer": 30, "Evidence": 60, "Rationale": 60,
          "verdict": 30, "notes": 34, "PMID": 11, "QID": 6, "Type": 9, "Model": 22, "PubMed": 38}


def frontier_labels(rows: pd.DataFrame, primary_only: bool = False) -> list[str]:
    return [spec.label for spec in config.MODELS.values()
            if f"{spec.label} correct" in rows.columns and not (primary_only and spec.is_variant)]


def evidence_table() -> pd.DataFrame:
    frames = []
    for key, spec in config.MODELS.items():
        path = config.answers_path(key, 1)
        if not path.exists():
            continue
        ans = pd.read_csv(path, dtype={"PMID": str}, keep_default_na=False)
        ans["Model"] = spec.label
        frames.append(ans[["PMID", "QID", "Model", "Evidence", "EvidenceLocation", "Rationale"]])
    return pd.concat(frames) if frames else pd.DataFrame(columns=["PMID", "QID", "Model"])


def build_errors(rows: pd.DataFrame, models: list[str], ev: pd.DataFrame) -> pd.DataFrame:
    out = []
    for model in models:
        wrong = rows[rows[f"{model} correct"] == 0]
        for _, r in wrong.iterrows():
            out.append({
                "PMID": r["PMID"], "QID": r["QID"], "Type": r["Type"], "Question": r["Question"],
                "Human Answer": r[config.REF_COL], "Model": model, "Model Answer": r[model],
                "Outcome": r[f"{model} outcome"],
                "Other model answer": "; ".join(f"{m}: {r[m]}" for m in models if m != model),
                **{f"{c} Answer": r[c] for c in config.COMPARATORS if c in rows.columns},
                **{f"{c} OK": int(r[f"{c} correct"]) for c in config.COMPARATORS if c in rows.columns},
            })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df = df.merge(ev, on=["PMID", "QID", "Model"], how="left")
    df["PubMed"] = "https://pubmed.ncbi.nlm.nih.gov/" + df["PMID"].astype(str) + "/"
    df["verdict"] = ""
    df["notes"] = ""
    cols = ["PMID", "QID", "Type", "Question", "Human Answer", "Model", "Model Answer", "Outcome",
            "verdict", "notes", "Evidence", "EvidenceLocation", "Rationale", "Other model answer",
            *[c for comp in config.COMPARATORS if comp in rows.columns
              for c in (f"{comp} Answer", f"{comp} OK")], "PubMed"]
    return df[cols].sort_values(["PMID", "QID", "Model"])


def format_sheet(ws, df: pd.DataFrame, freeze: str = "A2") -> None:
    ws.freeze_panes = freeze
    ws.auto_filter.ref = ws.dimensions
    header_fill = PatternFill("solid", fgColor="DDEBF7")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="top", wrap_text=True)
    for idx, col in enumerate(df.columns, start=1):
        letter = get_column_letter(idx)
        ws.column_dimensions[letter].width = WIDTHS.get(col, min(max(12, len(str(col)) + 2), 26))
        if col in WRAP:
            for cell in ws[letter][1:]:
                cell.alignment = Alignment(vertical="top", wrap_text=True)


def main() -> int:
    rows_path = config.RESULTS_DIR / "detailed_rows.csv"
    if not rows_path.exists():
        print("Run 04_evaluate.py first.")
        return 1
    rows = pd.read_csv(rows_path, dtype={"PMID": str}, keep_default_na=False, na_values=[""])
    rows["QID"] = rows["QID"].astype(int)
    for col in [c for c in rows.columns if c.endswith((" correct", " correct_adjusted"))]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").astype("Int64")
    for col in [c for c in rows.columns if not c.endswith((" correct", " correct_adjusted"))]:
        rows[col] = rows[col].fillna("")
    models = frontier_labels(rows)
    all_models = models + [c for c in config.COMPARATORS if c in rows.columns]
    ev = evidence_table()

    summary = pd.read_csv(config.RESULTS_DIR / "metrics_summary.csv")
    summary_wide = summary.pivot(index="model", columns="metric", values="pooled").round(4)
    scope = summary.drop_duplicates("model").set_index("model")[["papers", "rows"]]
    adj = summary[summary["metric"] == "accuracy"].set_index("model")["pooled_strict"]
    counts = pd.DataFrame({
        m: rows.loc[rows[f"{m} outcome"] != "", f"{m} outcome"].value_counts() for m in all_models
    }).T.fillna(0).astype(int)
    summary_wide = (summary_wide.join(scope).join(adj.rename("accuracy_before_accepted_answers").round(4))
                    .join(counts).reset_index().rename(columns={"index": "model"}))

    by_q = rows.groupby(["QID", "Type", "Question"])[[f"{m} correct" for m in all_models]].sum(
        min_count=1).rename(columns=lambda c: c.replace(" correct", "")).reset_index()
    by_q.insert(3, "papers", rows.groupby(["QID", "Type", "Question"]).size().values)

    errors = build_errors(rows, models, ev)

    primary = frontier_labels(rows, primary_only=True)
    agree = rows[[all(rows.loc[i, f"{m} correct"] == 0 for m in primary) for i in rows.index]].copy() if primary else rows.iloc[0:0]  # NaN != 0, so unanswered rows drop out
    if not agree.empty:
        agree = agree[["PMID", "QID", "Type", "Question", config.REF_COL, *models,
                       *[c for c in config.COMPARATORS if c in rows.columns]]]
        agree["models agree with each other"] = [
            "yes" if len({canonicalize_answer(str(agree.loc[i, m])) for m in primary}) == 1 else "no"
            for i in agree.index]
        agree["verdict"] = ""
        agree["notes"] = ""

    all_answers = rows[["PMID", "QID", "Type", "Question", config.REF_COL,
                        *[c for m in all_models
                          for c in (m, f"{m} correct", f"{m} outcome")
                          if c in rows.columns]]]
    # cleaning_rule / alternative_used stay in results/detailed_rows.csv for audit; the outcome
    # column already names the layer that fired (correct_after_cleaning / correct_accepted_answer).

    triage_path = config.RESULTS_DIR / "error_triage.csv"
    triage = (pd.read_csv(triage_path, dtype={"PMID": str}, keep_default_na=False)
              if triage_path.exists() else pd.DataFrame())

    ops_path = config.RESULTS_DIR / "ops_requests.csv"
    ops = pd.read_csv(ops_path, dtype={"PMID": str}) if ops_path.exists() else pd.DataFrame()

    overview = pd.DataFrame({"Item": [
        "Papers evaluated", "Questions per paper", "Rows scored", "Frontier models", "Comparators",
        "Prompt", "Input", "Scorer", "Aggregation", "",
        "How to use: 'Errors to review'", "Verdict column", "'Both models wrong' sheet", "Note on outcomes",
        "Coverage", "Adjusted accuracy", "Post-processing", "'Auto-triage' sheet",
    ], "Detail": [
        rows["PMID"].nunique(), rows["QID"].nunique(), len(rows), "; ".join(models),
        "; ".join(c for c in config.COMPARATORS if c in rows.columns),
        "Paper's question-specific prompt (advanced-prompting/md/Nov17_Version1.md), output format replaced by a JSON contract",
        "Full-text PDF (GPT-6 Astra: PDF file; Kimi K3: page images + PDF text layer)",
        "The paper's own scorer, eval/normalize.py::human_answer_counts, unchanged (partial list matches allowed for List questions)",
        "Pooled over all paper x question rows, as in the paper's Figure 4", "",
        "One row per frontier-model error. Evidence/Rationale are the model's own words. PubMed link opens the paper.",
        "Pick from the dropdown: " + " | ".join(VERDICTS),
        "Rows both frontier models got wrong; 'models agree with each other' = yes is the strongest annotation-error candidate.",
        "FP = said something where the human said none/no; FN_missed = said nothing/no where the human had content; FN_wrong_value = gave a different value",
        "The 'papers' column in Summary shows how many papers each model was scored on. Reasoning-effort variants were only run on the first 10 papers. A paper is scored only where every primary model returned a response (see Operations for failures/refusals).",
        "Metrics INCLUDE curator-accepted alternative answers (data/accepted_alternatives.csv plus the QID 10 Sanger convention), applied to every model equally. 'accuracy_before_accepted_answers' is the unadjusted score, kept for comparability with the paper.",
        "Answers are scored raw first; if that fails, the answer is re-scored with explanatory scaffolding removed (answer_cleaning.py) and then against accepted alternatives. Both layers apply to every model and can only rescue a row, never break one. Outcome 'correct_after_cleaning' / 'correct_accepted_answer' marks which layer fired.",
        "Machine-suggested cause per error, with checkable signals: evidence_in_pdf (the model's quoted evidence really occurs in the PDF text), human_answer_in_pdf (share of the human answer's words found in the PDF), models_agree, review_paper, annotation_hedge. Suggestions only - nothing is applied to scoring.",
    ]})

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        sheets = {
            "Overview": overview, "Summary": summary_wide, "By question": by_q,
            "Errors to review": errors, "Both models wrong": agree,
            "Auto-triage": triage, "All answers": all_answers, "Operations": ops,
        }
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            format_sheet(writer.sheets[name], frame)
        ws = writer.sheets["Errors to review"]
        if not errors.empty:
            dv = DataValidation(type="list", formula1='"' + ",".join(v.replace(",", ";") for v in VERDICTS) + '"', allow_blank=True)
            ws.add_data_validation(dv)
            col = get_column_letter(list(errors.columns).index("verdict") + 1)
            dv.add(f"{col}2:{col}{len(errors) + 1}")
            for row in ws.iter_rows(min_row=2, max_row=len(errors) + 1):
                for cell in row:
                    cell.alignment = Alignment(vertical="top", wrap_text=cell.column_letter in
                                               [get_column_letter(list(errors.columns).index(c) + 1) for c in WRAP if c in errors.columns])
        writer.sheets["Overview"].column_dimensions["A"].width = 32
        writer.sheets["Overview"].column_dimensions["B"].width = 120

    print(f"Wrote {OUT.relative_to(config.ROOT)}  ({len(errors)} error rows to review)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

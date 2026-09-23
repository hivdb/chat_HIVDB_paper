#!/usr/bin/env python3
"""Table 3 of the paper, for the frontier comparison, plus the same analysis by question type.

The paper's Table 3 lists, per question, precision and recall for each model family's base and
its FT / QSP variants, keeping only questions where a variant improved significantly. The test
is eval/statistics.py::compute_fisher_tests: a Fisher exact test on the per-question counts
(precision: TP vs FP; recall: TP vs FN), BH-adjusted within each question across all
comparisons of that metric. ** adjusted p < 0.01, * adjusted p < 0.05, only where the target
beats the base. build_table3() reproduces the paper's current Table 3 (the "Table3" sheet of
eval/results/statistical_tests_full150.xlsx) row for row and cell for cell (--validate).

Here the base is GPT-4o QSP (config.PRIMARY_COMPARATOR, the same prompt the frontier models
get), and the columns are GPT-4o FT, GPT-6 Astra QSP and Kimi K3 QSP. GPT-4o FT+QSP is tested
(it is part of the BH family) but not displayed, as in the paper. The "By question type" sheet
pools the counts of each type (Boolean / List / Number) and applies the same test.

Output: results/table3.xlsx with sheets "Table 3" (the questions where a model improves
significantly), "Table 3 complement" (all other questions, with significant declines marked
† / ‡) and "By question type".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.statistics import compute_fisher_tests  # noqa: E402
from frontier_compare import config  # noqa: E402

OUT = config.RESULTS_DIR / "table3.xlsx"
BASE = config.PRIMARY_COMPARATOR
DISPLAY = ["GPT-4o FT", "GPT-6 Astra QSP", "Kimi K3 QSP"]
TESTED = DISPLAY + ["GPT-4o FT+QSP"]
SHORT = {"GPT-4o QSP": "GPT-4o QSP (base)", "GPT-4o FT": "GPT-4o FT",
         "GPT-6 Astra QSP": "GPT-6 Astra", "Kimi K3 QSP": "Kimi K3"}
METRICS = ["precision", "recall"]


def fisher(counts: pd.DataFrame, comparisons: dict[str, tuple[str, str]], key: str = "QID") -> pd.DataFrame:
    """Long table (group, target, metric, key, base, value, p, adj_p) using the paper's test.

    `comparisons` maps a label to (base model, target model). Each label is passed to
    compute_fisher_tests as its own "family", so all comparisons share one BH family per
    question and metric, exactly as the paper's families do.
    """
    frame = counts.rename(columns={key: "QID"})
    fam = {label: {"base": base, "targets": [target]} for label, (base, target) in comparisons.items()}
    wide = compute_fisher_tests(frame, fam, METRICS)
    rows = []
    for _, r in wide.iterrows():
        base, target = comparisons[r["family"]]
        for q in sorted(frame["QID"].unique()):
            rows.append({"label": r["family"], "base_model": base, "target": target, "metric": r["metric"],
                         key: q, "base": r[f"base_qid_{q}"], "value": r[f"target_qid_{q}"],
                         "p": r[f"p_value_qid_{q}"], "adj_p": r[f"adj_p_qid_{q}"]})
    return pd.DataFrame(rows)


def star(row) -> str:
    """Paper's markers, improvements only: ** adjusted p < 0.01, * adjusted p < 0.05."""
    if pd.notna(row.adj_p) and row.value > row.base:
        return "**" if row.adj_p < 0.01 else ("*" if row.adj_p < 0.05 else "")
    return ""


def dagger(row) -> str:
    """Declines, used only in the complement tables: ‡ adjusted p < 0.01, † adjusted p < 0.05."""
    if pd.notna(row.adj_p) and row.value < row.base:
        return "‡" if row.adj_p < 0.01 else ("†" if row.adj_p < 0.05 else "")
    return ""


def build_table3(tests: pd.DataFrame, counts: pd.DataFrame, base: str, display: list[str],
                 key: str = "QID", only_significant: bool = True, labels: dict | None = None,
                 complement: bool = False) -> pd.DataFrame:
    """Rows = questions (or types) where a displayed target improves significantly on the base.

    complement=True returns the other questions instead, marking significant declines with
    † / ‡ so that a drop is not hidden by the improvement-only stars.
    """
    shown = tests[tests["target"].isin(display) & (tests["base_model"] == base)].copy()
    shown["star"] = shown.apply(star, axis=1)
    improved = set(shown.loc[shown["star"] != "", key])
    if complement:
        keep = sorted(set(shown[key]) - improved)
        shown["star"] = shown.apply(dagger, axis=1)
    else:
        keep = sorted(improved) if only_significant else sorted(shown[key].unique())
    value = counts.set_index([key, "model"])
    labels = labels or {}
    out = []
    for k in keep:
        rec = {key: k}
        if key == "QID":
            rec["Question"] = counts.loc[counts[key] == k, "Question"].iloc[0]
            rec["Type"] = counts.loc[counts[key] == k, "Type"].iloc[0]
        for metric in METRICS:
            rec[f"{metric} | {labels.get(base, base)}"] = f"{value.loc[(k, base), metric] * 100:.1f}"
            for t in display:
                s = shown[(shown[key] == k) & (shown["target"] == t) & (shown["metric"] == metric)]
                cell = f"{value.loc[(k, t), metric] * 100:.1f}"
                rec[f"{metric} | {labels.get(t, t)}"] = cell + (s["star"].iloc[0] if len(s) else "")
        out.append(rec)
    return pd.DataFrame(out)


def type_counts(qid_counts: pd.DataFrame) -> pd.DataFrame:
    pooled = qid_counts.groupby(["Type", "model"])[["tp", "tn", "fp", "fn"]].sum().reset_index()
    tp, tn, fp, fn = (pooled[c].astype(float) for c in ["tp", "tn", "fp", "fn"])
    pooled["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    pooled["precision"] = tp / (tp + fp)
    pooled["recall"] = tp / (tp + fn)
    pooled["n_questions"] = qid_counts.groupby(["Type", "model"])["QID"].nunique().values
    return pooled


def validate() -> int:
    """Rebuild the paper's Table 3 from its own per-question counts and compare every row and cell.

    The reference is the "Table3" sheet of eval/results/statistical_tests_full150.xlsx, which the
    paper's pipeline generates with the current scorer (April 2026). eval/results/
    Table3_formatted.xlsx predates that scorer update (December 2025) and differs on QIDs 6, 9, 15.
    """
    from eval.plots import FAMILY_COMPARISONS

    counts = pd.read_csv(config.ROOT / "eval/results/precision_recall_by_qid_full150.csv")
    comparisons = {f"{fam} {t.split(fam)[-1].strip()}": (m["base"], t)
                   for fam, m in FAMILY_COMPARISONS.items() for t in m["targets"]}
    tests = fisher(counts, comparisons)
    reference = pd.read_excel(config.ROOT / "eval/results/statistical_tests_full150.xlsx",
                              sheet_name="Table3", dtype=str)
    mismatches = checked = 0
    for family, m in FAMILY_COMPARISONS.items():
        ft = next(t for t in m["targets"] if t.endswith(" FT"))
        qsp = next(t for t in m["targets"] if t.endswith(" QSP") and "FT+QSP" not in t)
        mine = build_table3(tests, counts, m["base"], [ft, qsp])
        ref = reference[reference["Model"] == family]
        if sorted(mine["QID"]) != sorted(ref["QID"].astype(int)):
            mismatches += 1
            print(f"  {family}: rows {sorted(mine['QID'])} vs published {sorted(ref['QID'].astype(int))}")
        for _, r in ref.iterrows():
            row = mine[mine["QID"] == int(r["QID"])]
            if row.empty:
                continue
            row = row.iloc[0]
            pairs = [("base_prec", f"precision | {m['base']}"), ("FT_prec", f"precision | {ft}"),
                     ("QSP_prec", f"precision | {qsp}"), ("base_rec", f"recall | {m['base']}"),
                     ("FT_rec", f"recall | {ft}"), ("QSP_rec", f"recall | {qsp}")]
            for ref_col, my_col in pairs:
                checked += 1
                if str(r[ref_col]) != str(row[my_col]):
                    mismatches += 1
                    print(f"  {family} QID {r['QID']} {ref_col}: published {r[ref_col]!r}, rebuilt {row[my_col]!r}")
    print(f"Validated against statistical_tests_full150.xlsx 'Table3': {len(reference)} rows, "
          f"{checked} cells, {mismatches} mismatches")
    return 1 if mismatches else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true", help="check the builder against the paper's Table 3")
    args = parser.parse_args()
    if args.validate:
        return validate()

    counts = pd.read_csv(config.RESULTS_DIR / "metrics_by_qid.csv")
    comparisons = {t: (BASE, t) for t in TESTED}
    qid_tests = fisher(counts, comparisons)
    types = type_counts(counts)
    type_tests = fisher(types.rename(columns={"Type": "QID"}), comparisons).rename(columns={"QID": "Type"})
    by_type = build_table3(type_tests, types, BASE, DISPLAY, key="Type", only_significant=False, labels=SHORT)
    by_type.insert(1, "questions", [int(types.loc[types["Type"] == t, "n_questions"].iloc[0]) for t in by_type["Type"]])
    for m in [BASE, *DISPLAY]:
        by_type[f"accuracy | {SHORT[m]}"] = [f"{types.set_index(['Type', 'model']).loc[(t, m), 'accuracy'] * 100:.1f}"
                                             for t in by_type["Type"]]
    sheets = {
        "Table 3": build_table3(qid_tests, counts, BASE, DISPLAY, labels=SHORT),
        "Table 3 complement": build_table3(qid_tests, counts, BASE, DISPLAY, labels=SHORT, complement=True),
        "By question type": by_type,
    }
    for name, frame in sheets.items():
        print(f"\n== {name} ==")
        print(frame.drop(columns=[c for c in ["Question"] if c in frame]).to_string(index=False))

    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "B2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = 60 if col[0].value == "Question" else 16
        pd.DataFrame({"Notes": [
            f"Base: {BASE}. Fisher exact test per question (precision: TP vs FP; recall: TP vs FN), BH-adjusted "
            "within each question across the four comparisons vs the base (GPT-4o FT, GPT-4o FT+QSP, GPT-6 Astra, "
            "Kimi K3), as eval/statistics.py::compute_fisher_tests does for the paper's Table 3.",
            "** adjusted p < 0.01; * adjusted p < 0.05; marked only where the model beats the base.",
            "'Table 3' lists only questions where a displayed model improves significantly, as in the paper.",
            "'Table 3 complement' lists every remaining question. No model improves significantly on "
            "these; a significant DECLINE vs the base is marked ‡ (adjusted p < 0.01) or † (adjusted p < 0.05). The paper's "
            "Table 3 does not mark declines.",
            "'By question type' pools the TP/FP/TN/FN counts of all questions of each type and applies the same test "
            "(an extension; the paper does not pool by type). Rows from the same paper are not independent, so treat "
            "these p-values as approximate.",
            "Scores: the paper's evaluation with answer cleaning, all 150 papers (see 04_evaluate.py).",
        ]}).to_excel(writer, sheet_name="Notes", index=False)
    print(f"\nWrote {OUT.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

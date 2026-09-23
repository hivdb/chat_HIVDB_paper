#!/usr/bin/env python3
"""Consolidate every model's errors per (PMID, QID) and apply recorded adjudication verdicts.

Adjudication has two layers, which is what makes ~1,200 error rows tractable:

  A. Per (PMID, QID) - is the annotation sound? This is model-independent: if a row's annotation
     is wrong, ambiguous, or unanswerable, it is wrong for every model that missed it. Recorded in
     data/adjudication.csv as annotation_verdict in {ok, convention, wrong, ambiguous,
     unanswerable} with a reason.
  B. Per (PMID, QID, model) - only for rows whose annotation is sound: was this model's answer a
     genuine error, defensible-but-different (borderline), or a scoring artifact? Recorded in
     data/adjudication_overrides.csv; anything not listed defaults to "model error".

  --consolidate  writes results/adjudication_worksheet.txt: one block per distinct error row with
                 every model's answer, the frontier models' evidence, and PDF context - the input
                 for layer A.
  --apply        joins the recorded verdicts onto every model's error rows and writes
                 results/adjudicated_errors.csv plus a per-model summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier_compare import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
dossier = __import__("09_error_dossier")

ADJUDICATION = config.FC_DIR / "data/adjudication.csv"
OVERRIDES = config.FC_DIR / "data/adjudication_overrides.csv"
ANNOTATION_VERDICTS = {"ok", "convention", "wrong", "ambiguous", "unanswerable"}


def all_models(rows: pd.DataFrame) -> list[str]:
    frontier = [spec.label for spec in config.MODELS.values()
                if not spec.is_variant and f"{spec.label} correct" in rows.columns]
    return frontier + [c for c in config.COMPARATORS if f"{c} correct" in rows.columns]


def load_rows() -> pd.DataFrame:
    rows = pd.read_csv(config.RESULTS_DIR / "detailed_rows.csv", dtype={"PMID": str},
                       keep_default_na=False, na_values=[""])
    rows["QID"] = rows["QID"].astype(int)
    return rows


def evidence_map() -> dict[tuple[str, int, str], tuple[str, str, str]]:
    out = {}
    for key, spec in config.MODELS.items():
        path = config.answers_path(key, 1)
        if not path.exists():
            continue
        ans = pd.read_csv(path, dtype={"PMID": str}, keep_default_na=False)
        for r in ans.itertuples(index=False):
            out[(r.PMID, int(r.QID), spec.label)] = (r.Evidence, r.EvidenceLocation, r.Rationale)
    return out


def consolidate() -> int:
    rows = load_rows()
    models = all_models(rows)
    frontier = [m for m in models if m not in config.COMPARATORS]
    ev = evidence_map()
    cache: dict[str, str] = {}

    wrong_any = rows[[any(r.get(f"{m} correct") == 0 for m in models) for _, r in rows.iterrows()]]
    blocks, index = [], []
    for n, (_, r) in enumerate(wrong_any.sort_values(["QID", "PMID"]).iterrows(), start=1):
        pmid, qid = r["PMID"], int(r["QID"])
        text = dossier.load_pdf(pmid, cache)
        missed = [m for m in models if r.get(f"{m} correct") == 0]
        evidence = ""
        for m in frontier:
            if m in missed and (pmid, qid, m) in ev:
                quote, loc, _ = ev[(pmid, qid, m)]
                evidence = f"  {m} evidence [{loc}] (in pdf: {dossier.quote_found(quote, text)}): {quote[:240]}"
                break
        blocks.append(
            f"\n=== [{n}] PMID {pmid} QID {qid} ({r['Type']}) - missed by {len(missed)}/{len(models)} "
            f"{'ALL-MODELS-WRONG' if len(missed) == len(models) else ''}\n"
            f"Q: {r['Question']}\n"
            f"HUMAN: {r[config.REF_COL]}\n"
            + "\n".join(f"  {m:16} {str(r[m])[:90]}  {'X' if r.get(f'{m} correct') == 0 else 'ok'}" for m in models)
            + (f"\n{evidence}" if evidence else "")
            + "\n  PDF near HUMAN: " + (" | ".join(dossier.contexts(str(r[config.REF_COL]), text, width=110, limit=1)) or "(none)")
        )
        index.append({"n": n, "PMID": pmid, "QID": qid, "Type": r["Type"],
                      "Human Answer": r[config.REF_COL], "n_models_wrong": len(missed),
                      "all_models_wrong": len(missed) == len(models),
                      "annotation_verdict": "", "reason": ""})

    out = config.RESULTS_DIR / "adjudication_worksheet.txt"
    out.write_text("\n".join(blocks), encoding="utf-8")
    pd.DataFrame(index).to_csv(config.RESULTS_DIR / "adjudication_worksheet.csv", index=False)
    print(f"{len(index)} distinct error rows -> {out.relative_to(config.ROOT)}")
    return 0


def apply_verdicts() -> int:
    rows = load_rows()
    models = all_models(rows)
    if not ADJUDICATION.exists():
        print(f"Missing {ADJUDICATION}; run --consolidate and fill it in first.")
        return 1
    adj = pd.read_csv(ADJUDICATION, dtype={"PMID": str}, keep_default_na=False)
    bad = set(adj["annotation_verdict"]) - ANNOTATION_VERDICTS
    if bad:
        print(f"Unknown annotation_verdict values: {sorted(bad)}")
        return 1
    adj_map = {(r.PMID, int(r.QID)): (r.annotation_verdict, r.reason) for r in adj.itertuples(index=False)}
    over = (pd.read_csv(OVERRIDES, dtype={"PMID": str}, keep_default_na=False)
            if OVERRIDES.exists() else pd.DataFrame(columns=["PMID", "QID", "Model", "verdict", "note"]))
    over_map = {(r.PMID, int(r.QID), r.Model): (r.verdict, r.note) for r in over.itertuples(index=False)}

    out = []
    for model in models:
        for _, r in rows[rows[f"{model} correct"] == 0].iterrows():
            pmid, qid = r["PMID"], int(r["QID"])
            ann, reason = adj_map.get((pmid, qid), ("ok", ""))
            if (pmid, qid, model) in over_map:
                verdict, note = over_map[(pmid, qid, model)]
            elif ann in {"wrong", "ambiguous", "unanswerable", "convention"}:
                verdict, note = ("annotation/convention", f"{ann}: {reason}")
            else:
                verdict, note = ("model error", "")
            out.append({"Model": model, "PMID": pmid, "QID": qid, "Type": r["Type"],
                        "Question": r["Question"], "Human Answer": r[config.REF_COL],
                        "Model Answer": r[model], "annotation_verdict": ann,
                        "verdict": verdict, "note": note})

    adjudicated = pd.DataFrame(out)
    adjudicated.to_csv(config.RESULTS_DIR / "adjudicated_errors.csv", index=False)

    summary = []
    for model in models:
        sub = adjudicated[adjudicated["Model"] == model]
        scored = int(rows[f"{model} correct"].notna().sum())
        correct = int(rows[f"{model} correct"].sum())
        counts = sub["verdict"].value_counts()
        not_model = int(len(sub) - counts.get("model error", 0))
        summary.append({
            "model": model, "rows": scored, "errors": len(sub),
            "model error": counts.get("model error", 0),
            "annotation/convention": counts.get("annotation/convention", 0),
            "borderline": counts.get("borderline", 0),
            "scoring artifact": counts.get("scoring artifact", 0),
            "accuracy": round(correct / scored, 4),
            "accuracy_excl_annotation": round((correct + counts.get("annotation/convention", 0)) / scored, 4),
            "accuracy_excl_all_non_model": round((correct + not_model) / scored, 4),
        })
    table = pd.DataFrame(summary)
    table.to_csv(config.RESULTS_DIR / "adjudicated_summary.csv", index=False)
    print(table.to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--consolidate", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.consolidate:
        return consolidate()
    if args.apply:
        return apply_verdicts()
    parser.error("choose --consolidate or --apply")


if __name__ == "__main__":
    raise SystemExit(main())

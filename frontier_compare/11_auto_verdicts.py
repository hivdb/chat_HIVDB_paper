#!/usr/bin/env python3
"""First-pass adjudication: apply explicit rules, and flag what needs a human read.

Rules for the annotation layer (per PMID x QID), each with a stated justification:

  R1 type mismatch     A Boolean question annotated with something that is not yes/no-like, or a
                       Number question annotated with no number  -> unanswerable as scored.
  R2 hedged annotation The annotation itself flags uncertainty ("not stated", "unknown",
                       "(multicenter trial)", "(Review paper)")  -> ambiguous.
  R3 review paper      The paper is a review / meta-analysis (verified by reading, listed in
                       data/review_papers.csv) and a model missed a downstream question -> wrong
                       (the QSP rules say reviews get No/None). An earlier keyword version of
                       this rule also fired on 5 primary studies that merely cite a meta-analysis.
  R4 Sanger default    QID 10 annotated Sanger where the PDF never says Sanger  -> convention.
  R6 QID 5 denominator  The annotation's count and the model's count both appear in the PDF and
                       the model's is the larger -> convention. The paper states several counts
                       (enrolled / attempted / successfully sequenced / analysed); the annotation
                       uses one, the question text does not say which. Applies per model.
  R5 unanimous failure ALL 5 models miss the row, including the GPT-4o fine-tuned on these very
                       annotations -> ambiguous. Justification: if the fine-tuned model cannot
                       reproduce the annotation from the paper, the row is not answerable from the
                       text under a shared reading. Rows where >= 1 model succeeds are answerable,
                       so the annotation is treated as sound and the misses as model errors.
                       A sample of these rows was read individually to check the rule (see README).

Rules for the per-model layer (only where the annotation is sound):

  M1 over-extraction   The annotation is empty-ish ("Not reported"/"None") and the model's answer
                       words do occur in the PDF -> borderline (real text, wrong scope//threshold).
  M2 cascade           The model's own QID 1 answer was wrong and this is a downstream question
                       -> model error (its own inconsistency, not an annotation problem). M2 beats M1.

Everything else defaults to "model error". Output: data/adjudication_auto.csv and
data/adjudication_overrides_auto.csv; data/adjudication.csv (manual verdicts in
data/adjudication_manual.csv take precedence over the rules) and data/adjudication_overrides.csv,
which 10_adjudicate.py --apply reads; work/needs_review.txt for the flagged rows.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier_compare import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
dossier = __import__("09_error_dossier")
adjudicate = __import__("10_adjudicate")

BOOL_OK = re.compile(r"^\s*(yes|no|not\s+(reported|applicable|provided|specified|stated)|n/?a|unknown|unclear)\b", re.I)
HEDGE = re.compile(r"\b(not stated|not known|unknown|unclear|uncertain|assumed|presumed|multicenter|multicentre|review paper|supplementary)\b|\bor\b\s*\d|\d\s*and\s*\d", re.I)
EMPTYISH = re.compile(r"^\s*(not\s+(reported|applicable|provided|specified|stated)|none|no|n/?a|0)\s*$", re.I)
SCOPE_QIDS = {4, 6, 7, 9, 10, 11, 12, 14, 15, 16}


def main() -> int:
    rows = adjudicate.load_rows()
    models = adjudicate.all_models(rows)
    cache: dict[str, str] = {}
    reviews = config.review_pmids()
    # papers a model returned no answers for at all (a blocked request), scored as blanks
    unanswered = {(p, m) for m in models for p, g in rows.groupby("PMID") if g[m].isna().all()}
    ann_rows, overrides, needs_review = [], [], []

    for _, r in rows.iterrows():
        pmid, qid, qtype = r["PMID"], int(r["QID"]), str(r["Type"])
        missed = [m for m in models if r.get(f"{m} correct") == 0]
        if not missed:
            continue
        human = str(r[config.REF_COL])
        text = dossier.load_pdf(pmid, cache)
        verdict = reason = ""

        if qtype == "Boolean" and not BOOL_OK.match(human):
            verdict, reason = "unanswerable", f"Boolean question annotated {human!r}"
        elif qtype == "Number" and not re.search(r"\d", human) and not EMPTYISH.match(human):
            verdict, reason = "unanswerable", f"Number question annotated {human!r}"
        elif HEDGE.search(human):
            verdict, reason = "ambiguous", f"annotation hedges: {human!r}"
        elif pmid in reviews and qid != 1:
            verdict, reason = "wrong", "PDF is a systematic review/meta-analysis; QSP rules say No/None"
        elif qid == 10 and "sanger" in human.lower() and "sanger" not in dossier.flat(text):
            verdict, reason = "convention", "annotation defaults to Sanger; the PDF never mentions it"

        if not verdict and len(missed) == len(models):
            verdict, reason = "ambiguous", f"all {len(models)} models disagree with the annotation (incl. fine-tuned GPT-4o)"
        if verdict:
            ann_rows.append({"PMID": pmid, "QID": qid, "annotation_verdict": verdict, "reason": reason})
            continue
        if len(missed) >= 3:
            needs_review.append((pmid, qid, len(missed)))

        # annotation looks sound: classify each model's miss
        for m in missed:
            answer = str(r[m])
            if (pmid, m) in unanswered:
                overrides.append({"PMID": pmid, "QID": qid, "Model": m, "verdict": "model error",
                                  "note": "no answer: the request for this paper was blocked by the provider"})
                continue
            same_paper = rows[(rows.PMID == pmid)]
            q1_row = same_paper[same_paper.QID == 1]
            q1_wrong = bool(len(q1_row)) and q1_row.iloc[0].get(f"{m} correct") == 0
            if q1_wrong and qid in SCOPE_QIDS:
                overrides.append({"PMID": pmid, "QID": qid, "Model": m, "verdict": "model error",
                                  "note": "M2 cascade: the model's own QID 1 answer was wrong"})
            elif qid == 5 and re.search(r"\d", human) and re.search(r"\d", answer):
                flat_text = dossier.flat(text)
                h = int(re.sub(r"\D", "", human)[:9] or 0)
                a = int(re.sub(r"\D", "", answer)[:9] or 0)
                if h and a and a > h and str(h) in flat_text and str(a) in flat_text:
                    overrides.append({"PMID": pmid, "QID": qid, "Model": m,
                                      "verdict": "annotation/convention",
                                      "note": f"R6 QID 5 denominator: both {a} and {h} are stated in the paper"})
            elif EMPTYISH.match(human) and not EMPTYISH.match(answer):
                tokens = [t for t in dossier.flat(answer).split() if len(t) > 3][:6]
                flat_text = dossier.flat(text)
                if tokens and sum(t in flat_text for t in tokens) / len(tokens) >= 0.5:
                    overrides.append({"PMID": pmid, "QID": qid, "Model": m, "verdict": "borderline",
                                      "note": "M1: annotation empty but the model's answer is in the PDF text"})

    auto = pd.DataFrame(ann_rows).drop_duplicates(["PMID", "QID"])
    auto.to_csv(config.FC_DIR / "data/adjudication_auto.csv", index=False)
    pd.DataFrame(overrides).to_csv(config.FC_DIR / "data/adjudication_overrides_auto.csv", index=False)
    pd.DataFrame(overrides).to_csv(config.FC_DIR / "data/adjudication_overrides.csv", index=False)
    # data/adjudication.csv = manual verdicts, then rule-based ones for every other row
    manual = pd.read_csv(config.FC_DIR / "data/adjudication_manual.csv", dtype={"PMID": str}, keep_default_na=False)
    auto["PMID"] = auto["PMID"].astype(str)
    pd.concat([manual, auto]).drop_duplicates(["PMID", "QID"], keep="first").to_csv(
        config.FC_DIR / "data/adjudication.csv", index=False)
    Path(config.WORK_DIR / "needs_review.txt").write_text(
        "\n".join(f"{p} {q} missed_by={n}" for p, q, n in needs_review), encoding="utf-8")

    print(f"auto annotation verdicts : {len(ann_rows)}")
    print(f"auto per-model overrides : {len(overrides)}")
    print(f"flagged for manual review: {len(needs_review)} rows (>=3 of 5 models wrong)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

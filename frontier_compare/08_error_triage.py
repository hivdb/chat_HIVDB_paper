#!/usr/bin/env python3
"""Auto-triage every frontier-model error into candidate causes, with evidence from the PDF.

For each error row this computes signals that a human can check quickly:
  models_agree          - the primary frontier models gave the same (canonicalized) answer
  evidence_in_pdf       - the model's quoted evidence really appears in the PDF text layer
  human_answer_in_pdf   - share of the human answer's content tokens found anywhere in the PDF
  review_paper          - the PDF looks like a review / meta-analysis (QSP says answer "No")
  cleanup_would_pass    - a candidate post-processing rule turns the answer into a pass
  negation_tail         - the answer carries a caveat that the scorer reads as a negation

and proposes a category. These are SUGGESTIONS for the verdict column, never applied to scoring.
Writes results/error_triage.csv and prints a summary.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.normalize import (  # noqa: E402
    canonicalize_answer,
    contains_negation,
    human_answer_counts,
    is_empty_token,
)
from frontier_compare import config  # noqa: E402

HEDGE_PAT = re.compile(r"\((?:[^()]*\b(?:not stated|not specified|unclear|assumed|presumed|multicenter|multicentre)\b[^()]*)\)", re.I)
NONTEXT_EVIDENCE = {"figure", "table", "supplement", "not_found"}
STOP = {"not", "reported", "provided", "applicable", "none", "no", "yes", "and", "or", "the", "of",
        "in", "for", "with", "study", "paper", "data", "specified", "stated", "available"}
REVIEW_PAT = re.compile(r"systematic (literature )?review|meta-?analysis|PRISMA", re.I)


def norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("–", "-").replace("—", "-")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)  # 57,902 -> 57902 so counts match the annotation
    return re.sub(r"[^a-z0-9 ]", " ", re.sub(r"\s+", " ", text.lower())).strip()


def pdf_text(pmid: str, cache: dict[str, str]) -> str:
    if pmid not in cache:
        path = config.PDF_DIR / f"{pmid}.pdf"
        with pymupdf.open(path) as doc:
            cache[pmid] = norm_text(" ".join(page.get_text() for page in doc))
    return cache[pmid]


def evidence_in_pdf(evidence: str, text: str) -> float:
    """Share of the model's quoted sentences (>=6 words) found verbatim in the PDF."""
    quotes = re.findall(r"[\"“]([^\"”]{25,})[\"”]", evidence) or re.split(r"(?<=\.)\s+", evidence)
    checked = [norm_text(q) for q in quotes if len(q.split()) >= 6]
    if not checked:
        return float("nan")
    hits = sum(1 for q in checked if q[:120] in text)
    return round(hits / len(checked), 2)


def tokens_in_pdf(answer: str, text: str) -> float:
    toks = [t for t in norm_text(answer).split() if t not in STOP and len(t) > 2]
    if not toks:
        return float("nan")
    return round(sum(1 for t in toks if t in text) / len(toks), 2)


CLEANUP = [
    ("strip_parenthetical_caveat", lambda a: re.sub(r"\s*\((?:[^()]*\b(?:not|no|only|unclear|unspecified|prior|other)\b[^()]*)\)", "", a).strip()),
    ("strip_trailing_clause", lambda a: re.split(r"[.;]\s+(?=[A-Z])|\s+-\s+", a)[0].strip()),
    ("drop_explanatory_sentence", lambda a: a.split(". ")[0].strip() if len(a.split(". ")) > 1 else a),
]


def cleanup_pass(row: pd.Series, answer: str) -> tuple[str, str]:
    """Does a candidate post-processing rule make this answer score correct?"""
    for name, rule in CLEANUP:
        cleaned = rule(answer)
        if cleaned == answer or not cleaned:
            continue
        _, ok = human_answer_counts(
            row["Type"], canonicalize_answer(cleaned), canonicalize_answer(row[config.REF_COL]),
            question_text=row["Question"], ref_raw=row[config.REF_COL], pred_raw=cleaned,
            allow_partial_list=row["Type"].strip().lower() == "list",
        )
        if ok:
            return name, cleaned
    return "", ""


SCOPE_QIDS = {4, 6, 7, 9, 10, 11}  # details that only exist if patient samples were sequenced


def self_inconsistent(rows: pd.DataFrame, model: str) -> set[tuple[str, int]]:
    """Rows where the model said no patient sequences (QID 1 = No, or QID 5 = 0) yet still named
    sequencing details - typically lab-construct text read as if it described patient samples."""
    flagged: set[tuple[str, int]] = set()
    for pmid, grp in rows.groupby("PMID"):
        answers = {int(r["QID"]): str(r[model]).strip().lower() for _, r in grp.iterrows()}
        says_none = answers.get(1, "").startswith("no") or answers.get(5, "") in {"0", "none"}
        if not says_none:
            continue
        for qid in SCOPE_QIDS:
            value = answers.get(qid, "")
            if value and not re.match(r"^(no|none|not\b|n/?a)", value):
                flagged.add((str(pmid), qid))
    return flagged


def categorize(s: pd.Series) -> str:
    if s.get("self_inconsistent"):
        return "model error: out-of-scope evidence (model itself said no patient sequences)"
    qid = int(s["QID"])
    model_says_nothing = str(s["Model Answer"]).strip().lower() in {"no", "none", "not applicable", "not reported", "0"}
    if s["review_paper"] and model_says_nothing:
        return "annotation error: paper is a review/meta-analysis"
    if s["annotation_hedge"]:
        return "annotation hedge: human answer is explicitly uncertain"
    # Curator conventions the prompt never states (see README "Curation conventions")
    if qid == 10 and (s["human_answer_in_pdf"] or 0) == 0.0 and s["models_agree"]:
        return "curation convention: method defaulted (e.g. Sanger) though the paper never states it"
    if qid == 5 and s["Outcome"] == "FN_wrong_value":
        return "curation convention: which individuals to count"
    if s["review_paper"] and str(s["Model Answer"]).strip().lower() in {"no", "none", "not applicable", "0"}:
        return "annotation error: paper is a review/meta-analysis"
    if s["cleanup_would_pass"]:
        return "scoring artifact: cleanup rule fixes it"
    if s["negation_tail"]:
        return "scoring artifact: caveat read as negation"
    if s["models_agree"] and s["evidence_in_pdf"] == 1.0 and (s["human_answer_in_pdf"] or 0) < 0.5:
        return "annotation candidate: both models agree, evidence in PDF, human answer absent"
    if s["models_agree"] and (s["human_answer_in_pdf"] or 0) < 0.5:
        return "annotation candidate: both models agree, human answer absent from PDF"
    if s["evidence_in_pdf"] == 0.0 and str(s["EvidenceLocation"]).strip().lower() not in NONTEXT_EVIDENCE:
        return "model error: quoted evidence not found in PDF"
    if str(s["EvidenceLocation"]).strip().lower() in NONTEXT_EVIDENCE:
        return "needs review: evidence cited from a figure/table (not checkable in the text layer)"
    return "model error / needs review"


def main() -> int:
    rows = pd.read_csv(config.RESULTS_DIR / "detailed_rows.csv", dtype={"PMID": str},
                       keep_default_na=False, na_values=[""])
    rows["QID"] = rows["QID"].astype(int)
    primary = [spec.label for spec in config.MODELS.values()
               if not spec.is_variant and f"{spec.label} correct" in rows.columns]
    evidence = {}
    for key, spec in config.MODELS.items():
        path = config.answers_path(key, 1)
        if path.exists():
            ans = pd.read_csv(path, dtype={"PMID": str}, keep_default_na=False)
            for r in ans.itertuples(index=False):
                evidence[(r.PMID, int(r.QID), spec.label)] = (r.Evidence, r.EvidenceLocation, r.Rationale)

    cache: dict[str, str] = {}
    out = []
    for model in primary:
        inconsistent = self_inconsistent(rows[rows[model].notna()], model)
        for _, r in rows[rows[f"{model} correct"] == 0].iterrows():  # 'correct' already includes accepted answers
            text = pdf_text(r["PMID"], cache)
            answer = str(r[model])
            ev, loc, rat = evidence.get((r["PMID"], int(r["QID"]), model), ("", "", ""))
            rule, cleaned = cleanup_pass(r, answer)
            others = [m for m in primary if m != model]
            rec = {
                "PMID": r["PMID"], "QID": r["QID"], "Type": r["Type"], "Model": model,
                "Human Answer": r[config.REF_COL], "Model Answer": answer,
                "Outcome": r[f"{model} outcome"],
                "self_inconsistent": (str(r["PMID"]), int(r["QID"])) in inconsistent,
                "models_agree": all(canonicalize_answer(answer) == canonicalize_answer(str(r[m])) for m in others),
                "evidence_in_pdf": evidence_in_pdf(ev, text),
                "human_answer_in_pdf": tokens_in_pdf(str(r[config.REF_COL]), text),
                "review_paper": bool(REVIEW_PAT.search(text[:6000])),
                "cleanup_would_pass": bool(rule), "cleanup_rule": rule, "cleaned_answer": cleaned,
                "annotation_hedge": bool(HEDGE_PAT.search(str(r[config.REF_COL]))),
                "negation_tail": bool(contains_negation(answer) and not is_empty_token(canonicalize_answer(answer))),
                "EvidenceLocation": loc, "Evidence": ev[:400], "Rationale": rat[:400],
                "PubMed": f"https://pubmed.ncbi.nlm.nih.gov/{r['PMID']}/",
            }
            rec["suggested_category"] = categorize(pd.Series(rec))
            out.append(rec)

    triage = pd.DataFrame(out).sort_values(["suggested_category", "PMID", "QID"])
    triage.to_csv(config.RESULTS_DIR / "error_triage.csv", index=False)
    print(triage["suggested_category"].value_counts().to_string())
    print(f"\nrows: {len(triage)}  ->  {config.RESULTS_DIR.relative_to(config.ROOT)}/error_triage.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

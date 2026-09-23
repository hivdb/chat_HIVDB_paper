#!/usr/bin/env python3
"""Build a per-row evidence dossier for one model's errors, for manual adjudication.

For every error row it pulls, from the paper's own PDF:
  - the model's answer, evidence quote and rationale, and whether that quote is really in the PDF
  - the PDF context around the human answer's distinctive tokens (does the annotation have support?)
  - the PDF context around the model answer's distinctive tokens
  - what every other model answered
  - whether the paper looks like a review, and whether the model contradicted its own QID 1/5

Output: results/dossier_<model>.txt (for reading) and results/dossier_<model>.csv (for the sheet).
Verdicts are assigned by a human/agent reading the dossier, not by this script.

  python frontier_compare/09_error_dossier.py --model "GPT-6 Astra QSP"
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier_compare import config  # noqa: E402

STOP = {"not", "reported", "provided", "applicable", "none", "the", "of", "in", "for", "with",
        "and", "or", "study", "paper", "data", "specified", "stated", "available", "yes", "no",
        "sequencing", "sequenced", "samples", "individuals", "obtained", "from", "were", "was"}


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    for a, b in [("’", "'"), ("“", '"'), ("”", '"'), ("ﬁ", "fi"),
                 ("ﬂ", "fl"), ("–", "-"), ("—", "-")]:
        text = text.replace(a, b)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return re.sub(r"\s+", " ", text)


def flat(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", norm(text).lower())


def load_pdf(pmid: str, cache: dict[str, str]) -> str:
    if pmid not in cache:
        with pymupdf.open(config.PDF_DIR / f"{pmid}.pdf") as doc:
            cache[pmid] = norm(" ".join(page.get_text() for page in doc))
    return cache[pmid]


def contexts(answer: str, text: str, width: int = 150, limit: int = 2) -> list[str]:
    """Snippets of the PDF around the answer's distinctive tokens."""
    flat_text = flat(text)
    out: list[str] = []
    for token in dict.fromkeys(t for t in flat(answer).split() if t not in STOP and len(t) > 2):
        for match in list(re.finditer(re.escape(token), flat_text))[:1]:
            start = max(0, match.start() - width)
            out.append(f"[{token}] ...{text[start:match.end() + width]}...")
        if len(out) >= limit:
            break
    return out


def quote_found(evidence: str, text: str) -> str:
    quotes = re.findall(r"[\"“]([^\"”]{25,})[\"”]", evidence) or [evidence]
    flat_text = flat(text)
    hits = [flat(q)[:110] in flat_text for q in quotes if len(q.split()) >= 6]
    if not hits:
        return "n/a"
    return "yes" if all(hits) else ("partial" if any(hits) else "NO")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    rows = config.final_rows()
    model = args.model
    key = next((k for k, spec in config.MODELS.items() if spec.label == model), None)
    if key is None:  # cached GPT-4o comparator: answers only, no stored evidence/rationale
        key = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
        answers = pd.DataFrame(columns=["PMID", "QID", "Evidence", "EvidenceLocation", "Rationale"])
        answers = answers.set_index(["PMID", "QID"])
    else:
        answers = pd.read_csv(config.answers_path(key, 1), dtype={"PMID": str}, keep_default_na=False)
        answers = answers.set_index(["PMID", "QID"])
    others = [c[:-8] for c in rows.columns if c.endswith(" correct") and c[:-8] != model]

    cache: dict[str, str] = {}
    lines, records = [], []
    errors = rows[rows[f"{model} correct"] == 0].sort_values(["QID", "PMID"])
    for n, (_, r) in enumerate(errors.iterrows(), start=1):
        pmid, qid = r["PMID"], int(r["QID"])
        text = load_pdf(pmid, cache)
        ans = answers.loc[(pmid, qid)] if (pmid, qid) in answers.index else None
        evidence = str(ans["Evidence"]) if ans is not None else ""
        agree = [m for m in others if str(r[m]).strip().lower() == str(r[model]).strip().lower()]
        all_wrong = all(r.get(f"{m} correct") == 0 for m in others)
        same_paper = rows[rows.PMID == pmid]
        q1 = str(same_paper[same_paper.QID == 1][model].iloc[0]) if (same_paper.QID == 1).any() else ""
        q5 = str(same_paper[same_paper.QID == 5][model].iloc[0]) if (same_paper.QID == 5).any() else ""

        lines.append(
            f"\n=== [{n}] PMID {pmid}  QID {qid} ({r['Type']}) {'-' * 30}\n"
            f"Q: {r['Question']}\n"
            f"HUMAN : {r[config.REF_COL]}\n"
            f"MODEL : {r[model]}   [{ans['EvidenceLocation'] if ans is not None else ''}] "
            f"quote-in-pdf={quote_found(evidence, text)}\n"
            f"  evidence : {evidence[:260]}\n"
            f"  rationale: {str(ans['Rationale'])[:260] if ans is not None else ''}\n"
            f"OTHERS: " + "; ".join(f"{m}={str(r[m])[:40]}({'ok' if r.get(f'{m} correct') == 1 else 'X'})" for m in others) + "\n"
            f"  all-models-wrong={all_wrong}  agree-with-model={agree}\n"
            f"  model said Q1={q1[:20]!r} Q5={q5[:20]!r}\n"
            f"  PDF near HUMAN answer: " + (" | ".join(contexts(str(r[config.REF_COL]), text)) or "(no distinctive token found)") + "\n"
            f"  PDF near MODEL answer: " + (" | ".join(contexts(str(r[model]), text)) or "(no distinctive token found)")
        )
        records.append({"n": n, "PMID": pmid, "QID": qid, "Type": r["Type"], "Question": r["Question"],
                        "Human Answer": r[config.REF_COL], "Model Answer": r[model],
                        "quote_in_pdf": quote_found(evidence, text), "all_models_wrong": all_wrong,
                        "verdict": "", "note": ""})

    out_txt = config.WORK_DIR / f"dossier_{key}.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(records).to_csv(config.WORK_DIR / f"dossier_{key}.csv", index=False)
    print(f"{len(records)} error rows -> {out_txt.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

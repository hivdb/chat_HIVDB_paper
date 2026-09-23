#!/usr/bin/env python3
"""Score frontier-model answers against the human annotations with the paper's scorer.

Per-row scoring reuses eval.normalize.human_answer_counts exactly as the paper does (partial
list matches allowed for List questions). Aggregation follows the paper's Figure 4:
  pooled - PRIMARY. Metrics over all PMID x QID rows, including curator-accepted alternative
           answers (see below). 95% CI from a row bootstrap (5000, seed 42), as in
           eval/evaluation.py::build_bar_chart_confidence_intervals. This reproduces the paper's
           reported deltas (e.g. GPT-4o FT recall +11%, Llama-70B FT precision +16%).
  macro  - unweighted mean of the 16 per-QID values (reported for reference only)
Paired tests run on the 16 per-QID values (Wilcoxon signed-rank, as stated in the paper, plus
the paired t-test also in S5), BH-adjusted within each (metric, test) slice as in
eval/statistics.py. Exact McNemar on row correctness is a sensitivity analysis.
Cached GPT-4o comparators are re-scored on the same PMID subset as the frontier models.

Two post-processing layers sit between the model's answer and the score, both applied to EVERY
model equally (including the cached GPT-4o comparators), and both non-destructive - the raw
answer is scored first and a layer can only rescue a row, never break one:

1. Answer cleaning (answer_cleaning.py): strips explanatory scaffolding - a preamble, a trailing
   note, or a commentary parenthetical - that the scorer would otherwise read as hedging.
   Rescued rows are marked `correct_after_cleaning` and name the rule in `<model> cleaning_rule`.
2. Accepted alternative answers, which come from two places and are applied to EVERY model equally:
  - data/accepted_alternatives.csv: curator-approved answers for rows where the annotation is
    wrong or incomplete (e.g. a review paper annotated as a primary study; figure-only evidence)
  - convention_alternatives(): the QID 10 "Sanger by default" curation convention
Metrics including them are the primary numbers (`pooled`); the unadjusted score is kept
alongside as `pooled_strict` / `<model> correct_strict` for comparability with the paper.

Outputs (results/): detailed_rows.csv, metrics_by_qid.csv, metrics_summary.csv,
metrics_by_type.csv, pairwise_tests.csv, stability.csv (when >1 run exists).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.config import COLUMN_RENAMES  # noqa: E402
from eval.normalize import (  # noqa: E402
    _list_partial_match,
    canonicalize_answer,
    compare_lists,
    contains_negation,
    human_answer_counts,
    is_empty_token,
)
from eval.scoring import format_identifier  # noqa: E402
from frontier_compare import config  # noqa: E402
from frontier_compare.answer_cleaning import clean_answer  # noqa: E402

METRICS = ["accuracy", "precision", "recall", "f1"]
LABELS = ["tp", "tn", "fp", "fn"]
BOOTSTRAP_ITERATIONS = 5000  # matches eval/evaluation.py
SEED = 42


def load_reference() -> pd.DataFrame:
    df = pd.read_excel(config.MERGED_PATH, dtype=str, keep_default_na=False, na_filter=False)
    df = df.rename(columns=COLUMN_RENAMES)
    df["PMID"] = df["PMID"].apply(format_identifier)
    df["QID"] = df["QID"].apply(format_identifier).astype(int)
    return df[["PMID", "QID", "Question", "Type", "Category", config.REF_COL, *config.COMPARATORS]]


def attach_frontier(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Add one answer column per frontier model x run. Returns (df, primary columns, runs per model)."""
    primary, runs = [], {}
    for key, spec in config.MODELS.items():
        files = sorted(config.answers_path(key, 1).parent.glob(f"{key}_run*.csv"))
        for path in files:
            run_id = int(path.stem.rsplit("_run", 1)[1])
            col = spec.label if run_id == 1 else f"{spec.label} [run{run_id}]"
            ans = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
            ans["PMID"] = ans["PMID"].apply(format_identifier)
            ans["QID"] = ans["QID"].astype(int)
            df = df.merge(ans[["PMID", "QID", "Answer"]].rename(columns={"Answer": col}), on=["PMID", "QID"], how="left")
            runs.setdefault(spec.label, []).append(col)
            if run_id == 1:
                primary.append(col)
    return df, primary, runs


# Curation conventions the QSP prompt never states. Each rule adds accepted answers for rows
# that match it, for EVERY model, so the comparison stays symmetric.
NOT_REPORTED_FORMS = ["Not reported", "Not specified", "Not stated"]


def _pdf_mentions(pmid: str, needle: str, cache: dict[str, str]) -> bool:
    if pmid not in cache:
        path = config.PDF_DIR / f"{pmid}.pdf"
        if not path.exists():
            cache[pmid] = ""
        else:
            with pymupdf.open(path) as doc:
                cache[pmid] = " ".join(page.get_text() for page in doc).lower()
    return needle in cache[pmid]


def convention_alternatives(df: pd.DataFrame) -> dict[tuple[str, int], list[tuple[str, str]]]:
    """QID 10: curators record 'Sanger' for standard genotypic resistance testing even when the
    paper never says so (one annotation reads 'Sanger (not stated)'). Where the human answer says
    Sanger but the PDF never mentions it, 'Not reported' is equally defensible and is accepted."""
    out: dict[tuple[str, int], list[tuple[str, str]]] = {}
    cache: dict[str, str] = {}
    reason = ("Curation convention: the annotation defaults to Sanger for standard genotypic "
              "resistance testing, but the PDF never mentions Sanger, so 'Not reported' is "
              "also accepted (applied to every model).")
    for _, r in df[df["QID"] == 10].iterrows():
        if "sanger" in str(r[config.REF_COL]).lower() and not _pdf_mentions(str(r["PMID"]), "sanger", cache):
            out[(str(r["PMID"]), 10)] = [(form, reason) for form in NOT_REPORTED_FORMS]
    return out


def load_alternatives() -> dict[tuple[str, int], list[tuple[str, str]]]:
    """(PMID, QID) -> [(accepted answer, reason)] approved by a curator during error review."""
    if not config.ALTERNATIVES_PATH.exists():
        return {}
    alt = pd.read_csv(config.ALTERNATIVES_PATH, dtype=str, keep_default_na=False)
    out: dict[tuple[str, int], list[tuple[str, str]]] = {}
    for r in alt.itertuples(index=False):
        out.setdefault((str(r.PMID).strip(), int(r.QID)), []).append((r.accepted_answer, r.reason))
    return out


CONVENTION_ALTERNATIVES: dict[tuple[str, int], list[tuple[str, str]]] = {}


def score_rows(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per-row confusion label, correctness, and an error-analysis outcome."""
    ref_norm = df[config.REF_COL].map(canonicalize_answer)
    pred_raw = df[model].fillna("")
    pred_norm = pred_raw.map(canonicalize_answer)
    alternatives = load_alternatives()
    for key, value in CONVENTION_ALTERNATIVES.items():
        alternatives.setdefault(key, []).extend(value)
    labels, labels_adj, correct, outcome, adjusted, alt_used, cleaned_by = [], [], [], [], [], [], []
    for pmid, qid, qtype, question, rr, rn, pr, pn in zip(
        df["PMID"], df["QID"], df["Type"], df["Question"], df[config.REF_COL], ref_norm, pred_raw, pred_norm
    ):
        is_list = qtype.strip().lower() == "list"
        counts, ok = human_answer_counts(
            qtype, pn, rn, question_text=question, ref_raw=rr, pred_raw=pr, allow_partial_list=is_list
        )
        label = next(k for k, v in counts.items() if v)
        if ok and is_list and label == "tp" and not compare_lists(pn, rn) and _list_partial_match(pn, rn, pr):
            kind = "correct_partial_list"
        elif ok:
            kind = "correct"
        elif label == "fp":
            kind = "FP"
        elif is_empty_token(pn) or contains_negation(pr) or pn in {"no", ""}:
            kind = "FN_missed"
        else:
            kind = "FN_wrong_value"
        # Post-processing fallback: re-score the answer with explanatory scaffolding removed.
        # Non-destructive - only tried when the raw answer failed, so it can rescue but never break.
        ok_adj, used, rule_used = ok, "", ""
        if not ok:
            cleaned, rule = clean_answer(pr)
            if rule and cleaned:
                _, ok_clean = human_answer_counts(
                    qtype, canonicalize_answer(cleaned), rn, question_text=question,
                    ref_raw=rr, pred_raw=cleaned, allow_partial_list=is_list,
                )
                if ok_clean:
                    ok_adj, rule_used = True, rule
        # Also accept a curator-approved alternative reference answer.
        if not ok_adj:
            for alt_answer, reason in alternatives.get((str(pmid), int(qid)), []):
                _, alt_ok = human_answer_counts(
                    qtype, pn, canonicalize_answer(alt_answer), question_text=question,
                    ref_raw=alt_answer, pred_raw=pr, allow_partial_list=is_list,
                )
                if alt_ok:
                    ok_adj, used = True, reason
                    break
        # An accepted answer turns a miss into a hit: fn -> tp, fp -> tn.
        label_adj = label if ok_adj == ok else {"fn": "tp", "fp": "tn"}.get(label, label)
        if ok_adj and not ok:
            kind = "correct_after_cleaning" if rule_used else "correct_accepted_answer"
        labels.append(label)
        labels_adj.append(label_adj)
        correct.append(int(ok))
        outcome.append(kind)
        adjusted.append(int(ok_adj))
        alt_used.append(used)
        cleaned_by.append(rule_used)
    return pd.DataFrame({"label": labels, "label_adjusted": labels_adj, "correct": correct,
                         "outcome": outcome, "correct_adjusted": adjusted,
                         "alternative_used": alt_used, "cleaning_rule": cleaned_by}, index=df.index)


def metrics_from_counts(c: np.ndarray) -> dict[str, np.ndarray]:
    """c[..., 4] = tp, tn, fp, fn. Undefined ratios are 0, matching eval/evaluation.py."""
    tp, tn, fp, fn = (c[..., i].astype(float) for i in range(4))
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b != 0)  # noqa: E731
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": div(2 * precision * recall, precision + recall),
    }


def count_cube(rows: pd.DataFrame, pmids: list[str], model: str) -> np.ndarray:
    """(n_pmid, 16, 4) one-hot confusion labels for bootstrap resampling over papers."""
    cube = np.zeros((len(pmids), config.TOTAL_QUESTIONS, 4), dtype=int)
    pidx = {p: i for i, p in enumerate(pmids)}
    for pmid, qid, label in zip(rows["PMID"], rows["QID"], rows[f"{model} label"]):
        cube[pidx[pmid], qid - 1, LABELS.index(label)] = 1
    return cube


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-subset", action="store_true", help="Evaluate on PMIDs every frontier model answered.")
    args = parser.parse_args()

    df, frontier, runs = attach_frontier(load_reference())
    if not frontier:
        print("No frontier answers found; run 02_query_models.py and 03_parse_responses.py first.")
        return 1
    all_cols = frontier + [c for cols in runs.values() for c in cols if c not in frontier]
    variant_labels = {spec.label for spec in config.MODELS.values() if spec.is_variant}
    primary_cols = [c for c in frontier if c not in variant_labels] or frontier
    answered = df.groupby("PMID")[primary_cols].apply(lambda g: g.notna().all().all())
    pmids = sorted(answered[answered].index)
    n_total = df["PMID"].nunique()
    if len(pmids) < n_total and not args.allow_subset:
        print(f"Only {len(pmids)}/{n_total} PMIDs answered by all primary frontier models. Re-run with --allow-subset.")
        return 1
    df = df[df["PMID"].isin(pmids)].reset_index(drop=True)
    CONVENTION_ALTERNATIVES.update(convention_alternatives(df))
    if CONVENTION_ALTERNATIVES:
        print(f"Curation-convention alternatives applied to {len(CONVENTION_ALTERNATIVES)} rows (all models).")
    models = frontier + config.COMPARATORS
    print(f"Evaluating {len(models)} models on {len(pmids)} PMIDs x {config.TOTAL_QUESTIONS} QIDs")

    scored_masks: dict[str, pd.Series] = {}
    for model in models + [c for c in all_cols if c not in frontier]:
        scored_masks[model] = df[model].notna() if model in df.columns else pd.Series(True, index=df.index)
        scored = score_rows(df, model)
        df[f"{model} label"], df[f"{model} correct"], df[f"{model} outcome"] = (
            scored["label"], scored["correct"], scored["outcome"]
        )
        df[f"{model} label_strict"] = scored["label"]
        df[f"{model} label"] = scored["label_adjusted"]      # primary: accepted answers included
        df[f"{model} correct_strict"] = scored["correct"]
        df[f"{model} correct"] = scored["correct_adjusted"]  # primary
        df[f"{model} correct_adjusted"] = scored["correct_adjusted"]
        df[f"{model} alternative_used"] = scored["alternative_used"]
        df[f"{model} cleaning_rule"] = scored["cleaning_rule"]
        df.loc[~scored_masks[model], [f"{model} label", f"{model} label_strict", f"{model} correct",
                                      f"{model} correct_strict", f"{model} outcome",
                                      f"{model} correct_adjusted"]] = None
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "detailed_rows.csv", index=False)

    # Per-QID metrics
    qid_rows = []
    for model in models:
        model_rows = df[scored_masks[model]]
        for qid, grp in model_rows.groupby("QID"):
            counts = np.array([(grp[f"{model} label"] == lab).sum() for lab in LABELS])
            m = metrics_from_counts(counts)
            qid_rows.append(
                {"model": model, "QID": qid, "Type": grp["Type"].iloc[0], "Question": grp["Question"].iloc[0],
                 **{k: float(v) for k, v in m.items()}, **dict(zip(LABELS, counts.tolist()))}
            )
    by_qid = pd.DataFrame(qid_rows)
    by_qid.to_csv(config.RESULTS_DIR / "metrics_by_qid.csv", index=False)

    # Summary: pooled (paper Fig. 4) with row-bootstrap CIs; macro mean of per-QID values for reference
    rng = np.random.default_rng(SEED)
    summary_rows = []
    for model in models:
        model_rows = df[scored_masks[model]]
        model_pmids = sorted(model_rows["PMID"].unique())
        onehot = np.stack([(model_rows[f"{model} label"] == lab).to_numpy(int) for lab in LABELS], axis=1)
        boot_idx = rng.integers(0, len(model_rows), size=(BOOTSTRAP_ITERATIONS, len(model_rows)))
        pooled = metrics_from_counts(onehot.sum(axis=0))
        boot = metrics_from_counts(onehot[boot_idx].sum(axis=1))
        per_qid = metrics_from_counts(count_cube(model_rows, model_pmids, model).sum(axis=0))
        strict_onehot = np.stack([(model_rows[f"{model} label_strict"] == lab).to_numpy(int) for lab in LABELS], axis=1)
        strict = metrics_from_counts(strict_onehot.sum(axis=0))
        for metric in METRICS:
            summary_rows.append(
                {"model": model, "metric": metric, "papers": len(model_pmids), "rows": len(model_rows),
                 "pooled_strict": float(strict[metric]),
                 "pooled": float(pooled[metric]),
                 "pooled_ci_low": np.percentile(boot[metric], 2.5),
                 "pooled_ci_high": np.percentile(boot[metric], 97.5),
                 "macro": per_qid[metric].mean()}
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.RESULTS_DIR / "metrics_summary.csv", index=False)

    # By question type (macro over the QIDs of that type)
    by_qid.groupby(["model", "Type"])[METRICS].mean().reset_index().to_csv(
        config.RESULTS_DIR / "metrics_by_type.csv", index=False
    )

    # Frontier vs each comparator: paired tests over the 16 per-QID values (paper Fig. 4 stats),
    # plus exact McNemar on row correctness. BH within each (metric, test) slice.
    test_rows = []
    for f_model, comp in itertools.product(frontier, config.COMPARATORS):
        a = by_qid[by_qid["model"] == f_model].sort_values("QID")
        b = by_qid[by_qid["model"] == comp].sort_values("QID")
        for metric in METRICS:
            x, y = a[metric].to_numpy(), b[metric].to_numpy()
            diff = x - y
            base = {"frontier": f_model, "comparator": comp, "metric": metric, "mean_qid_diff": diff.mean(),
                    "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum())}
            same = not np.any(diff != 0)
            test_rows.append({**base, "test": "wilcoxon_qid", "p_raw": 1.0 if same else wilcoxon(x, y).pvalue})
            test_rows.append({**base, "test": "ttest_qid", "p_raw": 1.0 if same else ttest_rel(x, y).pvalue})
        both = scored_masks[f_model] & scored_masks.get(comp, pd.Series(True, index=df.index))
        fc, cc = df.loc[both, f"{f_model} correct"], df.loc[both, f"{comp} correct"]
        table = [[int(((fc == 1) & (cc == 1)).sum()), int(((fc == 1) & (cc == 0)).sum())],
                 [int(((fc == 0) & (cc == 1)).sum()), int(((fc == 0) & (cc == 0)).sum())]]
        test_rows.append({"frontier": f_model, "comparator": comp, "metric": "accuracy", "test": "mcnemar_rows",
                          "mean_qid_diff": fc.mean() - cc.mean(), "wins": table[0][1], "losses": table[1][0],
                          "p_raw": mcnemar(table, exact=True).pvalue})
    tests = pd.DataFrame(test_rows)
    tests["p_bh"] = tests.groupby(["metric", "test"])["p_raw"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
    tests.to_csv(config.RESULTS_DIR / "pairwise_tests.csv", index=False)

    # Run-to-run stability
    stab_rows = []
    for label, cols in runs.items():
        if len(cols) < 2:
            continue
        correct = np.column_stack([df[f"{c} correct"] for c in cols])
        answers = np.column_stack([df[c].fillna("").map(canonicalize_answer) for c in cols])
        accs = correct.mean(axis=0)
        stab_rows.append({"model": label, "n_runs": len(cols),
                          "accuracy_mean": accs.mean(), "accuracy_sd": accs.std(ddof=1),
                          "rows_same_correctness": float((correct == correct[:, :1]).all(axis=1).mean()),
                          "rows_same_normalized_answer": float((answers == answers[:, :1]).all(axis=1).mean())})
    if stab_rows:
        pd.DataFrame(stab_rows).to_csv(config.RESULTS_DIR / "stability.csv", index=False)

    print(summary.pivot(index="model", columns="metric", values="pooled").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

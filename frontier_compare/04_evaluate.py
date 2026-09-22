#!/usr/bin/env python3
"""Score frontier-model answers against the human annotations with the paper's scorer.

Per-row scoring reuses eval.normalize.human_answer_counts exactly as the paper does (partial
list matches allowed for List questions). Metrics are computed per QID across PMIDs, then
summarized across the 16 QIDs:
  macro  - unweighted mean of the 16 per-QID values (the proposal's primary aggregation)
  pooled - metrics over all PMID x QID rows (what eval/figures/full150-bar-chart.png plots)
Cached GPT-4o comparators are re-scored on the same PMID subset as the frontier models.

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
from scipy.stats import wilcoxon
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

METRICS = ["accuracy", "precision", "recall", "f1"]
LABELS = ["tp", "tn", "fp", "fn"]
BOOTSTRAP_ITERATIONS = 2000
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


def score_rows(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per-row confusion label, correctness, and an error-analysis outcome."""
    ref_norm = df[config.REF_COL].map(canonicalize_answer)
    pred_raw = df[model].fillna("")
    pred_norm = pred_raw.map(canonicalize_answer)
    labels, correct, outcome = [], [], []
    for qtype, question, rr, rn, pr, pn in zip(df["Type"], df["Question"], df[config.REF_COL], ref_norm, pred_raw, pred_norm):
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
        labels.append(label)
        correct.append(int(ok))
        outcome.append(kind)
    return pd.DataFrame({"label": labels, "correct": correct, "outcome": outcome}, index=df.index)


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
    answered = df.groupby("PMID")[frontier].apply(lambda g: g.notna().all().all())
    pmids = sorted(answered[answered].index)
    n_total = df["PMID"].nunique()
    if len(pmids) < n_total and not args.allow_subset:
        print(f"Only {len(pmids)}/{n_total} PMIDs answered by all frontier models. Re-run with --allow-subset.")
        return 1
    df = df[df["PMID"].isin(pmids)].reset_index(drop=True)
    models = frontier + config.COMPARATORS
    print(f"Evaluating {len(models)} models on {len(pmids)} PMIDs x {config.TOTAL_QUESTIONS} QIDs")

    for model in models + [c for c in all_cols if c not in frontier]:
        scored = score_rows(df, model)
        df[f"{model} label"], df[f"{model} correct"], df[f"{model} outcome"] = (
            scored["label"], scored["correct"], scored["outcome"]
        )
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "detailed_rows.csv", index=False)

    # Per-QID metrics
    qid_rows = []
    for model in models:
        for qid, grp in df.groupby("QID"):
            counts = np.array([(grp[f"{model} label"] == lab).sum() for lab in LABELS])
            m = metrics_from_counts(counts)
            qid_rows.append(
                {"model": model, "QID": qid, "Type": grp["Type"].iloc[0], "Question": grp["Question"].iloc[0],
                 **{k: float(v) for k, v in m.items()}, **dict(zip(LABELS, counts.tolist()))}
            )
    by_qid = pd.DataFrame(qid_rows)
    by_qid.to_csv(config.RESULTS_DIR / "metrics_by_qid.csv", index=False)

    # Summary: macro (mean of per-QID) and pooled, with paper-level bootstrap CIs on the macro value
    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, len(pmids), size=(BOOTSTRAP_ITERATIONS, len(pmids)))
    summary_rows = []
    for model in models:
        cube = count_cube(df, pmids, model)
        per_qid = metrics_from_counts(cube.sum(axis=0))
        pooled = metrics_from_counts(cube.sum(axis=(0, 1)))
        boot = metrics_from_counts(cube[boot_idx].sum(axis=1))  # (B, 16)
        for metric in METRICS:
            macro_boot = boot[metric].mean(axis=1)
            summary_rows.append(
                {"model": model, "metric": metric,
                 "macro": per_qid[metric].mean(),
                 "macro_ci_low": np.percentile(macro_boot, 2.5),
                 "macro_ci_high": np.percentile(macro_boot, 97.5),
                 "pooled": float(pooled[metric])}
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.RESULTS_DIR / "metrics_summary.csv", index=False)

    # By question type (macro over the QIDs of that type)
    by_qid.groupby(["model", "Type"])[METRICS].mean().reset_index().to_csv(
        config.RESULTS_DIR / "metrics_by_type.csv", index=False
    )

    # Frontier vs each comparator: Wilcoxon over 16 paired per-QID values (as in the paper's Fig. 4
    # stats) plus exact McNemar on pooled row correctness as a sensitivity analysis. BH across all.
    test_rows = []
    for f_model, comp in itertools.product(frontier, config.COMPARATORS):
        a = by_qid[by_qid["model"] == f_model].sort_values("QID")
        b = by_qid[by_qid["model"] == comp].sort_values("QID")
        for metric in METRICS:
            diff = a[metric].to_numpy() - b[metric].to_numpy()
            p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            test_rows.append({"frontier": f_model, "comparator": comp, "metric": metric, "test": "wilcoxon_qid",
                              "mean_diff": diff.mean(), "wins": int((diff > 0).sum()),
                              "losses": int((diff < 0).sum()), "p_raw": p})
        fc, cc = df[f"{f_model} correct"], df[f"{comp} correct"]
        table = [[int(((fc == 1) & (cc == 1)).sum()), int(((fc == 1) & (cc == 0)).sum())],
                 [int(((fc == 0) & (cc == 1)).sum()), int(((fc == 0) & (cc == 0)).sum())]]
        test_rows.append({"frontier": f_model, "comparator": comp, "metric": "accuracy", "test": "mcnemar_rows",
                          "mean_diff": fc.mean() - cc.mean(), "wins": table[0][1], "losses": table[1][0],
                          "p_raw": mcnemar(table, exact=True).pvalue})
    tests = pd.DataFrame(test_rows)
    tests["p_bh"] = multipletests(tests["p_raw"], method="fdr_bh")[1]
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

    print(summary.pivot(index="model", columns="metric", values="macro").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

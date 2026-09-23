#!/usr/bin/env python3
"""Score every model against the human annotations with the paper's evaluation.

Per-row scoring is eval/normalize.py::human_answer_counts, exactly as in the paper (partial list
matches allowed for List questions), on all 150 papers. A paper a model did not answer (GPT-6
Astra's request for PMID 36920025 was blocked) is scored as 16 blank answers, as the paper
scores missing answers.

Answer cleaning (answer_cleaning.py) is part of the evaluator: when a raw answer fails, it is
re-scored with explanatory commentary stripped (a preamble, a trailing note, or a wordy or
hedging parenthetical). It can rescue a row but never break one, and it runs for every model;
none of the cached GPT-4o answers needs it, so their numbers are the paper's.

Aggregation follows the paper's Figure 4: metrics pooled over all PMID x QID rows, 95% CIs from
a row bootstrap (5000, seed 42, one generator drawn in the paper's model order), which
reproduces eval/figures/full150-bar-chart-confidence-intervals.csv exactly for GPT-4o. Paired
tests run on the 16 per-QID values (Wilcoxon signed-rank as in the paper, plus the paired
t-test) and exact McNemar on rows, BH-adjusted within each (comparison set, metric, test) slice.

Outputs: results/metrics_summary.csv, results/metrics_by_qid.csv, results/statistical_tests.csv;
work/detailed_rows.csv (per-row answers, labels, correctness before and after cleaning),
work/metrics_by_type.csv.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
# eval/evaluation.py draws every model's bootstrap from one generator, in this order
PAPER_BOOTSTRAP_ORDER = ["GPT-4o base", "GPT-4o FT", "GPT-4o FT+QSP", "GPT-4o QSP"]


def load_reference() -> pd.DataFrame:
    df = pd.read_excel(config.MERGED_PATH, dtype=str, keep_default_na=False, na_filter=False)
    df = df.rename(columns=COLUMN_RENAMES)
    df["PMID"] = df["PMID"].apply(format_identifier)
    df["QID"] = df["QID"].apply(format_identifier).astype(int)
    return df[["PMID", "QID", "Question", "Type", "Category", config.REF_COL, *config.COMPARATORS]]


def attach_frontier(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add one answer column per primary frontier model (run 1)."""
    frontier = []
    for key, spec in config.MODELS.items():
        path = config.answers_path(key, 1)
        if spec.is_variant or not path.exists():
            continue
        ans = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
        ans["PMID"] = ans["PMID"].apply(format_identifier)
        ans["QID"] = ans["QID"].astype(int)
        df = df.merge(ans[["PMID", "QID", "Answer"]].rename(columns={"Answer": spec.label}),
                      on=["PMID", "QID"], how="left")
        frontier.append(spec.label)
    return df, frontier


def score_rows(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per-row confusion label and correctness (paper's scorer, then the cleaning fallback)."""
    ref_norm = df[config.REF_COL].map(canonicalize_answer)
    pred_raw = df[model].fillna("")
    pred_norm = pred_raw.map(canonicalize_answer)
    out = {"label_raw": [], "correct_raw": [], "label": [], "correct": [], "outcome": [], "cleaning_rule": []}
    for qtype, question, rr, rn, pr, pn in zip(
        df["Type"], df["Question"], df[config.REF_COL], ref_norm, pred_raw, pred_norm
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
        # Cleaning fallback: only tried when the raw answer fails, so it can rescue but never break
        ok_final, rule_used, label_final = ok, "", label
        if not ok:
            cleaned, rule = clean_answer(pr)
            if rule and cleaned:
                counts_c, ok_clean = human_answer_counts(
                    qtype, canonicalize_answer(cleaned), rn, question_text=question,
                    ref_raw=rr, pred_raw=cleaned, allow_partial_list=is_list,
                )
                if ok_clean:
                    ok_final, rule_used = True, rule
                    label_final = next(k for k, v in counts_c.items() if v)
                    kind = "correct_after_cleaning"
        out["label_raw"].append(label)
        out["correct_raw"].append(int(ok))
        out["label"].append(label_final)
        out["correct"].append(int(ok_final))
        out["outcome"].append(kind)
        out["cleaning_rule"].append(rule_used)
    return pd.DataFrame(out, index=df.index)


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


def bootstrap_indices(models: list[str], n_rows: int) -> dict[str, np.ndarray]:
    """Row-resampling indices per model, drawn exactly as eval/evaluation.py draws them."""
    rng = np.random.default_rng(SEED)
    order = PAPER_BOOTSTRAP_ORDER + [m for m in models if m not in PAPER_BOOTSTRAP_ORDER]
    return {m: rng.integers(0, n_rows, size=(BOOTSTRAP_ITERATIONS, n_rows)) for m in order}


def paired_tests(by_qid: pd.DataFrame, df: pd.DataFrame, models: list[str], frontier: list[str]) -> pd.DataFrame:
    """Two comparison sets, each its own BH family:
    figure4  - every model vs the base comparator (GPT-4o QSP), the brackets in Figure 4
    frontier - each frontier model vs each cached GPT-4o condition"""
    pairs = [("figure4", m, config.PRIMARY_COMPARATOR) for m in models if m != config.PRIMARY_COMPARATOR]
    pairs += [("frontier", m, c) for m, c in itertools.product(frontier, config.COMPARATORS)]
    out = []
    for comparison_set, model, comp in pairs:
        a = by_qid[by_qid["model"] == model].sort_values("QID")
        b = by_qid[by_qid["model"] == comp].sort_values("QID")
        for metric in METRICS:
            x, y = a[metric].to_numpy(), b[metric].to_numpy()
            diff = x - y
            base = {"comparison_set": comparison_set, "model": model, "comparator": comp, "metric": metric,
                    "mean_qid_diff": diff.mean(), "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum())}
            same = not np.any(diff != 0)
            out.append({**base, "test": "wilcoxon_qid", "p_raw": 1.0 if same else wilcoxon(x, y).pvalue})
            out.append({**base, "test": "ttest_qid", "p_raw": 1.0 if same else ttest_rel(x, y).pvalue})
        mc, cc = df[f"{model} correct"], df[f"{comp} correct"]
        table = [[int(((mc == 1) & (cc == 1)).sum()), int(((mc == 1) & (cc == 0)).sum())],
                 [int(((mc == 0) & (cc == 1)).sum()), int(((mc == 0) & (cc == 0)).sum())]]
        out.append({"comparison_set": comparison_set, "model": model, "comparator": comp, "metric": "accuracy",
                    "test": "mcnemar_rows", "mean_qid_diff": mc.mean() - cc.mean(),
                    "wins": table[0][1], "losses": table[1][0], "p_raw": mcnemar(table, exact=True).pvalue})
    tests = pd.DataFrame(out)
    tests["p_bh"] = tests.groupby(["comparison_set", "metric", "test"])["p_raw"].transform(
        lambda p: multipletests(p, method="fdr_bh")[1])
    return tests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    df, frontier = attach_frontier(load_reference())
    if not frontier:
        print("No frontier answers found; run 02_query_models.py and 03_parse_responses.py first.")
        return 1
    models = frontier + config.COMPARATORS
    unanswered = sorted(df.loc[df[frontier].isna().any(axis=1), "PMID"].unique())
    print(f"{df['PMID'].nunique()} papers; papers without a frontier answer (scored as blanks): {unanswered or 'none'}")

    for model in models:
        df[model] = df[model].fillna("")
        scored = score_rows(df, model)
        for col in scored.columns:
            df[f"{model} {col}"] = scored[col]

    qid_rows, summary_rows = [], []
    for model in models:
        for qid, grp in df.groupby("QID"):
            counts = np.array([(grp[f"{model} label"] == lab).sum() for lab in LABELS])
            qid_rows.append({"model": model, "QID": qid, "Type": grp["Type"].iloc[0], "Question": grp["Question"].iloc[0],
                             **{k: float(v) for k, v in metrics_from_counts(counts).items()},
                             **dict(zip(LABELS, counts.tolist()))})
    by_qid = pd.DataFrame(qid_rows)
    boot_idx = bootstrap_indices(models, len(df))
    for model in models:
        onehot = np.stack([(df[f"{model} label"] == lab).to_numpy(int) for lab in LABELS], axis=1)
        point = metrics_from_counts(onehot.sum(axis=0))
        boot = metrics_from_counts(onehot[boot_idx[model]].sum(axis=1))
        raw = metrics_from_counts(np.array([(df[f"{model} label_raw"] == lab).sum() for lab in LABELS]))
        macro = by_qid[by_qid["model"] == model][METRICS].mean()
        for metric in METRICS:
            summary_rows.append({"model": model, "metric": metric, "papers": df["PMID"].nunique(), "rows": len(df),
                                 "value": float(point[metric]),
                                 "ci_low": np.percentile(boot[metric], 2.5),
                                 "ci_high": np.percentile(boot[metric], 97.5),
                                 "value_before_cleaning": float(raw[metric]),
                                 "macro": float(macro[metric])})
    summary = pd.DataFrame(summary_rows)
    tests = paired_tests(by_qid, df, models, frontier)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.WORK_DIR / "detailed_rows.csv", index=False)
    by_qid.to_csv(config.RESULTS_DIR / "metrics_by_qid.csv", index=False)
    summary.to_csv(config.RESULTS_DIR / "metrics_summary.csv", index=False)
    tests.to_csv(config.RESULTS_DIR / "statistical_tests.csv", index=False)
    by_qid.groupby(["model", "Type"])[METRICS].mean().reset_index().to_csv(
        config.WORK_DIR / "metrics_by_type.csv", index=False)

    rescued = {m: int((df[f"{m} correct"] != df[f"{m} correct_raw"]).sum()) for m in models}
    print(f"Rows rescued by answer cleaning: {rescued}")
    print((summary.pivot(index="model", columns="metric", values="value")[METRICS] * 100).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

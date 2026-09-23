#!/usr/bin/env python3
"""Secondary analyses: error outcomes, question types, hard questions, ops metrics, failure-mode sheet.

Reads work/detailed_rows.csv, results/metrics_by_qid.csv and work/ops_requests.csv, and writes
results/operations.csv, work/secondary_*.csv and failure_modes/labeling_sheet.csv. The labeling sheet lists every
row a frontier model got wrong, alongside the model's evidence/rationale and the GPT-4o FT
answer, with blank columns for a curator to assign a failure mode from failure_modes/taxonomy.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier_compare import config  # noqa: E402

OUTCOMES = ["correct", "correct_partial_list", "FP", "FN_missed", "FN_wrong_value"]


def main() -> int:
    rows = config.final_rows().fillna({config.REF_COL: ""})
    by_qid = pd.read_csv(config.RESULTS_DIR / "metrics_by_qid.csv")
    models = list(by_qid["model"].unique())
    frontier = [m for m in models if m not in config.COMPARATORS]
    out = config.WORK_DIR

    # 1. Outcome distribution per model, overall and by question type
    dist = []
    for model in models:
        for qtype, grp in [("All", rows), *rows.groupby("Type")]:
            counts = grp[f"{model} outcome"].value_counts()
            dist.append({"model": model, "Type": qtype, "n": len(grp),
                         **{o: int(counts.get(o, 0)) for o in OUTCOMES}})
    pd.DataFrame(dist).to_csv(out / "secondary_error_outcomes.csv", index=False)

    # 2. Per-QID head-to-head vs the primary comparator (which questions improve/regress)
    wide = by_qid.pivot_table(index=["QID", "Type", "Question"], columns="model", values=["accuracy", "f1"])
    h2h = []
    for (qid, qtype, question), r in wide.iterrows():
        rec = {"QID": qid, "Type": qtype, "Question": question, "hard": config.HARD_QIDS.get(qid, "")}
        for m in frontier:
            for metric in ("accuracy", "f1"):
                rec[f"{m} {metric}"] = r[(metric, m)]
                rec[f"{m} {metric} delta vs {config.PRIMARY_COMPARATOR}"] = r[(metric, m)] - r[(metric, config.PRIMARY_COMPARATOR)]
        for c in config.COMPARATORS:
            rec[f"{c} accuracy"], rec[f"{c} f1"] = r[("accuracy", c)], r[("f1", c)]
        h2h.append(rec)
    pd.DataFrame(h2h).to_csv(out / "secondary_by_qid_vs_comparator.csv", index=False)

    # 3. Paired row-level flips vs the primary comparator (fixed / broken / both wrong)
    flips = []
    for m in frontier:
        f, c = rows[f"{m} correct"].astype(int), rows[f"{config.PRIMARY_COMPARATOR} correct"].astype(int)
        for qid, g in rows.assign(f=f, c=c).groupby("QID"):
            flips.append({"model": m, "QID": qid, "fixed": int(((g.f == 1) & (g.c == 0)).sum()),
                          "broken": int(((g.f == 0) & (g.c == 1)).sum()),
                          "both_wrong": int(((g.f == 0) & (g.c == 0)).sum())})
    pd.DataFrame(flips).to_csv(out / "secondary_flips_vs_comparator.csv", index=False)

    # 4. Operational metrics
    ops_path = config.WORK_DIR / "ops_requests.csv"
    if ops_path.exists():
        ops = pd.read_csv(ops_path, dtype={"PMID": str})
        agg = ops.groupby("model_key").agg(
            requests=("PMID", "size"),
            runs=("run_id", "nunique"),
            failed_requests=("ok", lambda s: int((~s.astype(bool)).sum())),
            invalid_json_strict=("json_status", lambda s: float((s != "strict").mean())),
            invalid_json_lenient=("json_status", lambda s: float(s.isin(["invalid", "no_response"]).mean())),
            incomplete_answer_sets=("n_answers", lambda s: int((s < config.TOTAL_QUESTIONS).sum())),
            truncated=("finish_reason", lambda s: int((s == "length").sum())),
            latency_median_s=("latency_s", "median"),
            latency_p90_s=("latency_s", lambda s: s.quantile(0.9)),
            prompt_tokens_mean=("prompt_tokens", "mean"),
            completion_tokens_mean=("completion_tokens", "mean"),
            cost_total_usd=("cost_usd", lambda s: s.sum(min_count=1)),
            cost_per_paper_usd=("cost_usd", "mean"),
        )
        stab_path = out / "stability.csv"
        if stab_path.exists():
            stab = pd.read_csv(stab_path)
            label_to_key = {spec.label: key for key, spec in config.MODELS.items()}
            stab["model_key"] = stab["model"].map(label_to_key)
            agg = agg.join(stab.set_index("model_key").drop(columns="model"))
        agg.reset_index().to_csv(config.RESULTS_DIR / "operations.csv", index=False)

    # 5. Failure-mode labeling sheet (frontier errors only)
    ev_cols = ["Evidence", "EvidenceLocation", "Rationale"]
    sheets = []
    for key, spec in config.MODELS.items():
        path = config.answers_path(key, 1)
        if spec.label not in frontier or not path.exists():
            continue
        ans = pd.read_csv(path, dtype={"PMID": str}, keep_default_na=False)
        ev = ans[["PMID", "QID", *[c for c in ev_cols if c in ans.columns]]]
        wrong = rows[rows[f"{spec.label} correct"].astype(int) == 0]
        sheet = wrong[["PMID", "QID", "Type", "Question", config.REF_COL, spec.label, f"{spec.label} outcome",
                       config.BEST_COMPARATOR, f"{config.BEST_COMPARATOR} correct"]].rename(
            columns={spec.label: "Model Answer", f"{spec.label} outcome": "Outcome",
                     config.BEST_COMPARATOR: f"{config.BEST_COMPARATOR} Answer",
                     f"{config.BEST_COMPARATOR} correct": f"{config.BEST_COMPARATOR} Correct"})
        sheet = sheet.merge(ev, on=["PMID", "QID"], how="left").assign(
            Model=spec.label, failure_mode="", annotation_error="", notes="")
        sheets.append(sheet)
    if sheets:
        config.FAILURE_DIR.mkdir(parents=True, exist_ok=True)
        pd.concat(sheets).to_csv(config.FAILURE_DIR / "labeling_sheet.csv", index=False)
        print(f"Failure-mode sheet: {sum(len(s) for s in sheets)} rows")

    print("Wrote work/secondary_*.csv and results/operations.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

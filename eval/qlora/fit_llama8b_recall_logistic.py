#!/usr/bin/env python3
"""Fit logistic regression models for Llama3.1-8B recall by rank."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit GEE logistic regression for recall using llama8b_regression_data.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("llama8b_regression_data.csv"),
        help="Recall-ready long CSV.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("regression/llama8b_recall_logistic"),
        help="Prefix for output files.",
    )
    return parser.parse_args()


def odds_ratio_table(result) -> pd.DataFrame:
    conf = result.conf_int()
    table = pd.DataFrame(
        {
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "z": result.tvalues.values,
            "p_value": result.pvalues.values,
            "or": [math.exp(value) for value in result.params.values],
            "or_ci_low": [math.exp(value) for value in conf[0].values],
            "or_ci_high": [math.exp(value) for value in conf[1].values],
        }
    )
    return table


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, encoding="utf-8-sig")

    required = {"item_id", "model", "rank", "ref_positive", "detected"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    recall_df = df[df["ref_positive"] == 1].copy()
    recall_df["rank"] = pd.to_numeric(recall_df["rank"], errors="raise")
    recall_df["detected"] = pd.to_numeric(recall_df["detected"], errors="raise").astype(int)

    recall_summary = (
        recall_df.groupby(["model", "rank"], as_index=False)["detected"]
        .agg(positive_cases="count", true_positives="sum")
        .sort_values("rank")
    )
    recall_summary["recall"] = recall_summary["true_positives"] / recall_summary["positive_cases"]

    trend_fit = smf.gee(
        "detected ~ rank",
        groups="item_id",
        data=recall_df,
        family=sm.families.Binomial(),
    ).fit()

    categorical_fit = smf.gee(
        "detected ~ C(rank, Treatment(reference=8))",
        groups="item_id",
        data=recall_df,
        family=sm.families.Binomial(),
    ).fit()

    summary_path = args.output_prefix.with_suffix(".summary.txt")
    or_path = args.output_prefix.with_suffix(".odds_ratios.csv")
    recall_summary_path = args.output_prefix.with_suffix(".recall_summary.csv")

    recall_summary.to_csv(recall_summary_path, index=False)
    odds_tables = pd.concat(
        [
            odds_ratio_table(trend_fit).assign(model_fit="numeric_rank_trend"),
            odds_ratio_table(categorical_fit).assign(model_fit="categorical_rank_vs_R8"),
        ],
        ignore_index=True,
    )
    odds_tables.to_csv(or_path, index=False)

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Recall summary\n")
        handle.write(recall_summary.to_string(index=False))
        handle.write("\n\nNumeric rank trend model: detected ~ rank\n")
        handle.write(trend_fit.summary().as_text())
        handle.write("\n\nCategorical rank model: detected ~ C(rank), reference rank=8\n")
        handle.write(categorical_fit.summary().as_text())
        handle.write("\n")

    print(f"Recall rows used: {len(recall_df)}")
    print(recall_summary.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {or_path}")
    print(f"Wrote {recall_summary_path}")


if __name__ == "__main__":
    main()

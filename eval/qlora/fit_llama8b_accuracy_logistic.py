#!/usr/bin/env python3
"""Fit logistic regression models for Llama3.1-8B accuracy by rank."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit GEE logistic regression for accuracy using llama8b_regression_data.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("llama8b_regression_data.csv"),
        help="Long CSV with per-item correctness by rank.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("regression/llama8b_accuracy_logistic"),
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

    required = {"item_id", "model", "rank", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    accuracy_df = df.copy()
    accuracy_df["rank"] = pd.to_numeric(accuracy_df["rank"], errors="raise")
    accuracy_df["correct"] = pd.to_numeric(accuracy_df["correct"], errors="raise").astype(int)

    accuracy_summary = (
        accuracy_df.groupby(["model", "rank"], as_index=False)["correct"]
        .agg(total_cases="count", correct_cases="sum")
        .sort_values("rank")
    )
    accuracy_summary["accuracy"] = accuracy_summary["correct_cases"] / accuracy_summary["total_cases"]

    trend_fit = smf.gee(
        "correct ~ rank",
        groups="item_id",
        data=accuracy_df,
        family=sm.families.Binomial(),
    ).fit()

    categorical_fit = smf.gee(
        "correct ~ C(rank, Treatment(reference=8))",
        groups="item_id",
        data=accuracy_df,
        family=sm.families.Binomial(),
    ).fit()

    summary_path = args.output_prefix.with_suffix(".summary.txt")
    or_path = args.output_prefix.with_suffix(".odds_ratios.csv")
    accuracy_summary_path = args.output_prefix.with_suffix(".accuracy_summary.csv")

    accuracy_summary.to_csv(accuracy_summary_path, index=False)
    odds_tables = pd.concat(
        [
            odds_ratio_table(trend_fit).assign(model_fit="numeric_rank_trend"),
            odds_ratio_table(categorical_fit).assign(model_fit="categorical_rank_vs_R8"),
        ],
        ignore_index=True,
    )
    odds_tables.to_csv(or_path, index=False)

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Accuracy summary\n")
        handle.write(accuracy_summary.to_string(index=False))
        handle.write("\n\nNumeric rank trend model: correct ~ rank\n")
        handle.write(trend_fit.summary().as_text())
        handle.write("\n\nCategorical rank model: correct ~ C(rank), reference rank=8\n")
        handle.write(categorical_fit.summary().as_text())
        handle.write("\n")

    print(f"Accuracy rows used: {len(accuracy_df)}")
    print(accuracy_summary.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {or_path}")
    print(f"Wrote {accuracy_summary_path}")


if __name__ == "__main__":
    main()

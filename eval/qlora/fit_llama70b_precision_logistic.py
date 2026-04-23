#!/usr/bin/env python3
"""Fit logistic regression models for Llama3.1-70B precision by rank."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit GEE logistic regression for precision using llama70b_recall_long.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("llama70b_recall_long.csv"),
        help="Long CSV with per-item TP/TN/FP/FN outcomes.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("llama70b_precision_logistic"),
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
    df = pd.read_csv(args.input, encoding="utf-8-sig")

    required = {"item_id", "model", "rank", "outcome"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Precision is TP / (TP + FP), so use the scorer's actual TP/FP outcomes.
    precision_df = df[df["outcome"].isin(["TP", "FP"])].copy()
    precision_df["rank"] = pd.to_numeric(precision_df["rank"], errors="raise")
    precision_df["precise"] = (precision_df["outcome"] == "TP").astype(int)

    precision_summary = (
        precision_df.groupby(["model", "rank"], as_index=False)["precise"]
        .agg(predicted_positive_cases="count", true_positives="sum")
        .sort_values("rank")
    )
    precision_summary["precision"] = (
        precision_summary["true_positives"] / precision_summary["predicted_positive_cases"]
    )

    trend_fit = smf.gee(
        "precise ~ rank",
        groups="item_id",
        data=precision_df,
        family=sm.families.Binomial(),
    ).fit()

    categorical_fit = smf.gee(
        "precise ~ C(rank, Treatment(reference=8))",
        groups="item_id",
        data=precision_df,
        family=sm.families.Binomial(),
    ).fit()

    summary_path = args.output_prefix.with_suffix(".summary.txt")
    or_path = args.output_prefix.with_suffix(".odds_ratios.csv")
    precision_summary_path = args.output_prefix.with_suffix(".precision_summary.csv")

    precision_summary.to_csv(precision_summary_path, index=False)
    odds_tables = pd.concat(
        [
            odds_ratio_table(trend_fit).assign(model_fit="numeric_rank_trend"),
            odds_ratio_table(categorical_fit).assign(model_fit="categorical_rank_vs_R8"),
        ],
        ignore_index=True,
    )
    odds_tables.to_csv(or_path, index=False)

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Precision summary\n")
        handle.write(precision_summary.to_string(index=False))
        handle.write("\n\nNumeric rank trend model: precise ~ rank\n")
        handle.write(trend_fit.summary().as_text())
        handle.write("\n\nCategorical rank model: precise ~ C(rank), reference rank=8\n")
        handle.write(categorical_fit.summary().as_text())
        handle.write("\n")

    print(f"Precision rows used: {len(precision_df)}")
    print(precision_summary.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {or_path}")
    print(f"Wrote {precision_summary_path}")


if __name__ == "__main__":
    main()

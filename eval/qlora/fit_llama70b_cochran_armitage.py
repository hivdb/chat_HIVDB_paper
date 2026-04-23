#!/usr/bin/env python3
"""Run Cochran-Armitage trend tests on llama70b_regression_data.csv."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Cochran-Armitage trend test on llama70b_regression_data.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("llama70b_regression_data.csv"),
        help="Long CSV with per-item rank outcomes.",
    )
    parser.add_argument(
        "--metric",
        choices=["accuracy", "recall", "precision"],
        default="accuracy",
        help="Binary endpoint to test across ordered ranks.",
    )
    parser.add_argument(
        "--alternative",
        choices=["two-sided", "increasing", "decreasing"],
        default="two-sided",
        help="Alternative hypothesis for the trend test.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("regression/llama70b_cochran_armitage"),
        help="Prefix for output files.",
    )
    return parser.parse_args()


def cochran_armitage_test(
    successes: list[int],
    totals: list[int],
    scores: list[float],
    alternative: str,
) -> dict[str, float]:
    if len(successes) != len(totals) or len(successes) != len(scores):
        raise ValueError("successes, totals, and scores must have the same length")
    if len(successes) < 2:
        raise ValueError("need at least two ordered groups")

    total_n = sum(totals)
    total_successes = sum(successes)
    if total_n <= 1:
        raise ValueError("at least two total observations are required")
    if total_successes == 0 or total_successes == total_n:
        raise ValueError("test undefined when all outcomes are identical")

    weighted_score_mean = sum(n_i * score for n_i, score in zip(totals, scores)) / total_n
    statistic = sum(
        (score - weighted_score_mean) * success
        for score, success in zip(scores, successes)
    )
    variance = (
        total_successes
        * (total_n - total_successes)
        / (total_n * (total_n - 1))
        * sum(
            n_i * (score - weighted_score_mean) ** 2
            for n_i, score in zip(totals, scores)
        )
    )
    if variance <= 0:
        raise ValueError("variance is zero; the test cannot be computed")

    z_stat = statistic / math.sqrt(variance)
    normal = NormalDist()
    if alternative == "two-sided":
        p_value = 2 * (1 - normal.cdf(abs(z_stat)))
    elif alternative == "increasing":
        p_value = 1 - normal.cdf(z_stat)
    else:
        p_value = normal.cdf(z_stat)

    return {
        "statistic": statistic,
        "variance": variance,
        "z": z_stat,
        "p_value": p_value,
    }


def build_metric_frame(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str]:
    if metric == "accuracy":
        required = {"model", "rank", "correct"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns for accuracy: {sorted(missing)}")
        metric_df = df.copy()
        metric_df["success"] = pd.to_numeric(metric_df["correct"], errors="raise").astype(int)
        return metric_df, "correct"

    if metric == "recall":
        required = {"model", "rank", "ref_positive", "detected"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns for recall: {sorted(missing)}")
        metric_df = df.copy()
        metric_df["ref_positive"] = pd.to_numeric(metric_df["ref_positive"], errors="raise").astype(int)
        metric_df = metric_df[metric_df["ref_positive"] == 1].copy()
        metric_df["success"] = pd.to_numeric(metric_df["detected"], errors="raise").astype(int)
        return metric_df, "detected"

    required = {"model", "rank", "outcome"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for precision: {sorted(missing)}")
    metric_df = df[df["outcome"].isin(["TP", "FP"])].copy()
    metric_df["success"] = (metric_df["outcome"] == "TP").astype(int)
    return metric_df, "precise"


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    metric_df, success_label = build_metric_frame(df, args.metric)
    if metric_df.empty:
        raise ValueError(f"No rows available for metric {args.metric}")

    metric_df["rank"] = pd.to_numeric(metric_df["rank"], errors="raise")

    summary = (
        metric_df.groupby(["model", "rank"], as_index=False)["success"]
        .agg(total_cases="count", success_cases="sum")
        .sort_values("rank")
    )
    summary[args.metric] = summary["success_cases"] / summary["total_cases"]

    scores = summary["rank"].astype(float).tolist()
    successes = summary["success_cases"].astype(int).tolist()
    totals = summary["total_cases"].astype(int).tolist()
    result = cochran_armitage_test(successes, totals, scores, args.alternative)

    results_table = pd.DataFrame(
        [
            {
                "metric": args.metric,
                "success_definition": success_label,
                "alternative": args.alternative,
                "scores": ",".join(
                    str(int(score)) if float(score).is_integer() else str(score)
                    for score in scores
                ),
                "successes": ",".join(str(value) for value in successes),
                "totals": ",".join(str(value) for value in totals),
                "statistic": result["statistic"],
                "variance": result["variance"],
                "z": result["z"],
                "p_value": result["p_value"],
            }
        ]
    )

    summary_path = args.output_prefix.with_name(
        f"{args.output_prefix.name}.{args.metric}_summary.csv"
    )
    result_path = args.output_prefix.with_name(
        f"{args.output_prefix.name}.{args.metric}_result.csv"
    )
    text_path = args.output_prefix.with_name(
        f"{args.output_prefix.name}.{args.metric}.txt"
    )

    summary.to_csv(summary_path, index=False)
    results_table.to_csv(result_path, index=False)

    with text_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Cochran-Armitage trend test for {args.metric}\n")
        handle.write(f"Alternative: {args.alternative}\n")
        handle.write(f"Success definition: {success_label}\n\n")
        handle.write(summary.to_string(index=False))
        handle.write("\n\n")
        handle.write(f"Statistic: {result['statistic']:.10f}\n")
        handle.write(f"Variance: {result['variance']:.10f}\n")
        handle.write(f"Z: {result['z']:.10f}\n")
        handle.write(f"P-value: {result['p_value']:.10g}\n")
        handle.write(
            "\nNote: this test treats rows within each rank as independent and does not model"
            " repeated measures across the same item_id.\n"
        )

    print(summary.to_string(index=False))
    print(f"\nCochran-Armitage test for {args.metric}")
    print(f"Alternative: {args.alternative}")
    print(f"Z = {result['z']:.6f}")
    print(f"p-value = {result['p_value']:.6g}")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {result_path}")
    print(f"Wrote {text_path}")


if __name__ == "__main__":
    main()

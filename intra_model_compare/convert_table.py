#!/usr/bin/env python3
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert evaluation metrics table into wide format by model/QID."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="evaluation_metrics_by_qid_full150.csv",
        help="Input CSV file path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="evaluation_metrics_by_qid_full150_converted.csv",
        help="Output CSV file path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    parts = df["model"].str.split(" ", n=1, expand=True)
    df["model_name"] = parts[0]
    df["variant"] = parts[1]

    metric_map = {
        "accuracy": "A",
        "precision": "P",
        "recall": "R",
        "f1": "F",
    }

    long_df = df.melt(
        id_vars=["QID", "model_name", "variant"],
        value_vars=list(metric_map.keys()),
        var_name="metric",
        value_name="value",
    )
    long_df["metric"] = long_df["metric"].map(metric_map)
    long_df["col"] = long_df["variant"] + " " + long_df["metric"]

    wide_df = (
        long_df.pivot_table(
            index=["QID", "model_name"],
            columns="col",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    variant_order = ["base", "FT", "QSP", "FT+QSP"]
    metric_order = ["A", "P", "R", "F"]
    ordered_cols = [
        f"{variant} {metric}"
        for variant in variant_order
        for metric in metric_order
        if f"{variant} {metric}" in wide_df.columns
    ]
    wide_df = wide_df[["QID", "model_name"] + ordered_cols]

    wide_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

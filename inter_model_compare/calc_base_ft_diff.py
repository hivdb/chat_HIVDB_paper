from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_table(path: Path) -> tuple[pd.DataFrame, Path]:
    """
    Load the input table from Excel or CSV.

    Prefers the provided path. If the Excel file is missing, the same name with
    a .csv extension is used as a fallback.
    """
    if path.exists():
        df = pd.read_excel(path)
        return df, path

    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df, csv_path

    raise FileNotFoundError(
        f"Could not find {path} or {csv_path}. Place the file next to this script."
    )


def compute_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pair base/FT rows by name1 + metric, then compute FT - base for columns 1-16.

    A standard deviation across the 16 differences is provided as std_diff.
    """
    metric_cols = ['Q'+str(i) for i in range(1, 17)]
    required_cols = {"name1", "name2", "metric", *metric_cols}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Normalize casing in case the file uses different capitalization.
    df = df.copy()
    df["name2"] = df["name2"].str.lower()

    base = df[df["name2"] == "base"]
    ft = df[df["name2"] == "ft"]

    key_cols = ["name1", "metric"]
    merged = base.merge(
        ft,
        on=key_cols,
        suffixes=("_base", "_ft"),
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError("No base/FT pairs found when grouping by name1 + metric.")

    output = merged[key_cols].copy()
    for col in metric_cols:
        output[col] = merged[f"{col}_ft"] - merged[f"{col}_base"]

    output["std_diff"] = output[metric_cols].std(axis=1, ddof=1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute FT-base differences and std across columns 1-16."
    )
    parser.add_argument(
        "--input",
        default="metrics_by_model_and_metric.xlsx",
        type=Path,
        help="Input Excel file (falls back to .csv with the same name if missing).",
    )
    parser.add_argument(
        "--output",
        default="base_ft_diff.csv",
        type=Path,
        help="Where to write the difference table.",
    )
    args = parser.parse_args()

    df, source = load_table(args.input)
    diff_df = compute_differences(df)
    diff_df.to_csv(args.output, index=False)

    print(f"Loaded data from: {source}")
    print(f"Saved differences to: {args.output}")
    # print(diff_df)


if __name__ == "__main__":
    main()

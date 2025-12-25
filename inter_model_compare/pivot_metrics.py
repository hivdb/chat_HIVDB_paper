"""Build a QID-wide metrics table from evaluation_metrics_by_qid.csv.

The script filters to `scenario == "Overall - partial match"`, then for each
model writes four rows (accuracy, precision, recall, f1). Each QID becomes a
column so the final CSV looks like:

model,metric,Q1,Q2,...,Q16
GPT-5 base,accuracy,0.93,...
GPT-5 base,precision,...
...
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


METRIC_COLUMNS: Sequence[str] = ("accuracy", "precision", "recall", "f1")

NAME1_CODE = {"GPT-4o": "1", "Llama3.1-70B": "2", "Llama3.1-8B": "3"}
NAME2_CODE = {"base": "1", "FT": "2", "QSP": "3", "FT+QSP": "4"}
METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}
METRIC_CODE = {"accuracy": "1", "precision": "2", "recall": "3", "f1": "4"}


def split_model(model: str) -> tuple[str, str]:
    """Split model name into name1 and name2 parts."""

    parts = model.split(" ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], ""


def build_name_code(name1: str, metric: str, name2: str) -> str:
    """Return the numeric code derived from name1/metric/name2."""

    return (
        NAME1_CODE.get(name1, "")
        + METRIC_CODE.get(metric, "")
        + NAME2_CODE.get(name2, "")
    )


def build_wide_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table keyed by model/metric with QIDs spread into columns."""

    filtered = df

    qids = sorted(filtered["QID"].unique())
    rows = []

    # groupby preserves the incoming order of models for readability
    for model, group in filtered.groupby("model", sort=False):
        name1, name2 = split_model(model)
        values_by_qid = group.set_index("QID")
        for metric in METRIC_COLUMNS:
            display_metric = METRIC_DISPLAY.get(metric, metric.capitalize())
            name_code = build_name_code(name1, metric, name2)
            row = [model, name1, name2, display_metric, name_code]
            # fill missing QIDs with blanks to keep table rectangular
            raw_series = values_by_qid.get(metric, default=pd.Series())
            scaled = (raw_series * 100).round(1)
            # Keep blanks for missing QIDs; otherwise format with one decimal place
            series = scaled.reindex(qids).apply(
                lambda x: "" if pd.isna(x) else f"{x:.1f}"
            )
            row.extend(series.tolist())
            rows.append(row)

    columns = ["model", "name1", "name2", "metric", "name"] + [
        f"Q{qid}" for qid in qids
    ]
    return pd.DataFrame(rows, columns=columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation_metrics_by_qid_full150.csv"),
        help="Path to the source CSV (default: evaluation_metrics_by_qid_full150.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics_by_model_and_metric.csv"),
        help="Where to write the transformed table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    wide = build_wide_table(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.output, index=False)
    print(f"Wrote {len(wide)} rows to {args.output}")


if __name__ == "__main__":
    main()

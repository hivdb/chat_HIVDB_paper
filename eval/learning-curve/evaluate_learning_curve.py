#!/usr/bin/env python3
"""Score learning-curve runs with the standard evaluation toolkit."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

from eval import config  # type: ignore
from eval.scoring import (  # type: ignore
    build_detail_rows,
    ensure_norm,
    evaluate_group,
    format_identifier,
    load_dataset,
)


@dataclass
class RunSpec:
    label: str
    path: Path
    column: str


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH for --responses.")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip())
    if not label:
        raise argparse.ArgumentTypeError("Response label cannot be empty.")
    return label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        type=parse_run,
        action="append",
        metavar="LABEL=PATH",
        help="Map a label (e.g., size050) to its response CSV.",
    )
    parser.add_argument(
        "--column-prefix",
        type=str,
        default="GPT-4o LC",
        help="Prefix used when naming new evaluation columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/learning-curve/results"),
        help="Directory for metrics/details/summary outputs.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on evaluation rows.")
    return parser.parse_args()


def discover_default_responses() -> List[Tuple[str, Path]]:
    base_dir = Path("eval/learning-curve/responses")
    if not base_dir.exists():
        return []
    candidates = sorted(base_dir.glob("*_responses.csv"))
    responses: List[Tuple[str, Path]] = []
    for path in candidates:
        stem = path.stem
        if not stem.endswith("_responses"):
            continue
        label = stem[: -len("_responses")]
        if label:
            responses.append((label, path))
    return responses


def integrate_responses(df: pd.DataFrame, path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Response CSV missing: {path}")
    df_resp = pd.read_csv(path, dtype={"PMID": str})
    required = {"PMID", "QID", "Answer"}
    if missing := required - set(df_resp.columns):
        raise ValueError(f"{path} missing required columns: {', '.join(sorted(missing))}")
    df_resp["PMID"] = df_resp["PMID"].apply(format_identifier)
    df_resp["QID"] = df_resp["QID"].apply(format_identifier)
    df_resp["sample_id"] = df_resp["PMID"] + "-" + df_resp["QID"]
    mapping = df_resp.set_index("sample_id")["Answer"].to_dict()
    return df["sample_id"].map(mapping).fillna("")


def scenario_copy(models: Sequence[str]) -> List[dict]:
    overrides: List[dict] = []
    for scenario in config.SCENARIOS:
        clone = dict(scenario)
        clone["models"] = list(models)
        overrides.append(clone)
    return overrides


def evaluate(df: pd.DataFrame, scenarios: Iterable[dict]) -> Tuple[pd.DataFrame, List[dict]]:
    cache: dict = {}
    scenario_frames: List[pd.DataFrame] = []
    details: List[dict] = []
    for scenario in scenarios:
        scenario_df = df
        if filter_type := scenario.get("filter_type"):
            scenario_df = df[df["Type"] == filter_type].copy()
        if scenario_df.empty:
            continue
        convert = scenario["convert_special_no"]
        relevant_columns = [scenario["reference"], *scenario["models"]]
        norm_lookup = {
            col: ensure_norm(df, col, convert, cache)
            for col in relevant_columns
            if col in df.columns
        }
        subset = evaluate_group(
            scenario_df,
            scenario["models"],
            scenario["reference"],
            scenario["title"],
            norm_lookup,
            convert,
            allow_partial_list=scenario.get("allow_partial_list", False),
        )
        if subset.empty:
            continue
        if scenario.get("include_details", True):
            details.extend(build_detail_rows(scenario_df, scenario, norm_lookup))
        scenario_frames.append(subset)
    metrics = pd.concat(scenario_frames, ignore_index=True) if scenario_frames else pd.DataFrame()
    return metrics, details


def main() -> int:
    args = parse_args()
    responses = args.responses or discover_default_responses()
    if not responses:
        raise SystemExit(
            "No responses provided. Run with --responses LABEL=PATH or store CSVs under eval/learning-curve/responses/."
        )
    df = load_dataset()
    if args.limit:
        df = df.head(args.limit)
    df["sample_id"] = df["PMID"] + "-" + df["QID"]

    runs: List[RunSpec] = []
    for label, path in responses:
        column = f"{args.column_prefix} {label}".strip()
        df[column] = integrate_responses(df, path)
        runs.append(RunSpec(label=label, path=path, column=column))

    if not runs:
        raise SystemExit("No response files were integrated.")

    scenarios = scenario_copy([run.column for run in runs])
    metrics, detail_rows = evaluate(df, scenarios)
    if metrics.empty:
        raise SystemExit("No metrics produced. Check response coverage.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "learning_curve_metrics.csv"
    details_path = args.output_dir / "learning_curve_details.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(detail_rows).to_csv(details_path, index=False, encoding="utf-8-sig")

    summary = []
    overall = metrics[metrics["scenario"] == "Overall"]
    for run in runs:
        row = overall[overall["model"] == run.column]
        summary.append(
            {
                "label": run.label,
                "responses": str(run.path),
                "column": run.column,
                "accuracy": float(row["accuracy"].iloc[0]) if not row.empty else None,
                "precision": float(row["precision"].iloc[0]) if not row.empty else None,
                "recall": float(row["recall"].iloc[0]) if not row.empty else None,
                "f1": float(row["f1"].iloc[0]) if not row.empty else None,
            }
        )
    summary_path = args.output_dir / "learning_curve_summary.json"
    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump({"runs": summary}, outfile, indent=2)

    print(f"Metrics written to {metrics_path}")
    print(f"Details written to {details_path}")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

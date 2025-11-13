#!/usr/bin/env python3
"""Evaluate model outputs against human answers with lenient matching."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

from eval import config  # type: ignore
from eval.plots import generate_figures  # type: ignore
from eval.scoring import build_detail_rows, ensure_norm, evaluate_group, load_dataset  # type: ignore


def run(limit: int | None) -> Tuple[pd.DataFrame, List[dict], List[Tuple[str, str, pd.DataFrame]]]:
    df = load_dataset()
    if limit:
        df = df.head(limit)
    scenario_frames: List[pd.DataFrame] = []
    detail_rows: List[dict] = []
    figure_specs: List[Tuple[str, str, pd.DataFrame]] = []

    cache: dict = {}
    for scenario in config.SCENARIOS:
        scenario_df = df
        if filter_type := scenario.get("filter_type"):
            scenario_df = df[df["Type"] == filter_type].copy()
        if scenario_df.empty:
            continue

        missing_models = [model for model in scenario["models"] if model not in df.columns]
        if missing_models:
            logging.warning("Scenario '%s' missing models: %s", scenario["title"], ", ".join(missing_models))

        convert = scenario["convert_special_no"]
        norm_lookup = {col: ensure_norm(df, col, convert, cache) for col in [scenario["reference"], *scenario["models"]] if col in df.columns}
        subset = evaluate_group(
            scenario_df,
            scenario["models"],
            scenario["reference"],
            scenario["title"],
            norm_lookup,
            convert,
        )
        if subset.empty:
            continue
        detail_rows.extend(build_detail_rows(scenario_df, scenario, norm_lookup))
        scenario_frames.append(subset)
        figure_specs.append((scenario["title"], scenario["footnote"], subset))
    combined = pd.concat(scenario_frames, ignore_index=True) if scenario_frames else pd.DataFrame()
    return combined, detail_rows, figure_specs


def write_outputs(metrics: pd.DataFrame, details: List[dict]) -> None:
    metrics.to_csv(config.OUTPUT_METRICS, index=False)
    detail_df = pd.DataFrame(details)
    detail_df.sort_values(["Scenario", "sort_key"], inplace=True)
    detail_df.drop(columns=["sort_key"], inplace=True)
    detail_df.to_csv(config.DETAIL_METRICS_HUMAN, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics, details, figures = run(args.limit)
    if metrics.empty:
        logging.error("No scenarios produced metrics.")
        return 1
    write_outputs(metrics, details)
    logging.info("Wrote metrics to %s", config.OUTPUT_METRICS)
    logging.info("Wrote detail rows to %s", config.DETAIL_METRICS_HUMAN)
    for title, footnote, subset in figures:
        generate_figures(subset, title, footnote, config.OUTPUT_TABLE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

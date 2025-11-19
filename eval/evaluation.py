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
from eval.normalize import match_scenario_label  # type: ignore


LIST_SCENARIO_TITLES = {
    "List questions (exact match)",
    "List questions (partial match)",
}


def run(limit: int | None) -> Tuple[
    pd.DataFrame,
    List[dict],
    List[Tuple[str, str, pd.DataFrame]],
    List[dict],
    List[dict],
]:
    df = load_dataset()
    if limit:
        df = df.head(limit)
    scenario_frames: List[pd.DataFrame] = []
    detail_rows: List[dict] = []
    figure_specs: List[Tuple[str, str, pd.DataFrame]] = []
    exact_partial_details: List[dict] = []
    partial_only_details: List[dict] = []

    cache: dict = {}
    scenario_results: dict[str, pd.DataFrame] = {}
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
        allow_partial = scenario.get("allow_partial_list", False)
        match_label = match_scenario_label(allow_partial)
        detail_type_filter = scenario.get("detail_types")
        detail_types = set(detail_type_filter) if detail_type_filter else None
        norm_lookup = {col: ensure_norm(df, col, convert, cache) for col in [scenario["reference"], *scenario["models"]] if col in df.columns}
        subset = evaluate_group(
            scenario_df,
            scenario["models"],
            scenario["reference"],
            scenario["title"],
            norm_lookup,
            convert,
            allow_partial_list=allow_partial,
        )
        if subset.empty:
            continue
        scenario_id = scenario.get("scenario_id")
        if scenario.get("compare_to"):
            base_id = scenario.get("compare_to")
            base_subset = scenario_results.get(base_id)
            if base_subset is None:
                logging.warning("Scenario '%s' requires compare_to '%s', which is missing.", scenario["title"], base_id)
                continue
            metrics_cols = ["accuracy", "precision", "recall", "f1"]
            merged = subset.merge(
                base_subset[["model", *metrics_cols]],
                on="model",
                suffixes=("", "_base"),
            )
            for col in metrics_cols:
                merged[col] = merged[col] - merged[f"{col}_base"]
                merged.drop(columns=f"{col}_base", inplace=True)
            subset = merged
        if scenario_id:
            scenario_results[scenario_id] = subset.copy()
        if scenario.get("include_details", True):
            detail_rows.extend(
                build_detail_rows(
                    scenario_df,
                    scenario,
                    norm_lookup,
                    match_label,
                    detail_types,
                )
            )
        if scenario["title"] in LIST_SCENARIO_TITLES:
            exact_partial_details.extend(build_detail_rows(scenario_df, scenario, norm_lookup, match_label))
            if scenario["title"] == "List questions (partial match)":
                partial_only_details.extend(build_detail_rows(scenario_df, scenario, norm_lookup, match_label))
        scenario_frames.append(subset)
        figure_specs.append((scenario["title"], scenario["footnote"], subset))
    combined = pd.concat(scenario_frames, ignore_index=True) if scenario_frames else pd.DataFrame()
    return combined, detail_rows, figure_specs, exact_partial_details, partial_only_details


def write_outputs(metrics: pd.DataFrame, details: List[dict], extra_details: List[dict], partial_details: List[dict]) -> None:
    metrics.to_csv(config.OUTPUT_METRICS, index=False, encoding="utf-8-sig")
    detail_df = pd.DataFrame(details)
    detail_df.sort_values(["sort_key"], inplace=True)
    detail_df.drop(columns=["sort_key"], inplace=True, errors="ignore")
    detail_df.to_csv(config.DETAIL_METRICS_HUMAN, index=False, encoding="utf-8-sig")
    if extra_details:
        extra_df = pd.DataFrame(extra_details)
        sort_cols = ["Scenario Title", "Scenario", "sort_key"]
        present_cols = [col for col in sort_cols if col in extra_df.columns]
        if present_cols:
            extra_df.sort_values(present_cols, inplace=True)
        extra_df.drop(columns=["sort_key"], inplace=True, errors="ignore")
        extra_df.to_csv(config.EXACT_VS_PARTIAL_DETAILS, index=False, encoding="utf-8-sig")
    if partial_details:
        partial_df = pd.DataFrame(partial_details)
        partial_df.sort_values(["sort_key"], inplace=True)
        partial_df.drop(columns=["sort_key", "Scenario Title"], inplace=True, errors="ignore")
        partial_df.to_csv(config.DETAIL_METRICS_PARTIAL, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics, details, figures, extra_details, partial_details = run(args.limit)
    if metrics.empty:
        logging.error("No scenarios produced metrics.")
        return 1
    write_outputs(metrics, details, extra_details, partial_details)
    logging.info("Wrote metrics to %s", config.OUTPUT_METRICS)
    logging.info("Wrote detail rows to %s", config.DETAIL_METRICS_HUMAN)
    if extra_details:
        logging.info("Wrote list scenario details to %s", config.EXACT_VS_PARTIAL_DETAILS)
    if partial_details:
        logging.info("Wrote partial-list detail rows to %s", config.DETAIL_METRICS_PARTIAL)
    for title, footnote, subset in figures:
        generate_figures(subset, title, footnote, config.OUTPUT_TABLE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

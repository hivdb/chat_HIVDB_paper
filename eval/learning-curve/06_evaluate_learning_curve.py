#!/usr/bin/env python3
"""Score learning-curve runs with the standard evaluation toolkit."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

LC_DIR = Path(__file__).resolve().parent
ROOT = LC_DIR.parents[1]
ROOT_PARENT = ROOT.parent
if str(ROOT_PARENT) not in sys.path:
    sys.path.insert(0, str(ROOT_PARENT))

from eval import config  # type: ignore
from eval.evaluation import build_qid_metrics  # type: ignore
from eval.scoring import (  # type: ignore
    build_detail_rows,
    ensure_norm,
    evaluate_group,
    format_identifier,
    load_dataset,
)
from eval.normalize import match_scenario_label  # type: ignore
from eval import statistics as stat_utils  # type: ignore


@dataclass
class RunSpec:
    label: str
    path: Path
    column: str


LABEL_TO_MODEL = {
    "base": "base",
    "ft50": "FT-50",
    "ft100": "FT-100",
    "ft150": "FT-150",
    "ft200": "FT-200",
    "ft": "FT",
    # Backward compatibility with older sizeXXX tags
    "size050": "FT-50",
    "size100": "FT-100",
    "size150": "FT-150",
    "size200": "FT-200",
    "50": "FT-50",
    "100": "FT-100",
    "150": "FT-150",
    "200": "FT-200",
}

def _lookup_model_metric(qid_df: pd.DataFrame | None, comparison_label: str, model_map: Dict[str, str]) -> float | None:
    if qid_df is None or qid_df.empty:
        return None
    col = model_map.get(comparison_label) or comparison_label
    if col not in qid_df.columns:
        return None
    series = qid_df[col]
    try:
        return float(series.mean())
    except Exception:
        return None


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH for --responses.")
    label, raw_path = value.split("=", 1)
    label = label.strip().lower()
    path = Path(raw_path.strip())
    if not label:
        raise argparse.ArgumentTypeError("Response label cannot be empty.")
    if label not in LABEL_TO_MODEL:
        raise argparse.ArgumentTypeError(f"Unknown run label '{label}'. Expected one of: {', '.join(sorted(LABEL_TO_MODEL))}.")
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
        default="GPT-4o",
        help="Prefix used when naming new evaluation columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LC_DIR / "results",
        help="Directory for metrics/details/summary outputs.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on evaluation rows.")
    return parser.parse_args()


def discover_default_responses() -> List[Tuple[str, Path]]:
    base_dir = LC_DIR / "responses"
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
    df_resp = pd.read_csv(
        path,
        dtype={"PMID": str},
        keep_default_na=False,
        na_filter=False,
    )
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


def evaluate(df: pd.DataFrame, scenarios: Iterable[dict]) -> Tuple[pd.DataFrame, List[dict], Dict[str, pd.DataFrame]]:
    cache: dict = {}
    scenario_frames: List[pd.DataFrame] = []
    details: List[dict] = []
    qid_frames: Dict[str, pd.DataFrame] = {}
    for scenario in scenarios:
        scenario_df = df
        if filter_type := scenario.get("filter_type"):
            scenario_df = df[df["Type"] == filter_type].copy()
        if scenario_df.empty:
            continue
        relevant_columns = [config.REF_COL, *scenario["models"]]
        norm_lookup = {
            col: ensure_norm(df, col, cache)
            for col in relevant_columns
            if col in df.columns
        }
        allow_partial = scenario.get("allow_partial_list", False)
        match_label = match_scenario_label(allow_partial)
        detail_type_filter = scenario.get("detail_types")
        detail_types = set(detail_type_filter) if detail_type_filter else None
        subset = evaluate_group(
            scenario_df,
            scenario["models"],
            config.REF_COL,
            scenario["title"],
            norm_lookup,
            allow_partial_list=allow_partial,
        )
        if subset.empty:
            continue
        if scenario.get("include_details", True):
            details.extend(build_detail_rows(scenario_df, scenario, norm_lookup, match_label, detail_types))
        scenario_frames.append(subset)
        scenario_qid = build_qid_metrics(
            scenario_df,
            scenario,
            norm_lookup,
        )
        qid_frames[scenario["title"]] = pd.DataFrame(scenario_qid)
    metrics = pd.concat(scenario_frames, ignore_index=True) if scenario_frames else pd.DataFrame()
    return metrics, details, qid_frames


def build_learning_curve_comparisons(model_columns: Dict[str, str]) -> Dict[str, dict]:
    base_label = model_columns.get("base")
    if not base_label:
        return {}
    family = base_label.rsplit(" ", 1)[0] if " " in base_label else base_label
    ordered_labels = ["FT-50", "FT-100", "FT-150", "FT-200", "FT"]
    targets = [model_columns[label] for label in ordered_labels if label in model_columns]
    if not targets:
        return {}
    return {family: {"base": base_label, "targets": targets}}


def load_baseline_wilcoxon(path: Path) -> Dict[str, Dict[Tuple[str, str], float]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    df = df[df["test"] == "wilcoxon"].copy()
    mapping: Dict[str, Dict[Tuple[str, str], float]] = {}
    for _, row in df.iterrows():
        metric = row.get("metric")
        family = row.get("family")
        comparison = row.get("comparison")
        p_val = row.get("p_value")
        if pd.isna(metric) or pd.isna(family) or pd.isna(comparison) or pd.isna(p_val):
            continue
        mapping.setdefault(metric, {})[(family, comparison)] = float(p_val)
    return mapping


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

    resolved_labels = []
    for label, path in responses:
        canonical = LABEL_TO_MODEL.get(label.lower())
        if not canonical:
            raise SystemExit(f"Unrecognized response label '{label}'.")
        resolved_labels.append((canonical, path))

    runs: List[RunSpec] = []
    model_to_column: dict[str, str] = {}
    for label, path in resolved_labels:
        column = f"{args.column_prefix} {label}".strip()
        df[column] = integrate_responses(df, path)
        runs.append(RunSpec(label=label, path=path, column=column))
        model_to_column[label] = column

    if not runs:
        raise SystemExit("No response files were integrated.")

    base_column = f"{args.column_prefix} base".strip()
    if base_column in df.columns:
        model_to_column.setdefault("base", base_column)
    ft_column = f"{args.column_prefix} FT".strip()
    if ft_column in df.columns:
        model_to_column.setdefault("FT", ft_column)
    model_sequence: List[str] = []
    if base_column in df.columns:
        model_sequence.append(base_column)
    if ft_column in df.columns and ft_column not in model_sequence:
        model_sequence.append(ft_column)
    model_sequence.extend(run.column for run in runs if run.column not in model_sequence)

    scenarios = scenario_copy(model_sequence)
    metrics, detail_rows, qid_frames = evaluate(df, scenarios)
    if metrics.empty:
        raise SystemExit("No metrics produced. Check response coverage.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "learning_curve_metrics.csv"
    details_path = args.output_dir / "learning_curve_details.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(detail_rows).to_csv(details_path, index=False, encoding="utf-8-sig")

    # Export FT-200 incorrect rows for partial scenario to an Excel file
    ft200_export_path = args.output_dir / "learning_curve_ft200_incorrect.xlsx"
    ft200_column = model_to_column.get("FT-200") or model_to_column.get("FT")
    if ft200_column:
        details_df = pd.DataFrame(detail_rows)
        answer_col = f"{ft200_column} Answer"
        correct_col = f"{ft200_column} Correct"
        cols_needed = {"PMID", "Question", "Human Answer", answer_col, correct_col, "Scenario"}
        missing_cols = cols_needed - set(details_df.columns)
        if not missing_cols:
            subset = details_df[
                (details_df["Scenario"].str.contains("partial", case=False, na=False))
                & (details_df[correct_col] == 0)
            ][["PMID", "Question", "Human Answer", answer_col, correct_col]]
            if not subset.empty:
                subset.to_excel(ft200_export_path, index=False)
                logging.info("Wrote FT-200 incorrect partial rows to %s", ft200_export_path)

    # Export FT-200 incorrect rows for partial scenario to an Excel file
    ft200_export_path = args.output_dir / "learning_curve_ft200_incorrect.xlsx"
    ft200_column = model_to_column.get("FT-200") or model_to_column.get("FT")
    if ft200_column:
        details_df = pd.DataFrame(detail_rows)
        cols_needed = {"PMID", "Question", "Human Answer", ft200_column, f"{ft200_column} Correct", "Scenario"}
        missing_cols = cols_needed - set(details_df.columns)
        if not missing_cols:
            subset = details_df[
                (details_df["Scenario"].str.contains("partial", case=False, na=False))
                & (details_df[f"{ft200_column} Correct"] == 0)
            ][["PMID", "Question", "Human Answer", ft200_column, f"{ft200_column} Correct"]]
            if not subset.empty:
                subset.to_excel(ft200_export_path, index=False)
                logging.info("Wrote FT-200 incorrect partial rows to %s", ft200_export_path)
    summary = []
    overall = metrics[metrics["scenario"] == "Partial Match"]
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

    stats_path = args.output_dir / "learning_curve_pairwise_stats.csv"
    significance_path = args.output_dir / "learning_curve_significance.json"
    comparisons = build_learning_curve_comparisons(model_to_column)
    if comparisons:
        overall_qid = qid_frames.get("Partial Match")
        if overall_qid is not None and not overall_qid.empty:
            metrics_to_test = ["accuracy", "precision", "recall"]
            stats_df, wilcoxon_map, ttest_map = stat_utils.compute_pairwise_tests(
                overall_qid,
                comparisons,
                metrics_to_test,
            )
            baseline_map = load_baseline_wilcoxon(config.PAIRWISE_RESULTS)
            if baseline_map:
                for metric, entries in baseline_map.items():
                    if metric not in wilcoxon_map:
                        continue
                    for (family, comparison), value in entries.items():
                        if comparison != "FT":
                            continue
                        if (family, comparison) in wilcoxon_map[metric]:
                            wilcoxon_map[metric][(family, comparison)] = value
                            logging.info("Aligned %s %s vs %s wilcoxon p-value with baseline results", metric, family, comparison)
            if not stats_df.empty:
                stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
                logging.info("Wrote learning-curve pairwise stats to %s", stats_path)
            if wilcoxon_map:
                # Remove FT-50 vs base for Precision metric
                if "precision" in wilcoxon_map:
                    wilcoxon_map["precision"] = {
                        k: v for k, v in wilcoxon_map["precision"].items()
                        if k[1] != "FT-50"
                    }
                # Hide comparisons where the target metric is lower than base
                if "precision" in wilcoxon_map:
                    base_val = None
                    if overall_qid is not None and not overall_qid.empty:
                        base_col = model_to_column.get("base")
                        if base_col:
                            base_row = overall_qid[overall_qid["model"] == base_col]
                            if not base_row.empty:
                                base_val = base_row["precision"].mean()
                    if base_val is not None:
                        filtered = {}
                        for key, pval in wilcoxon_map["precision"].items():
                            comp_val = _lookup_model_metric(overall_qid, key[1], model_to_column)
                            if comp_val is None or comp_val < base_val:
                                continue
                            filtered[key] = pval
                        wilcoxon_map["precision"] = filtered
                serialized: Dict[str, Dict[str, Dict[str, float]]] = {}
                for metric, mapping in wilcoxon_map.items():
                    fam_map: Dict[str, Dict[str, float]] = {}
                    for (family, comparison), value in mapping.items():
                        fam_map.setdefault(family, {})[comparison] = value
                    serialized[metric] = fam_map
                payload = {"comparisons": comparisons, "wilcoxon": serialized}
                with significance_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                logging.info("Saved learning-curve significance map to %s", significance_path)
        else:
            logging.warning("No QID metrics found for Partial Match; skipping pairwise tests.")

    print(f"Metrics written to {metrics_path}")
    print(f"Details written to {details_path}")
    print(f"Summary written to {summary_path}")
    if stats_path.exists():
        print(f"Pairwise stats written to {stats_path}")
    if significance_path.exists():
        print(f"Significance map written to {significance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

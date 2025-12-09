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
from eval import statistics as stat_utils  # type: ignore
from eval.plots import generate_figures  # type: ignore
from eval.scoring import build_detail_rows, ensure_norm, evaluate_group, evaluate_model, load_dataset  # type: ignore
from eval.normalize import match_scenario_label  # type: ignore
# Statistical helpers
from scipy.stats import fisher_exact, ttest_rel, wilcoxon
import numpy as np

from eval.normalize import slugify


LIST_SCENARIO_TITLES = {
    "Exact Match",
    "Partial Match",
}

FAMILY_COMPARISONS = {
    "GPT-4o": {
        "base": "GPT-4o base",
        "targets": ["GPT-4o FT", "GPT-4o QSP"],
    },
    "Llama3.1-70B": {
        "base": "Llama3.1-70B base",
        "targets": ["Llama3.1-70B FT", "Llama3.1-70B QSP"],
    },
    "Llama3.1-8B": {
        "base": "Llama3.1-8B base",
        "targets": ["Llama3.1-8B FT", "Llama3.1-8B QSP"],
    },
}


def build_qid_metrics(
    scenario_df: pd.DataFrame,
    scenario: dict,
    norm_lookup: dict[str, str],
) -> List[dict]:
    rows: List[dict] = []
    if scenario_df.empty:
        return rows
    allow_partial = scenario.get("allow_partial_list", False)
    ref_col = config.REF_COL
    ref_norm = norm_lookup.get(ref_col)
    if not ref_norm:
        return rows
    for qid, qid_df in scenario_df.groupby("QID"):
        q_type = qid_df.get("Type", pd.Series([""])).iloc[0]
        question = qid_df.get("Question", pd.Series([""])).iloc[0]
        for model in scenario["models"]:
            pred_norm = norm_lookup.get(model)
            if not pred_norm or model not in qid_df.columns:
                continue
            metrics = evaluate_model(
                qid_df,
                model,
                ref_col,
                pred_norm,
                ref_norm,
                allow_partial_list=allow_partial,
            )
            metrics.update(
                {
                    "model": model,
                    "scenario": scenario["title"],
                    "QID": qid,
                    "Type": q_type,
                    "Question": question,
                }
            )
            rows.append(metrics)
    return rows


def run(limit: int | None) -> Tuple[
    pd.DataFrame,
    List[dict],
    List[Tuple[str, str, pd.DataFrame]],
    List[dict],
    List[dict],
    List[dict],
]:
    df = load_dataset()
    if limit:
        df = df.head(limit)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenario_frames: List[pd.DataFrame] = []
    detail_rows: List[dict] = []
    figure_specs: List[Tuple[str, str, pd.DataFrame]] = []
    exact_partial_details: List[dict] = []
    partial_only_details: List[dict] = []
    qid_metrics_rows: List[dict] = []

    cache: dict = {}
    scenario_results: dict[str, pd.DataFrame] = {}
    scenario_qid_frames: dict[str, pd.DataFrame] = {}
    for scenario in config.SCENARIOS:
        scenario_df = df
        if filter_type := scenario.get("filter_type"):
            scenario_df = df[df["Type"] == filter_type].copy()
        if scenario_df.empty:
            continue

        missing_models = [model for model in scenario["models"] if model not in df.columns]
        if missing_models:
            logging.warning("Scenario '%s' missing models: %s", scenario["title"], ", ".join(missing_models))

        allow_partial = scenario.get("allow_partial_list", False)
        match_label = match_scenario_label(allow_partial)
        detail_type_filter = scenario.get("detail_types")
        detail_types = set(detail_type_filter) if detail_type_filter else None
        norm_lookup = {col: ensure_norm(df, col, cache) for col in [config.REF_COL, *scenario["models"]] if col in df.columns}
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
        scenario_qid = build_qid_metrics(
            scenario_df,
            scenario,
            norm_lookup,
        )
        qid_metrics_rows.extend(scenario_qid)
        scenario_qid_df = pd.DataFrame(scenario_qid)
        scenario_qid_frames[scenario["title"]] = scenario_qid_df
        if scenario["title"] in LIST_SCENARIO_TITLES:
            exact_partial_details.extend(build_detail_rows(scenario_df, scenario, norm_lookup, match_label))
            if scenario["title"] == "List questions - partial match":
                partial_only_details.extend(build_detail_rows(scenario_df, scenario, norm_lookup, match_label))
        scenario_frames.append(subset)
        def _format_title(title: str) -> str:
            if " - " in title:
                head, tail = title.split(" - ", 1)
                return f"{head} ({tail})"
            return title

        figure_specs.append((_format_title(scenario["title"]), scenario["title"], subset, scenario_qid_df))
    combined = pd.concat(scenario_frames, ignore_index=True) if scenario_frames else pd.DataFrame()
    return combined, detail_rows, figure_specs, exact_partial_details, partial_only_details, qid_metrics_rows, scenario_qid_frames


def _write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


def _dataset_label_from_suffix(suffix: str) -> str:
    normalized = suffix.strip("_").lower()
    if not normalized:
        return "Full 150"
    if "full150" in normalized:
        return "Full 150"
    if "new30" in normalized:
        return "New 30"
    if "original120" in normalized or "orig120" in normalized:
        return "Original 120"
    return normalized.replace("_", " ").title()


def _aggregate_fisher_summary(
    qid_df: pd.DataFrame,
    comparisons: dict,
    fisher_metrics: list[str],
) -> pd.DataFrame:
    """Compute aggregated Fisher exact tests per family/target across all QIDs."""
    if qid_df is None or qid_df.empty:
        return pd.DataFrame()

    def _sum_counts(model: str) -> tuple[int, int, int, int] | None:
        subset = qid_df[qid_df["model"] == model]
        if subset.empty or not {"tp", "fp", "tn", "fn"}.issubset(subset.columns):
            return None
        totals = subset[["tp", "fp", "tn", "fn"]].sum()
        return int(totals["tp"]), int(totals["fp"]), int(totals["tn"]), int(totals["fn"])

    rows: list[dict] = []
    for family, mapping in comparisons.items():
        base_model = mapping.get("base")
        if not base_model:
            continue
        base_counts = _sum_counts(base_model)
        if base_counts is None:
            continue
        for target_model in mapping.get("targets", []):
            target_counts = _sum_counts(target_model)
            if target_counts is None:
                continue
            base_tp, base_fp, base_tn, base_fn = base_counts
            target_tp, target_fp, target_tn, target_fn = target_counts
            for metric in fisher_metrics:
                if metric == "accuracy":
                    base_pos, base_neg = base_tp + base_tn, base_fp + base_fn
                    target_pos, target_neg = target_tp + target_tn, target_fp + target_fn
                elif metric == "precision":
                    base_pos, base_neg = base_tp, base_fp
                    target_pos, target_neg = target_tp, target_fp
                elif metric == "recall":
                    base_pos, base_neg = base_tp, base_fn
                    target_pos, target_neg = target_tp, target_fn
                elif metric == "f1":
                    base_pos, base_neg = 2 * base_tp, base_fp + base_fn
                    target_pos, target_neg = 2 * target_tp, target_fp + target_fn
                else:
                    continue
                table = np.array([[base_pos, base_neg], [target_pos, target_neg]])
                try:
                    _, p_value = fisher_exact(table)
                except ValueError:
                    p_value = np.nan
                rows.append(
                    {
                        "family": family,
                        "comparison": target_model.replace(f"{family} ", ""),
                        "base_model": base_model,
                        "target_model": target_model,
                        "metric": metric,
                        "p_value": float(p_value),
                    }
                )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    # Adjust p-values within each metric
    summary["adj_p_value"] = np.nan
    for metric in fisher_metrics:
        mask = summary["metric"] == metric
        metric_rows = summary.loc[mask]
        adjusted = stat_utils.benjamini_hochberg(metric_rows["p_value"].tolist())
        summary.loc[mask, "adj_p_value"] = adjusted
    return summary


def _build_fisher_qid_sheet(fisher: pd.DataFrame) -> pd.DataFrame:
    """Expand fisher results into a long per-QID dataframe."""
    if fisher is None or fisher.empty:
        return pd.DataFrame()
    records: list[dict] = []
    for _, row in fisher.iterrows():
        family = row.get("family")
        comparison = row.get("comparison")
        metric = row.get("metric")
        test_name = row.get("test")
        for col in fisher.columns:
            if not col.startswith("p_value_qid_"):
                continue
            qid = int(col.split("_")[-1])
            adj_col = f"adj_p_qid_{qid}"
            base_col = f"base_qid_{qid}"
            target_col = f"target_qid_{qid}"
            records.append(
                {
                    "family": family,
                    "comparison": comparison,
                    "metric": metric,
                    "test": test_name,
                    "QID": qid,
                    "base": row.get(base_col),
                    "target": row.get(target_col),
                    "p_value": row.get(col),
                    "adj_p": row.get(adj_col),
                }
            )
    return pd.DataFrame(records)


def _drop_qid_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove per-QID columns for compact summary output."""
    return df[
        [
            c
            for c in df.columns
            if not (
                c.startswith("base_qid_")
                or c.startswith("target_qid_")
                or c.startswith("p_value_qid_")
                or c.startswith("adj_p_qid_")
            )
        ]
    ].copy()


def _build_table3(
    fisher_qid_df: pd.DataFrame,
    qid_metrics_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Construct Table3: precision/recall with significance markers for select QIDs."""
    if fisher_qid_df is None or fisher_qid_df.empty or qid_metrics_df is None or qid_metrics_df.empty:
        return pd.DataFrame()

    qid_question = qid_metrics_df.groupby("QID")["Question"].first().to_dict()

    # Identify where any metric is significant (p<0.05 and target > base) per family/target/QID
    sig_map: dict[tuple[str, str, int], set[str]] = {}
    sig_any: set[tuple[str, int]] = set()
    sig_info: dict[tuple[str, str, str, int], tuple[float, float]] = {}
    for _, row in fisher_qid_df.iterrows():
        family = row.get("family")
        comparison = row.get("comparison")
        metric = row.get("metric")
        qid = row.get("QID")
        try:
            p_val = float(row.get("p_value"))
        except (TypeError, ValueError):
            continue
        try:
            base_val = float(row.get("base"))
            target_val = float(row.get("target"))
        except (TypeError, ValueError):
            continue
        if p_val < 0.05 and target_val > base_val:
            sig_map.setdefault((str(family), str(comparison), int(qid)), set()).add(str(metric))
            sig_any.add((str(family), int(qid)))
            sig_info[(str(family), str(comparison), str(metric), int(qid))] = (
                p_val,
                float(row.get("adj_p")) if row.get("adj_p") is not None else float("nan"),
            )

    def _metric_value(qid: int, model: str, metric: str) -> float | None:
        subset = qid_metrics_df[(qid_metrics_df["QID"] == qid) & (qid_metrics_df["model"] == model)]
        if subset.empty or metric not in subset:
            return None
        try:
            return float(subset[metric].iloc[0])
        except Exception:
            return None

    # Ordering of QIDs to keep the layout stable
    qid_order = {
        "GPT-4o": [1, 2, 6, 7, 9, 11, 12, 14, 15, 16],
        "Llama3.1-70B": [14, 15, 16, 11, 3, 6],
    }

    def _target_model(family: str, label: str) -> str | None:
        mapping = FAMILY_COMPARISONS.get(family, {})
        for target in mapping.get("targets", []):
            if label == "FT" and "FT" in target and "QSP" not in target:
                return target
            if label == "QSP" and "QSP" in target:
                return target
        return None

    records: list[dict] = []
    for family, order in qid_order.items():
        base_model = FAMILY_COMPARISONS.get(family, {}).get("base")
        ft_model = _target_model(family, "FT")
        qsp_model = _target_model(family, "QSP")
        if not base_model:
            continue
        for qid in order:
            if (family, qid) not in sig_any:
                continue
            row = {
                "Model": family,
                "QID": qid,
                "Question": qid_question.get(qid, ""),
                "base_prec": _metric_value(qid, base_model, "precision"),
                "FT_prec": "",
                "QSP_prec": "",
                "base_rec": _metric_value(qid, base_model, "recall"),
                "FT_rec": "",
                "QSP_rec": "",
            }
            # Evaluate FT/QSP with significance thresholds
            for target_label, model_label in [("FT", ft_model), ("QSP", qsp_model)]:
                if not model_label:
                    continue
                tgt_prec = _metric_value(qid, model_label, "precision")
                tgt_rec = _metric_value(qid, model_label, "recall")
                sig_metrics = sig_map.get((family, target_label, qid), set())
                if not sig_metrics and (family, qid) not in sig_any:
                    continue
                for metric, tgt_val, field in [
                    ("precision", tgt_prec, f"{target_label}_prec"),
                    ("recall", tgt_rec, f"{target_label}_rec"),
                ]:
                    if tgt_val is None:
                        continue
                    suffix = ""
                    if metric in sig_metrics:
                        p_val, adj_val = sig_info.get((family, target_label, metric, qid), (None, None))
                        if p_val is not None:
                            if p_val < 0.001:
                                suffix = "***"
                            elif p_val < 0.01:
                                suffix = "**"
                            elif p_val < 0.05:
                                suffix = "*"
                        if adj_val is not None and not np.isnan(adj_val) and adj_val > 0.05:
                            suffix = f"{suffix}\u2020" if suffix else "\u2020"
                    row[field] = f"{tgt_val * 100:.1f}{suffix}"
            # Format base values as percentages
            if row["base_prec"] is not None:
                row["base_prec"] = f"{row['base_prec'] * 100:.1f}"
            else:
                row["base_prec"] = ""
            if row["base_rec"] is not None:
                row["base_rec"] = f"{row['base_rec'] * 100:.1f}"
            else:
                row["base_rec"] = ""
            records.append(row)
    df = pd.DataFrame(records, columns=["Model", "QID", "Question", "base_prec", "FT_prec", "QSP_prec", "base_rec", "FT_rec", "QSP_rec"])
    if df.empty:
        return df
    model_order = {"GPT-4o": 0, "Llama3.1-70B": 1}
    df["__order"] = df["Model"].map(model_order).fillna(99)
    df.sort_values(["__order", "QID"], inplace=True)
    df.drop(columns="__order", inplace=True)
    return df


def write_outputs(
    metrics: pd.DataFrame,
    details: List[dict],
    extra_details: List[dict],
    partial_details: List[dict],
    scenario_qid_frames: dict[str, pd.DataFrame],
) -> None:
    # Metrics: write CSV only (avoid duplicate Excel outputs)
    metrics.to_csv(config.OUTPUT_METRICS, index=False, encoding="utf-8-sig")

    # Detail rows workbook
    detail_df = pd.DataFrame(details)
    detail_df.sort_values(["sort_key"], inplace=True)
    detail_df.drop(columns=["sort_key"], inplace=True, errors="ignore")
    detail_sheets = {title: detail_df[detail_df.get("Scenario") == title] for title in ["Exact Match", "Partial Match"]}
    _write_excel(config.DETAIL_METRICS_HUMAN.with_suffix(".xlsx"), detail_sheets)

    # Per-QID metrics workbook
    combined_qid = pd.concat(scenario_qid_frames.values(), ignore_index=True) if scenario_qid_frames else pd.DataFrame()
    if not combined_qid.empty:
        combined_qid.to_csv(config.OUTPUT_METRICS_BY_QID, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows.")
    parser.add_argument("--merged-path", type=Path, default=None, help="Override merged answers path.")
    parser.add_argument("--gpt5-path", type=Path, default=None, help="Override GPT-5 responses path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for metrics/results outputs.")
    parser.add_argument("--figures-dir", type=Path, default=None, help="Directory for figure outputs.")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix appended to output filenames (e.g., new30).")
    args = parser.parse_args()

    suffix = args.output_suffix.strip()
    suffix = f"_{suffix}" if suffix else ""
    if args.merged_path:
        config.MERGED_PATH = args.merged_path
    if args.gpt5_path:
        config.GPT5_PATH = args.gpt5_path
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    config.OUTPUT_METRICS = config.OUTPUT_DIR / f"evaluation_metrics{suffix}.csv"
    config.OUTPUT_METRICS_BY_QID = config.OUTPUT_DIR / f"evaluation_metrics_by_qid{suffix}.csv"
    config.STAT_RESULTS = config.OUTPUT_DIR / f"statistical_tests{suffix}.xlsx"
    config.DETAIL_METRICS_HUMAN = config.OUTPUT_DIR / f"detailed_evaluation{suffix}.csv"
    config.DETAIL_METRICS_PARTIAL = config.OUTPUT_DIR / f"detailed_evaluation_partial_list_matches{suffix}.csv"
    config.EXACT_VS_PARTIAL_DETAILS = config.OUTPUT_DIR / f"exact_vs_partial_evaluation{suffix}.csv"
    if args.figures_dir:
        config.OUTPUT_TABLE_DIR = args.figures_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    (
        metrics,
        details,
        figures,
        extra_details,
        partial_details,
        qid_rows,
        scenario_qid_frames,
    ) = run(args.limit)
    if metrics.empty:
        logging.error("No scenarios produced metrics.")
        return 1
    write_outputs(metrics, details, extra_details, partial_details, scenario_qid_frames)
    overall_stats = {}
    fisher_df = pd.DataFrame()
    pair_df = pd.DataFrame()
    overall_qid_df = scenario_qid_frames.get("Partial Match")
    exact_qid_df = scenario_qid_frames.get("Exact Match")
    fisher_metrics = ["accuracy", "precision", "recall", "f1"]
    if overall_qid_df is not None and not overall_qid_df.empty:
        paired_metrics = ["accuracy", "precision", "recall", "f1"]
        fisher_df = stat_utils.compute_fisher_tests(
            overall_qid_df,
            FAMILY_COMPARISONS,
            fisher_metrics,
        )
        pair_df, overall_stats, _ = stat_utils.compute_pairwise_tests(
            overall_qid_df,
            FAMILY_COMPARISONS,
            paired_metrics,
        )
    exact_stats = {}
    if exact_qid_df is not None and not exact_qid_df.empty:
        _, exact_stats, _ = stat_utils.compute_pairwise_tests(
            exact_qid_df,
            FAMILY_COMPARISONS,
            ["accuracy", "precision", "recall", "f1"],
        )
    logging.info("Wrote metrics to %s", config.OUTPUT_METRICS)
    logging.info("Wrote detail rows to %s", config.DETAIL_METRICS_HUMAN)
    if extra_details:
        logging.info("Wrote list scenario details to %s", config.EXACT_VS_PARTIAL_DETAILS)
    if partial_details:
        logging.info("Wrote partial-list detail rows to %s", config.DETAIL_METRICS_PARTIAL)
    if not pair_df.empty or not fisher_df.empty:
        fisher_qid_sheet = _build_fisher_qid_sheet(fisher_df)
        table3_df = _build_table3(fisher_qid_sheet, overall_qid_df)
        with pd.ExcelWriter(config.STAT_RESULTS, engine="openpyxl") as writer:
            if not pair_df.empty:
                _drop_qid_cols(pair_df).to_excel(writer, sheet_name="Paired Tests", index=False)
            if not fisher_df.empty:
                qid_sheet = fisher_qid_sheet
                if not qid_sheet.empty:
                    qid_sheet.to_excel(writer, sheet_name="Fisher Exact Test", index=False)
            if table3_df is not None and not table3_df.empty:
                table3_df.to_excel(writer, sheet_name="Table3", index=False)
        logging.info("Wrote combined statistical tests to %s", config.STAT_RESULTS)
    # Export Partial Match metrics with aggregated Fisher results
    fisher_summary = _aggregate_fisher_summary(
        overall_qid_df if overall_qid_df is not None else pd.DataFrame(),
        FAMILY_COMPARISONS,
        fisher_metrics,
    )
    partial_metrics = metrics[metrics["scenario"] == "Partial Match"].copy()
    fisher_col_map = {
        "accuracy": ("p_acc_fisher", "adj_p_acc_fisher"),
        "precision": ("p_prec_fisher", "adj_p_prec_fisher"),
        "recall": ("p_rec_fisher", "adj_p_rec_fisher"),
        "f1": ("p_f1_fisher", "adj_p_f1_fisher"),
    }
    desired_order = [
        "scenario",
        "samples",
        "model",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "p_acc_fisher",
        "adj_p_acc_fisher",
        "precision",
        "p_prec_fisher",
        "adj_p_prec_fisher",
        "recall",
        "p_rec_fisher",
        "adj_p_rec_fisher",
        "f1",
        "p_f1_fisher",
        "adj_p_f1_fisher",
    ]
    if not partial_metrics.empty:
        for metric_name, (p_col, adj_col) in fisher_col_map.items():
            partial_metrics[p_col] = np.nan
            partial_metrics[adj_col] = np.nan
    if not partial_metrics.empty and not fisher_summary.empty:
        for _, row in fisher_summary.iterrows():
            target_model = row["target_model"]
            metric_name = row["metric"]
            p_val = row.get("p_value")
            adj_val = row.get("adj_p_value")
            mask = partial_metrics["model"] == target_model
            if mask.any():
                p_col, adj_col = fisher_col_map.get(metric_name, (None, None))
                if p_col:
                    partial_metrics.loc[mask, p_col] = p_val
                if adj_col:
                    partial_metrics.loc[mask, adj_col] = adj_val
    if not partial_metrics.empty:
        for col in desired_order:
            if col not in partial_metrics.columns:
                partial_metrics[col] = np.nan
        partial_metrics = partial_metrics.reindex(columns=desired_order)
        fisher_metrics_path = config.OUTPUT_DIR / f"evaluation_metrics_fisher{suffix}.xlsx"
        partial_metrics.to_excel(fisher_metrics_path, index=False)
        logging.info("Wrote Partial Match metrics with Fisher p-values to %s", fisher_metrics_path)
    dataset_label = _dataset_label_from_suffix(args.output_suffix)
    for display_title, scenario_title, subset, scenario_qid_df in figures:
        sig = overall_stats if scenario_title == "Partial Match" else exact_stats if scenario_title == "Exact Match" else None
        base_name = slugify(scenario_title)
        if suffix:
            base_name = f"{base_name}{suffix}"
        full_title = f"{scenario_title} ({dataset_label})"
        generate_figures(
            subset,
            scenario_title,
            config.OUTPUT_TABLE_DIR,
            significance=sig,
            comparisons=FAMILY_COMPARISONS,
            base_name=base_name,
            display_title=full_title,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

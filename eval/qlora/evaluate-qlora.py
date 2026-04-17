#!/usr/bin/env python3
"""Evaluate the QLoRA comparison models and save summary metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

QLORA_DIR = Path(__file__).resolve().parent
ROOT = QLORA_DIR.parents[1]
DEFAULT_MERGED_PATH = ROOT / "advanced-prompting/csv/merged_answers_full_150.xlsx"
TARGET_MODELS = [
    "Llama3.1-70B FT",
    "Llama3.1-70B R16",
    "Llama3.1-70B R32",
    "Llama3.1-8B FT",
    "Llama3.1-8B R16",
    "Llama3.1-8B R32",
]
OUTPUT_CSV = QLORA_DIR / "evaluation_metrics.csv"
OUTPUT_XLSX = QLORA_DIR / "evaluation_metrics.xlsx"
QID_CSV = QLORA_DIR / "evaluation_metrics_by_qid_full150.csv"
QID_XLSX = QLORA_DIR / "evaluation_metrics_by_qid_full150.xlsx"
STATS_XLSX = QLORA_DIR / "statistical_tests_full150.xlsx"
CORRECT_CSV = QLORA_DIR / "merged_answers_with_correct.csv"
CORRECT_XLSX = QLORA_DIR / "merged_answers_with_correct.xlsx"
BAR_CHART = QLORA_DIR / "full150-bar-chart.png"
LOCAL_COLUMN_RENAMES = {
    "llama3.1-70B R16": "Llama3.1-70B R16",
    "llama3.1-70B R32": "Llama3.1-70B R32",
    "llama3.1-8B R16": "Llama3.1-8B R16",
    "llama3.1-8B R32": "Llama3.1-8B R32",
}

for path in (ROOT, ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval import config  # type: ignore
from eval.evaluation import build_qid_metrics  # type: ignore
from eval.plots import plot_metric_panels  # type: ignore
from eval import statistics as stat_utils  # type: ignore
from eval.scoring import build_detail_rows, ensure_norm, evaluate_group, format_identifier  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-path",
        type=Path,
        default=DEFAULT_MERGED_PATH,
        help="Merged answers file to evaluate.",
    )
    parser.add_argument(
        "--wilcoxon-only",
        action="store_true",
        default=True,
        help="Only keep Wilcoxon rows in the statistical tests workbook.",
    )
    parser.add_argument(
        "--include-t-test",
        dest="wilcoxon_only",
        action="store_false",
        help="Include both t-test and Wilcoxon rows in the statistical tests workbook.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Merged answers file not found: {path}")

    df = pd.read_excel(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    if getattr(config, "COLUMN_RENAMES", None):
        df.rename(columns=config.COLUMN_RENAMES, inplace=True)
    df.rename(columns=LOCAL_COLUMN_RENAMES, inplace=True)

    required = {"PMID", "QID", "Question", "Type", config.REF_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Merged answers file is missing required columns: {', '.join(sorted(missing))}")

    df["PMID"] = df["PMID"].apply(format_identifier)
    df["QID"] = df["QID"].apply(format_identifier).astype(int)
    return df


def evaluate_models(df: pd.DataFrame) -> pd.DataFrame:
    available_models = [model for model in TARGET_MODELS if model in df.columns]
    if not available_models:
        raise SystemExit(f"None of the target model columns were found: {', '.join(TARGET_MODELS)}")

    cache: dict[str, str] = {}
    norm_lookup = {
        config.REF_COL: ensure_norm(df, config.REF_COL, cache),
    }
    for model in available_models:
        norm_lookup[model] = ensure_norm(df, model, cache)

    metrics_df = evaluate_group(
        df,
        available_models,
        config.REF_COL,
        norm_lookup,
        allow_partial_list=True,
    )
    if metrics_df.empty:
        raise SystemExit("No metrics were produced.")

    ordered_columns = [
        "model",
        "samples",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    metrics_df = metrics_df[ordered_columns].copy()
    metrics_df["display_order"] = metrics_df["model"].map({model: idx for idx, model in enumerate(TARGET_MODELS)})
    return metrics_df.sort_values("display_order").reset_index(drop=True)


def add_correct_columns(df: pd.DataFrame) -> pd.DataFrame:
    available_models = [model for model in TARGET_MODELS if model in df.columns]
    if not available_models:
        raise SystemExit(f"None of the target model columns were found: {', '.join(TARGET_MODELS)}")

    cache: dict[str, str] = {}
    norm_lookup = {
        config.REF_COL: ensure_norm(df, config.REF_COL, cache),
    }
    for model in available_models:
        norm_lookup[model] = ensure_norm(df, model, cache)

    detail_rows = build_detail_rows(
        df,
        {"models": available_models, "allow_partial_list": True},
        norm_lookup,
    )
    details_df = pd.DataFrame(detail_rows)
    if details_df.empty:
        return df

    keep_columns = ["PMID", "QID", *[f"{model} Correct" for model in available_models]]
    details_df = details_df[keep_columns]
    merged_df = df.merge(details_df, on=["PMID", "QID"], how="left")
    norm_columns = [column for column in merged_df.columns if column.endswith("__norm")]
    if norm_columns:
        merged_df = merged_df.drop(columns=norm_columns)
    return merged_df


def evaluate_models_by_qid(df: pd.DataFrame) -> pd.DataFrame:
    available_models = [model for model in TARGET_MODELS if model in df.columns]
    if not available_models:
        raise SystemExit(f"None of the target model columns were found: {', '.join(TARGET_MODELS)}")

    cache: dict[str, str] = {}
    norm_lookup = {
        config.REF_COL: ensure_norm(df, config.REF_COL, cache),
    }
    for model in available_models:
        norm_lookup[model] = ensure_norm(df, model, cache)

    qid_rows = build_qid_metrics(
        df,
        {"models": available_models, "allow_partial_list": True},
        norm_lookup,
    )
    qid_df = pd.DataFrame(qid_rows)
    if qid_df.empty:
        raise SystemExit("No per-QID metrics were produced.")

    ordered_columns = [
        "samples",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
        "model",
        "QID",
        "Type",
        "Question",
    ]
    qid_df = qid_df[ordered_columns].copy()
    qid_df["display_order"] = qid_df["model"].map({model: idx for idx, model in enumerate(TARGET_MODELS)})
    return qid_df.sort_values(["QID", "display_order"]).drop(columns="display_order").reset_index(drop=True)


def build_statistical_tests(qid_df: pd.DataFrame, wilcoxon_only: bool = False) -> pd.DataFrame:
    comparisons = {
        "Llama3.1-70B": {
            "base": "Llama3.1-70B FT",
            "targets": ["Llama3.1-70B R16", "Llama3.1-70B R32"],
        },
        "Llama3.1-8B": {
            "base": "Llama3.1-8B FT",
            "targets": ["Llama3.1-8B R16", "Llama3.1-8B R32"],
        },
    }
    stats_df, _, _ = stat_utils.compute_pairwise_tests(
        qid_df,
        comparisons,
        ["accuracy", "precision", "recall", "f1"],
    )
    if stats_df.empty:
        raise SystemExit("No statistical tests were produced.")
    if wilcoxon_only:
        stats_df = stats_df[stats_df["test"] == "wilcoxon"].copy()
        if stats_df.empty:
            raise SystemExit("No Wilcoxon statistical tests were produced.")

    keep_columns = [
        column
        for column in stats_df.columns
        if not (
            column.startswith("base_qid_")
            or column.startswith("target_qid_")
            or column.startswith("p_value_qid_")
            or column.startswith("adj_p_qid_")
        )
    ]
    return stats_df[keep_columns].reset_index(drop=True)


def main() -> int:
    args = parse_args()
    df = load_dataset(args.merged_path)
    metrics_df = evaluate_models(df)
    qid_df = evaluate_models_by_qid(df)
    stats_df = build_statistical_tests(qid_df, wilcoxon_only=args.wilcoxon_only)
    correct_df = add_correct_columns(df)
    metrics_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    metrics_df.to_excel(OUTPUT_XLSX, index=False)
    qid_df.to_csv(QID_CSV, index=False, encoding="utf-8-sig")
    qid_df.to_excel(QID_XLSX, index=False)
    stats_df.to_excel(STATS_XLSX, index=False)
    correct_df.to_csv(CORRECT_CSV, index=False, encoding="utf-8-sig")
    correct_df.to_excel(CORRECT_XLSX, index=False)
    plot_metric_panels(
        metrics_df,
        qid_df=None,
        title="Full 150",
        output_path=BAR_CHART,
        significance=None,
        comparisons=None,
        layout="vertical",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare exact vs partial list scoring on a per-question basis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from eval import config
from eval.normalize import match_scenario_label
from eval.scoring import ensure_norm, evaluate_model, load_dataset


def _scenario_metrics(
    list_df: pd.DataFrame,
    models: Iterable[str],
    norm_lookup: dict[str, str],
    allow_partial: bool,
) -> pd.DataFrame:
    """Compute metrics for each QID/model under a given matching rule."""
    rows: list[dict] = []
    ref_norm = norm_lookup.get(config.REF_COL)
    if not ref_norm or list_df.empty:
        return pd.DataFrame()

    for qid, qid_df in list_df.groupby("QID"):
        question = qid_df.get("Question", pd.Series([""])).iloc[0]
        q_type = qid_df.get("Type", pd.Series([""])).iloc[0]
        for model in models:
            pred_norm = norm_lookup.get(model)
            if not pred_norm or model not in qid_df.columns:
                continue
            metrics = evaluate_model(
                qid_df,
                model,
                config.REF_COL,
                pred_norm,
                ref_norm,
                allow_partial_list=allow_partial,
            )
            rows.append(
                {
                    "scenario": match_scenario_label(allow_partial),
                    "QID": int(qid),
                    "Question": question,
                    "Type": q_type,
                    "model": model,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _rename_metrics(df: pd.DataFrame, prefix: str, metric_cols: list[str]) -> pd.DataFrame:
    rename_map = {col: f"{prefix}_{col}" for col in metric_cols}
    renamed = df.rename(columns=rename_map).drop(columns=["scenario"], errors="ignore")
    for col in metric_cols:
        target = f"{prefix}_{col}"
        if target not in renamed.columns:
            renamed[target] = pd.NA
    return renamed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-path", type=Path, default=None, help="Override merged answers path.")
    parser.add_argument("--gpt5-path", type=Path, default=None, help="Override GPT-5 responses path.")
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix for the output filename (e.g., full150).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.EVAL_DIR / "exact_vs_partial",
        help="Directory for the comparison CSV.",
    )
    args = parser.parse_args()

    if args.merged_path:
        config.MERGED_PATH = args.merged_path
    if args.gpt5_path:
        config.GPT5_PATH = args.gpt5_path

    df = load_dataset()
    models = [model for group in config.MODEL_GROUPS.values() for model in group]

    # Pre-compute normalized columns once per model/reference (in-place on df).
    cache: dict[str, str] = {}
    norm_lookup = {}
    for column in [config.REF_COL, *models]:
        if column in df.columns:
            norm_lookup[column] = ensure_norm(df, column, cache)

    list_df = df[df["Type"].str.lower() == "list"].copy()

    metric_cols = ["samples", "accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
    exact_df = _scenario_metrics(list_df, models, norm_lookup, allow_partial=False)
    partial_df = _scenario_metrics(list_df, models, norm_lookup, allow_partial=True)

    exact_wide = _rename_metrics(exact_df, "exact", metric_cols)
    partial_wide = _rename_metrics(partial_df, "partial", metric_cols)

    merged = partial_wide.merge(
        exact_wide,
        on=["model", "QID", "Question", "Type"],
        how="outer",
    )

    for metric in ["accuracy", "precision", "recall", "f1"]:
        merged[f"{metric}_delta"] = merged[f"partial_{metric}"] - merged[f"exact_{metric}"]

    merged.sort_values(["QID", "model"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.output_suffix.strip()
    suffix = f"_{suffix}" if suffix else ""
    output_path = output_dir / f"list_metrics_by_scenario{suffix}.csv"
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote comparison metrics to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import math
import logging
from typing import Dict, Iterable, List, Set

import pandas as pd

from . import config
from .normalize import (
    canonicalize_answer,
    human_answer_counts,
)


def format_identifier(value: str | int | float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def load_dataset() -> pd.DataFrame:
    merged = pd.read_excel(
        config.MERGED_PATH,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    if getattr(config, "COLUMN_RENAMES", None):
        merged.rename(columns=config.COLUMN_RENAMES, inplace=True)
    merged["PMID"] = merged["PMID"].apply(format_identifier)
    merged["QID"] = merged["QID"].apply(format_identifier)
    # Convert QID to int for proper numeric sorting (critical fix for alignment)
    merged["QID"] = merged["QID"].astype(int)

    gpt5 = pd.read_csv(
        config.GPT5_PATH,
        dtype={"PMID": str},
        keep_default_na=False,
        na_filter=False,
    ).rename(columns={"Answer": "GPT-5 base"})
    gpt5["PMID"] = gpt5["PMID"].apply(format_identifier)
    gpt5["QID"] = gpt5["QID"].apply(format_identifier)
    gpt5["QID"] = gpt5["QID"].astype(int)

    df = merged.merge(gpt5[["PMID", "QID", "GPT-5 base"]], on=["PMID", "QID"], how="left")

    for column, path in getattr(config, "LEARNING_CURVE_RESPONSES", {}).items():
        if not path.exists():
            logging.warning("Learning-curve response file missing for %s: %s", column, path)
            continue
        extra = pd.read_csv(
            path,
            dtype={"PMID": str},
            keep_default_na=False,
            na_filter=False,
        )
        if "Answer" not in extra.columns:
            logging.warning("Response file %s missing 'Answer' column.", path)
            continue
        extra["PMID"] = extra["PMID"].apply(format_identifier)
        extra["QID"] = extra["QID"].apply(format_identifier)
        extra["QID"] = extra["QID"].astype(int)
        extra = extra.rename(columns={"Answer": column})
        df = df.merge(extra[["PMID", "QID", column]], on=["PMID", "QID"], how="left")

    for column in getattr(config, "ALL_MODEL_COLUMNS", []):
        if column not in df.columns:
            logging.warning("Column '%s' missing from merged answers; filling with blanks.", column)
            df[column] = ""
    df = df[(df["PMID"] != "") & (df["QID"] != "")]
    # Ensure QID is int for proper sorting
    df["QID"] = df["QID"].astype(int)
    df["sort_key"] = range(len(df))
    df["sample_id"] = df["PMID"] + "-" + df["QID"].astype(str)
    return df


def ensure_norm(df: pd.DataFrame, column: str, cache: dict) -> str:
    key = column
    if key in cache:
        return cache[key]
    norm_col = f"{column}__norm"
    if norm_col not in df.columns:
        df[norm_col] = df[column].apply(canonicalize_answer)
    cache[key] = norm_col
    return norm_col


def evaluate_model(
    data: pd.DataFrame,
    model_col: str,
    ref_col: str,
    pred_norm_col: str,
    ref_norm_col: str,
    allow_partial_list: bool = False,
) -> Dict[str, float]:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for _, row in data.iterrows():
        row_allow_partial = bool(allow_partial_list)
        row_counts, _ = human_answer_counts(
            row.get("Type", ""),
            row.get(pred_norm_col, ""),
            row.get(ref_norm_col, ""),
            question_text=row.get("Question", ""),
            ref_raw=row.get(ref_col, ""),
            pred_raw=row.get(model_col, ""),
            allow_partial_list=row_allow_partial,
        )
        for key, value in row_counts.items():
            counts[key] += value
    total = sum(counts.values())
    if not total:
        return {"samples": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, **counts}
    accuracy = (counts["tp"] + counts["tn"]) / total
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"samples": total, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, **counts}


def evaluate_group(
    df: pd.DataFrame,
    models: Iterable[str],
    ref_col: str,
    norm_lookup: dict[str, str],
    allow_partial_list: bool = False,
) -> pd.DataFrame:
    rows = []
    ref_norm = norm_lookup[ref_col]
    for model in models:
        if model not in df.columns:
            continue
        pred_norm = norm_lookup.get(model)
        if not pred_norm:
            continue
        metrics = evaluate_model(df, model, ref_col, pred_norm, ref_norm, allow_partial_list)
        metrics.update({"model": model})
        rows.append(metrics)
    return pd.DataFrame(rows)


def build_detail_rows(
    df: pd.DataFrame,
    scenario: dict,
    norm_lookup: dict[str, str],
    detail_types: Set[str] | None = None,
) -> List[dict]:
    records = []
    ref_col = config.REF_COL
    ref_norm = norm_lookup[ref_col]
    allowed_types = {value.lower() for value in detail_types} if detail_types else None
    for _, row in df.iterrows():
        question_type = str(row.get("Type", ""))
        if allowed_types and question_type.lower() not in allowed_types:
            continue
        base = {
            "PMID": row["PMID"],
            "QID": row["QID"],
            "Question": row.get("Question", ""),
            "Type": row.get("Type", ""),
            "Human Answer": row.get("Human Answer", ""),
            "sort_key": row.get("sort_key", 0),
        }
        for model in scenario["models"]:
            answer = row.get(model, "")
            base[f"{model} Answer"] = answer
            pred_norm = row.get(norm_lookup.get(model, ""), "")
            ref_norm_value = row.get(ref_norm, "")
            row_allow_partial = scenario.get("allow_partial_list", False) and question_type.strip().lower() == "list"
            _, correct = human_answer_counts(
                row.get("Type", ""),
                pred_norm,
                ref_norm_value,
                question_text=row.get("Question", ""),
                ref_raw=row.get(ref_col, ""),
                pred_raw=answer,
                allow_partial_list=row_allow_partial,
            )
            base[f"{model} Correct"] = int(correct)
        records.append(base)
    return records

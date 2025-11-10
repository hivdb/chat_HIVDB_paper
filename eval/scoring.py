from __future__ import annotations

import math
from typing import Dict, Iterable, List

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
    merged = pd.read_excel(config.MERGED_PATH)
    merged["PMID"] = merged["PMID"].apply(format_identifier)
    merged["QID"] = merged["QID"].apply(format_identifier)

    gpt5 = pd.read_csv(config.GPT5_PATH, dtype={"PMID": str}).rename(columns={"Answer": "gpt5-mini"})
    gpt5["PMID"] = gpt5["PMID"].apply(format_identifier)
    gpt5["QID"] = gpt5["QID"].apply(format_identifier)

    df = merged.merge(gpt5[["PMID", "QID", "gpt5-mini"]], on=["PMID", "QID"], how="left")
    df = df[(df["PMID"] != "") & (df["QID"] != "")]
    df["sort_key"] = range(len(df))
    df["sample_id"] = df["PMID"] + "-" + df["QID"]
    return df


def ensure_norm(df: pd.DataFrame, column: str, convert: bool, cache: dict) -> str:
    key = (column, convert)
    if key in cache:
        return cache[key]
    norm_col = f"{column}__norm__{'cno' if convert else 'raw'}"
    if norm_col not in df.columns:
        df[norm_col] = df[column].apply(lambda value: canonicalize_answer(value, convert_special_no=convert))
    cache[key] = norm_col
    return norm_col


def evaluate_model(
    data: pd.DataFrame,
    model_col: str,
    ref_col: str,
    pred_norm_col: str,
    ref_norm_col: str,
) -> Dict[str, float]:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for _, row in data.iterrows():
        row_counts, _ = human_answer_counts(
            row.get("Type", ""),
            row.get(pred_norm_col, ""),
            row.get(ref_norm_col, ""),
            question_text=row.get("Question", ""),
            ref_raw=row.get(ref_col, ""),
            pred_raw=row.get(model_col, ""),
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
    scenario: str,
    norm_lookup: dict[str, str],
    convert_special_no: bool,
) -> pd.DataFrame:
    rows = []
    ref_norm = norm_lookup[ref_col]
    for model in models:
        if model not in df.columns:
            continue
        pred_norm = norm_lookup.get(model)
        if not pred_norm:
            continue
        metrics = evaluate_model(df, model, ref_col, pred_norm, ref_norm)
        metrics.update({"model": model, "scenario": scenario, "reference": ref_col, "convert_no": convert_special_no})
        rows.append(metrics)
    return pd.DataFrame(rows)


def build_detail_rows(df: pd.DataFrame, scenario: dict, norm_lookup: dict[str, str]) -> List[dict]:
    records = []
    ref_col = scenario["reference"]
    ref_norm = norm_lookup[ref_col]
    for _, row in df.iterrows():
        base = {
            "Scenario": scenario["title"],
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
            _, correct = human_answer_counts(
                row.get("Type", ""),
                pred_norm,
                ref_norm_value,
                question_text=row.get("Question", ""),
                ref_raw=row.get(ref_col, ""),
                pred_raw=answer,
            )
            base[f"{model} Correct"] = int(correct)
        records.append(base)
    return records

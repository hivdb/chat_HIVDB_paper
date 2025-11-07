#!/usr/bin/env python3
"""Evaluate multiple model outputs against Human Answers with lenient matching."""

from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:  # pragma: no cover
    plt = None
    HAS_MPL = False
import pandas as pd


MERGED_PATH = Path("advanced-prompting/merged_answers.xlsx")
# UPDATED_PATH = Path("eval/updated_human_answers.csv")
GPT5_PATH = Path("eval/gpt5_responses.csv")
JUDGE_RESULTS_PATH = Path("eval/judge_gpt5_results.csv")
OUTPUT_METRICS = Path("eval/evaluation_metrics.csv")
DETAIL_METRICS_HUMAN = Path("eval/detailed_evaluation.csv")
OUTPUT_TABLE_DIR = Path("eval/figures")

SPECIAL_NO = {
    "none",
    "not reported",
    "not applicable",
    "not available",
    "not provided",
    "na",
    "n/a",
    "not stated",
    "nr",
    "no data",
    "0",
}
YES_SYNONYMS = {"yes", "y", "true", "reported", "present"}
NO_SYNONYMS = {"no", "false", "not", "absent"}

ARV_SYNONYMS = {
    "tfv": "tenofovir",
    "tdf": "tenofovir",
    "tenofovir disoproxil fumarate": "tenofovir",
    "taf": "tenofovir alafenamide",
    "ftc": "emtricitabine",
    "3tc": "lamivudine",
    "lamivudine (3tc)": "lamivudine",
    "azt": "zidovudine",
    "abc": "abacavir",
    "efv": "efavirenz",
    "efavirenz (efv)": "efavirenz",
    "nvp": "nevirapine",
    "dtg": "dolutegravir",
    "ral": "raltegravir",
    "bik": "bictegravir",
    "bic": "bictegravir",
    "lpv": "lopinavir",
    "rtv": "ritonavir",
    "etv": "etravirine",
    "etr": "etravirine",
    "rpv": "rilpivirine",
    "rilpivirine (rpv)": "rilpivirine",
    "cab": "cabotegravir",
    "cabotegravir (cab)": "cabotegravir",
    "d4t": "stavudine",
    "mvc": "maraviroc",
    "evg": "elvitegravir",
    "elvitegravir (evg)": "elvitegravir",
    "efv, nvp": "efavirenz | nevirapine",
}

LIST_DELIM = re.compile(r",|;|/|\band\b|\bor\b")
NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
LEADING_YES_NO = re.compile(r"^(yes|no)\b")
def detect_terms(text: str, synonyms: dict[str, str], min_len: int = 3) -> List[str]:
    lowered = text.lower()
    found: set[str] = set()
    for key, canonical in synonyms.items():
        if len(key) < min_len:
            continue
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            found.add(canonical)
    return sorted(found)
GENE_SYNONYMS = {
    "integrase": "in",
    "in": "in",
    "reverse transcriptase": "rt",
    "rt": "rt",
    "protease": "pr",
    "pr": "pr",
    "capsid": "ca",
    "ca": "ca",
    "full genome": "full genome",
    "near full length genome": "nflg",
    "nflg": "nflg",
}

MODEL_GROUPS = {
    "gpt4_family": ["GPT-4o base", "GPT-4o FT", "GPT-4o AP", "gpt5-mini"],
    "llama_70b": ["Llama3.1-70B base", "Llama3.1-70B FT", "llama-3.1-70B AP"],
}

ADDITIONAL_AP_MODELS = ["GPT-4o AP", "llama-3.1-70B AP", "gpt5-mini"]

SCENARIOS = [
    {
        "title": "Human Answer",
        "reference": "Human Answer",
        "models": MODEL_GROUPS["gpt4_family"] + MODEL_GROUPS["llama_70b"],
        "convert_special_no": True,
        "footnote": (
            "*AI Answers compared against Human Answers after "
            "normalizing case, stripping punctuation, sorting list-valued entries, "
            "and collapsing None/Not reported/Not applicable/0 into 'No'."
        ),
    },
    {
        "title": "Human Answer – Yes/No questions",
        "reference": "Human Answer",
        "models": MODEL_GROUPS["gpt4_family"] + MODEL_GROUPS["llama_70b"],
        "convert_special_no": True,
        "footnote": (
            "*AI Answers compared against Human Answers after "
            "collapsing None/Not reported/Not applicable/0 into 'No'."
        ),
        "filter_type": "Boolean",
    },
]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def configure_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("evaluation")


def token_synonym(token: str) -> str:
    token = token.lower().strip()
    if token in ARV_SYNONYMS:
        token = ARV_SYNONYMS[token]
    if token in GENE_SYNONYMS:
        token = GENE_SYNONYMS[token]
    return token


def normalize_list(text: str) -> str:
    parts = [token_synonym(part.strip()) for part in LIST_DELIM.split(text) if part.strip()]
    if len(parts) < 2:
        return text
    cleaned = [" ".join(part.split()) for part in parts]
    cleaned = [NON_ALPHANUM.sub(" ", part).strip() for part in cleaned]
    cleaned = [part for part in cleaned if part]
    if not cleaned:
        return text
    cleaned = sorted(set(cleaned))
    return " | ".join(cleaned)


def canonicalize_answer(
    text: str | float | None,
    *,
    convert_special_no: bool,
) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    raw = str(text).strip()
    if not raw:
        return ""

    lowered = raw.lower().strip()
    if lowered.startswith("answer:"):
        lowered = lowered.split("answer:", 1)[1].strip()

    leading = LEADING_YES_NO.match(lowered)
    if leading:
        return leading.group(1)

    if lowered in YES_SYNONYMS:
        return "yes"
    if lowered in NO_SYNONYMS:
        return "no"
    if convert_special_no:
        if lowered in SPECIAL_NO:
            return "no"
        for special in SPECIAL_NO:
            if lowered.startswith(f"{special} "):
                return "no"

    if lowered.isdigit():
        if convert_special_no and lowered == "0":
            return "no"
        return str(int(lowered))
    if any(char.isdigit() for char in lowered) and " or " in lowered:
        numbers = []
        for token in re.split(r"[^\d]+", lowered):
            if token.isdigit():
                numbers.append(token)
        if numbers:
            return " or ".join(sorted(numbers))

    normalized = normalize_list(lowered)
    if "|" in normalized:
        sanitized = " | ".join(token.strip() for token in normalized.split("|"))
    else:
        normalized = NON_ALPHANUM.sub(" ", normalized).strip()
        sanitized = " ".join(normalized.split())

    if convert_special_no and sanitized in SPECIAL_NO:
        return "no"
    if sanitized in YES_SYNONYMS:
        return "yes"
    if sanitized in NO_SYNONYMS:
        return "no"
    arv_terms = detect_terms(raw, ARV_SYNONYMS)
    if arv_terms:
        return " | ".join(sorted(arv_terms))
    gene_terms = detect_terms(sanitized, GENE_SYNONYMS, min_len=3)
    if gene_terms:
        return " | ".join(sorted(gene_terms))
    return sanitized


def load_data() -> pd.DataFrame:
    merged = pd.read_excel(MERGED_PATH)
    merged["PMID"] = merged["PMID"].astype(str)

    gpt5 = pd.read_csv(GPT5_PATH, dtype={"PMID": str})
    gpt5 = gpt5.rename(columns={"Answer": "gpt5-mini"})

    df = merged.merge(gpt5[["PMID", "QID", "gpt5-mini"]], on=["PMID", "QID"], how="left")
    if "Updated Human Answer" not in df.columns:
        df["Updated Human Answer"] = ""
    df["sample_id"] = df["PMID"].astype(str) + "-" + df["QID"].astype(str)
    return df


def load_judge_results() -> pd.DataFrame | None:
    if not JUDGE_RESULTS_PATH.exists():
        return None
    df = pd.read_csv(JUDGE_RESULTS_PATH, dtype={"PMID": str})
    df["sample_id"] = df["PMID"].astype(str) + "-" + df["QID"].astype(str)
    return df


def judge_lookup_map(judge_df: pd.DataFrame | None) -> dict[str, dict]:
    if judge_df is None:
        return {}
    return judge_df.set_index("sample_id").to_dict("index")


def evaluate_model(
    df: pd.DataFrame,
    model_col: str,
    ref_col: str,
    convert_special_no: bool,
) -> Dict[str, float]:
    preds = []
    refs = []

    for _, row in df.iterrows():
        pred = row.get(model_col, "")
        ref = row.get(ref_col, "")
        if (pred is None or (isinstance(pred, float) and math.isnan(pred))) or (
            ref is None or (isinstance(ref, float) and math.isnan(ref))
        ):
            continue
        pred_norm = canonicalize_answer(pred, convert_special_no=convert_special_no)
        ref_norm = canonicalize_answer(ref, convert_special_no=convert_special_no)
        if not pred_norm and not ref_norm:
            continue
        preds.append(pred_norm)
        refs.append(ref_norm)

    if not preds:
        return {"samples": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    accuracy = sum(1 for p, r in zip(preds, refs) if p == r) / len(preds)
    precision, recall, f1 = macro_metrics(refs, preds)

    return {
        "samples": len(preds),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compare_answers(
    pred: str | float | None,
    ref: str | float | None,
    convert_special_no: bool,
) -> bool:
    if pred is None or (isinstance(pred, float) and math.isnan(pred)):
        return False
    if ref is None or (isinstance(ref, float) and math.isnan(ref)):
        return False
    pred_norm = canonicalize_answer(pred, convert_special_no=convert_special_no)
    ref_norm = canonicalize_answer(ref, convert_special_no=convert_special_no)
    if not pred_norm and not ref_norm:
        return False
    if pred_norm == ref_norm:
        return True
    # Handle numeric ranges: treat "6 or 7" as matching "6" or "7"
    if " or " in ref_norm:
        options = {opt.strip() for opt in ref_norm.split("or")}
        if pred_norm in options:
            return True
    if " or " in pred_norm:
        options = {opt.strip() for opt in pred_norm.split("or")}
        if ref_norm in options:
            return True
    if "|" in pred_norm or "|" in ref_norm:
        pred_set = {tok.strip() for tok in pred_norm.split("|") if tok.strip()}
        ref_set = {tok.strip() for tok in ref_norm.split("|") if tok.strip()}
        if pred_set and ref_set:
            if pred_set == ref_set:
                return True
            if ref_set.issubset(pred_set):
                return True
    def normalize_gene_list(text: str) -> str:
        parts = [part.strip() for part in text.split("|")]
        mapped = []
        for part in parts:
            mapped.append(GENE_SYNONYMS.get(part, part))
        mapped = [item for item in mapped if item]
        if not mapped:
            return ""
        return " | ".join(sorted(mapped))

    gene_pred = normalize_gene_list(pred_norm)
    gene_ref = normalize_gene_list(ref_norm)
    if gene_pred and gene_ref and gene_pred == gene_ref:
        return True

    return False


def judge_value_for_scenario(
    sample_id: str,
    scenario_name: str,
    judge_lookup: dict[str, dict],
) -> bool | None:
    info = judge_lookup.get(sample_id)
    if not info:
        return None
    return info.get("judge_human_correct") if scenario_name.startswith("Human Answer") else None


def macro_metrics(refs: List[str], preds: List[str]) -> Tuple[float, float, float]:
    labels = sorted(set(refs))
    prec_values = []
    rec_values = []
    f1_values = []

    for label in labels:
        tp = sum(1 for p, r in zip(preds, refs) if p == label and r == label)
        fp = sum(1 for p, r in zip(preds, refs) if p == label and r != label)
        fn = sum(1 for p, r in zip(preds, refs) if p != label and r == label)

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        prec_values.append(precision)
        rec_values.append(recall)
        f1_values.append(f1)

    precision = sum(prec_values) / len(prec_values)
    recall = sum(rec_values) / len(rec_values)
    f1 = sum(f1_values) / len(f1_values)
    return precision, recall, f1


def evaluate_group(
    df: pd.DataFrame,
    models: Iterable[str],
    ref_col: str,
    convert_special_no: bool,
    scenario: str,
) -> pd.DataFrame:
    rows = []
    for model in models:
        if model not in df.columns:
            continue
        metrics = evaluate_model(df, model, ref_col, convert_special_no)
        metrics.update({"model": model, "scenario": scenario, "reference": ref_col, "convert_no": convert_special_no})
        rows.append(metrics)
    return pd.DataFrame(rows)


def build_detail_rows(
    data: pd.DataFrame,
    scenario: dict,
    judge_lookup: dict[str, dict],
) -> List[dict]:
    rows: List[dict] = []
    models = scenario["models"]
    ref_col = scenario["reference"]
    convert = scenario["convert_special_no"]
    scenario_name = scenario["title"]

    for _, record in data.iterrows():
        base = {
            "Scenario": scenario_name,
            "PMID": record["PMID"],
            "QID": record["QID"],
            "Question": record.get("Question", ""),
            "Type": record.get("Type", ""),
            "Human Answer": record.get("Human Answer", ""),
            "Updated Human Answer": record.get("Updated Human Answer", ""),
        }
        sample_id = record.get("sample_id", "")
        for model in models:
            answer = record.get(model, "")
            base[f"{model} Answer"] = answer
            judge_val = judge_value_for_scenario(sample_id, scenario_name, judge_lookup)
            if model == "gpt5-mini" and judge_val is not None:
                correct = bool(judge_val)
            else:
                correct = compare_answers(answer, record.get(ref_col, ""), convert)
            base[f"{model} Correct"] = int(bool(correct))
        rows.append(base)
    return rows


def apply_judge_override(
    df: pd.DataFrame,
    scenario_name: str,
    judge_df: pd.DataFrame | None,
    scenario_records: pd.DataFrame,
) -> None:
    if judge_df is None:
        return
    if not scenario_name.startswith("Human Answer"):
        return
    column = "judge_human_correct"

    scenario_ids = set(scenario_records["sample_id"])
    subset = judge_df[judge_df["sample_id"].isin(scenario_ids) & judge_df[column].notna()]
    if subset.empty:
        return

    samples = len(subset)
    accuracy = float(subset[column].mean())
    metrics = [samples, accuracy, accuracy, accuracy, accuracy]

    mask = df["model"] == "gpt5-mini"
    if mask.any():
        df.loc[mask, ["samples", "accuracy", "precision", "recall", "f1"]] = metrics


def plot_bar_chart(df: pd.DataFrame, title: str, path: Path, footnote: str | None = None) -> None:
    if not HAS_MPL:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    models = df["model"]
    accuracy = df["accuracy"]
    positions = range(len(models))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(positions, accuracy, color="#4c72b0")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    for bar, value in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom")
    if footnote:
        fig.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=8, wrap=True)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_table(df: pd.DataFrame, title: str, path: Path, footnote: str | None = None) -> None:
    if not HAS_MPL:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = 0.7 + 0.4 * len(df)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=df[["model", "accuracy", "precision", "recall", "f1"]].round(3).values,
        colLabels=["Model", "Accuracy", "Precision", "Recall", "F1"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title(title, fontweight="bold", pad=10)
    if footnote:
        fig.text(0.01, 0.01, footnote, ha="left", va="bottom", fontsize=8, wrap=True)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate model outputs against human answers.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of records evaluated.")
    args = parser.parse_args()

    logger = configure_logger()
    if not MERGED_PATH.exists():
        logger.error("Missing merged answers at %s", MERGED_PATH)
        return 1

    df = load_data()
    judge_df = load_judge_results()
    judge_lookup = judge_lookup_map(judge_df)
    if args.limit is not None:
        df = df.head(args.limit)

    scenario_frames: List[pd.DataFrame] = []
    figure_specs: List[Tuple[str, str, pd.DataFrame]] = []
    detail_rows: List[dict] = []

    for scenario in SCENARIOS:
        scenario_df = df
        filter_type = scenario.get("filter_type")
        if filter_type:
            scenario_df = df[df["Type"] == filter_type]
        if scenario_df.empty:
            continue

        subset = evaluate_group(
            scenario_df,
            scenario["models"],
            scenario["reference"],
            scenario["convert_special_no"],
            scenario["title"],
        )
        if subset.empty:
            continue
        apply_judge_override(subset, scenario["title"], judge_df, scenario_df)
        detail_rows.extend(build_detail_rows(scenario_df, scenario, judge_lookup))
        scenario_frames.append(subset)
        figure_specs.append((scenario["title"], scenario["footnote"], subset))

    if not scenario_frames:
        logger.error("No scenarios produced metrics; aborting.")
        return 1

    combined = pd.concat(scenario_frames, ignore_index=True)
    combined.to_csv(OUTPUT_METRICS, index=False)
    logger.info("Wrote metrics to %s", OUTPUT_METRICS)

    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        detail_df.sort_values(["Scenario", "PMID", "QID"], inplace=True)
        DETAIL_METRICS_HUMAN.parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(DETAIL_METRICS_HUMAN, index=False)
        logger.info("Wrote detail rows to %s", DETAIL_METRICS_HUMAN)

    if not HAS_MPL:
        logger.warning("matplotlib not available; skipping figure generation.")
    else:
        for title, footnote, subset in figure_specs:
            slug = slugify(title)
            chart_path = OUTPUT_TABLE_DIR / f"{slug}_accuracy.png"
            plot_bar_chart(subset, f"{title} Accuracy", chart_path, footnote=footnote)

            table_path = OUTPUT_TABLE_DIR / f"{slug}_table.png"
            save_table(subset, f"{title} Metrics", table_path, footnote=footnote)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

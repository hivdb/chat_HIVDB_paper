from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import binomtest


FAMILY_COMPARISONS = {
    "GPT-4o": {
        "base": "GPT-4o base",
        "targets": ["GPT-4o FT", "GPT-4o QSP", "GPT-4o FT+QSP"],
    },
    "Llama3.1-70B": {
        "base": "Llama3.1-70B base",
        "targets": ["Llama3.1-70B FT", "Llama3.1-70B QSP", "Llama3.1-70B FT+QSP"],
    },
    "Llama3.1-8B": {
        "base": "Llama3.1-8B base",
        "targets": ["Llama3.1-8B FT", "Llama3.1-8B QSP", "Llama3.1-8B FT+QSP"],
    },
}

MODEL_BASE_COLORS = {
    "GPT-4o": "#ff7f0e",
    "Llama3.1-70B": "#2ca02c",
    "Llama3.1-8B": "#d62728",
}
VARIANT_TINTS = {"base": 0.65, "FT": 0.05, "QSP": 0.35, "FT+QSP": 0.2}
VARIANT_ORDER = ["base", "FT", "QSP", "FT+QSP"]
BOOTSTRAP_SEED = 20260420
BOOTSTRAP_SAMPLES = 10_000


def benjamini_hochberg(p_values: Iterable[float]) -> List[float]:
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return []
    order = np.argsort(values)
    ranked = np.arange(1, len(values) + 1, dtype=float)
    adjusted = np.empty_like(values)
    adjusted[order] = values[order] * len(values) / ranked
    temp = adjusted[order]
    temp = np.minimum.accumulate(temp[::-1])[::-1]
    adjusted[order] = temp
    return np.clip(adjusted, 0.0, 1.0).tolist()


def _variant_label(model_label: str, family: str) -> str:
    suffix = model_label.replace(family, "", 1).strip()
    return suffix if suffix else "base"


def _format_p_value_display(value: float) -> str:
    if value < 0.001:
        return "< 0.001"
    return f"{value:.3f}"


def _round_half_up(value: float, places: int) -> str:
    quant = Decimal(str(value)).quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return f"{quant:.{places}f}"


def _plot_p_label(value: float) -> str:
    if value < 0.001:
        return "p < 0.001"
    return f"p = {value:.3f}"


def _tint_color(hex_color: str, tint: float) -> tuple[float, float, float]:
    base = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    tint = max(0.0, min(1.0, tint))
    return tuple(channel + (1 - channel) * tint for channel in base)


def _bootstrap_delta_ci(
    base_values: np.ndarray,
    target_values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float]:
    paired_delta = target_values.astype(float) - base_values.astype(float)
    sample_size = paired_delta.size
    indices = rng.integers(0, sample_size, size=(n_bootstrap, sample_size))
    samples = paired_delta[indices].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _build_article_level_frame(detail_df: pd.DataFrame) -> pd.DataFrame:
    correct_columns = [column for column in detail_df.columns if column.endswith(" Correct")]
    article_counts = detail_df.groupby("PMID")[correct_columns].sum()
    question_counts = detail_df.groupby("PMID")["QID"].nunique().rename("question_count")
    article_df = article_counts.join(question_counts)
    article_df.index = article_df.index.astype(str)
    for column in correct_columns:
        exact_match_column = column.replace(" Correct", " Exact Match")
        article_df[exact_match_column] = (article_df[column] == article_df["question_count"]).astype(int)
    return article_df.reset_index().rename(columns={"index": "PMID"})


def _comparison_status(base_value: int, target_value: int) -> str:
    if base_value == 1 and target_value == 1:
        return "both_exact"
    if base_value == 0 and target_value == 0:
        return "neither_exact"
    if base_value == 0 and target_value == 1:
        return "target_only"
    return "base_only"


def _build_summary(article_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows: list[dict] = []
    article_sheet = article_df.copy()
    n_articles = len(article_sheet)

    for family, mapping in FAMILY_COMPARISONS.items():
        base_model = mapping["base"]
        base_exact_col = f"{base_model} Exact Match"
        base_array = article_sheet[base_exact_col].to_numpy(dtype=int)
        for target_model in mapping["targets"]:
            target_exact_col = f"{target_model} Exact Match"
            target_array = article_sheet[target_exact_col].to_numpy(dtype=int)

            both_exact = int(np.sum((base_array == 1) & (target_array == 1)))
            neither_exact = int(np.sum((base_array == 0) & (target_array == 0)))
            target_only = int(np.sum((base_array == 0) & (target_array == 1)))
            base_only = int(np.sum((base_array == 1) & (target_array == 0)))

            discordant = target_only + base_only
            raw_p = (
                float(binomtest(min(target_only, base_only), n=discordant, p=0.5).pvalue)
                if discordant
                else 1.0
            )
            base_rate = float(base_array.mean())
            target_rate = float(target_array.mean())
            ci_low, ci_high = _bootstrap_delta_ci(base_array, target_array, rng=rng)

            summary_rows.append(
                {
                    "family": family,
                    "comparison": _variant_label(target_model, family),
                    "base_model": base_model,
                    "target_model": target_model,
                    "n_articles": n_articles,
                    "base_exact_count": int(base_array.sum()),
                    "base_exact_rate": base_rate,
                    "target_exact_count": int(target_array.sum()),
                    "target_exact_rate": target_rate,
                    "delta_exact_rate": target_rate - base_rate,
                    "delta_ci_low": ci_low,
                    "delta_ci_high": ci_high,
                    "target_only": target_only,
                    "base_only": base_only,
                    "both_exact": both_exact,
                    "neither_exact": neither_exact,
                    "discordant_pairs": discordant,
                    "raw_p": raw_p,
                }
            )

            status_column = f"{family} {_variant_label(target_model, family)} vs base"
            article_sheet[status_column] = [
                _comparison_status(base_value, target_value)
                for base_value, target_value in zip(base_array, target_array, strict=True)
            ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df["adj_p"] = benjamini_hochberg(summary_df["raw_p"].tolist())
    summary_df["significant_bh_0_05"] = summary_df["adj_p"] < 0.05
    summary_df["raw_p_display"] = summary_df["raw_p"].map(_format_p_value_display)
    summary_df["adj_p_display"] = summary_df["adj_p"].map(_format_p_value_display)
    return summary_df, article_sheet


def _build_pooled_qid_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    summary_rows: list[dict] = []
    n_pairs = len(detail_df)

    for family, mapping in FAMILY_COMPARISONS.items():
        base_model = mapping["base"]
        base_array = detail_df[f"{base_model} Correct"].to_numpy(dtype=int)
        for target_model in mapping["targets"]:
            target_array = detail_df[f"{target_model} Correct"].to_numpy(dtype=int)

            both_correct = int(np.sum((base_array == 1) & (target_array == 1)))
            neither_correct = int(np.sum((base_array == 0) & (target_array == 0)))
            target_only = int(np.sum((base_array == 0) & (target_array == 1)))
            base_only = int(np.sum((base_array == 1) & (target_array == 0)))
            discordant = target_only + base_only

            raw_p = (
                float(binomtest(min(target_only, base_only), n=discordant, p=0.5).pvalue)
                if discordant
                else 1.0
            )
            base_rate = float(base_array.mean())
            target_rate = float(target_array.mean())
            ci_low, ci_high = _bootstrap_delta_ci(base_array, target_array, rng=rng)

            summary_rows.append(
                {
                    "family": family,
                    "comparison": _variant_label(target_model, family),
                    "base_model": base_model,
                    "target_model": target_model,
                    "n_pairs": n_pairs,
                    "base_correct_count": int(base_array.sum()),
                    "base_correct_rate": base_rate,
                    "target_correct_count": int(target_array.sum()),
                    "target_correct_rate": target_rate,
                    "delta_correct_rate": target_rate - base_rate,
                    "delta_ci_low": ci_low,
                    "delta_ci_high": ci_high,
                    "target_only": target_only,
                    "base_only": base_only,
                    "both_correct": both_correct,
                    "neither_correct": neither_correct,
                    "discordant_pairs": discordant,
                    "raw_p": raw_p,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df["adj_p"] = benjamini_hochberg(summary_df["raw_p"].tolist())
    summary_df["significant_bh_0_05"] = summary_df["adj_p"] < 0.05
    summary_df["raw_p_display"] = summary_df["raw_p"].map(_format_p_value_display)
    summary_df["adj_p_display"] = summary_df["adj_p"].map(_format_p_value_display)
    return summary_df


def _make_workbook_payload(
    article_summary_df: pd.DataFrame,
    pooled_summary_df: pd.DataFrame,
    article_df: pd.DataFrame,
    payload_path: Path,
) -> None:
    def _to_json_value(value: object) -> object:
        if pd.isna(value):
            return None
        if isinstance(value, (np.integer, np.int64)):
            return int(value)
        if isinstance(value, (np.floating, np.float64)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    summary_cols = [
        "family",
        "comparison",
        "base_model",
        "target_model",
        "n_articles",
        "base_exact_count",
        "base_exact_rate",
        "target_exact_count",
        "target_exact_rate",
        "delta_exact_rate",
        "delta_ci_low",
        "delta_ci_high",
        "target_only",
        "base_only",
        "both_exact",
        "neither_exact",
        "discordant_pairs",
        "raw_p",
        "adj_p",
        "significant_bh_0_05",
    ]
    summary_headers = [
        "Family",
        "Comparison",
        "Base Model",
        "Target Model",
        "Articles",
        "Base Exact-Match Count",
        "Base Exact-Match Rate",
        "Target Exact-Match Count",
        "Target Exact-Match Rate",
        "Delta Exact-Match Rate",
        "Delta 95% CI Low",
        "Delta 95% CI High",
        "Target Only",
        "Base Only",
        "Both Exact",
        "Neither Exact",
        "Discordant Pairs",
        "Raw Exact McNemar p",
        "BH-Adjusted p",
        "BH < 0.05",
    ]

    article_summary_rows: list[list[object]] = [
        ["McNemar experiment: article-level exact match (all 16 QIDs correct per PMID)"],
        [
            "Source: eval/results/detailed_evaluation_full150.xlsx | Delta CIs are paired bootstrap 95% intervals | p-values use exact McNemar and BH correction across all 9 comparisons."
        ],
        [],
        summary_headers,
    ]
    for row in article_summary_df[summary_cols].itertuples(index=False):
        article_summary_rows.append([_to_json_value(value) for value in row])

    pooled_summary_cols = [
        "family",
        "comparison",
        "base_model",
        "target_model",
        "n_pairs",
        "base_correct_count",
        "base_correct_rate",
        "target_correct_count",
        "target_correct_rate",
        "delta_correct_rate",
        "delta_ci_low",
        "delta_ci_high",
        "target_only",
        "base_only",
        "both_correct",
        "neither_correct",
        "discordant_pairs",
        "raw_p",
        "adj_p",
        "significant_bh_0_05",
    ]
    pooled_summary_headers = [
        "Family",
        "Comparison",
        "Base Model",
        "Target Model",
        "Article-QID Pairs",
        "Base Correct Count",
        "Base Correct Rate",
        "Target Correct Count",
        "Target Correct Rate",
        "Delta Correct Rate",
        "Delta 95% CI Low",
        "Delta 95% CI High",
        "Target Only",
        "Base Only",
        "Both Correct",
        "Neither Correct",
        "Discordant Pairs",
        "Raw Exact McNemar p",
        "BH-Adjusted p",
        "BH < 0.05",
    ]
    pooled_summary_rows: list[list[object]] = [
        ["McNemar experiment: pooled article-question pairs (no QID stratification)"],
        [
            "Each PMID × QID pair is treated as one paired observation across strategies; this is a sensitivity analysis and does not collapse to one article-level outcome."
        ],
        [],
        pooled_summary_headers,
    ]
    for row in pooled_summary_df[pooled_summary_cols].itertuples(index=False):
        pooled_summary_rows.append([_to_json_value(value) for value in row])

    article_headers = list(article_df.columns)
    article_rows = [article_headers]
    for row in article_df.itertuples(index=False):
        article_rows.append([_to_json_value(value) for value in row])

    payload = {
        "sheets": [
            {
                "name": "Article Exact Match",
                "rows": article_summary_rows,
                "formats": [
                    {"range": "G5:G13", "numberFormat": "0.0%"},
                    {"range": "I5:I13", "numberFormat": "0.0%"},
                    {"range": "J5:L13", "numberFormat": "0.0%"},
                ],
            },
            {
                "name": "Pooled QID Pairs",
                "rows": pooled_summary_rows,
                "formats": [
                    {"range": "G5:G13", "numberFormat": "0.0%"},
                    {"range": "I5:I13", "numberFormat": "0.0%"},
                    {"range": "J5:L13", "numberFormat": "0.0%"},
                ],
            },
            {
                "name": "Article Outcomes",
                "rows": article_rows,
                "formats": [],
            },
        ]
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_figure(
    summary_df: pd.DataFrame,
    output_path: Path,
    *,
    rate_kind: str,
    y_label: str,
    footer_note: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models: list[str] = []
    rates: dict[str, float] = {}
    significant_rows = summary_df[summary_df["significant_bh_0_05"]].copy()
    significant_map = {
        (row["family"], row["comparison"]): row["adj_p"]
        for _, row in significant_rows.iterrows()
    }

    for family, mapping in FAMILY_COMPARISONS.items():
        family_models = [mapping["base"], *mapping["targets"]]
        models.extend(family_models)
        for model in family_models:
            if rate_kind == "article_exact":
                rate_column = "base_exact_rate" if model == mapping["base"] else "target_exact_rate"
            elif rate_kind == "pooled_correct":
                rate_column = "base_correct_rate" if model == mapping["base"] else "target_correct_rate"
            else:
                raise ValueError(f"Unsupported rate_kind: {rate_kind}")
            match_rows = summary_df.loc[
                (summary_df["family"] == family)
                & (
                    (summary_df["base_model"] == model)
                    | (summary_df["target_model"] == model)
                )
            ]
            rates[model] = float(match_rows.iloc[0][rate_column])

    positions: list[float] = []
    family_ranges: dict[str, tuple[float, float]] = {}
    x_pos = 0.0
    gap = 1.8
    for family, mapping in FAMILY_COMPARISONS.items():
        family_models = [mapping["base"], *mapping["targets"]]
        family_positions = []
        for model in family_models:
            positions.append(x_pos)
            family_positions.append(x_pos)
            x_pos += 1.0
        family_ranges[family] = (family_positions[0], family_positions[-1])
        x_pos += gap

    fig, ax = plt.subplots(figsize=(14, 8))
    metric_values = []
    bracket_height = 1.6
    text_offset = 1.0
    bracket_step = 6.4
    bracket_top_padding = 4.0
    for position, model in zip(positions, models, strict=True):
        family = next(name for name in FAMILY_COMPARISONS if model.startswith(name))
        variant = _variant_label(model, family)
        color = _tint_color(MODEL_BASE_COLORS[family], VARIANT_TINTS[variant])
        value = rates[model] * 100
        metric_values.append(value)
        ax.bar(position, value, width=0.82, color=color, edgecolor="white", linewidth=1.5)
        ax.text(position, value + 1.3, _round_half_up(value, 0), ha="center", va="bottom", fontsize=16)

    max_required_y = max(metric_values) + 8
    for family, mapping in FAMILY_COMPARISONS.items():
        family_peak = max(rates[model] * 100 for model in [mapping["base"], *mapping["targets"]])
        n_sig = sum((family, _variant_label(target, family)) in significant_map for target in mapping["targets"])
        family_required = family_peak + 6.5 + bracket_height + text_offset + max(0, n_sig - 1) * bracket_step + bracket_top_padding
        max_required_y = max(max_required_y, family_required)

    ax.set_title("Full 150", fontsize=30, pad=18)
    ax.set_ylabel(y_label, fontsize=24)
    ax.set_ylim(0, max(50, math.ceil(max_required_y / 5) * 5))
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [_variant_label(model, next(name for name in FAMILY_COMPARISONS if model.startswith(name))) for model in models],
        rotation=25,
        ha="right",
        fontsize=18,
    )
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for family, (start, end) in family_ranges.items():
        center = (start + end) / 2
        ax.text(
            center,
            -0.15,
            family,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=20,
            fontweight="bold",
        )

    pos_lookup = dict(zip(models, positions, strict=True))
    y_top = ax.get_ylim()[1]
    for family, mapping in FAMILY_COMPARISONS.items():
        family_peak = max(rates[model] * 100 for model in [mapping["base"], *mapping["targets"]])
        annotation_y = min(y_top - 2.5, family_peak + 6.5)
        base_x = pos_lookup[mapping["base"]]
        for target in mapping["targets"]:
            comparison = _variant_label(target, family)
            adj_p = significant_map.get((family, comparison))
            if adj_p is None:
                continue
            target_x = pos_lookup[target]
            x1, x2 = sorted([base_x, target_x])
            ax.plot([x1, x1, x2, x2], [annotation_y, annotation_y + bracket_height, annotation_y + bracket_height, annotation_y], color="black", linewidth=1.0)
            ax.text((x1 + x2) / 2, annotation_y + bracket_height + text_offset, _plot_p_label(adj_p), ha="center", va="bottom", fontsize=14)
            annotation_y += bracket_step

    fig.text(
        0.5,
        0.01,
        footer_note,
        ha="center",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.99, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _run_workbook_builder(
    payload_path: Path,
    workbook_path: Path,
    node_bin: str,
    artifact_tool_path: str | None,
) -> None:
    builder_path = Path(__file__).with_name("build_workbook.mjs")
    env = dict(os.environ)
    if artifact_tool_path:
        env["ARTIFACT_TOOL_PATH"] = artifact_tool_path
    subprocess.run(
        [node_bin, str(builder_path), str(payload_path), str(workbook_path)],
        check=True,
        env=env,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval/results/detailed_evaluation_full150.xlsx"),
        help="Detailed evaluation workbook used as source of truth.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mcnemar"),
        help="Output directory for the experiment artifacts.",
    )
    parser.add_argument(
        "--node-bin",
        type=str,
        default="node",
        help="Node.js executable for the workbook builder.",
    )
    parser.add_argument(
        "--artifact-tool-path",
        type=str,
        default=None,
        help="Optional module path for @oai/artifact-tool/dist/artifact_tool.mjs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detail_df = pd.read_excel(args.input)
    article_df = _build_article_level_frame(detail_df)
    article_summary_df, article_outcomes_df = _build_summary(article_df)
    pooled_summary_df = _build_pooled_qid_summary(detail_df)

    article_summary_path = args.output_dir / "mcnemar_article_exact_comparisons_full150.csv"
    pooled_summary_path = args.output_dir / "mcnemar_pooled_qid_comparisons_full150.csv"
    article_path = args.output_dir / "mcnemar_article_outcomes_full150.csv"
    article_figure_path = args.output_dir / "full150-mcnemar-article-exact-bar-chart.png"
    pooled_figure_path = args.output_dir / "full150-mcnemar-pooled-qid-bar-chart.png"
    payload_path = args.output_dir / "mcnemar_workbook_payload_full150.json"
    workbook_path = args.output_dir / "mcnemar_results_full150.xlsx"

    article_summary_df.to_csv(article_summary_path, index=False, encoding="utf-8-sig")
    pooled_summary_df.to_csv(pooled_summary_path, index=False, encoding="utf-8-sig")
    article_outcomes_df.to_csv(article_path, index=False, encoding="utf-8-sig")
    _make_workbook_payload(article_summary_df, pooled_summary_df, article_outcomes_df, payload_path)
    _build_figure(
        article_summary_df,
        article_figure_path,
        rate_kind="article_exact",
        y_label="Article Exact-Match (%)",
        footer_note="Brackets show BH-adjusted exact McNemar p-values for target vs family base.",
    )
    _build_figure(
        pooled_summary_df,
        pooled_figure_path,
        rate_kind="pooled_correct",
        y_label="Pooled Correctness (%)",
        footer_note="Brackets show BH-adjusted exact McNemar p-values with all article-question pairs pooled across QIDs.",
    )
    _run_workbook_builder(payload_path, workbook_path, args.node_bin, args.artifact_tool_path)

    print(f"Wrote article-level summary CSV to {article_summary_path}")
    print(f"Wrote pooled-QID summary CSV to {pooled_summary_path}")
    print(f"Wrote article-level CSV to {article_path}")
    print(f"Wrote article-level figure to {article_figure_path}")
    print(f"Wrote pooled-QID figure to {pooled_figure_path}")
    print(f"Wrote workbook to {workbook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

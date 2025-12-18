from __future__ import annotations

from pathlib import Path
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from decimal import Decimal, ROUND_HALF_UP

from .constants import MODEL_BASE_COLORS, VARIANT_TINTS
from .normalize import slugify

METRIC_COLUMNS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
]

TITLE_FONT_SIZE = 32
AXIS_LABEL_SIZE = 24
AXIS_TICK_SIZE = 18
BAR_LABEL_SIZE = 20
ANNOTATION_FONT_SIZE = 12
FAMILY_LABEL_SIZE = 18

MAX_Y_LIM = 100  # cap at 100% since metrics are proportions

METRIC_PALETTE = {
    "accuracy": "#2a9d8f",
    "precision": "#e76f51",
    "recall": "#f4a261",
    "f1": "#264653",
}

FAMILY_COMPARISONS = {
    "GPT-4o": {
        "base": "GPT-4o base",
        "targets": ["GPT-4o FT", "GPT-4o FT+QSP", "GPT-4o QSP"],
    },
    "Llama3.1-70B": {
        "base": "Llama3.1-70B base",
        "targets": ["Llama3.1-70B FT", "Llama3.1-70B FT+QSP", "Llama3.1-70B QSP"],
    },
    "Llama3.1-8B": {
        "base": "Llama3.1-8B base",
        "targets": ["Llama3.1-8B FT", "Llama3.1-8B FT+QSP", "Llama3.1-8B QSP"],
    },
}

# Preferred ordering within each family: base, FT, QSP, FT+QSP, then any others.
VARIANT_ORDER = ["base", "FT", "QSP", "FT+QSP"]

def _round_half_up(value: float, places: int) -> str:
    # Use string conversion to avoid binary float artifacts (e.g., 0.835 -> 0.84).
    quant = Decimal(str(value)).quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)
    return f"{quant:.{places}f}"


def _variant_from_label(label: str) -> str:
    lowered = label.lower()
    if "lc" in lowered:
        return "LC"
    if "rag" in lowered:
        return "RAG"
    if "bm25" in lowered:
        return "BM25"
    if "question-specific" in lowered or "pv1" in lowered:
        return "QSP"
    if "ap" in lowered and "base" not in lowered:
        return "AP"
    if " ft" in lowered or lowered.endswith("ft") or " ft " in lowered or lowered.startswith("ft") or lowered.startswith("ft"):
        return "FT"
    if "ft" in lowered:
        return "FT"
    return "base"


def _family_from_label(label: str) -> str:
    for family in MODEL_BASE_COLORS:
        if label.startswith(family):
            return family
    return ""


def _tint_color(hex_color: str, tint: float) -> tuple[float, float, float]:
    base = mcolors.to_rgb(hex_color)
    tint = max(0.0, min(1.0, tint))
    return tuple(channel + (1 - channel) * tint for channel in base)


def _color_for_model(label: str) -> tuple[float, float, float]:
    family = _family_from_label(label)
    base_color = MODEL_BASE_COLORS.get(family, "#4c72b0")
    variant = _variant_from_label(label)
    tint = VARIANT_TINTS.get(variant, 0.0)
    return _tint_color(base_color, tint)


def _variant_handles() -> list[Patch]:
    handles: list[Patch] = []
    base_gray = "#888888"
    for variant, tint in VARIANT_TINTS.items():
        color = _tint_color(base_gray, tint)
        handles.append(Patch(facecolor=color, label=f"{variant} scenario".title()))
    return handles


def _family_handles() -> list[Patch]:
    return [Patch(facecolor=color, label=family) for family, color in MODEL_BASE_COLORS.items()]


def _variant_label(label: str) -> str:
    for family in MODEL_BASE_COLORS:
        if label.startswith(family):
            suffix = label[len(family):].strip()
            return suffix if suffix else "base"
    return label


def _ordered_models(models: list[str]) -> list[str]:
    """Order models within each family as base, FT, QSP, FT+QSP, then others (stable by input order)."""
    families_in_order: list[str] = []
    for model in models:
        fam = _family_from_label(model)
        if fam and fam not in families_in_order:
            families_in_order.append(fam)
    index_map = {m: i for i, m in enumerate(models)}
    ordered: list[str] = []
    for fam in families_in_order:
        fam_models = [m for m in models if _family_from_label(m) == fam]
        def _rank(model: str) -> tuple[int, int]:
            var = _variant_label(model).lower()
            try:
                v_idx = [v.lower() for v in VARIANT_ORDER].index(var)
            except ValueError:
                v_idx = len(VARIANT_ORDER)
            return (v_idx, index_map.get(model, 0))
        fam_sorted = sorted(fam_models, key=_rank)
        ordered.extend(fam_sorted)
    # Append any models with no family match
    for m in models:
        if m not in ordered:
            ordered.append(m)
    return ordered


def _group_positions(models: list[str]) -> tuple[list[float], dict[str, tuple[float, float]]]:
    positions: list[float] = []
    family_ranges: dict[str, list[float]] = {}
    gap = 2.0  # extra breathing room between families and labels
    x = 0.0
    previous_family = None
    for model in models:
        family = _family_from_label(model)
        if previous_family is not None and family != previous_family:
            x += gap
        positions.append(x)
        family_ranges.setdefault(family, []).append(x)
        x += 1.0
        previous_family = family
    family_bounds = {fam: (min(pos_list), max(pos_list)) for fam, pos_list in family_ranges.items()}
    return positions, family_bounds


def _short_question_label(qid: int, question: str, max_len: int = 28) -> str:
    if qid in QID_TOPICS:
        return f"Q{qid}: {QID_TOPICS[qid]}"
    clean = question.strip().replace("\n", " ")
    if len(clean) > max_len:
        clean = clean[: max_len - 1].rstrip() + "…"
    return f"Q{qid}: {clean}"


def _annotate_families(ax, family_bounds: dict[str, tuple[float, float]]) -> None:
    if len(family_bounds) <= 1:
        return
    for family, (start, end) in family_bounds.items():
        center = (start + end) / 2
        ax.text(
            center,
            -0.45,
            family,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=FAMILY_LABEL_SIZE,
            fontweight="bold",
        )


def _variant_label(label: str) -> str:
    for family in MODEL_BASE_COLORS:
        if label.startswith(family):
            suffix = label[len(family):].strip()
            return suffix if suffix else "base"
    return label


def _position_lookup(models: list[str]) -> dict[str, float]:
    positions, _ = _group_positions(models)
    return dict(zip(models, positions))


def _annotate_significance(
    ax,
    models: list[str],
    pos_lookup: dict[str, float],
    significance: dict | None,
    metric: str,
    comparisons: dict | None = None,
    offset_shift: float = 0.0,
    metric_values: dict[str, float] | None = None,
) -> None:
    if not significance:
        return
    metric_map = significance.get(metric)
    if not metric_map:
        return
    y_max = ax.get_ylim()[1]
    # Use fixed spacing constants for consistent, non-overlapping annotations
    base_margin = 35.0
    bracket_height = 5.0
    text_padding = 3.0
    base_offset_step = 35.0  # Increased spacing between stacked annotations
    lift = 15.0
    max_offset = max(0.0, y_max - bracket_height - text_padding - 0.5)
    family_offsets: dict[str, float] = {}
    comparison_map = comparisons or FAMILY_COMPARISONS
    for family, mapping in comparison_map.items():
        step = base_offset_step
        base_label = mapping["base"]
        base_x = pos_lookup.get(base_label)
        base_val = None if metric_values is None else metric_values.get(base_label)
        if base_x is None:
            continue
        # Keep brackets well above value labels on bars.
        value_label_padding = max(10.0, y_max * 0.1)
        margin_up = max(10.0, y_max * 0.12)
        family_max_height = 0.0
        highest_label_y = None
        if metric_values is not None:
            for candidate in [mapping["base"], *mapping["targets"]]:
                val = metric_values.get(candidate)
                if val is not None:
                    family_max_height = max(family_max_height, val)
            if family_max_height > 0.0:
                # Match the value label placement used above to guarantee clearance.
                highest_label_y = min(y_max - 1.0, family_max_height + value_label_padding)
        target_rows: list[tuple[float, str, float, str]] = []
        for target in mapping["targets"]:
            target_x = pos_lookup.get(target)
            if target_x is None:
                continue
            target_suffix = target.replace(f"{family} ", "")
            p_value = metric_map.get((family, target_suffix))
            if p_value is None:
                continue
            # Blanket policy: skip annotation when target metric is lower than base
            if metric_values is not None:
                target_val = metric_values.get(target)
                if base_val is not None and target_val is not None and target_val < base_val:
                    continue
            distance = abs(target_x - base_x)
            target_rows.append((distance, target, target_x, target_suffix))
        if not target_rows:
            continue
        target_rows.sort(key=lambda item: item[0])
        start_offset = family_offsets.get(family)
        if start_offset is None:
            comparisons_needed = len(target_rows) - 1
            top = max_offset
            if comparisons_needed <= 0:
                step = base_offset_step
                start_offset = min(top, max(base_margin, top - 6.0))
            else:
                span = max(0.0, top - base_margin)
                step = min(base_offset_step, max(10.0, span / max(1, comparisons_needed)))
                start_offset = max(base_margin, top - step * comparisons_needed)
        start_offset += lift
        if highest_label_y is not None:
            # Push the starting bracket high enough to avoid colliding with bar value labels.
            clearance = max(1.0, y_max * 0.01)
            min_offset = highest_label_y + clearance - (bracket_height + text_padding)
            start_offset = max(start_offset, min_offset)
        offset = max(base_margin, min(max_offset, start_offset))
        for _, target, target_x, target_suffix in target_rows:
            p_value = metric_map.get((family, target_suffix))
            if p_value is None:
                continue
            # Only show adjusted p-values < 0.05
            if float(p_value) >= 0.05:
                continue
            if p_value < 0.001:
                label = "p<0.001"
            elif p_value > 0.009:
                # For larger p-values, display only two decimal places (e.g., 0.01 instead of 0.011)
                label = f"p={p_value:.2f}"
            else:
                label = f"p={p_value:.3f}"
            # Use fixed vertical spacing - no collision detection needed
            ax.plot(
                [base_x, base_x, target_x, target_x],
                [offset, offset + bracket_height, offset + bracket_height, offset],
                color="black",
                linewidth=1,
            )
            text_y = offset + bracket_height + text_padding
            ax.text(
                (base_x + target_x) / 2,
                text_y,
                label,
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_FONT_SIZE,
            )
            offset = min(offset + step, max_offset)
        family_offsets[family] = offset


def plot_metric_panels(
    df,
    qid_df,
    title: str,
    output_path: Path,
    significance: dict | None = None,
    comparisons: dict | None = None,
    layout: str = "vertical",
) -> None:
    if df.empty:
        return
    df = df.copy()
    for metric_col, _ in METRIC_COLUMNS:
        if metric_col in df.columns:
            df[metric_col] = df[metric_col] * 100.0
    y_limit = MAX_Y_LIM
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "display_order" in df.columns:
        df = df.sort_values("display_order").reset_index(drop=True)
        models = df["model"].tolist()
    else:
        models = _ordered_models(df["model"].tolist())
        df = df.set_index("model").loc[models].reset_index()
    positions, family_bounds = _group_positions(models)
    pos_lookup = _position_lookup(models)
    if "color_override" in df.columns:
        colors = df["color_override"].tolist()
    else:
        colors = [_color_for_model(model) for model in models]
    if "display_label" in df.columns:
        variant_labels = df["display_label"].tolist()
    else:
        variant_labels = [_variant_label(model) for model in models]
    if layout == "combined":
        width = max(18, 0.7 * len(models) * len(METRIC_COLUMNS))
        height = 10
        fig, ax = plt.subplots(figsize=(width, height))
        total_width = 0.8
        bar_width = total_width / len(METRIC_COLUMNS)
        handles: list[Patch] = []
        ax.set_ylim(0, y_limit)
        for idx, (metric, label) in enumerate(METRIC_COLUMNS):
            metric_color = METRIC_PALETTE.get(metric, "#4c72b0")
            offsets = [pos - total_width / 2 + bar_width * (idx + 0.5) for pos in positions]
            values = df[metric].tolist()
            bars = ax.bar(offsets, values, color=metric_color, width=bar_width * 0.9)
            for bar, value in zip(bars, values):
                label_y = min(y_limit - 1.0, value + 4.0)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    _round_half_up(value, 0),
                    ha="center",
                    va="bottom",
                    fontsize=BAR_LABEL_SIZE,
            )
            handles.append(Patch(facecolor=metric_color, label=label))
            value_map = dict(zip(models, values))
            _annotate_significance(
                ax,
                models,
                pos_lookup,
                significance,
                metric,
                comparisons,
                offset_shift=idx * 0.35,
                metric_values=value_map,
            )
        ax.set_ylim(0, y_limit)
        ax.set_ylabel("Percentage", fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=AXIS_TICK_SIZE)
        ax.tick_params(axis="x", pad=12)
        ax.set_xticks(positions)
        ax.set_xticklabels(variant_labels, rotation=25, ha="right", fontsize=AXIS_TICK_SIZE)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=handles, fontsize=AXIS_TICK_SIZE, frameon=False)
        axes = [ax]
    else:
        width = max(14, 0.7 * len(models))
        height = 16
        # hspace is a fraction of the average axis height; 0.4 ~= 40% gap between stacked plots.
        fig, axes = plt.subplots(
            len(METRIC_COLUMNS),
            1,
            figsize=(width, height),
            sharex=True,
            gridspec_kw={"hspace": 0.4},
        )
        for ax in axes:
            ax.tick_params(axis="x", pad=12)
        bar_width = 0.8
        if len(METRIC_COLUMNS) == 1:
            axes = [axes]
        for ax, (metric, label) in zip(axes, METRIC_COLUMNS):
            values = df[metric].tolist()
            metric_map = significance.get(metric) if significance else None
            metric_headroom = 50.0 if metric_map else 0.0
            y_limit_for_metric = max(y_limit, max(values, default=0) + metric_headroom)
            # Shade base/FT/QSP/FT+QSP groups differently when style_group present
            if "style_group" in df.columns:
                shade_map = {"base": 0.35, "ft": 0.0, "ft_qsp": -0.08, "qsp": 0.18}
                base_colors = [_color_for_model(model) for model in models]
                bar_colors = []
                for base_color, group in zip(base_colors, df["style_group"].tolist()):
                    tint = shade_map.get(group, 0.0)
                    bar_colors.append(_tint_color(mcolors.to_hex(base_color), tint))
            else:
                bar_colors = colors
            bars = ax.bar(positions, values, color=bar_colors, width=bar_width)
            ax.set_ylim(0, y_limit_for_metric)
            ax.set_ylabel(f"{label} (%)", fontsize=AXIS_LABEL_SIZE)
            ax.tick_params(axis="both", labelsize=AXIS_TICK_SIZE)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for bar, value in zip(bars, values):
                label_y = min(y_limit - 1.0, value + 4.0)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    _round_half_up(value, 0),
                    ha="center",
                    va="bottom",
                    fontsize=BAR_LABEL_SIZE,
                )
            value_map = dict(zip(models, values))
            _annotate_significance(ax, models, pos_lookup, significance, metric, comparisons, metric_values=value_map)
        axes[-1].set_xticks(positions)
        axes[-1].set_xticklabels(variant_labels, rotation=25, ha="right", fontsize=AXIS_TICK_SIZE)
    _annotate_families(axes[-1], family_bounds)
    fig.suptitle(title, fontsize=TITLE_FONT_SIZE)
    if layout == "combined":
        fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_table(df, title: str, path: Path) -> None:
    if df.empty:
        return
    df = df.copy()
    metric_cols = ["accuracy", "precision", "recall", "f1"]
    for metric_col in metric_cols:
        if metric_col in df.columns:
            df[metric_col] = df[metric_col] * 100.0
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = 1.0 + 0.6 * len(df)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    rounded_metrics = df[metric_cols].round(0).astype(int)
    data = rounded_metrics.copy()
    data.insert(0, "model", df["model"].str.replace("Llama3.1", "L3.1", regex=False))
    data = data.values
    table = ax.table(
        cellText=data,
        colLabels=["Model", "Accuracy", "Precision", "Recall", "F1"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(22)
    table.scale(1.4, 2.2)
    ax.set_title(title, fontweight="bold", pad=10, fontsize=TITLE_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_metric_by_qid(
    qid_df,
    metric: str,
    output_path: Path,
    family_models: dict[str, list[str]] | None = None,
) -> None:
    """Plot per-question metric (precision/recall) for each model family."""
    if qid_df is None or qid_df.empty or metric not in qid_df.columns:
        return
    family_models = family_models or {
        "GPT-4o": [
            "GPT-4o base",
            "GPT-4o FT",
            "GPT-4o QSP",
        ],
        "Llama3.1-70B": [
            "Llama3.1-70B base",
            "Llama3.1-70B FT",
            "Llama3.1-70B QSP",
        ],
        "Llama3.1-8B": [
            "Llama3.1-8B base",
            "Llama3.1-8B FT",
            "Llama3.1-8B QSP",
        ],
    }
    questions = (
        qid_df[["QID", "Question"]]
        .drop_duplicates()
        .sort_values("QID")
    )
    qids = questions["QID"].tolist()
    labels = [_short_question_label(qid, question) for qid, question in questions.itertuples(index=False)]

    fig, axes = plt.subplots(len(family_models), 1, figsize=(22, 18), sharey=True)
    if len(family_models) == 1:
        axes = [axes]

    bar_width = 0.22
    x_base = np.arange(len(qids))
    legend_handles: list = []
    legend_labels: list[str] = []

    for ax, (family, models) in zip(axes, family_models.items()):
        fam_df = qid_df[qid_df["model"].isin(models)].copy()
        if fam_df.empty:
            ax.set_visible(False)
            continue
        for idx, model in enumerate(models):
            series = fam_df[fam_df["model"] == model].set_index("QID")[metric] * 100
            heights = [series.get(qid, np.nan) for qid in qids]
            offsets = (idx - (len(models) - 1) / 2) * bar_width
            positions = x_base + offsets
            color = _color_for_model(model)
            label = _variant_label(model)
            bars = ax.bar(positions, heights, width=bar_width, label=label, color=color, edgecolor="black")
            if label not in legend_labels and len(bars) > 0:
                legend_handles.append(bars[0])
                legend_labels.append(label)
            ax.set_title(family, fontsize=TITLE_FONT_SIZE - 6, pad=12, fontweight="bold")
            ax.set_ylim(0, 105)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[-1].set_xticks(x_base)
    axes[-1].set_xticklabels(labels, rotation=50, ha="right", fontsize=AXIS_TICK_SIZE)
    for ax in axes[:-1]:
        ax.set_xticks([])
    for ax in axes:
        ax.set_ylabel(f"{metric.title()} (%)", fontsize=AXIS_LABEL_SIZE)

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels), fontsize=AXIS_TICK_SIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_figures(
    subset: pd.DataFrame,
    scenario: str,
    output_dir: Path,
    significance=None,
    comparisons=None,
    base_name: str | None = None,
    display_title: str | None = None,
) -> None:
    """Render accuracy bar chart and metrics table per scenario."""
    if subset.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = base_name or slugify(scenario)
    title = display_title or scenario
    plot_metric_panels(
        subset,
        qid_df=None,
        title=title,
        output_path=output_dir / f"{slug}-bar-chart.png",
        significance=significance,
        comparisons=comparisons or FAMILY_COMPARISONS,
        layout="vertical",
    )
    save_table(subset, f"{title} Metrics", output_dir / f"{slug}-table.png")
QID_TOPICS = {
    1: "Patient Sequences?",
    2: "In vitro Drug Susceptibility?",
    3: "Open Access?",
    4: "GenBank IDs",
    5: "# Patients Sequenced",
    6: "Countries",
    7: "Sampling Years",
    8: "Were Samples Cloned?",
    9: "HIV Genes",
    10: "Sequencing Methods",
    11: "Sample Types",
    12: "VF on Therapy?",
    13: "Clinical Study?",
    14: "Prior ARV Use?",
    15: "Drug Classes",
    16: "Drugs",
}

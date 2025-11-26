from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ttest_rel, wilcoxon


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


def _contingency_counts(row, metric: str, base: str, target: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    try:
        base_tp = row["tp"][base]
        base_fp = row["fp"][base]
        base_tn = row["tn"][base]
        base_fn = row["fn"][base]
        target_tp = row["tp"][target]
        target_fp = row["fp"][target]
        target_tn = row["tn"][target]
        target_fn = row["fn"][target]
    except KeyError:
        return None, None

    if metric == "accuracy":
        base_pos = base_tp + base_tn
        base_neg = base_fp + base_fn
        target_pos = target_tp + target_tn
        target_neg = target_fp + target_fn
    elif metric == "precision":
        base_pos, base_neg = base_tp, base_fp
        target_pos, target_neg = target_tp, target_fp
    elif metric == "recall":
        base_pos, base_neg = base_tp, base_fn
        target_pos, target_neg = target_tp, target_fn
    else:
        return None, None

    base_counts = (base_pos, base_neg)
    target_counts = (target_pos, target_neg)
    if any(math.isnan(value) for value in (*base_counts, *target_counts)):
        return None, None
    return base_counts, target_counts


def compute_fisher_tests(
    qid_df: pd.DataFrame,
    comparisons: dict,
    metrics: Iterable[str],
) -> pd.DataFrame:
    if qid_df is None or qid_df.empty:
        return pd.DataFrame()

    pivot = qid_df.pivot_table(index="QID", columns="model", values=["tp", "fp", "tn", "fn"])
    records: List[dict] = []
    for family, mapping in comparisons.items():
        base_label = mapping["base"]
        for target in mapping["targets"]:
            comparison_name = f"{family} base vs {target.split()[-1]}"
            for qid, row in pivot.iterrows():
                for metric in metrics:
                    counts = _contingency_counts(row, metric, base_label, target)
                    if counts is None:
                        continue
                    (base_pos, base_neg), (target_pos, target_neg) = counts
                    table = np.array([[base_pos, base_neg], [target_pos, target_neg]])
                    try:
                        _, p_value = fisher_exact(table)
                    except ValueError:
                        p_value = 1.0
                    records.append(
                        {
                            "QID": qid,
                            "family": family,
                            "comparison": target.split()[-1],
                            "metric": metric,
                            "p_value": float(p_value),
                        }
                    )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["adj_p"] = 1.0
    for (metric, comparison), group in df.groupby(["metric", "comparison"]):
        adjusted = benjamini_hochberg(group["p_value"].tolist())
        df.loc[group.index, "adj_p"] = adjusted
    return df


def compute_pairwise_tests(
    qid_df: pd.DataFrame,
    comparisons: dict,
    metrics: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[Tuple[str, str], float]], Dict[str, Dict[Tuple[str, str], float]]]:
    if qid_df is None or qid_df.empty:
        return pd.DataFrame(), {}, {}

    metric_pivots = {
        metric: qid_df.pivot_table(index="QID", columns="model", values=metric)
        for metric in metrics
    }

    records: List[dict] = []
    wilcoxon_map: Dict[str, Dict[Tuple[str, str], float]] = {metric: {} for metric in metrics}
    ttest_map: Dict[str, Dict[Tuple[str, str], float]] = {metric: {} for metric in metrics}

    for metric in metrics:
        pivot = metric_pivots[metric]
        for family, mapping in comparisons.items():
            base_label = mapping["base"]
            if base_label not in pivot:
                continue
            for target_label in mapping["targets"]:
                if target_label not in pivot:
                    continue
                aligned = pivot[[base_label, target_label]].dropna()
                if aligned.empty:
                    continue
                base_vals = aligned[base_label].to_numpy(dtype=float)
                target_vals = aligned[target_label].to_numpy(dtype=float)
                try:
                    t_stat, t_p = ttest_rel(base_vals, target_vals)
                except ValueError:
                    t_stat, t_p = np.nan, 1.0
                try:
                    w_stat, w_p = wilcoxon(base_vals, target_vals)
                except ValueError:
                    w_stat, w_p = np.nan, 1.0
                target_suffix = target_label.replace(f"{family} ", "")
                records.append(
                    {
                        "family": family,
                        "comparison": target_suffix,
                        "metric": metric,
                        "test": "t-test",
                        "p_value": float(t_p),
                    }
                )
                records.append(
                    {
                        "family": family,
                        "comparison": target_suffix,
                        "metric": metric,
                        "test": "wilcoxon",
                        "p_value": float(w_p),
                    }
                )
                ttest_map.setdefault(metric, {})[(family, target_suffix)] = float(t_p)
                wilcoxon_map.setdefault(metric, {})[(family, target_suffix)] = float(w_p)

    stats_df = pd.DataFrame(records)
    if stats_df.empty:
        return stats_df, wilcoxon_map, ttest_map

    stats_df["adj_p"] = 1.0
    for metric in metrics:
        for test_name in ["t-test", "wilcoxon"]:
            mask = (stats_df["metric"] == metric) & (stats_df["test"] == test_name)
            if not mask.any():
                continue
            adjusted = benjamini_hochberg(stats_df.loc[mask, "p_value"].tolist())
            stats_df.loc[mask, "adj_p"] = adjusted

    return stats_df, wilcoxon_map, ttest_map

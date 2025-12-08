from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ttest_rel, wilcoxon


def _round_sig(val: float) -> float:
    try:
        return float(f"{float(val):.3g}")
    except Exception:
        return float(val)


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

    def _metric_value(tp, fp, tn, fn, metric_name):
        total = tp + fp + tn + fn
        if metric_name == "accuracy":
            return (tp + tn) / total if total else math.nan
        if metric_name == "precision":
            denom = tp + fp
            return tp / denom if denom else math.nan
        if metric_name == "recall":
            denom = tp + fn
            return tp / denom if denom else math.nan
        if metric_name == "f1":
            denom = 2 * tp + fp + fn
            return (2 * tp) / denom if denom else math.nan
        return math.nan

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
    elif metric == "f1":
        # Scale TP by 2 so that f1 = pos / (pos + neg) for Fisher calculation
        base_pos, base_neg = 2 * base_tp, base_fp + base_fn
        target_pos, target_neg = 2 * target_tp, target_fp + target_fn
    else:
        return None, None

    base_metric_val = _metric_value(base_tp, base_fp, base_tn, base_fn, metric)
    target_metric_val = _metric_value(target_tp, target_fp, target_tn, target_fn, metric)

    base_counts = (base_pos, base_neg, base_metric_val)
    target_counts = (target_pos, target_neg, target_metric_val)
    if any(math.isnan(value) for value in (*base_counts[:2], *target_counts[:2])):
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
    all_qids = sorted(pivot.index.tolist())
    records: List[dict] = []
    for family, mapping in comparisons.items():
        base_label = mapping["base"]
        for target in mapping["targets"]:
            for metric in metrics:
                base_values = {}
                target_values = {}
                p_values = {}
                for qid, row in pivot.iterrows():
                    counts = _contingency_counts(row, metric, base_label, target)
                    if counts is None:
                        continue
                    (base_pos, base_neg, base_val), (target_pos, target_neg, target_val) = counts
                    table = np.array([[base_pos, base_neg], [target_pos, target_neg]])
                    try:
                        _, p_value = fisher_exact(table)
                    except ValueError:
                        p_value = 1.0
                    base_values[qid] = _round_sig(base_val)
                    target_values[qid] = _round_sig(target_val)
                    p_values[qid] = float(p_value)
                if not p_values:
                    continue
                # Adjust p-values across QIDs for this metric/comparison
                adj_map = {}
                adjusted = benjamini_hochberg(list(p_values.values()))
                for qid, adj in zip(p_values.keys(), adjusted):
                    adj_map[qid] = _round_sig(adj)
                record = {
                    "family": family,
                    "comparison": target.split()[-1],
                    "metric": metric,
                    "test": "fisher",
                }
                for qid in all_qids:
                    record[f"base_qid_{qid}"] = _round_sig(base_values.get(qid, math.nan))
                    record[f"target_qid_{qid}"] = _round_sig(target_values.get(qid, math.nan))
                    record[f"p_value_qid_{qid}"] = _round_sig(p_values.get(qid, math.nan))
                    record[f"adj_p_qid_{qid}"] = _round_sig(adj_map.get(qid, math.nan))
                records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    # Round numeric columns for readability
    for col in df.columns:
        if df[col].dtype.kind in {"f", "i"}:
            df[col] = df[col].apply(_round_sig)
    return df


def compute_pairwise_tests(
    qid_df: pd.DataFrame,
    comparisons: dict,
    metrics: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[Tuple[str, str], float]], Dict[str, Dict[Tuple[str, str], float]]]:
    if qid_df is None or qid_df.empty:
        return pd.DataFrame(), {}, {}

    all_qids = sorted(qid_df["QID"].unique())
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
                base_mean = float(np.mean(base_vals))
                target_mean = float(np.mean(target_vals))
                base_std = float(np.std(base_vals, ddof=1)) if base_vals.size > 1 else 0.0
                target_std = float(np.std(target_vals, ddof=1)) if target_vals.size > 1 else 0.0
                # Capture per-QID metric values for transparency
                base_qid_values = {
                    f"base_qid_{qid}": _round_sig(float(aligned.at[qid, base_label])) if qid in aligned.index else math.nan
                    for qid in all_qids
                }
                target_qid_values = {
                    f"target_qid_{qid}": _round_sig(float(aligned.at[qid, target_label])) if qid in aligned.index else math.nan
                    for qid in all_qids
                }
                def _round_p(val: float) -> float:
                    try:
                        return float(f"{float(val):.3g}")
                    except Exception:
                        return float(val)
                t_p = _round_p(t_p)
                w_p = _round_p(w_p)
                base_mean_r = _round_sig(base_mean)
                target_mean_r = _round_sig(target_mean)
                base_std_r = _round_sig(base_std)
                target_std_r = _round_sig(target_std)
                records.append(
                    {
                        "family": family,
                        "comparison": target_suffix,
                        "metric": metric,
                        "test": "t-test",
                        "p_value": float(t_p),
                        "base_mean": base_mean_r,
                        "target_mean": target_mean_r,
                        "base_std": base_std_r,
                        "target_std": target_std_r,
                        **base_qid_values,
                        **target_qid_values,
                    }
                )
                records.append(
                    {
                        "family": family,
                        "comparison": target_suffix,
                        "metric": metric,
                        "test": "wilcoxon",
                        "p_value": float(w_p),
                        "base_mean": base_mean_r,
                        "target_mean": target_mean_r,
                        "base_std": base_std_r,
                        "target_std": target_std_r,
                        **base_qid_values,
                        **target_qid_values,
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
            stats_df.loc[mask, "adj_p"] = [ _round_sig(val) for val in adjusted ]

    # Round numeric columns to 3 significant digits for readability
    for col in stats_df.columns:
        if stats_df[col].dtype.kind in {"f", "i"}:
            stats_df[col] = stats_df[col].apply(_round_sig)

    return stats_df, wilcoxon_map, ttest_map

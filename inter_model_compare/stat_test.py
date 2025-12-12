from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as stats


METRICS = ["Accuracy", "Precision", "Recall", "F1"]
NAME2_VARIANTS = ("base", "FT")
SOURCE = "metrics_by_model_and_metric.csv"


def metric_values(row, q_columns):
    """Return aligned numeric Q-values for the two rows."""

    values = pd.DataFrame(
        {
            "a": pd.to_numeric(row[0][q_columns], errors="coerce"),
            "b": pd.to_numeric(row[1][q_columns], errors="coerce"),
        }
    ).dropna()
    return values["a"].to_numpy(float), values["b"].to_numpy(float)


def format_label(row):
    return f"{row['name1']} {row['name2']} {row['metric']}"


def add_result(label, row_a, row_b, q_columns, results):
    pre, post = metric_values((row_a, row_b), q_columns)
    if len(pre) == 0 or len(post) == 0:
        return

    if np.allclose(pre, post):
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0
    else:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(pre, post, zero_method="zsplit")

    ttest = stats.ttest_rel(pre, post, nan_policy="omit")
    t_stat = 0.0 if np.isnan(ttest.statistic) else ttest.statistic
    t_p = 1.0 if np.isnan(ttest.pvalue) else ttest.pvalue

    results.append(
        {
            "Set": label,
            "Group A": format_label(row_a),
            "Group B": format_label(row_b),
            "Wilcoxon statistic": wilcoxon_stat,
            "Wilcoxon p-value": wilcoxon_p,
            "T-test statistic": t_stat,
            "T-test p-value": t_p,
        }
    )


def load_table(path):
    df = pd.read_csv(path)
    q_columns = [c for c in df.columns if c.startswith("Q")]
    return df, q_columns


def get_row(df, name1, name2, metric):
    subset = df[
        (df["name1"] == name1) & (df["name2"] == name2) & (df["metric"] == metric)
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def compare_base_vs_ft(df, q_columns):
    results = []
    for name1 in sorted(df["name1"].unique()):
        for metric in METRICS:
            base_row = get_row(df, name1, "base", metric)
            ft_row = get_row(df, name1, "FT", metric)
            if base_row is None or ft_row is None:
                continue
            label = f"{name1}: base vs FT ({metric})"
            add_result(label, base_row, ft_row, q_columns, results)
    return results


def compare_between_models(df, q_columns):
    results = []
    for name1_a, name1_b in combinations(sorted(df["name1"].unique()), 2):
        for name2 in NAME2_VARIANTS:
            for metric in METRICS:
                row_a = get_row(df, name1_a, name2, metric)
                row_b = get_row(df, name1_b, name2, metric)
                if row_a is None or row_b is None:
                    continue
                label = f"{name1_a} vs {name1_b} ({name2}, {metric})"
                add_result(label, row_a, row_b, q_columns, results)
    return results


def main():
    df, q_columns = load_table(SOURCE)
    results = []
    results.extend(compare_base_vs_ft(df, q_columns))
    results.extend(compare_between_models(df, q_columns))

    output_path = "Inter_model_Stat_results.xlsx"
    pd.DataFrame(results).to_excel(output_path, index=False)
    print(f"Saved Wilcoxon results to {output_path}")


if __name__ == "__main__":
    main()

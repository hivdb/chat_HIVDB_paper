# Regression Analysis

This directory contains scripts for preparing per-paper-question Llama3.1-70B rank data and fitting regression models for recall and precision.

## 1. Scripts To Call

Run the full pipeline with one command:

```bash
./run_llama70b_regression_pipeline.sh
```

Or run the steps manually:

Run scripts from this directory:

```bash
cd /Users/kaimingtao/HIVDB/chat_HIVDB_paper/eval/qlora
```

First build the TP/TN/FP/FN long dataframe used for the regression analyses:

```bash
uv run python prepare_llama70b_regression_data.py
```

Output:

```text
llama70b_regression_data.csv
```

Validate that the long file matches the original correctness columns in `merged_answers_with_correct.csv`:

```bash
uv run python validate_llama70b_regression_data.py
```

Expected result:

```text
Validation passed: llama70b_regression_data.csv matches merged_answers_with_correct.csv.
```

Fit recall regression:

```bash
uv run python fit_llama70b_recall_logistic.py
```

Outputs:

```text
llama70b_recall_logistic.summary.txt
llama70b_recall_logistic.odds_ratios.csv
llama70b_recall_logistic.recall_summary.csv
```

Fit precision regression:

```bash
uv run python fit_llama70b_precision_logistic.py
```

Outputs:

```text
llama70b_precision_logistic.summary.txt
llama70b_precision_logistic.odds_ratios.csv
llama70b_precision_logistic.precision_summary.csv
```

Plot the regression results with the numeric rank-trend p-value shown in a box:

```bash
uv run python plot_llama70b_regression.py
```

Output:

```text
llama70b_regression_plot.png
```

## 2. Why Use This Method

The same 2400 paper-question pairs are tested under four LoRA ranks: `R8`, `R16`, `FT(R25)`, and `R32`. Therefore, rows from the same paper-question pair are paired/repeated observations, not independent observations. Ordinary logistic regression would incorrectly treat all rows as independent and can give misleading standard errors and p-values.

The scripts use GEE logistic regression:

```python
smf.gee(
    "detected ~ rank",
    groups="item_id",
    data=recall_df,
    family=sm.families.Binomial(),
).fit()
```

GEE means generalized estimating equations. Here it is used with a binomial family and logit link, so it is a logistic regression model that accounts for correlation among repeated rows from the same paper-question pair.

For recall, the model uses only true-positive reference cases:

```text
ref_positive == 1
```

The binary outcome is:

```text
detected = 1 for TP
detected = 0 for FN
```

So the recall model estimates:

```text
P(model detects a truly positive paper-question pair | rank)
```

For precision, the model uses only predicted-positive cases:

```text
outcome in {TP, FP}
```

The binary outcome is:

```text
precise = 1 for TP
precise = 0 for FP
```

So the precision model estimates:

```text
P(predicted-positive paper-question pair is truly positive | rank)
```

Each regression script fits two versions:

```text
numeric rank trend: rank as 8, 16, 25, 32
categorical rank comparison: rank as categories with R8 as reference
```

The numeric model tests whether there is an overall linear trend with increasing rank. The categorical model avoids assuming linearity and estimates differences for `R16`, `FT(R25)`, and `R32` relative to `R8`.

Report odds ratios from the `.odds_ratios.csv` files. An odds ratio below 1 means higher rank is associated with lower odds of the target outcome. An odds ratio above 1 means higher rank is associated with higher odds of the target outcome.

## 3. Paper Discussion Text

To assess whether LoRA rank was associated with changes in Llama3.1-70B performance, we fitted GEE logistic regression models with paper-question pair as the clustering variable, using recall detection among reference-positive cases and precision among predicted-positive cases as binary outcomes. Increasing rank was associated with significantly lower recall odds (p < 0.001), while precision odds increased significantly with rank (p < 0.001). These results support the observed trade-off that higher ranks improve precision but reduce recall in the Llama3.1-70B rank series.

# Evaluation Guide

## Quickstart (reproduce all suffixed results)
From the repo root:
```bash
make -C eval
```
This runs the suffixed evaluation and learning-curve pipelines for Full 150, New 30, and Original 120, regenerating metrics, figures, and statistical test workbooks. Unsuffixed artifacts are not produced.

This folder scores model outputs against human answers and emits metrics, details, figures, and statistical tests. All outputs are suffix-specific (e.g., `full150`, `new30`, `original120`) to avoid ambiguity.

## Key scripts
- `evaluation.py`: end-to-end scoring for a merged answer sheet. Outputs per-suffix metrics CSVs, detailed evaluation CSVs, figures (`eval/figures/*_<suffix>-*.png`), and a combined stats workbook `statistical_tests_<suffix>.xlsx` (sheets: Paired Tests, Fisher Exact Test).
- `learning-curve/06_evaluate_learning_curve.py`: same stack for learning-curve runs; writes `learning_curve_metrics_<suffix>.csv`, `learning_curve_details_<suffix>.csv`, `learning_curve_summary_<suffix>.json`, `statistical_tests<suffix>.xlsx`, and significance JSON.
- `learning-curve/07_plot_learning_curve.py`: renders learning-curve figures (`learning-curve_<suffix>-bar-chart.png`/`table.png`) from the suffixed metrics/significance files.

## Core helpers
- `config.py`: paths and model groupings.
- `constants.py`: synonym maps and normalization vocab.
- `normalize.py`: canonicalization and matching utilities.
- `scoring.py`: dataset loading, per-row scoring, aggregation.
- `plots.py`: shared figure rendering (titles include dataset label derived from the suffix).

## Make targets (suffixed only)
Run from repo root: `make -C eval`
- `evaluation_full150`, `evaluation_new30`, `evaluation_original120`
- `learning_curve_full150`, `learning_curve_new30`
- `plot_learning_curve_full150`, `plot_learning_curve_new30`
All of these recompute metrics, figures, and the stats workbooks for their suffix.

## Inputs
- Merged answers: `advanced-prompting/csv/merged_answers_full_150.xlsx`, `merged_answers_new30.xlsx`, `merged_answers_original_120.xlsx`
- GPT-5 responses: `eval/gpt-5/gpt5_responses.csv`
- Learning-curve responses: `eval/learning-curve/responses/*`

## Outputs (examples)
- `eval/results/evaluation_metrics_full150.csv`
- `eval/results/detailed_evaluation_full150.csv`
- `eval/results/statistical_tests_full150.xlsx`
- `eval/figures/full150-bar-chart.png`, `full150-table.png`
- `eval/learning-curve/results/learning_curve_metrics_full150.csv`
- `eval/learning-curve/results/statistical_tests_full150.xlsx`
- `eval/learning-curve/figures/learning-curve_full150-bar-chart.png`

Use the suffix variants (`*_new30`, `*_original120`) for the other datasets. Unsuffixed artifacts have been removed and are no longer generated.***

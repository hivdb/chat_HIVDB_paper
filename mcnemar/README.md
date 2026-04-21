# McNemar experiment

This folder contains a standalone McNemar experiment for Reviewer 3.

Definition used here:
- A PMID is counted as an article-level success for a model only when all 16 question judgments for that PMID are correct.
- Each intervention (`FT`, `QSP`, `FT+QSP`) is then compared against its family `base` model with an exact McNemar test.
- `BH` correction is applied across all 9 base-vs-intervention comparisons.
- The summary table also includes paired bootstrap 95% confidence intervals for the exact-match rate delta (`target - base`).
- A second sensitivity analysis pools all `PMID × QID` pairs and repeats McNemar without QID stratification.

Figures:
- `full150-mcnemar-article-exact-bar-chart.png` shows the article-level exact-match rate for each model on the Full 150 set.
  Here, each of the 150 PMIDs contributes one binary outcome per model: `1` only if all 16 QIDs for that PMID were answered correctly, otherwise `0`.
  The bar height is therefore `(# PMIDs with all 16 QIDs correct) / 150`.
  The brackets show `BH`-adjusted exact McNemar p-values for each intervention (`FT`, `QSP`, `FT+QSP`) versus its family `base` model using those paired per-PMID binary outcomes.
- `full150-mcnemar-pooled-qid-bar-chart.png` shows the pooled correctness rate when all article-question pairs are analyzed together without QID stratification.
  Here, each `PMID × QID` pair contributes one binary outcome per model, for a total of 2400 paired observations (`150 × 16`).
  The bar height is therefore `(# correct PMID × QID pairs) / 2400`.
  The brackets show `BH`-adjusted exact McNemar p-values for each intervention versus its family `base` model using those paired pooled binary outcomes.

Tables and workbooks:
- `mcnemar_article_exact_comparisons_full150.csv` is the main article-level comparison table.
  Each row is one intervention-vs-base comparison within a model family.
  It reports the number and rate of article-level exact matches for base and target, the paired delta with bootstrap 95% confidence interval, the McNemar discordant-pair counts (`target_only`, `base_only`), the raw exact McNemar p-value, and the `BH`-adjusted p-value.
- `mcnemar_pooled_qid_comparisons_full150.csv` is the pooled `PMID × QID` sensitivity-analysis table.
  Each row is one intervention-vs-base comparison within a model family.
  It reports the number and rate of correct pooled article-question pairs for base and target, the paired delta with bootstrap 95% confidence interval, the discordant-pair counts, the raw exact McNemar p-value, and the `BH`-adjusted p-value.
- `mcnemar_article_outcomes_full150.csv` is the article-level intermediate table used to build the exact-match McNemar analysis.
  Each row is one PMID.
  It includes, for every model, the total number of correct QIDs for that PMID, the derived `Exact Match` indicator for that model, and per-comparison status columns such as `target_only`, `base_only`, `both_exact`, and `neither_exact`.
- `mcnemar_results_full150.xlsx` is the consolidated workbook version of the outputs.
  The `Article Exact Match` sheet contains the article-level comparison table.
  The `Pooled QID Pairs` sheet contains the pooled no-QID-stratification sensitivity-analysis table.
  The `Article Outcomes` sheet contains the PMID-level intermediate outcomes used for the article-level exact-match analysis.

Run from the repo root:

```bash
python mcnemar/run_mcnemar_analysis.py \
  --node-bin <path-to-node> \
  --artifact-tool-path <path-to-artifact_tool.mjs>
```

The Python analysis is repo-local, but the workbook export step depends on a Node runtime and the `@oai/artifact-tool` module available in the local environment. Those paths should be supplied by whoever reruns the pipeline in their own setup rather than hard-coded in the repository.

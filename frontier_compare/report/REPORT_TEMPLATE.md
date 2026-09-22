# Frontier LLMs vs GPT-4o on HIVDB extraction: short report

*Status: template. Fill in from `results/` and `figures/` once all runs are complete.*

## Setup (1 paragraph)
- Models: GPT-6 Astra (`gpt-6-astra`, PDF input, run dates); Kimi K3 (`moonshotai/kimi-k3`, Moonshot AI host, page images + text layer, run dates).
- Input: full-text PDFs for N/150 papers (list exclusions). Prompt: paper QSP (Nov17_Version1) with a JSON output contract.
- Comparators: cached GPT-4o (gpt-4o-mini-2024-07-18) FT, QSP, and FT+QSP, re-scored on the same PMIDs with the same scorer.
- Replicates: R runs per model.

## Q1. Do frontier prompted models exceed GPT-4o?
- Figure: `figures/figure4_frontier_pooled.png` (pooled over all paper × question pairs, as in the paper).
- Table: pooled accuracy/precision/recall/F1 with 95% row-bootstrap CIs (`results/metrics_summary.csv`).
- Tests: Wilcoxon over the 16 per-QID values vs each GPT-4o condition, BH-adjusted; McNemar on rows as a sensitivity check (`results/pairwise_tests.csv`).
- Answer: yes/no per model and metric, with effect size and CI, not only the p-value.

## Q2. On which questions?
- Per-QID deltas vs GPT-4o FT (`results/secondary_by_qid_vs_comparator.csv`), fixed/broken counts (`secondary_flips_vs_comparator.csv`).
- By type: Boolean / List / Number (`results/metrics_by_type.csv`).
- Previously hard questions: QID 8 (cloning), 15 (drug classes), 16 (drugs).

## Q3. Are the remaining errors scientifically meaningful?
- Outcome mix: FP / FN_missed / FN_wrong_value / partial lists (`results/secondary_error_outcomes.csv`).
- Failure modes from the curator-labeled sheet (`failure_modes/labeling_sheet.csv`, taxonomy in `failure_modes/taxonomy.md`): counts by mode and model, and the share that would corrupt an HIVDB record.
- Annotation errors found (reported separately, not rescored).
- Did PDF access help? Share of correct answers whose `EvidenceLocation` is table/figure, and remaining `MISSED_TABLE_FIG` errors.

## Q4. Autonomous entry or curator-assisted review?
- Decision criteria (agree before looking at results): e.g. autonomous entry requires >= X% precision on every QID that writes to HIVDB and zero `UNSUPPORTED`/`FABRICATED_EVIDENCE` errors on identity fields (accessions, drugs); otherwise curator-assisted.
- Operational fit: cost per paper, latency, invalid-JSON rate, run-to-run agreement (`results/secondary_operational.csv`).
- Recommendation per question group.

## Caveats
- Input modality differs from the cached GPT-4o runs (PDF vs markdown), so the headline comparison mixes model and input effects. See the markdown-ablation arm, if run.
- The human annotations were made against the paper versions HIVDB curators used; supplements may not be in the PDF.
- Frontier models may have seen these papers (published <= 2025) in pretraining. That leakage risk applies to all models but is larger for newer ones.

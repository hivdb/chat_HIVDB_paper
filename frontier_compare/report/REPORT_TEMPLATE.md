# Frontier LLMs vs GPT-4o on HIVDB extraction: short report

*Status: template. Fill in from `results/` and `figures/` once all runs are complete.*

## Setup (1 paragraph)
- Models: GPT-6 Astra (`gpt-6-astra`, PDF input, run dates); Kimi K3 (`moonshotai/kimi-k3`, Moonshot AI host, page images + text layer, run dates).
- Input: full-text PDFs for N/150 papers (list exclusions). Prompt: paper QSP (Nov17_Version1) with a JSON output contract.
- Comparators: cached GPT-4o (gpt-4o-mini-2024-07-18) FT, QSP, and FT+QSP, re-scored on the same PMIDs with the same scorer.
- One run per paper per model (no replicates; run-to-run stability not assessed).

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
- Operational fit: cost per paper, latency, invalid-JSON rate (`results/secondary_operational.csv`).
- Recommendation per question group.

## Limitations of prompting without curator guidelines

The QSP prompt is the paper's, unchanged, for every model. Two curator conventions are not in it,
and no prompted model can infer them. Both were measured on the 40-paper pilot:

- **QID 5 (how many individuals):** the annotations count individuals whose sequencing *succeeded
  and is reported in this paper*; the models count individuals *sampled*, which the question text
  equally supports. 16 error rows. A one-line prompt clause raised QID 5 from 5/10 to 7/10 on the
  first 10 papers but did not separate from the base prompt at 39 papers; it was not adopted, and
  no prompt variant is part of the study.
- **QID 10 (sequencing method):** the annotations default to "Sanger" for standard genotypic
  resistance testing even where the paper never says so - one annotation reads "Sanger (not
  stated)". Handled in scoring instead: where the PDF never mentions Sanger, "Not reported" is
  accepted for every model.

This is an argument about what a fine-tuned model buys: GPT-4o FT learned these conventions from
the training annotations, while a prompted model can only follow what the prompt says. State it
as a limitation of the comparison, and as an argument for curator-assisted review over autonomous
entry until such conventions are written down.

## Post-processing

Two layers sit between the model's answer and the score, both applied to every model and both
non-destructive (raw answer scored first):
1. **Answer cleaning** (`answer_cleaning.py`): removes explanatory scaffolding. Worth 3 rows on
   the pilot, all Kimi K3; zero for Astra and GPT-4o. Report the count per model - it is an
   artifact of answer style, not extraction quality.
2. **Accepted alternatives**: see below.

## Annotation quality

Accepted alternative answers (`data/accepted_alternatives.csv`, applied to every model) cover rows
where the annotation is wrong or incomplete. The primary metrics include them; the unadjusted
score is reported alongside as `pooled_strict`. Report:
- how many rows were adjusted and why (review paper, figure-only evidence, curation convention),
- that the effect is asymmetric: it raises frontier models more than GPT-4o, because GPT-4o
  sometimes matched a wrong annotation by producing plausible detail (e.g. 12/16 on the
  mis-annotated review paper PMID 40872801, vs 8/16 for the frontier models).

## Caveats
- Input modality differs from the cached GPT-4o runs (PDF vs markdown), so the headline comparison mixes model and input effects. See the markdown-ablation arm, if run.
- The human annotations were made against the paper versions HIVDB curators used; supplements may not be in the PDF.
- Frontier models may have seen these papers (published <= 2025) in pretraining. That leakage risk applies to all models but is larger for newer ones.

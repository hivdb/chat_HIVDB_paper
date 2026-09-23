# Frontier LLM comparison (subproject)

**Objective.** Test whether current frontier LLMs using question-specific prompting (QSP) can
match or beat the cached GPT-4o results (FT, QSP, FT+QSP) on the 16 HIVDB extraction questions.

| | |
|---|---|
| Models | GPT-6 Astra (`gpt-6-astra`, closed, OpenAI API); Kimi K3 (`moonshotai/kimi-k3`, open weights, OpenRouter, first-party Moonshot AI host pinned). Qwen3.8 2.4T A95B was dropped because OpenRouter lists it as text-only. |
| Eval set | Full 150 held-out papers (original 120 + new 30) with the paper's human annotations |
| Protocol | Same as the paper ([PLOS ONE, Methods](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0351631#sec006)), **except** the model gets the full-text **PDF** instead of the markdown conversion, so tables and figures are available |
| Comparators | Cached GPT-4o FT, QSP, FT+QSP from `advanced-prompting/csv/merged_answers_full_150.xlsx`. "GPT-4o" in the paper is `gpt-4o-mini-2024-07-18`. |
| Scoring | The paper's scorer (`eval/normalize.py::human_answer_counts`), unchanged. It reproduces the paper's per-QID metrics exactly (verified: max abs diff 0.0 for all three GPT-4o conditions). |

## Pipeline

Run from the repo root with the project venv (`.venv/bin/python` or `uv run`):

```bash
python frontier_compare/00_smoke_test.py --real-pmid 40596906        # can each model see text, tables, figures?
python frontier_compare/01a_fetch_pdfs.py                              # open-access PDFs: PMC OA on S3, then publisher
python frontier_compare/01_pdf_manifest.py --stage                    # inventory and stage PDFs -> pdfs/<PMID>.pdf
python frontier_compare/02_query_models.py --model gpt6-astra --run 1 --dry-run --limit 3
python frontier_compare/02_query_models.py --model gpt6-astra --run 1  # one run per paper
python frontier_compare/02_query_models.py --model kimi-k3    --run 1
python frontier_compare/03_parse_responses.py                          # raw JSONL -> answers + ops table
python frontier_compare/04_evaluate.py                                 # metrics and tests (paper's evaluation + answer cleaning)
python frontier_compare/05_figure4.py                                  # updated Figure 4, paper style
python frontier_compare/06_secondary.py                                # operations table, error analysis, failure-mode sheet
python frontier_compare/07_detailed_evaluation.py                      # per-row workbook, as in the paper
python frontier_compare/08_error_triage.py                             # evidence-based signals per error
python frontier_compare/11_auto_verdicts.py                            # rule-based adjudication -> data/adjudication*.csv
python frontier_compare/10_adjudicate.py --apply                       # adjudicated errors, summary and tests
python frontier_compare/12_table3.py                                   # the paper's Table 3, by question and by type
```

### Evaluation

One evaluation: the paper's scorer (`eval/normalize.py::human_answer_counts`) on all 150 papers,
with answer cleaning as part of the evaluator (`answer_cleaning.py`: when a raw answer fails, it
is re-scored with explanatory commentary stripped; it can rescue a row, never break one). GPT-6
Astra's blocked request for PMID 36920025 is scored as 16 blank answers, as the paper scores
missing answers. Cleaning runs for every model but only changes frontier rows (Astra 1, Kimi 5),
so **the cached GPT-4o numbers reproduce the paper exactly**: pooled metrics and their bootstrap
CIs equal `eval/figures/full150-bar-chart-confidence-intervals.csv` (the bootstrap draws follow
the paper's model order: GPT-4o base, FT, FT+QSP, QSP), and the GPT-4o columns of
`results/detailed_evaluation.xlsx` equal `eval/results/detailed_evaluation_full150.xlsx`.

Annotation problems (review papers, the QID 10 Sanger default, figure-only evidence, data-entry
errors) do not change the scores; they are handled in the error adjudication (steps 08-11).

### Where the outputs are

`results/` holds only the final, reader-facing outputs:

| File | Contents |
|---|---|
| `detailed_evaluation.xlsx` | Sheet "All": every PMID × QID row with the human answer and each model's answer and 1/0 correctness, in the layout of `eval/results/detailed_evaluation_full150.xlsx`. "Answer cleaning": the 6 rows cleaning rescued. |
| `metrics_summary.csv` | Pooled accuracy / precision / recall / F1 per model with 95% bootstrap CIs (the Figure 4 numbers), the value before cleaning, and the per-QID mean for reference |
| `metrics_by_qid.csv` | The same four metrics per question, with TP/FP/TN/FN counts |
| `statistical_tests.csv` | Paired tests, in BH families by `comparison_set`: `figure4` = every model vs GPT-4o QSP (the figure's brackets); `frontier` = each frontier model vs each GPT-4o condition; `frontier_adjudicated` = the same after removing non-model errors |
| `table3.xlsx` | The paper's Table 3 for this comparison: per-question precision and recall vs GPT-4o QSP with Fisher exact tests (BH within each question), for the questions where a model improves significantly ("Table 3"), every other question ("Table 3 complement", significant declines marked † / ‡), and the same test on counts pooled by question type. `12_table3.py --validate` reproduces the paper's current Table 3 exactly. |
| `adjudicated_errors.csv` | Every error of every model with its adjudication verdict |
| `adjudicated_summary.csv` | Per-model error breakdown and adjusted accuracy |
| `operations.csv` | Per-model cost, latency, JSON validity, failures |

`figures/figure4_frontier.png` is the updated Figure 4. The written report is `report/frontier-model-evals-09-23-26.docx`
(edited by hand). Everything else is a pipeline
intermediate in `work/`: parsed answers (`work/answers/`), per-row scores with every
post-processing column (`work/detailed_rows.csv`), per-request operations, error triage,
evidence dossiers, and the adjudication worksheets.

| Step | Output |
|---|---|
| 01 | `data/pdf_manifest.csv` (status, pages, size, and hash per PMID) |
| 02 | `runs/<model>/run<N>.jsonl`: raw response, latency, usage, and cost per request (resumable) |
| 03 | `work/answers/<model>_run<N>.csv`, `work/ops_requests.csv` |
| 04 | `results/metrics_summary.csv`, `metrics_by_qid.csv`, `statistical_tests.csv`; `work/detailed_rows.csv` (labels and correctness before and after cleaning), `work/metrics_by_type.csv` |
| 05 | `figures/figure4_frontier.png` |
| 06 | `results/operations.csv`, `work/secondary_*.csv`, `failure_modes/labeling_sheet.csv` |
| 07 | `results/detailed_evaluation.xlsx` |
| 12 | `results/table3.xlsx` |
| 08-11 | `work/error_triage.csv`, `work/dossier_*`, `work/adjudication_worksheet.*`, `results/adjudicated_*.csv` |

## Smoke test (2026-09-21)

`00_smoke_test.py` builds a one-page synthetic PDF whose facts are split across body text, a
ruled text table, and a **raster** bar chart. The chart's values, colours and a corner code
exist only as pixels (confirmed absent from the text layer). It also probes a real eval paper
(PMID 40596906). Results were checked by hand against the rendered pages.

| | GPT-6 Astra (PDF file input) | Kimi K3 (page images + text layer) |
|---|---|---|
| Synthetic text / table / figure checks | 7/7, 7 s | 7/7, 16 s |
| Real paper: Table 1 first row (page 6) | correct | correct |
| Real paper: Figure 1 values (EFV 10/8, ABC 6/6), colours | correct | correct; also spotted the axis label "AZT" vs "ZDV" in the caption |
| Real-paper latency | 13 s | 59 s |

Rejected input path for Kimi: PDF file + OpenRouter `file-parser` (mistral-ocr) + page images.
With the plugin active, only page 1 of 10 images reached the model. It missed Table 1 and
spent 13k reasoning tokens (417 s, $0.26) searching for it. Kimi therefore gets every page
rendered at 110 dpi plus the PDF's own text layer, extracted locally with PyMuPDF.

## PDF acquisition

`01a_fetch_pdfs.py` maps PMIDs to PMCID/DOI (NCBI ID converter; PubMed for DOIs), then downloads
from the **PMC Open Access dataset on AWS S3**, falling back to the publisher PDF link registered
with Crossref. PMC and Europe PMC web PDF links sit behind bot challenges and are not used.
Every file is checked for a `%PDF` header. The script fetched 104/150 (all 30 new + 74 of the
original 120); the remaining 46 were downloaded by hand from PubMed/PMC, because PMC's PDF link
serves a JavaScript bot challenge that a script cannot (and should not) pass.

**All 150 PDFs are now staged and verified**: every file opens, yields extractable text, and its
PubMed title appears in the first two pages (8 initial mismatches were ligature/prime/"Brief
Report:" artifacts, checked by hand). 1,606 pages total. Short PDFs (2–4 pages) are complete
research letters or case reports.

## Results (full run, 150 papers, 2026-09-22)

| | accuracy (95% CI) | precision | recall | F1 |
|---|---|---|---|---|
| Kimi K3 | 92.2 (91.1-93.3) | 91.5 | 90.4 | 91.0 |
| GPT-6 Astra | 91.0 (89.8-92.1) | 91.2 | 87.7 | 89.4 |
| GPT-4o FT | 90.3 (89.1-91.5) | 92.1 | 84.9 | 88.3 |
| GPT-4o FT+QSP | 88.0 (86.7-89.3) | 88.3 | 83.4 | 85.8 |
| GPT-4o QSP | 84.5 (83.1-85.9) | 83.5 | 80.1 | 81.7 |

Against GPT-4o QSP (Figure 4, BH-adjusted Wilcoxon), both frontier models are significantly
better on accuracy and precision, and Kimi K3 also on F1; no model differs significantly on
recall, because the per-question test weights all 16 questions equally and GPT-4o QSP's
over-answering of Yes gives it higher recall on several small Boolean questions (Table 3 shows
the per-question recall gains on list questions). Kimi K3 also beats GPT-4o FT+QSP on accuracy
(p = 0.017). Neither frontier model is distinguishable from GPT-4o FT (Kimi p = 0.11, Astra
p = 0.96). Operationally: Astra $0.378/paper, 53 s median, 99% strict JSON, 1 policy block
(HTTP 400, code `bio_policy`, "flagged for possible biological risk", on both requests for
PMID 36920025; an exploratory prompt-variant request for the same paper succeeded, so the block
is intermittent or prompt-dependent); Kimi $0.187/paper, 133 s median, 95% strict JSON, 1 empty
response (retried). Total spend $84.30.

## Adjudication of every error, every model

`09_error_dossier.py` builds a per-row evidence dossier (the model's quote checked verbatim
against the PDF, PDF context around both answers, the other models' answers, self-consistency).
`10_adjudicate.py` consolidates the error rows into distinct (PMID, QID) rows and
applies verdicts in two layers - annotation soundness first (model-independent), then per-model
classification. `11_auto_verdicts.py` applies the explicit rules:

| rule | effect |
|---|---|
| R1 type mismatch (Boolean annotated non-yes/no, Number with no number) | unanswerable |
| R2 annotation hedges ("not stated", "uncertain", "(multicenter trial)", "8 and 10") | ambiguous |
| R3 paper is a review / meta-analysis, from the list verified by reading (`data/review_papers.csv`) | annotation wrong |
| R4 QID 10 annotated Sanger where the PDF never says Sanger | convention |
| R5 all 5 models miss the row, incl. the fine-tuned GPT-4o | ambiguous |
| R6 QID 5 where both the annotation's count and the model's count are stated in the paper | convention |
| M1 annotation empty but the model's answer is in the PDF text | borderline |
| M2 the model's own QID 1 answer was wrong (cascade) | model error |

Everything else defaults to **model error**. `data/adjudication.csv` is the manual verdicts
(`data/adjudication_manual.csv`) laid over the rule-based ones; `11_auto_verdicts.py` writes it.
Results:

| | errors | model error | annotation/convention | borderline | accuracy | excl. annotation | excl. all non-model |
|---|---|---|---|---|---|---|---|
| Kimi K3 | 187 | 88 (47%) | 71 | 28 | 0.922 | 0.952 | 0.963 |
| GPT-6 Astra | 216 | 118 (55%) | 81 | 17 | 0.910 | 0.944 | 0.951 |
| GPT-4o FT | 233 | 139 (60%) | 76 | 18 | 0.903 | 0.935 | 0.942 |
| GPT-4o FT+QSP | 288 | 182 (63%) | 70 | 36 | 0.880 | 0.909 | 0.924 |
| GPT-4o QSP | 372 | 266 (72%) | 67 | 39 | 0.845 | 0.873 | 0.889 |

GPT-6 Astra's 4 wrong blank answers for the blocked paper count as model errors. The ranking
survives adjudication and Kimi vs GPT-4o FT stays non-significant (BH p = 0.088,
`comparison_set = frontier_adjudicated` in `results/statistical_tests.csv`).

**R3 was corrected on 2026-09-23.** Its first version matched "systematic review",
"meta-analysis" or "PRISMA" anywhere in the first 6,000 characters and fired on nine papers,
five of which are primary studies that merely cite a meta-analysis (37976080, 37976185,
40596906, 41057785, 41129268). Reading all nine found four real reviews (37880705, 37910452,
40872801, 41140464), now listed with evidence in `data/review_papers.csv`.

**How much to trust these splits.** The rules are deliberately conservative: anything not matched
by a rule counts as a model error. Hand-adjudicating GPT-6 Astra's 99 errors on the 79-paper
subset gave 44% model errors, where the rules give 57%. So the rule-based numbers **under-count**
annotation problems, and the "excl. annotation" columns are lower bounds on how much of each
model's error rate is really the dataset. Manual verdicts for rows read individually are in
`data/adjudication_manual.csv`; a sample of the rule-assigned rows was read to check the default.

### Earlier hand-adjudication (GPT-6 Astra, 79-paper subset)

`09_error_dossier.py` builds an evidence dossier per error row - the model's answer, evidence
quote and rationale, whether that quote is verbatim in the PDF, PDF context around both the human
and model answers, what the other four models said, and whether the model contradicted its own
QID 1/5. All 99 Astra error rows were then read and adjudicated one by one; verdicts and reasons
are in `work/dossier_gpt6-astra.csv` and `results/adjudicated_errors.csv`.

| Verdict | rows | share |
|---|---|---|
| genuine model error | 44 | 44% |
| annotation error or unstated curation convention | 38 | 38% |
| borderline (both readings defensible) | 16 | 16% |
| scoring artifact | 1 | 1% |

**So roughly half of Astra's "errors" are not extraction failures.** Accuracy on the 79-paper
pilot moves from 0.9217 as scored, to 0.9517 if the annotation/convention rows are counted
correct, to 0.9652 if the borderline rows are allowed too. The residual genuine-error rate is
**3.5%** (44/1264).

Where each kind sits:
- **Annotation/convention rows cluster at QID 5** (12 of 38) - the sampled-vs-sequenced counting
  convention - then QID 9 (5) and QID 8 (4).
- **Genuine model errors cluster at QID 15/16** (10 of 44: incomplete or over-inclusive ARV lists)
  and QID 5 (5: wrong denominator).
- **15 of the 44 genuine errors are cascades from just 3 papers**, where Astra's own QID 1 answer
  ("does this paper report patient sequences?") was wrong and it then answered "Not applicable" -
  or over-extracted - for every downstream question. Astra's QID 1 judgement is the single
  highest-leverage failure: 2 wrong QID 1 answers cost 15 further rows.
- **Borderline rows are inference-from-context** (country from an ethics committee or institution,
  sample type from routine genotyping) and in-vitro-data judgement calls at QID 2.

## Annotation issues and evaluation artifacts (40-paper pilot)

`08_error_triage.py` classifies every frontier error using signals checkable against the PDF:
does the model's quoted evidence actually occur in the text layer, how much of the human answer
occurs anywhere in the PDF, do both models agree, is the paper a review, is the annotation itself
hedged. Output: `work/error_triage.csv`.
Suggestions only - nothing is applied to scoring.

| Suggested cause | rows (of 110) |
|---|---|
| model error / needs review | 53 |
| annotation error: paper is a review/meta-analysis | 16 |
| curation convention: which individuals to count (QID 5) | 16 |
| model error: quoted evidence not found in PDF | 9 |
| annotation hedge: human answer explicitly uncertain | 5 |
| needs review: evidence cited from a figure/table | 5 |
| curation convention: sequencing method defaulted (QID 10) | 4 |
| scoring artifact: cleanup rule fixes it | 2 |

**PMID 40872801 is mis-annotated.** The PDF is a PRISMA systematic review and meta-analysis of
Tanzanian studies; the QSP rules say reviews get "No"/"None". The annotation instead records
primary-study details ("Sanger", "Plasma") whose words appear **zero** times in the PDF. It costs
every frontier model 8 of 16 rows. Note GPT-4o QSP/FT+QSP score 12/16 on this paper: producing
plausible detail matches a wrong annotation, so the error rewards the weaker behaviour.

**Out-of-scope evidence on lab-only papers (47 rows across all five models; 17 frontier).** Both frontier models sometimes
answer QID 9/10 from text describing *laboratory constructs* rather than patient samples. Example:
PMID 30803972 - both models correctly answer "No" to QID 1 and "0" to QID 5, then cite a table
footnote ("The integrase-encoding region of each plasmid clone was confirmed by automated Sanger
DNA sequencing") for QID 10. The quote is real and the models read the table correctly; it just
describes pNL4-3/pROD9 clones, not patient material. This is the lab-vs-clinical failure mode, and
a counter-example to "PDF access always helps": the extra table text caused the error. It is not
frontier-specific - GPT-4o FT answers "IN" on the same row, and the failure is in fact *more*
common for the cached comparators (30 rows) than for the frontier models (17), because
`08_error_triage.py` now triages every model, not just the frontier ones (evidence-based signals
are blank for the cached answers, which have no stored evidence).

A consistency rule (force "Not reported" on QID 4/6/7/9/10/11 when the model itself said there
were no patient sequences) was measured and **rejected**: Astra +6/-17, Kimi +8/-16,
GPT-4o FT +4/-23. The reason is the annotations themselves: of the 26 pilot papers annotated
"no patient sequences" (QID 1 = No), 9 still record genes for QID 9 and 5 record a method for
QID 10 - i.e. curators sometimes do record lab-construct sequencing details. On those papers
every model scores worse on QID 9/10 (Astra 85%, Kimi 83%, GPT-4o FT 75%) than on the rest
(95%, 95%, 83%). Report this as an annotation-consistency limitation, not a model failure alone.

**Rows every model gets wrong (22 distinct rows in the pilot).** When all five models - including
the GPT-4o fine-tuned on these very annotations - disagree with the annotation, the question or
the annotation is usually the problem, not the models. `08_error_triage.py` flags these as
`annotation ambiguous`. Example: PMID 31988104 (Gilead in-vitro study) tests site-directed mutants
**and** 14 "patient-derived mutants ... cloned from clinical plasma samples" and reports their TAM
profiles. The QSP rule says answer Yes when a paper "reported lists of mutations, directly from
clinical samples", and No only if *only* lab strains/SDMs were studied - so every model answered
Yes. The curators applied a stricter standard (were new patient sequences generated and reported?)
and answered No, while still recording "RT" for QID 9 on the same paper. This is a third unstated
convention: what counts as "reporting sequences".

**Annotation data-entry errors.** A type check over all 150 papers x 16 questions found 3:
QID 14 (Boolean) annotated "14" for PMID 31988104; QID 8 (Boolean) annotated "Not known" for
PMID 41140464; QID 5 (Number) annotated "Not Reported" for PMID 28559249. The first is
unanswerable as scored - no model can produce "14" for a yes/no question.

**Two curator conventions the prompt never states**, both systematic rather than capability gaps:
- QID 5: annotations count individuals *successfully sequenced*; the models count individuals
  *sampled* (16 rows).
- QID 10: annotations default to "Sanger" for standard genotypic resistance testing even when the
  paper never says so - one annotation literally reads "Sanger (not stated)". None of the three
  papers involved contains the word "Sanger" (4 rows).

**Evaluation artifacts are small.** Answer cleaning (below) recovers **3 rows in total, all
Kimi's** - caveats like "(paper states either Sanger sequencing or NGS)" that the
scorer reads as a negation. Astra and all GPT-4o conditions gain nothing. A destructive cleanup
(rewriting answers before scoring) is worse than useless: it gains 3 and loses 3, because stripping
a parenthetical sometimes removes the text the scorer was matching. Near-miss list answers are
*more* common for GPT-4o (23) than for Astra (5) or Kimi (8), so any leniency must be applied to
all models or it flatters the comparator.

**Sensitivity of the headline numbers** (accuracy on the 39-paper pilot):

| | all rows | excl. 40872801 | excl. review + conventions + hedges |
|---|---|---|---|
| Kimi K3 | 0.918 | 0.929 | 0.946 |
| Astra (Q5 v2) | 0.915 | 0.924 | 0.943 |
| GPT-6 Astra | 0.905 | 0.916 | 0.939 |
| GPT-4o FT | 0.886 | 0.895 | 0.912 |
| GPT-4o FT+QSP | 0.877 | 0.880 | 0.893 |
| GPT-4o QSP | 0.841 | 0.844 | 0.856 |

Every model gains and the ranking is unchanged, so these issues do not create the frontier lead -
but they do compress it, and roughly a quarter of all "errors" are not model errors.

**Decisions taken after the pilot review (2026-09-22):**
- The 10 proposed alternatives were approved and moved into `accepted_alternatives.csv` (since removed from scoring; see "Annotation problems").
- For PMID 40872801 both the annotated answers and the models' "No"/"Not applicable" now count.
- The QID 10 Sanger default is handled in scoring (`convention_alternatives()` in
  `04_evaluate.py`): where the human answer says Sanger and the PDF never mentions it,
  "Not reported" is accepted too - for every model.
- Accepted answers now feed the **primary** metrics; the unadjusted score is kept as
  `pooled_strict` / `<model> correct_strict`.
- **No prompt variants.** The protocol keeps the paper's prompt unchanged for every model. The
  pilot's effort and Q5-wording probes are archived in `runs/_variants_archive/` and are not part
  of the study; the conventions they exposed are reported as limitations instead.

Pilot results with accepted answers applied (39 papers, 624 rows):

| | accuracy | precision | recall | F1 | before cleaning + accepted answers |
|---|---|---|---|---|---|
| Kimi K3 | 0.942 | 0.945 | 0.932 | 0.938 | 0.918 |
| GPT-6 Astra | 0.925 | 0.922 | 0.918 | 0.920 | 0.905 |
| GPT-4o FT | 0.894 | 0.942 | 0.827 | 0.880 | 0.886 |
| GPT-4o FT+QSP | 0.886 | 0.915 | 0.837 | 0.874 | 0.877 |
| GPT-4o QSP | 0.849 | 0.876 | 0.793 | 0.832 | 0.841 |

## Q5 counting convention (pilot follow-up)

Astra's question-5 errors were a convention mismatch, not a capability gap: it counts individuals
**sampled**, the annotations count individuals **sequenced**. `prompts/q5_convention*.md` replace
only the Question 5 block of the QSP prompt (`ModelSpec.prompt_variant`); every other word is the
paper's. Measured on the 10 pilot papers:

| Astra variant | QID 5 correct | total /160 | precision | recall |
|---|---|---|---|---|
| original prompt | 5/10 | 143 | 0.918 | 0.886 |
| `q5_convention` (+2 bullets) | 7/10 | 144 | 0.962 | 0.852 |
| `q5_convention_v2` (+1 bullet) | 7/10 | 144 | 0.919 | 0.898 |

v1's second bullet ("exclude groups reported elsewhere") backfired on PMID 37340869: Astra answered
0 for QID 5 and then cascaded "Not applicable" through QID 1/12/14/15/16, losing 5 correct rows.
**v2 is the version to consider adopting.** Two caveats: the edit was derived from rows already
scored (fitting to the test set), and the +1 total is within single-run variation — two of the 13
changed rows are reworded list answers unrelated to QID 5, since temperature is left at the API
default. Only the QID 5 effect is systematic.

Adopting it for the headline comparison would mean the frontier models get a prompt the cached
GPT-4o runs never saw. Options: keep the original prompt for the primary result and report v2 as a
prompt-sensitivity analysis, or re-run everything on v2 (a second full pass, ~$95).

## Answer cleaning (post-processing)

`answer_cleaning.py` separates the answer proper from explanatory scaffolding the scorer would
otherwise read as hedging: a preamble ("The paper reports that: X"), a trailing note ("X; the
paper does not specify which"), or a commentary parenthetical ("X (paper states either X or Y)").
Abbreviations such as "(3TC)" are preserved - a parenthetical is only dropped when it is wordy
(>=4 words) or contains a hedge word.

It is part of the evaluator and **non-destructive at scoring time**: `04_evaluate.py` scores the raw answer first and only
falls back to the cleaned form, so a rule can rescue a row but never break one. The rule that
fired is recorded per row (`<model> cleaning_rule`, outcome `correct_after_cleaning`). The same
rules run for every model, including the cached GPT-4o comparators.

Effect on the full run: 6 rows (Kimi K3 5, GPT-6 Astra 1, GPT-4o none). An earlier destructive variant (rewriting every
answer before scoring) was rejected: it gained 3 rows and lost 3, because stripping a
parenthetical sometimes removes the text the scorer was matching.

## Annotation problems (handled in adjudication, not in the scores)

An earlier version of the analysis accepted curator-approved alternative answers in the scores.
They are now recorded as adjudication verdicts instead, so the scores stay the paper's:

- **Review papers annotated with primary-study details**: R3 with `data/review_papers.csv`
  (40872801, 41140464, 37880705; 37910452 is annotated consistently), plus a manual verdict for
  37880705 QID 1, which R3 does not cover.
- **QID 10 Sanger default**: R4.
- **41130593 QID 7** (years only in the Figure 3 footnotes) and **41091504 QID 8** (computational
  paper annotated "Yes (site-directed mutants)"): manual verdicts in `data/adjudication_manual.csv`.

## Design decisions

- **One request per paper** answers all 16 questions, as in the paper's QSP runs. The system prompt
  is `advanced-prompting/md/Nov17_Version1.md` verbatim. Only its final "format your answer"
  section is replaced with a JSON contract that keeps the same fields (question, evidence,
  rationale, answer) and adds `evidence_location` (main_text / table / figure / supplement),
  which feeds the missed-table/figure analysis.
- **Input by model.** GPT-6 Astra takes the PDF natively (OpenAI extracts text and page images).
  Kimi K3 has no native PDF ingestion, so it gets page images plus the text layer (see Smoke test).
- **No enforced JSON schema.** Structured-output mode would push the invalid-JSON rate to ~0 by
  construction, so the model is only *asked* for JSON. JSON validity is logged at two levels:
  `strict` (parses as-is) and `lenient` (parses after stripping fences and prose). Answers use the
  lenient parse. Unparseable responses score as wrong, as missing answers did in the paper.
- **Aggregation = the paper's Figure 4.** Bars pool all PMID × QID rows, with 95% CIs from a
  row bootstrap (5,000 resamples, seed 42, one generator drawn in the paper's model order).
  This is what `eval/evaluation.py` plots and what reproduces the paper's reported deltas. The
  mean of the per-QID values does not reproduce them, so it is kept only as a reference column.
  The cached GPT-4o values and CIs match the paper exactly.
- **Tests = the paper's Fig. 4 statistics.** Wilcoxon signed-rank (primary, as stated in the
  paper) and paired t-test (also in S5) over the 16 per-QID values, with BH applied within each
  (comparison set, metric, test) slice as in `eval/statistics.py`. Exact McNemar on row correctness is a
  sensitivity analysis.
- **Base comparator in Figure 4** is GPT-4o QSP: the same prompt the frontier models get, so the
  brackets answer "what does a frontier model add over prompted GPT-4o?". GPT-4o FT and FT+QSP are
  tested against the same base, mirroring the paper's figure where each family's variants are
  compared with its base model. Frontier vs GPT-4o FT (the strongest cached condition) is in the
  `frontier` comparison set and in the report.
- **Undefined precision/recall** (no positives) is set to 0, matching `eval/evaluation.py`.
- **Reproducibility.** Kimi K3 is pinned to the first-party Moonshot AI endpoint with no
  fallbacks, because OpenRouter otherwise routes across about 20 hosts with fp4/fp8/bf16
  quantizations. The serving provider is logged per request.

## Open issues

1. **Confound: model vs input.** The cached GPT-4o runs used markdown, so frontier-vs-GPT-4o
   differences mix model and modality effects (41130593 QID 7 is a clear case). A cheap ablation
   is to run one frontier model on the same markdown (`<PMID>.checked.md`) to separate the two.
2. **Supplements.** Some human answers (accessions, long drug lists) come from supplementary
   files that are not in the PDFs.
3. **Annotation fixes to report to the curators**: the four review papers, the data-entry errors
   (31988104 QID 14 "14", 41140464 QID 8 "Not known", 28559249 QID 5 "Not Reported"), and the
   unstated conventions (QID 5 denominator, QID 10 Sanger default). Details in the report, section 2.
4. **R6 (QID 5 denominator) is generous**: it excuses any larger model count that appears in the
   paper, e.g. GPT-6 Astra's 5357 on 33855437 (annotated 6).
5. **Contamination.** All 150 papers predate the frontier models' training cutoffs. The new-30
   subset (2025 papers) is the least exposed.
6. **GPT-6 Astra pricing** is not returned by the API, so cost is computed from token usage at the
   published rates ($10 / $1 cached / $50 per 1M tokens).

## Deliverable

`report/frontier-model-evals-09-23-26.docx` answers the four study questions: do frontier models
beat GPT-4o, on which questions, are the remaining errors scientifically meaningful, and is
performance adequate for autonomous HIVDB entry or only curator-assisted review.

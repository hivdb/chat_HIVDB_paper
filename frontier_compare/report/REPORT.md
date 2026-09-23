# Frontier LLMs vs GPT-4o on HIVDB extraction: results

*Run 2026-09-21/22. All numbers reproducible from `frontier_compare/` (see README for the pipeline).*

## Setup

| | |
|---|---|
| Models | **GPT-6 Astra** (`gpt-6-astra`, PDF input) and **Kimi K3** (`moonshotai/kimi-k3`, Moonshot AI host, page images + PDF text layer), one run per paper |
| Prompt | The paper's question-specific prompt (`Nov17_Version1.md`) **unchanged**; only the output-format section is replaced by a JSON contract. No prompt variants. |
| Input | Full-text **PDFs** for all 150 papers (the paper used markdown conversions) |
| Comparators | Cached GPT-4o (`gpt-4o-mini-2024-07-18`) FT, QSP, FT+QSP, re-scored on the same rows |
| Scored | 149 papers x 16 questions = **2,384 rows per model**. PMID 36920025 is excluded: GPT-6 Astra refuses it on policy grounds. |
| Scoring | The paper's own scorer, plus two non-destructive post-processing layers (answer cleaning; curator-accepted alternatives) applied identically to every model |

## Q1. Do frontier prompted models exceed GPT-4o?

**Yes on every metric, but the margin over the fine-tuned GPT-4o is not statistically significant.**

| | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| **Kimi K3 QSP** | **0.927** | 0.917 | **0.914** | **0.915** |
| **GPT-6 Astra QSP** | 0.917 | 0.914 | 0.892 | 0.903 |
| GPT-4o FT | 0.905 | **0.921** | 0.854 | 0.886 |
| GPT-4o FT+QSP | 0.883 | 0.883 | 0.841 | 0.862 |
| GPT-4o QSP | 0.846 | 0.835 | 0.806 | 0.820 |

Wilcoxon over the 16 per-question values, BH-adjusted (`results/pairwise_tests.csv`):

| comparison | accuracy | F1 |
|---|---|---|
| Kimi K3 vs GPT-4o **FT** | +0.022, 10 wins / 5 losses, p = 0.077 | p = 0.42 |
| Kimi K3 vs GPT-4o FT+QSP | +0.044, 13/3, **p = 0.016** | p = 0.067 |
| Kimi K3 vs GPT-4o QSP | +0.080, 15/1, **p = 0.004** | **p = 0.008** |
| Astra vs GPT-4o **FT** | +0.012, 8/8, p = 0.64 | p = 0.74 |
| Astra vs GPT-4o QSP | +0.070, 12/3, **p = 0.016** | **p = 0.047** |

So: both frontier models beat *prompted* GPT-4o decisively, and beat *fine-tuned* GPT-4o on the
point estimate only. Fine-tuning on 150 curated papers remains competitive with a frontier model
that has never seen the annotation conventions.

The trade is consistent: GPT-4o FT has the best precision (0.921); the frontier models buy recall
(+0.04 to +0.06) at a small precision cost. For curation, recall is usually the expensive side -
a missed field is silent, a wrong field is visible in review.

## Q2. On which questions?

Biggest gains are exactly the questions the original paper found hard:

| QID | question | Kimi | Astra | GPT-4o FT | Kimi − FT |
|---|---|---|---|---|---|
| 9 | HIV genes sequenced | 0.906 | 0.899 | 0.745 | **+0.161** |
| 16 | ARV drugs before sequencing | 0.879 | 0.886 | 0.805 | **+0.074** |
| 15 | ARV drug classes | 0.913 | 0.893 | 0.859 | +0.054 |
| 11 | sample types | 0.886 | 0.886 | 0.852 | +0.034 |
| 8 | cloning | 0.940 | 0.946 | 0.973 | −0.033 |
| 6 | countries | 0.899 | 0.893 | 0.919 | −0.020 |

By question type: List questions are where frontier models win (Kimi 0.916 / Astra 0.918 vs
GPT-4o FT 0.876); Boolean is a tie (0.950 / 0.933 vs 0.948); **Number (QID 5) is the worst
question for every model** (0.79-0.85).

## Q3. Are the remaining errors scientifically meaningful?

Every error of every model was adjudicated (`results/adjudicated_errors.csv`), using explicit
rules plus manual reading; see README for the rule list and its limitations.

| | errors | model error | annotation/convention | borderline | accuracy after removing non-model errors |
|---|---|---|---|---|---|
| Kimi K3 | 175 | 89 (51%) | 59 | 27 | 0.963 |
| GPT-6 Astra | 199 | 114 (57%) | 69 | 16 | 0.952 |
| GPT-4o FT | 227 | 133 (59%) | 77 | 17 | 0.944 |
| GPT-4o FT+QSP | 280 | 179 (64%) | 65 | 36 | 0.925 |
| GPT-4o QSP | 366 | 255 (70%) | 72 | 39 | 0.893 |

**Roughly 40-50% of all "errors" are not extraction failures.** They are annotation errors,
unstated curation conventions, or rows where both readings are defensible. The ranking is
unchanged after adjudication, and Kimi vs GPT-4o FT remains non-significant (p = 0.13).

Three curation conventions the prompt never states, which no prompted model can infer:
1. **QID 5 counting** - annotations count individuals *successfully sequenced*; models count
   individuals *sampled*. Both fit the question text. The largest single error class.
2. **QID 10 method default** - annotations record "Sanger" for standard genotypic resistance
   testing even where the paper never says so (one annotation literally reads "Sanger (not stated)").
3. **What counts as "reporting sequences"** - for in-vitro studies using patient-derived clones,
   the QSP rule says Yes and the curators say No (e.g. PMID 31988104, where all five models say Yes).

Annotation problems found and reported to the curators:
- **PMID 40872801** is a PRISMA systematic review annotated as a primary study; its own QID 5
  annotation says "0 (Review paper)" while QID 6-15 record primary-study details.
- **3 data-entry errors** across the 150x16 grid: a Boolean question annotated "14"
  (PMID 31988104 QID 14), a Boolean annotated "Not known" (41140464 QID 8), a Number annotated
  "Not Reported" (28559249 QID 5). The first is unanswerable as scored.
- **60 rows where all five models disagree with the annotation**, including the GPT-4o fine-tuned
  on these annotations. If the fine-tuned model cannot reproduce a label from the paper, the row
  is unlikely to be answerable from the text.

The genuinely meaningful model errors that remain are:
- **Wrong denominator** (QID 5) - would put a wrong N into HIVDB.
- **Out-of-scope evidence** - quoting lab-construct text for patient-sample questions
  (47 rows across all models; more common in the GPT-4o conditions than the frontier ones).
- **Incomplete or over-inclusive ARV lists** (QID 15/16).
- **Cascades** - when a model gets QID 1 wrong, the downstream answers follow it. For Astra,
  15 of its adjudicated model errors trace to 2 wrong QID 1 answers.

## Q4. Autonomous entry or curator-assisted review?

**Curator-assisted review, not autonomous entry.**

- At ~0.93 accuracy (0.96 after removing non-model errors), roughly **1 in 14 answers** still needs
  correction. Per paper, that is about one wrong field in every two papers across 16 questions.
- The errors are not uniformly distributed: QID 5 (counts) and QID 15/16 (drug lists) carry most
  of them, and those write the numeric and identity fields a database cares about.
- **QID 1 should gate the rest.** It decides whether a paper contributes anything, and a wrong
  QID 1 propagates to every downstream field.
- Three curation conventions are not in the prompt. Until they are written down, no prompted model
  can match curator behaviour on QID 5, QID 10, or the in-vitro boundary - and a fine-tuned model
  learns them only from examples.

A workable split: let a frontier model draft all 16 answers with its evidence quotes, auto-accept
the high-agreement Boolean questions (QID 3, 4, 13: >= 0.97), and route QID 5, 15 and 16 - plus
any paper where QID 1 is uncertain - to a curator. The models' evidence quotes are reliable enough
to make that review fast: where a quote was checkable against the PDF text, it was verbatim.

## Operational

| | cost/paper | total (150) | median latency | p90 | valid JSON first try | failures |
|---|---|---|---|---|---|---|
| GPT-6 Astra | $0.378 | $56.30 | 53 s | 63 s | 99% | 1 policy refusal |
| Kimi K3 | $0.187 | $28.01 | 133 s | 241 s | 95% | 1 empty response (retried) |

Total spend for the study: **$84.30**.

**GPT-6 Astra refuses PMID 36920025** ("flagged for possible biological risk") - an eLife paper on
HIV-1 resistance to protease inhibitors. Reproducible across requests. Kimi answered it normally.
For an HIV drug-resistance pipeline, a closed model that intermittently refuses on-topic virology
papers is an operational risk that belongs in any deployment decision.

Kimi is 2x cheaper but 2.5x slower, and needs page images because it cannot ingest PDFs directly.

## Limitations

1. **Input modality is confounded with the model.** The frontier models read PDFs; the cached
   GPT-4o answers came from markdown conversions. Some of the frontier advantage is the input, not
   the model. The clean ablation (a frontier model on the same markdown) was not run.
2. **One run per model.** No run-to-run variance estimate; observed differences of ~1 point are
   within what a rerun could move.
3. **Adjudication is mine, not a curator's.** Rule-based first pass plus manual reading of the
   heavily-failed rows. Hand-adjudicating one model's 99 errors earlier gave 44% model errors,
   where the rules give 57% - so these rule-based figures **under-count** annotation problems.
4. **Contamination.** All 150 papers predate both models' training cutoffs.
5. **Supplementary material** is not in the PDFs; some annotations depend on it.

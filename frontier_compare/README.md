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
python frontier_compare/04_evaluate.py                                 # metrics, tests, stability
python frontier_compare/05_figure4.py                                  # updated Figure 4 (pooled, as in the paper)
python frontier_compare/06_secondary.py                                # error analysis, ops, failure-mode sheet
```

Or run `make -C frontier_compare all` after the query step.

Keys are read from `advanced-prompting/.env` (also `.env` or `frontier_compare/.env`):
`OPENAI_API_KEY`, `OPENROUTER_API_KEY`. Model IDs can be overridden with `FC_GPT6_MODEL_ID` and
`FC_KIMI_MODEL_ID`. GPT-6 Astra cost is computed from usage at published rates in `config.py`, since
OpenAI doesn't return cost; OpenRouter reports the billed cost per request.

Concurrency defaults come from each model's `max_concurrency` in `config.py` (48 for GPT-6 Astra,
whose account limit is 15k RPM / 40M TPM; 24 for Kimi K3). 429s honor `Retry-After`.

| Step | Output |
|---|---|
| 01 | `data/pdf_manifest.csv` (status, pages, size, and hash per PMID) |
| 02 | `runs/<model>/run<N>.jsonl`: raw response, latency, usage, and cost per request (resumable) |
| 03 | `results/answers/<model>_run<N>.csv`, `results/ops_requests.csv` |
| 04 | `results/detailed_rows.csv`, `metrics_by_qid.csv`, `metrics_summary.csv`, `metrics_by_type.csv`, `pairwise_tests.csv`, `stability.csv` |
| 05 | `figures/figure4_frontier_pooled.{png,tiff}` (`--aggregation macro` for the per-QID mean) |
| 06 | `results/secondary_*.csv`, `failure_modes/labeling_sheet.csv` |

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

## Pilot (40 papers, 2026-09-22)

Stratified random sample: 20 original120 + 20 new30 (the first 10 from seed 0, 30 more from
seed 1). 39 papers x 16 questions = 624 rows scored; PMID 36920025 is excluded because
GPT-6 Astra refuses it (see below). Reasoning-effort variants were only run on the first 10.

| | papers | accuracy | precision | recall | F1 | adj. acc. |
|---|---|---|---|---|---|---|
| Kimi K3 | 39 | 0.918 | 0.939 | 0.884 | 0.911 | 0.920 |
| GPT-6 Astra (Q5 convention v2) | 39 | 0.915 | 0.935 | 0.881 | 0.907 | 0.917 |
| GPT-6 Astra | 39 | 0.905 | 0.915 | 0.881 | 0.898 | 0.907 |
| GPT-4o FT | 39 | 0.886 | 0.941 | 0.810 | 0.870 | 0.886 |
| GPT-4o FT+QSP | 39 | 0.877 | 0.913 | 0.816 | 0.862 | 0.877 |
| GPT-4o QSP | 39 | 0.841 | 0.874 | 0.776 | 0.822 | 0.841 |

The gap narrowed as the sample grew (Kimi 0.931 -> 0.918; GPT-4o FT 0.862 -> 0.886). **No
comparison is significant**: Wilcoxon over the 16 per-QID values vs GPT-4o FT gives BH-adjusted
p >= 0.11 for every frontier model and metric (best: Kimi accuracy, raw p = 0.051). The consistent
pattern is recall: every frontier model gains recall (+0.06 to +0.07 mean per-QID) and gives up
precision, and Kimi beats GPT-4o FT on 6 of 16 questions while losing only 1.

**Operational (40 requests per model):** Astra $0.38/paper, 51 s median, 39/40 strict JSON;
Kimi $0.19/paper, 131 s median, 37/40 strict JSON (3 needed fence-stripping, 0 unparseable).

**Policy refusal.** `gpt-6-astra` returns HTTP 400 "flagged for possible biological risk" for
PMID 36920025 (eLife, selection of HIV-1 for resistance to fifth-generation protease inhibitors).
Reproducible with the base prompt; the same paper succeeded under the Q5-variant prompt, so the
filter is not deterministic across requests. Kimi K3 answered it normally. Failed requests
produce no answer rows (they are an operational failure in `ops_requests.csv`, not 16 wrong
answers), and the PMID drops out of the evaluated set for every model.

At 39 papers the Q5 prompt edit no longer separates from the base prompt on QID 5 itself
(29/39 both) but still gains overall (+0.01 accuracy, +0.02 precision), which is within noise.

## Annotation issues and evaluation artifacts (40-paper pilot)

`08_error_triage.py` classifies every frontier error using signals checkable against the PDF:
does the model's quoted evidence actually occur in the text layer, how much of the human answer
occurs anywhere in the PDF, do both models agree, is the paper a review, is the annotation itself
hedged. Output: `results/error_triage.csv` (also an "Auto-triage" sheet in the workbook).
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

**Two curator conventions the prompt never states**, both systematic rather than capability gaps:
- QID 5: annotations count individuals *successfully sequenced*; the models count individuals
  *sampled* (16 rows).
- QID 10: annotations default to "Sanger" for standard genotypic resistance testing even when the
  paper never says so - one annotation literally reads "Sanger (not stated)". None of the three
  papers involved contains the word "Sanger" (4 rows).

**Evaluation artifacts are small.** Scoring an answer correct if *either* its raw or
scaffolding-stripped form matches (applied to every model) recovers **3 rows in
total, all Kimi's** - caveats like "(paper states either Sanger sequencing or NGS)" that the
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
- The 10 proposed alternatives were approved and moved into `accepted_alternatives.csv`.
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

| | accuracy | precision | recall | F1 | before accepted answers |
|---|---|---|---|---|---|
| Kimi K3 | 0.938 | 0.944 | 0.922 | 0.933 | 0.918 |
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

## Annotation gaps: accepted alternatives

`data/accepted_alternatives.csv` holds curator-approved alternative reference answers for rows
where the annotation is known to be incomplete — typically evidence that exists only in a figure,
which the curators' markdown conversion dropped. Each row records the accepted answer, the reason,
and the source.

Scoring is layered, never overwritten: `correct` uses the paper's scorer against the human answer
only; `correct_adjusted` also accepts a listed alternative. Both appear in `detailed_rows.csv`,
and `metrics_summary.csv` carries `row_accuracy_adjusted`. Primary metrics and figures use the
strict score. Alternatives apply to every model equally.

Seeded with one entry from the pilot: PMID 41130593 QID 7, where both frontier models read
"August 2021-September 2023" from a Figure 3 footnote while the annotation says "Not provided".

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
  row bootstrap (5,000 resamples, seed 42). This is what `eval/evaluation.py` plots and what
  reproduces the paper's reported deltas (GPT-4o FT recall +11%; Llama-70B / 8B FT precision
  +16% / +8%). The mean of the per-QID values does not reproduce them (e.g. +12% for 70B), so
  it is kept only as a reference column. Checked: pooled values for the cached GPT-4o
  conditions match `eval/figures/full150-bar-chart-confidence-intervals.csv` exactly, and CIs
  agree to within 0.1 point.
- **Tests = the paper's Fig. 4 statistics.** Wilcoxon signed-rank (primary, as stated in the
  paper) and paired t-test (also in S5) over the 16 per-QID values, with BH applied within each
  (metric, test) slice as in `eval/statistics.py`. Exact McNemar on row correctness is a
  sensitivity analysis.
- **Primary comparator** is GPT-4o FT, the best cached condition (pooled accuracy 0.903, F1 0.883).
  All three GPT-4o conditions are plotted.
- **Undefined precision/recall** (no positives) is set to 0, matching `eval/evaluation.py`.
- **Reproducibility.** Kimi K3 is pinned to the first-party Moonshot AI endpoint with no
  fallbacks, because OpenRouter otherwise routes across about 20 hosts with fp4/fp8/bf16
  quantizations. The serving provider is logged per request.

## Open issues

1. **46 PDFs need manual download.** See `data/pdfs_to_download.csv` and save each as
   `frontier_compare/pdfs/<PMID>.pdf`, then re-run `01_pdf_manifest.py`. Ideally use the same
   version the curators annotated.
2. **GPT-6 Astra pricing** is not returned by the API, so cost is computed from token usage at the published rates ($10 / $1 cached / $50 per 1M tokens).
3. **OpenRouter budget.** The key has a $50 cap. A single Kimi run over 150 papers is estimated at
   $28–48 depending on reasoning length (input: 5.1M tokens, $15.35), so it is tight.
4. **Confound: model vs input.** The cached GPT-4o runs used markdown, so frontier-vs-GPT-4o
   differences mix model and modality effects. A cheap ablation is to run one frontier model on
   the same markdown (`<PMID>.checked.md`) to separate the two.
5. **Supplements.** Some human answers (accessions, long drug lists) come from supplementary
   files. Decide whether to merge supplements into the PDFs. Either way, label such errors
   `SUPPLEMENT`.
6. **Contamination.** All 150 papers predate the frontier models' training cutoffs. Note this
   in the report. The new-30 subset (2025 papers) is the least exposed.

## Deliverable

`report/REPORT_TEMPLATE.md` answers four questions: do frontier models beat GPT-4o, on which
questions, are the remaining errors scientifically meaningful, and is performance adequate for
autonomous HIVDB entry or only curator-assisted review. The acceptance criteria for "autonomous"
should be fixed **before** the results are seen.

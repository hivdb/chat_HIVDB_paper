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
python frontier_compare/02_query_models.py --model gpt6-astra --run 1  # repeat with --run 2, 3 for stability
python frontier_compare/02_query_models.py --model kimi-k3    --run 1
python frontier_compare/03_parse_responses.py                          # raw JSONL -> answers + ops table
python frontier_compare/04_evaluate.py                                 # metrics, tests, stability
python frontier_compare/05_figure4.py                                  # updated Figure 4 (pooled, as in the paper)
python frontier_compare/06_secondary.py                                # error analysis, ops, failure-mode sheet
```

Or run `make -C frontier_compare all` after the query step.

Keys are read from `advanced-prompting/.env` (also `.env` or `frontier_compare/.env`):
`OPENAI_API_KEY`, `OPENROUTER_API_KEY`. Model IDs can be overridden with `FC_GPT6_MODEL_ID` and
`FC_KIMI_MODEL_ID`. GPT-6 Astra cost needs `FC_GPT6_PRICE_IN/OUT` (USD per 1M tokens), since
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
Every file is checked for a `%PDF` header. Result: **104/150 staged** (all 30 new + 74 of the
original 120). The other 46 are listed in `data/pdfs_to_download.csv` for manual download:
25 are in PMC but outside the open-access subset, and 21 have no PMC copy (JAC, CID, HIV
Medicine, JMV, Elsevier titles). Short PDFs (2–4 pages) were checked and are complete research
letters or case reports.

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
2. **GPT-6 Astra pricing** is not returned by the API. Set `FC_GPT6_PRICE_IN/OUT` for cost metrics.
3. **OpenRouter budget.** The key has a $50 cap. Kimi costs about $0.07–0.10 per paper, so
   3 runs × 150 papers comes to about $35–45.
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

# Frontier LLM comparison (subproject)

**Objective.** Test whether current frontier LLMs using question-specific prompting (QSP) can
match or beat the cached GPT-4o results (FT, QSP, FT+QSP) on the 16 HIVDB extraction questions.

| | |
|---|---|
| Models | GPT-6 Astra (closed, OpenAI API); Qwen3.8 2.4T A95B (open weights, OpenRouter) |
| Eval set | Full 150 held-out papers (original 120 + new 30) with the paper's human annotations |
| Protocol | Same as the paper ([PLOS ONE, Methods](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0351631#sec006)), **except** the model gets the full-text **PDF** instead of the markdown conversion, so tables and figures are available |
| Comparators | Cached GPT-4o FT, QSP, FT+QSP from `advanced-prompting/csv/merged_answers_full_150.xlsx`. "GPT-4o" in the paper is `gpt-4o-mini-2024-07-18`. |
| Scoring | The paper's scorer (`eval/normalize.py::human_answer_counts`), unchanged. It reproduces the paper's per-QID metrics exactly (verified: max abs diff 0.0 for all three GPT-4o conditions). |

## Pipeline

Run from the repo root with the project venv (`.venv/bin/python` or `uv run`):

```bash
python frontier_compare/01_pdf_manifest.py --stage                    # inventory and stage PDFs -> pdfs/<PMID>.pdf
python frontier_compare/02_query_models.py --model gpt6-astra --run 1 --dry-run --limit 3
python frontier_compare/02_query_models.py --model gpt6-astra --run 1  # repeat with --run 2, 3 for stability
python frontier_compare/02_query_models.py --model qwen3.8   --run 1
python frontier_compare/03_parse_responses.py                          # raw JSONL -> answers + ops table
python frontier_compare/04_evaluate.py                                 # metrics, tests, stability
python frontier_compare/05_figure4.py                                  # updated Figure 4 (macro; --aggregation pooled)
python frontier_compare/06_secondary.py                                # error analysis, ops, failure-mode sheet
```

Or run `make -C frontier_compare all` after the query step.

Keys go in `.env` or `frontier_compare/.env`: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`. Model IDs and
prices can be overridden with `FC_GPT6_MODEL_ID`, `FC_QWEN_MODEL_ID`, `FC_GPT6_PRICE_IN/OUT`
(USD per 1M tokens), and `FC_QWEN_PDF_ENGINE`.

| Step | Output |
|---|---|
| 01 | `data/pdf_manifest.csv` (status, pages, size, and hash per PMID) |
| 02 | `runs/<model>/run<N>.jsonl`: raw response, latency, usage, and cost per request (resumable) |
| 03 | `results/answers/<model>_run<N>.csv`, `results/ops_requests.csv` |
| 04 | `results/detailed_rows.csv`, `metrics_by_qid.csv`, `metrics_summary.csv`, `metrics_by_type.csv`, `pairwise_tests.csv`, `stability.csv` |
| 05 | `figures/figure4_frontier_{macro,pooled}.{png,tiff}` |
| 06 | `results/secondary_*.csv`, `failure_modes/labeling_sheet.csv` |

## Design decisions

- **One request per paper** answers all 16 questions, as in the paper's QSP runs. The system prompt
  is `advanced-prompting/md/Nov17_Version1.md` verbatim. Only its final "format your answer"
  section is replaced with a JSON contract that keeps the same fields (question, evidence,
  rationale, answer) and adds `evidence_location` (main_text / table / figure / supplement),
  which feeds the missed-table/figure analysis.
- **No enforced JSON schema.** Structured-output mode would push the invalid-JSON rate to ~0 by
  construction, so the model is only *asked* for JSON. JSON validity is logged at two levels:
  `strict` (parses as-is) and `lenient` (parses after stripping fences and prose). Answers use the
  lenient parse. Unparseable responses score as wrong, as missing answers did in the paper.
- **Aggregation.** The proposal specifies per-QID metrics averaged across the 16 questions
  (`macro`). The existing paper figure (`eval/figures/full150-bar-chart.png`) plots **pooled**
  counts over all 2,400 rows, and uses per-QID values only for the Wilcoxon tests. Both are
  reported. CIs come from a paper-level bootstrap (2,000 resamples of PMIDs).
- **Tests.** Wilcoxon signed-rank over the 16 paired per-QID values (as in the paper's Fig. 4
  statistics), plus exact McNemar on row-level correctness. BH correction across all frontier ×
  comparator × metric tests.
- **Primary comparator** is GPT-4o FT, the best cached condition (pooled accuracy 0.903, F1 0.883).
  All three GPT-4o conditions are plotted.
- **Undefined precision/recall** (no positives) is set to 0, matching `eval/evaluation.py`.

## Open issues (resolve before the full run)

1. **PDFs are missing for the original 120 papers.** `01_pdf_manifest.py` finds PDFs only for the
   30 new papers. The original 120 exist locally only as HTML/markdown under
   `advanced-prompting/papers/` (gitignored). PDFs must be obtained and placed at
   `frontier_compare/pdfs/<PMID>.pdf`, ideally the same version the curators annotated. Until
   then, `04_evaluate.py --allow-subset` evaluates the new-30 subset only.
2. **Model IDs and pricing are unverified placeholders** (`config.py`). Confirm the exact API IDs
   for GPT-6 Astra and Qwen3.8 2.4T A95B, and whether each accepts PDF `file` inputs through Chat
   Completions.
3. **The open model's PDF path matters.** If Qwen3.8 isn't natively multimodal, OpenRouter parses
   the PDF first. `pdf-text` drops figures entirely and undercuts the reason for using PDFs, so use
   `native` or `mistral-ocr` and record which one was used.
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

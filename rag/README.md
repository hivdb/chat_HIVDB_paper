# RAG Baselines

This directory contains the reviewer-aligned per-paper RAG pipelines.

The paper markdown lives in:
- `advanced-prompting/papers`
- `advanced-prompting/papers_2025_30`

This pipeline keeps those source files in place, but stores RAG-specific code and artifacts under:
- `rag/scripts`
- `rag/jsonl`
- `rag/csv`
- `rag/eval`

The instruction scaffold is intentionally the same base prompt used by the main runs:
- `eval/gpt-5/gpt-5-mini-prompt.md`

The only change for RAG is the context: instead of supplying the full paper, the prompt appends a shared, deduplicated evidence pool built from question-specific retrieval over that same paper.

The retrieval settings are recorded in:
- `rag/run_manifest.json`

The manifest stores repo-relative paths so it can be committed without exposing local machine paths.

Evaluation outputs now live under:
- `rag/eval/results_new30`
- `rag/eval/results_original120`
- `rag/eval/results_full150`
- `rag/eval/figures_new30`
- `rag/eval/figures_original120`
- `rag/eval/figures_full150`

Verification-only CSV artifacts live under:
- `rag/csv/verification`

Artifact CSVs generated or consumed by the RAG pipeline live under `rag/csv/`.
Evaluator output CSVs remain under `rag/eval/results_{suffix}` because those directories are owned by the shared evaluation stack.

Current defaults:
- retrieved passages per question: `top_k = 5`
- chunk target size: `chunk_chars = 1800`
- chunk overlap: `1` paragraph
- chunking is section-aware
- chunking stops at `References` / `Bibliography`
- retrieval is per paper only
- prompt assembly uses a shared evidence pool per paper
- the evidence pool is deduplicated by `chunk_id`
- evidence pool ranking is `hit_count` then reciprocal-rank fusion
- prompt assembly trims the lowest-ranked pooled passages until the RAG prompt is shorter than the corresponding base full-paper prompt

BM25-specific defaults:
- tokenizer regex: `[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*`
- `k1 = 1.5`
- `b = 0.75`

Semantic-specific defaults:
- embedding model: `text-embedding-3-small`
- embedding batch size: `64`
- similarity: cosine similarity on normalized embeddings

When either generator is run, `rag/run_manifest.json` is updated with:
- the exact parameters used
- dataset-specific output paths
- paper counts
- chunk-count statistics per dataset

## 1. Generate prompts

```bash
python rag/scripts/01_create_bm25_rag_prompts.py
```

Outputs:
- `rag/jsonl/pmid_prompts_bm25_rag_original120.jsonl`
- `rag/jsonl/pmid_prompts_bm25_rag_new30.jsonl`
- `rag/csv/log/bm25_rag_retrieval_original120.csv`
- `rag/csv/log/bm25_rag_retrieval_new30.csv`
- `rag/csv/log/bm25_rag_pool_original120.csv`
- `rag/csv/log/bm25_rag_pool_new30.csv`
- `rag/run_manifest.json`

## 1b. Generate semantic-retrieval prompts

This uses the `OPENAI_API_KEY` from `advanced-prompting/.env` by default.

```bash
python rag/scripts/01_create_semantic_rag_prompts.py
```

Outputs:
- `rag/jsonl/pmid_prompts_semantic_rag_original120.jsonl`
- `rag/jsonl/pmid_prompts_semantic_rag_new30.jsonl`
- `rag/csv/log/semantic_rag_retrieval_original120.csv`
- `rag/csv/log/semantic_rag_retrieval_new30.csv`
- `rag/csv/log/semantic_rag_pool_original120.csv`
- `rag/csv/log/semantic_rag_pool_new30.csv`
- `rag/run_manifest.json`

## 2. Run a model on the prompts

For GPT-4o base via the existing OpenAI runner:

```bash
python advanced-prompting/02_call_openai.py \
  --job GPT4O_BM25_RAG_original120 rag/jsonl/pmid_prompts_bm25_rag_original120.jsonl rag/jsonl/pmid_responses_bm25_rag_gpt4o_original120.jsonl gpt-4o-mini-2024-07-18 \
  --job GPT4O_BM25_RAG_new30 rag/jsonl/pmid_prompts_bm25_rag_new30.jsonl rag/jsonl/pmid_responses_bm25_rag_gpt4o_new30.jsonl gpt-4o-mini-2024-07-18 \
  --job GPT4O_SEMANTIC_RAG_original120 rag/jsonl/pmid_prompts_semantic_rag_original120.jsonl rag/jsonl/pmid_responses_semantic_rag_gpt4o_original120.jsonl gpt-4o-mini-2024-07-18 \
  --job GPT4O_SEMANTIC_RAG_new30 rag/jsonl/pmid_prompts_semantic_rag_new30.jsonl rag/jsonl/pmid_responses_semantic_rag_gpt4o_new30.jsonl gpt-4o-mini-2024-07-18
```

Fine-tuned GPT-4o or external Llama runs can use the same prompt files.

### 2b. Llama response files

The committed Llama runs in this folder were produced outside this repo and then saved back into:
- `rag/csv/llama3.1/70B_RAG_BM25_120.csv`
- `rag/csv/llama3.1/70B_RAG_BM25_30.csv`
- `rag/csv/llama3.1/70B_RAG_Semantic_120.csv`
- `rag/csv/llama3.1/70B_RAG_Semantic_30.csv`
- `rag/csv/llama3.1/8B_RAG_BM25_120.csv`
- `rag/csv/llama3.1/8B_RAG_BM25_30.csv`
- `rag/csv/llama3.1/8B_RAG_Semantic_120.csv`
- `rag/csv/llama3.1/8B_RAG_Semantic_30.csv`

These CSVs are expected to contain at least:
- `PMID`
- `FT Answer`

where `FT Answer` is the full multi-question response for one PMID, using the same `Question / Evidence / Rationale / Answer` block structure expected by the downstream parser.

The exact inference stack used to generate those CSVs is not implemented in this repo. This repo starts from the prompt JSONLs and then documents how to parse, merge, validate, and evaluate external model outputs once they have been saved into this format.

## 3. Parse structured responses

```bash
python rag/scripts/02_extract_structured_responses.py \
  --responses-jsonl rag/jsonl/pmid_responses_bm25_rag_gpt4o_original120.jsonl \
  --output-csv rag/csv/parsed/gpt-4o_bm25_rag_original120_parsed.csv
```

Repeat for the new 30-paper responses.

### 3b. Parse Llama CSV outputs

The Llama runs in this folder were parsed with the dedicated wrapper scripts:
- `rag/scripts/extract_llama3.1-70B_RAG_BM25_120_csv.py`
- `rag/scripts/extract_llama3.1-70B_RAG_BM25_30_csv.py`
- `rag/scripts/extract_llama3.1-70B_RAG_Semantic_120_csv.py`
- `rag/scripts/extract_llama3.1-70B_RAG_Semantic_30_csv.py`
- `rag/scripts/extract_llama3.1-8B_RAG_BM25_120_csv.py`
- `rag/scripts/extract_llama3.1-8B_RAG_BM25_30_csv.py`
- `rag/scripts/extract_llama3.1-8B_RAG_Semantic_120_csv.py`
- `rag/scripts/extract_llama3.1-8B_RAG_Semantic_30_csv.py`

Each wrapper script:
- reads one fixed CSV under `rag/csv/llama3.1/`
- uses `advanced-prompting/csv/S4Table.xlsx` for canonical metadata
- uses `eval/gpt-5/gpt-5-mini-prompt.md` to recover canonical question ordering
- extracts one answer per `PMID/QID`
- writes a parsed CSV under `rag/csv/parsed/` with a name like `llama3.1-70B_RAG_BM25_120_parsed.csv`

Example:

```bash
python rag/scripts/extract_llama3.1-70B_RAG_BM25_120_csv.py
```

This writes:
- `rag/csv/parsed/llama3.1-70B_RAG_BM25_120_parsed.csv`

The same parsing logic is used across all eight wrappers; they are separate only because each one is pinned to one concrete Llama result file.

## 4. Merge into the evaluation workbook

```bash
python rag/scripts/03_merge_model_answers.py \
  --source rag/csv/parsed/gpt-4o_bm25_rag_original120_parsed.csv \
  --column-name "GPT-4o BM25 RAG"
```

For the embedding-retrieval baseline, merge with:

```bash
python rag/scripts/03_merge_model_answers.py \
  --source rag/csv/parsed/gpt-4o_semantic_rag_original120_parsed.csv \
  --column-name "GPT-4o Semantic RAG"
```

If you want full 150-paper columns, first combine original120 and new30 parsed outputs, then merge the combined file.

### 4b. Merge Llama parsed outputs

The committed Llama parsed CSVs were merged with:
- `rag/scripts/run_merge_llama3.1_parsed.sh`

That helper script:
- scans `rag/csv/parsed/` for `llama3.1-*_parsed.csv`
- infers whether each file belongs to `original120` or `new30`
- converts the filename into the evaluator-facing column label
- merges each parsed file into both:
  - `rag/eval/merged_answers_original120.xlsx` or `rag/eval/merged_answers_new30.xlsx`
  - `rag/eval/merged_answers_full150.xlsx`

The resulting column names are:
- `Llama3.1-70B BM25 RAG`
- `Llama3.1-70B Semantic RAG`
- `Llama3.1-8B BM25 RAG`
- `Llama3.1-8B Semantic RAG`

Run it with:

```bash
bash rag/scripts/run_merge_llama3.1_parsed.sh
```

## 5. Audit hard list-question retrieval

```bash
python rag/scripts/04_audit_list_retrievals.py
```

This writes retrieval-audit files under `rag/csv/verification/`, including:
- `rag/csv/verification/retrieval_audit_focus_q9_q15_q16.csv`
- `rag/csv/verification/retrieval_audit_all.csv`

The audit reuses the normalization and list-matching logic from `eval/normalize.py`, so ARV aliases and gene synonyms are scored consistently with the downstream evaluation code.

## 6. Validate response completeness

Response-validation summaries are also kept under `rag/csv/verification/`:
- `rag/csv/verification/response_validation_bm25_new30.csv`
- `rag/csv/verification/response_validation_bm25_original120.csv`
- `rag/csv/verification/response_validation_semantic_new30.csv`
- `rag/csv/verification/response_validation_semantic_original120.csv`

These are verification artifacts only. They are useful for checking that each response JSONL contains one complete 16-answer response per PMID, but they are not part of the main prompt / response / eval workflow.

For GPT-based JSONL outputs, the helper scripts are:
- `rag/scripts/05_validate_rag_responses.py`
- `rag/scripts/06_compact_valid_rag_responses.py`

`05_validate_rag_responses.py` checks that each response file contains one complete 16-answer response per PMID and writes the verification CSVs listed above.

`06_compact_valid_rag_responses.py` is a cleanup helper for retry-heavy JSONLs. It keeps one preferred row per PMID, preferring the latest valid response when retries are present.

Llama CSV outputs were not validated with those JSONL-specific scripts because they were already delivered as one CSV row per PMID.

## 6b. Re-run local evaluations

Once parsed outputs have been merged into the RAG workbooks, the local evaluation helper is:
- `rag/scripts/run_local_evaluations.sh`

It runs:
- `eval/evaluation.py --merged-path rag/eval/merged_answers_full150.xlsx ...`
- `eval/evaluation.py --merged-path rag/eval/merged_answers_original120.xlsx ...`
- `eval/evaluation.py --merged-path rag/eval/merged_answers_new30.xlsx ...`

and rewrites:
- `rag/eval/results_full150`, `rag/eval/figures_full150`
- `rag/eval/results_original120`, `rag/eval/figures_original120`
- `rag/eval/results_new30`, `rag/eval/figures_new30`

Run it with:

```bash
bash rag/scripts/run_local_evaluations.sh
```

This is the main entry point to refresh the RAG-side workbooks, metrics, statistical test files, and bar/table figures after new model columns are merged.

## 7. Why The Current `full150` Bar Chart Differs From The Manuscript Figure

The current RAG-era `full150` chart:
- `rag/eval/figures_full150/full150-bar-chart.png`

is not expected to be numerically identical to the manuscript-era chart:
- `eval/figures/full150-bar-chart.png`

There are three concrete reasons.

1. A small number of legacy model rows now score differently because normalization was improved.
   This happened in two distinct ways:
   - sentence-form list answers such as `Participants received integrase inhibitors (INSTIs) and nucleos(t)ide reverse transcriptase inhibitors (NRTIs)` now normalize to compact list forms like `INSTI, NRTI`, which lets them be treated consistently with terse list answers that were already receiving partial-match credit;
   - `Q10` sequencing-method aliases were expanded so that Sanger-platform answers such as `ABI Prism BigDye Terminator cycle sequencing`, `BigDye terminator v3.1 cycle sequencing kit`, `ABI PRISM 3130xl DNA Analyser`, and `ViroSeq Genotyping System` are treated as equivalent to `Sanger sequencing`.
   The scoring rule itself did not become more lenient; the evaluator now just recognizes more semantically equivalent phrasings. After this update, the RAG evaluation outputs under `rag/eval/results_{suffix}` and `rag/eval/figures_{suffix}` were regenerated so the stored workbooks and plots reflect the current normalization logic.

2. The bar-chart significance labels come from paired Wilcoxon tests over per-QID metrics, not from the aggregated Fisher table.
   In `eval/evaluation.py`, the figure significance map is built from the adjusted Wilcoxon p-values in the `Paired Tests` sheet of `statistical_tests_*.xlsx`. So the p-value labels shown above the bars are driven by the per-question paired tests, not by `evaluation_metrics_fisher_*.xlsx`.

3. In the current evaluation run, Benjamini-Hochberg adjustment is performed over a larger comparison set because RAG targets are present.
   `eval/evaluation.py` now builds `FAMILY_COMPARISONS` using `config.FAMILY_OPTIONAL_TARGETS`, which adds `BM25 RAG` and `Semantic RAG` targets for each family when present. The plotting code still draws brackets only for the original `FT`, `QSP`, and `FT+QSP` comparisons, but the adjusted p-values being plotted were computed in the expanded comparison universe. As a result, even when a raw Wilcoxon p-value stayed the same or moved only slightly, its adjusted p-value could change more noticeably.

So the current `rag/eval/figures_full150/full150-bar-chart.png` differs from the manuscript-era figure because it reflects:
- the updated normalization fixes,
- recomputed per-QID paired tests,
- and multiple-testing correction in the presence of the extra RAG comparisons.

If manuscript-exact p-values are needed for the non-RAG models, the figure should be regenerated with BH adjustment restricted to the original three targets per family (`FT`, `QSP`, `FT+QSP`) and without the RAG rows participating in the correction set.

# RAG Baselines

This directory contains the reviewer-aligned per-paper RAG pipelines.

The paper markdown lives in:
- `advanced-prompting/papers`
- `advanced-prompting/papers_2025_30`

This pipeline keeps those source files in place, but stores RAG-specific prompts and logs under `rag/`.

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

Verification-only artifacts live under:
- `rag/verification`

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
python rag/01_create_bm25_rag_prompts.py
```

Outputs:
- `rag/jsonl/pmid_prompts_bm25_rag_original120.jsonl`
- `rag/jsonl/pmid_prompts_bm25_rag_new30.jsonl`
- `rag/log/bm25_rag_retrieval_original120.csv`
- `rag/log/bm25_rag_retrieval_new30.csv`
- `rag/log/bm25_rag_pool_original120.csv`
- `rag/log/bm25_rag_pool_new30.csv`
- `rag/run_manifest.json`

## 1b. Generate semantic-retrieval prompts

This uses the `OPENAI_API_KEY` from `advanced-prompting/.env` by default.

```bash
python rag/01_create_semantic_rag_prompts.py
```

Outputs:
- `rag/jsonl/pmid_prompts_semantic_rag_original120.jsonl`
- `rag/jsonl/pmid_prompts_semantic_rag_new30.jsonl`
- `rag/log/semantic_rag_retrieval_original120.csv`
- `rag/log/semantic_rag_retrieval_new30.csv`
- `rag/log/semantic_rag_pool_original120.csv`
- `rag/log/semantic_rag_pool_new30.csv`
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

## 3. Parse structured responses

```bash
python rag/02_extract_structured_responses.py \
  --responses-jsonl rag/jsonl/pmid_responses_bm25_rag_gpt4o_original120.jsonl \
  --output-csv rag/gpt-4o_bm25_rag_original120_parsed.csv
```

Repeat for the new 30-paper responses.

## 4. Merge into the evaluation workbook

```bash
python rag/03_merge_model_answers.py \
  --source rag/gpt-4o_bm25_rag_original120_parsed.csv \
  --column-name "GPT-4o BM25 RAG"
```

For the embedding-retrieval baseline, merge with:

```bash
python rag/03_merge_model_answers.py \
  --source rag/gpt-4o_semantic_rag_original120_parsed.csv \
  --column-name "GPT-4o Semantic RAG"
```

If you want full 150-paper columns, first combine original120 and new30 parsed outputs, then merge the combined file.

## 5. Audit hard list-question retrieval

```bash
python rag/04_audit_list_retrievals.py
```

This writes retrieval-audit files under `rag/verification/`, including:
- `rag/verification/retrieval_audit_focus_q9_q15_q16.csv`
- `rag/verification/retrieval_audit_all.csv`

The audit reuses the normalization and list-matching logic from `eval/normalize.py`, so ARV aliases and gene synonyms are scored consistently with the downstream evaluation code.

## 6. Validate response completeness

Response-validation summaries are also kept under `rag/verification/`:
- `rag/verification/response_validation_bm25_new30.csv`
- `rag/verification/response_validation_bm25_original120.csv`
- `rag/verification/response_validation_semantic_new30.csv`
- `rag/verification/response_validation_semantic_original120.csv`

These are verification artifacts only. They are useful for checking that each response JSONL contains one complete 16-answer response per PMID, but they are not part of the main prompt / response / eval workflow.

## 7. Why The Current `full150` Bar Chart Differs From The Manuscript Figure

The current RAG-era `full150` chart:
- `rag/eval/figures_full150/full150-bar-chart.png`

is not expected to be numerically identical to the manuscript-era chart:
- `eval/figures/full150-bar-chart.png`

There are three concrete reasons.

1. A small number of legacy model rows now score differently because list normalization was improved.
   The scoring rule itself did not become more lenient. Instead, sentence-form list answers such as `Participants received integrase inhibitors (INSTIs) and nucleos(t)ide reverse transcriptase inhibitors (NRTIs)` now normalize to compact list forms like `INSTI, NRTI`, which lets them be treated consistently with terse list answers that were already receiving partial-match credit. This caused a small number of baseline rows to flip from incorrect to correct.

2. The bar-chart significance labels come from paired Wilcoxon tests over per-QID metrics, not from the aggregated Fisher table.
   In `eval/evaluation.py`, the figure significance map is built from the adjusted Wilcoxon p-values in the `Paired Tests` sheet of `statistical_tests_*.xlsx`. So the p-value labels shown above the bars are driven by the per-question paired tests, not by `evaluation_metrics_fisher_*.xlsx`.

3. In the current evaluation run, Benjamini-Hochberg adjustment is performed over a larger comparison set because RAG targets are present.
   `eval/evaluation.py` now builds `FAMILY_COMPARISONS` using `config.FAMILY_OPTIONAL_TARGETS`, which adds `BM25 RAG` and `Semantic RAG` targets for each family when present. The plotting code still draws brackets only for the original `FT`, `QSP`, and `FT+QSP` comparisons, but the adjusted p-values being plotted were computed in the expanded comparison universe. As a result, even when a raw Wilcoxon p-value stayed the same or moved only slightly, its adjusted p-value could change more noticeably.

So the current `rag/eval/figures_full150/full150-bar-chart.png` differs from the manuscript-era figure because it reflects:
- the updated normalization fixes,
- recomputed per-QID paired tests,
- and multiple-testing correction in the presence of the extra RAG comparisons.

If manuscript-exact p-values are needed for the non-RAG models, the figure should be regenerated with BH adjustment restricted to the original three targets per family (`FT`, `QSP`, `FT+QSP`) and without the RAG rows participating in the correction set.

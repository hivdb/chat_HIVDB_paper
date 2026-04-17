# RAG Baselines

This directory contains the reviewer-aligned per-paper RAG pipelines.

The paper markdown lives in:
- `advanced-prompting/papers`
- `advanced-prompting/papers_2025_30`

This pipeline keeps those source files in place, but stores RAG-specific prompts and logs under `rag/`.

The instruction scaffold is intentionally the same base prompt used by the main runs:
- `eval/gpt-5/gpt-5-mini-prompt.md`

The only change for RAG is the context: instead of supplying the full paper, the prompt appends question-specific retrieved passages from that same paper.

The retrieval settings are recorded in:
- `rag/run_manifest.json`

The manifest stores repo-relative paths so it can be committed without exposing local machine paths.

Current defaults:
- retrieved passages per question: `top_k = 5`
- chunk target size: `chunk_chars = 1800`
- chunk overlap: `1` paragraph
- chunking is section-aware
- chunking stops at `References` / `Bibliography`
- retrieval is per paper only

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

This writes `rag/list_retrieval_audit.csv`, comparing BM25 and semantic retrieval on hard list questions (`Q9`, `Q15`, `Q16`).

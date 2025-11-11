# Evaluation Package Overview

This package scores multiple model outputs against human‐curated answers and produces tabular/graphical summaries. The files work together as follows.

## Configuration (`eval/config.py`)
- Defines file locations for the merged dataset, GPT‑5 responses, and all outputs.
- Groups model columns into families (`MODEL_GROUPS`) and lists evaluation scenarios (`SCENARIOS`). Each scenario specifies:
  * The human reference column.
  * Which model columns to compare.
  * Whether to collapse “special no” answers (None/Not reported/0/etc.) into `no`.
  * Optional filtering on question type (e.g., Boolean only).
  * Footnotes that are appended to generated figures/tables.

## Constants (`eval/constants.py`)
- Provides the vocabulary that powers normalization and scoring:
  * Synonym sets for `yes`/`no`, negation phrases, lab-only contexts, text/drug/gene variants, and “special no” tokens.
  * Numeric word mappings and scale words to recover numbers written as text.
  * Thresholds for partial list credit (`LIST_PARTIAL_THRESHOLD`, `LIST_PARTIAL_MIN_TOKENS`) and cue words that implicitly signal positive Boolean answers (`BOOLEAN_POSITIVE_CUES`).

## Normalization & Scoring Helpers (`eval/normalize.py`)
- `canonicalize_answer` cleans every reference/model answer by:
  * Pulling off leading “Answer:” prefixes.
  * Collapsing recognized Boolean variants to `yes`/`no`.
  * Normalizing numbers or year ranges (with optional conversion of `0`/“None” to `no`).
  * Sorting and deduplicating list tokens, while mapping drug/gene synonyms to canonical forms.
- Extra utilities detect negations, lab-only phrases, numeric mentions (digits or words), year references, and synonym-expanded list matches. These signals support lenient scoring.
- `human_answer_counts` dispatches each question to a handler based on `Type`:
  * **Boolean:** Positive human answers grant TP if either the normalized token is `yes` or the raw model text contains scenario-specific positive cues; negatives are TN unless the model says `yes`.
  * **List:** Non-empty human lists award TP for exact matches, full token coverage, partial matches over the 60 %/4-token threshold, or matching year mentions. Empty human answers become TN when the model is empty, negated, or restricted to lab-only contexts; otherwise FP.
  * **Number:** Non-zero references require any overlapping number between human/model texts (or exact list equivalence) to earn TP; zero/empty references produce TN when the model negates or is also zero.
  * **Generic fallback:** Empty human answers + empty/negated model → TN; otherwise TP requires list equality, else FN.

## Dataset Loading & Metric Aggregation (`eval/scoring.py`)
- `load_dataset` merges the human spreadsheet with GPT‑5 responses, normalizes identifiers, and adds convenience columns.
- `ensure_norm` caches canonicalized columns per `(column, convert_special_no)` combination so multiple scenarios reuse the work.
- `evaluate_model` iterates rows, invokes `human_answer_counts`, accumulates TP/FP/TN/FN, and derives accuracy/precision/recall/F1.
- `evaluate_group` applies `evaluate_model` to every model in a scenario, tagging results with scenario metadata.
- `build_detail_rows` produces a per-question audit table containing raw answers and a binary correctness flag per model.

## Orchestration (`eval/evaluation.py`)
- Loads the dataset, applies optional row limits, and loops through all scenarios.
- For each scenario it filters rows (when requested), normalizes the necessary columns, evaluates each model, appends detailed per-question rows, and queues figure specifications.
- After all scenarios run, it writes:
  * `evaluation_metrics.csv` — stacked model/scenario metrics (including TP/FP/TN/FN counts).
  * `detailed_evaluation.csv` — per-question correctness table for manual inspection.

## Visualization (`eval/plots.py`)
- `generate_figures` creates, per scenario, a bar chart of accuracies and a table of accuracy/precision/recall/F1.
- Footnotes from the scenario configuration are rendered beneath both outputs, and files are saved under `eval/figures/` using slugified titles.

## Typical Workflow
1. Ensure `advanced-prompting/merged_answers.xlsx` and `eval/gpt5_responses.csv` are up to date.
2. Run `python eval/evaluation.py` (optionally with `--limit N` during debugging).
3. Inspect `eval/evaluation_metrics.csv`, `eval/detailed_evaluation.csv`, and the generated figures for insights or auditing borderline TP/FN cases.

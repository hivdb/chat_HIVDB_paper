#!/usr/bin/env bash

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)"
EVAL_PY="$RAG_ROOT/../eval/evaluation.py"

run_eval() {
  local merged_path="$1"
  local output_dir="$2"
  local figures_dir="$3"
  local suffix="$4"
  local per_metric_dir="${5:-}"

  echo "Evaluating $(basename "$merged_path")"
  local cmd=(
    uv run python "$EVAL_PY"
    --merged-path "$merged_path"
    --output-dir "$output_dir"
    --figures-dir "$figures_dir"
    --output-suffix "$suffix"
  )
  if [ -n "$per_metric_dir" ]; then
    cmd+=(--per-metric-figures-dir "$per_metric_dir")
  fi
  "${cmd[@]}"
}

run_eval \
  "$RAG_ROOT/eval/merged_answers_full150.xlsx" \
  "$RAG_ROOT/eval/results_full150" \
  "$RAG_ROOT/eval/figures_full150" \
  "full150" \
  "$RAG_ROOT/eval/figures_full150/metrics"

run_eval \
  "$RAG_ROOT/eval/merged_answers_original120.xlsx" \
  "$RAG_ROOT/eval/results_original120" \
  "$RAG_ROOT/eval/figures_original120" \
  "original120"

run_eval \
  "$RAG_ROOT/eval/merged_answers_new30.xlsx" \
  "$RAG_ROOT/eval/results_new30" \
  "$RAG_ROOT/eval/figures_new30" \
  "new30"

#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_PY="$ROOT/../eval/evaluation.py"

run_eval() {
  local merged_path="$1"
  local output_dir="$2"
  local figures_dir="$3"
  local suffix="$4"

  echo "Evaluating $(basename "$merged_path")"
  uv run python "$EVAL_PY" \
    --merged-path "$merged_path" \
    --output-dir "$output_dir" \
    --figures-dir "$figures_dir" \
    --output-suffix "$suffix"
}

run_eval \
  "$ROOT/eval/merged_answers_full150.xlsx" \
  "$ROOT/eval/results_full150" \
  "$ROOT/eval/figures_full150" \
  "full150"

run_eval \
  "$ROOT/eval/merged_answers_original120.xlsx" \
  "$ROOT/eval/results_original120" \
  "$ROOT/eval/figures_original120" \
  "original120"

run_eval \
  "$ROOT/eval/merged_answers_new30.xlsx" \
  "$ROOT/eval/results_new30" \
  "$ROOT/eval/figures_new30" \
  "new30"

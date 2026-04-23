#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run_step() {
  local label="$1"
  shift
  echo
  echo "==> $label"
  "$@"
}

run_step "Prepare regression data" \
  uv run python prepare_llama8b_regression_data.py

run_step "Validate regression data" \
  uv run python validate_llama8b_regression_data.py

run_step "Fit recall regression" \
  uv run python fit_llama8b_recall_logistic.py

run_step "Fit precision regression" \
  uv run python fit_llama8b_precision_logistic.py

run_step "Fit accuracy regression" \
  uv run python fit_llama8b_accuracy_logistic.py

run_step "Plot regression results" \
  uv run python plot_llama8b_regression.py

echo
echo "Pipeline completed."

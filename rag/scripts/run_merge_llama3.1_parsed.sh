#!/usr/bin/env bash

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)"
TARGET_FULL150="$RAG_ROOT/eval/merged_answers_full150.xlsx"
TARGET_120="$RAG_ROOT/eval/merged_answers_original120.xlsx"
TARGET_30="$RAG_ROOT/eval/merged_answers_new30.xlsx"
shopt -s nullglob

files=("$RAG_ROOT"/csv/parsed/llama3.1-*.csv)

if [ "${#files[@]}" -eq 0 ]; then
  echo "No llama3.1-*.csv files found in $RAG_ROOT/csv/parsed"
  exit 1
fi

for file in "${files[@]}"; do
  base="$(basename "$file" .csv)"
  if [[ "$base" == *_120_parsed ]]; then
    target="$TARGET_120"
  elif [[ "$base" == *_30_parsed ]]; then
    target="$TARGET_30"
  else
    echo "Skipping unrecognized file name: $base.csv"
    continue
  fi

  label="${base%_parsed}"
  label="${label%_120}"
  label="${label%_30}"
  label="${label//_/ }"
  label="${label//-/ }"
  label="L${label:1}"
  label="${label/Llama3.1 70B/Llama3.1-70B}"
  label="${label/Llama3.1 8B/Llama3.1-8B}"
  label="${label// RAG/}"
  label="$label RAG"

  echo "Merging $base.csv as '$label' into $(basename "$target") and $(basename "$TARGET_FULL150")"
  uv run python "$SCRIPT_ROOT/03_merge_model_answers.py" \
    --source "$file" \
    --column-name "$label" \
    --target "$target"
  uv run python "$SCRIPT_ROOT/03_merge_model_answers.py" \
    --source "$file" \
    --column-name "$label" \
    --target "$TARGET_FULL150"
done

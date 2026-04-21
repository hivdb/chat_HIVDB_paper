#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_FULL150="$ROOT/eval/merged_answers_full150.xlsx"
TARGET_120="$ROOT/eval/merged_answers_original120.xlsx"
TARGET_30="$ROOT/eval/merged_answers_new30.xlsx"
shopt -s nullglob

files=("$ROOT"/llama3.1-*.csv)

if [ "${#files[@]}" -eq 0 ]; then
  echo "No llama3.1-*.csv files found in $ROOT"
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
  uv run python "$ROOT/03_merge_model_answers.py" \
    --source "$file" \
    --column-name "$label" \
    --target "$target"
  uv run python "$ROOT/03_merge_model_answers.py" \
    --source "$file" \
    --column-name "$label" \
    --target "$TARGET_FULL150"
done

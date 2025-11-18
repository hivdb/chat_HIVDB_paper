# Learning-Curve Toolkit

This folder contains everything needed to reproduce the HIVDB learning-curve study for the OpenAI fine-tunes that were previously launched under job `ftjob-KcO0ZDfs21Hq688zDsZwvtDN`. The workflow mirrors the original pipeline: create nested subsets of the 200-example instruction dataset, fine-tune `gpt-4o-mini-2024-07-18` (or whatever base model was used in that job), gather model answers with the same prompt runner as `eval/gpt-5/query_gpt5.py`, and evaluate the outputs with the standard scoring helpers.

## 1. Build the subset files

```bash
python eval/learning-curve/prepare_subsets.py --sizes 50 100 150 --seed 7
```

* Reads `advanced-prompting/train_val/train_set.jsonl`, shuffles it once using the supplied seed, and emits nested subsets under `eval/learning-curve/data/`.
* `subset_manifest.json` captures the permutation so that future runs can reuse the exact same examples.

## 2. Launch the fine-tunes

```bash
python eval/learning-curve/finetune.py \
  --train-dir eval/learning-curve/data \
  --reference-job ftjob-KcO0ZDfs21Hq688zDsZwvtDN \
  --val-file advanced-prompting/train_val/val_set.jsonl
```

* Loads the reference job to inherit `model`, `n_epochs`, `batch_size`, and `learning_rate_multiplier`. Use `--base-model/--n-epochs/...` if the reference job is unreachable and you need to supply values manually.
* Uploads each subset and creates a fine-tune with suffix `hivdb-lc-050`, `hivdb-lc-100`, and `hivdb-lc-150`.
* Appends job metadata to `eval/learning-curve/finetune_jobs.jsonl` so you have a local log of file IDs and job IDs.
* Use `--dry-run` to verify settings without calling the API.

## 3. Generate answers with each fine-tuned model

```bash
# One command to run all entries in finetune_jobs.jsonl
python eval/learning-curve/query_finetuned.py \
  --jobs-file eval/learning-curve/finetune_jobs.jsonl \
  --max-concurrency 8

# ...or run a single model/tag explicitly
python eval/learning-curve/query_finetuned.py \
  --model ft:gpt-4o-mini-2024-07-18:... \
  --tag size050 \
  --max-concurrency 8
```

* Reuses every helper from `eval/gpt-5/query_gpt5.py`, so throttling, resume logic, parsed output schema, and logging all behave the same way.
* When `--jobs-file` is supplied, the script looks up each `result_model` (or fetches it from OpenAI if missing), derives a tag like `size050`, and emits `{tag}_responses.csv` / `{tag}_raw.jsonl` under `eval/learning-curve/responses/`.
* Single-run mode remains available for ad-hoc models via `--model/--tag`.
* Every run automatically rebuilds the CSV from the raw JSONL log (to fix quoting issues), and failed PMIDs are written to `{tag}_failures.jsonl`. Subsequent runs will prioritize those failed PMIDs until they succeed, so you can safely re-run the command to clean up parse/syntax problems without duplicating already-completed rows.

## 4. Score the runs with the existing evaluation stack

```bash
python eval/learning-curve/evaluate_learning_curve.py \
  --responses size050=eval/learning-curve/responses/size050_responses.csv \
             size100=eval/learning-curve/responses/size100_responses.csv \
             size150=eval/learning-curve/responses/size150_responses.csv
```

* Loads the merged answer sheet via `eval.scoring.load_dataset`, injects each run as a new column (prefix `GPT-4o LC` by default), and reuses the same scenarios defined in `eval/config.py` (overall, Boolean only, list exact, and list partial).
* Produces three outputs under `eval/learning-curve/results/`:
  * `learning_curve_metrics.csv` — stacked metrics for every scenario.
  * `learning_curve_details.csv` — per-question correctness flags for auditing.
  * `learning_curve_summary.json` — shortcut view of overall accuracy/precision/recall/F1 per subset size.
* If you omit `--responses`, the script auto-discovers every `*_responses.csv` under `eval/learning-curve/responses/`, so the one-liner `python eval/learning-curve/evaluate_learning_curve.py` will “just work” after you gather the three runs.

## Environment requirements

* The OpenAI SDK reads credentials via the usual mechanisms (`OPENAI_API_KEY`, `OPENAI_ORG_ID`, etc.), so make sure your `.env` or shell exports them before running the scripts. All commands also respect `python-dotenv` for convenience.
* Network access is required for the fine-tuning and querying steps. `prepare_subsets.py` and `evaluate_learning_curve.py` are purely local.

With these pieces in place you can iterate on additional points in the learning curve by adding sizes to `prepare_subsets.py --sizes ...` and rerunning the remaining steps. The manifest ensures that every subset remains a prefix of the next one, guaranteeing a legitimate learning-curve analysis.

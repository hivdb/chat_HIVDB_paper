#!/usr/bin/env python3
"""Launch OpenAI fine-tuning jobs for learning-curve subsets."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.fine_tuning import FineTuningJob


SUBSET_PATTERN = re.compile(r"(?:train_)?subset_(\d+)\.jsonl$|train_subset_(\d+)\.jsonl$")


@dataclass(frozen=True)
class SubsetSpec:
    size: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("eval/learning-curve/data"),
        help="Directory containing subset JSONL files (default: eval/learning-curve/data).",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Only launch jobs for the listed subset sizes.",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("advanced-prompting/train_val/val_set.jsonl"),
        help="Validation file uploaded once per run.",
    )
    parser.add_argument(
        "--validation-file-id",
        type=str,
        default=None,
        help="Existing validation file ID to reuse (skips upload).",
    )
    parser.add_argument(
        "--reference-job",
        type=str,
        default="ftjob-KcO0ZDfs21Hq688zDsZwvtDN",
        help="Existing job ID whose model/hyperparameters should be reused.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override the base model used for fine-tuning (default: inherits from reference job).",
    )
    parser.add_argument("--n-epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument(
        "--learning-rate-multiplier",
        type=float,
        default=None,
        help="Override the learning rate multiplier.",
    )
    parser.add_argument(
        "--suffix-prefix",
        type=str,
        default="hivdb-lc",
        help="Prefix used when deriving fine-tune suffixes.",
    )
    parser.add_argument(
        "--suffix-template",
        type=str,
        default="{prefix}-{size:03d}",
        help="Python format string for suffixes (fields: prefix, size).",
    )
    parser.add_argument(
        "--jobs-file",
        type=Path,
        default=Path("eval/learning-curve/finetune_jobs.jsonl"),
        help="Where to append job metadata.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions that would be taken without calling the API.",
    )
    return parser.parse_args()


def discover_subsets(directory: Path, allowed_sizes: Optional[Iterable[int]]) -> List[SubsetSpec]:
    allowed = {size for size in allowed_sizes} if allowed_sizes else None
    subsets: List[SubsetSpec] = []
    for path in sorted(directory.glob("*.jsonl")):
        match = SUBSET_PATTERN.match(path.name)
        if not match:
            continue
        size_str = next(filter(None, match.groups()))
        size = int(size_str)
        if allowed and size not in allowed:
            continue
        subsets.append(SubsetSpec(size=size, path=path))
    return subsets


def upload_file(client: OpenAI, path: Path) -> str:
    with path.open("rb") as infile:
        file_obj = client.files.create(file=infile, purpose="fine-tune")
    logging.info("Uploaded %s -> %s", path, file_obj.id)
    return file_obj.id


def append_job_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(record) + "\n")


def get_hparam(source: object | None, key: str) -> object | None:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def resolve_hyperparams(args: argparse.Namespace, reference: FineTuningJob | None) -> Dict[str, object]:
    result: Dict[str, object] = {}
    default_hp = getattr(reference, "hyperparameters", None)
    mappings = {
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "learning_rate_multiplier": args.learning_rate_multiplier,
    }
    for key, override in mappings.items():
        if override is not None:
            result[key] = override
        else:
            inherited = get_hparam(default_hp, key)
            if inherited is not None:
                result[key] = inherited
    return result


def create_job(
    client: OpenAI,
    base_model: str,
    train_file_id: str,
    val_file_id: str | None,
    suffix: str,
    hyperparams: Dict[str, object],
) -> FineTuningJob:
    kwargs = {
        "model": base_model,
        "training_file": train_file_id,
        "suffix": suffix,
    }
    if val_file_id:
        kwargs["validation_file"] = val_file_id
    if hyperparams:
        kwargs["hyperparameters"] = hyperparams
    return client.fine_tuning.jobs.create(**kwargs)  # type: ignore[arg-type]


def main() -> int:
    args = parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    subsets = discover_subsets(args.train_dir, args.sizes)
    if not subsets:
        logging.error("No subset files found in %s", args.train_dir)
        return 1

    client = OpenAI()
    reference_job: FineTuningJob | None = None
    if args.reference_job:
        try:
            reference_job = client.fine_tuning.jobs.retrieve(args.reference_job)
            logging.info("Loaded reference job %s (model=%s)", args.reference_job, reference_job.model)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Unable to retrieve reference job %s: %s", args.reference_job, exc)
            reference_job = None

    base_model = args.base_model or (reference_job.model if reference_job else None)
    if not base_model:
        logging.error("Base model is unknown. Provide --base-model or a retrievable --reference-job.")
        return 1

    hyperparams = resolve_hyperparams(args, reference_job)
    logging.info(
        "Using base model %s with hyperparameters: %s",
        base_model,
        json.dumps(hyperparams) if hyperparams else "{}",
    )

    val_file_id = args.validation_file_id
    if args.val_file and not val_file_id:
        if args.dry_run:
            logging.info("[dry-run] Would upload validation file %s", args.val_file)
            val_file_id = "dry-run-validation"
        else:
            val_file_id = upload_file(client, args.val_file)
    elif not args.val_file:
        logging.info("Validation file not provided; jobs will train without validation splits.")

    for subset in subsets:
        suffix = args.suffix_template.format(prefix=args.suffix_prefix, size=subset.size)
        logging.info("Processing subset %s (%d examples)", subset.path, subset.size)

        if args.dry_run:
            logging.info("[dry-run] Would upload %s and launch job with suffix %s", subset.path, suffix)
            job_info = {
                "size": subset.size,
                "train_file": str(subset.path),
                "suffix": suffix,
                "base_model": base_model,
                "reference_job": args.reference_job,
                "validation_file_id": val_file_id,
                "hyperparameters": hyperparams,
                "dry_run": True,
            }
            append_job_record(args.jobs_file, job_info)
            continue

        train_file_id = upload_file(client, subset.path)
        job = create_job(client, base_model, train_file_id, val_file_id, suffix, hyperparams)
        logging.info("Started job %s for %s (%s)", job.id, subset.path, suffix)
        record = {
            "size": subset.size,
            "train_file": str(subset.path),
            "train_file_id": train_file_id,
            "validation_file_id": val_file_id,
            "job_id": job.id,
            "status": job.status,
            "created_at": job.created_at,
            "base_model": base_model,
            "hyperparameters": hyperparams,
            "reference_job": args.reference_job,
            "result_model": getattr(job, "fine_tuned_model", None),
            "suffix": suffix,
        }
        append_job_record(args.jobs_file, record)

    logging.info("Job metadata appended to %s", args.jobs_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

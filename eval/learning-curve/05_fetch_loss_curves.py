#!/usr/bin/env python3
"""Download loss curves for completed fine-tuning jobs and plot them."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from openai import OpenAI, OpenAIError


MPLCONFIG_DEFAULT = ".mplconfig"


def load_env(files: Iterable[str]) -> None:
    for env_path in files:
        path = Path(env_path)
        if not path.exists():
            continue
        for key, value in dotenv_values(path).items():
            if value is not None:
                os.environ.setdefault(key, value)


def read_jobs(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def list_events(client: OpenAI, job_id: str) -> List[dict]:
    collected: List[dict] = []
    after: str | None = None
    while True:
        kwargs = {"limit": 100}
        if after:
            kwargs["after"] = after
        response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, **kwargs)
        collected.extend(response.data)
        if not response.has_more:
            break
        after = response.data[-1].id
    return collected


def extract_metrics(events: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for event in events:
        if event.type != "metrics":
            continue
        data = event.data or {}
        step = data.get("step")
        if step is None:
            continue
        rows.append({
            "created_at": event.created_at,
            "step": step,
            "training_loss": data.get("training_loss", data.get("train_loss")),
            "validation_loss": data.get("validation_loss", data.get("valid_loss")),
            "full_validation_loss": data.get("full_valid_loss"),
            "training_accuracy": data.get("training_mean_token_accuracy", data.get("train_mean_token_accuracy")),
            "validation_accuracy": data.get("validation_mean_token_accuracy", data.get("valid_mean_token_accuracy")),
            "full_validation_accuracy": data.get("full_valid_mean_token_accuracy"),
        })
    if not rows:
        return pd.DataFrame(columns=["created_at", "step", "training_loss", "validation_loss"])
    frame = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return frame


def plot_series(ax, steps, values, label, marker):
    ax.plot(steps, values, marker=marker, linestyle="-", linewidth=1.2, markersize=3, label=label)


def compute_cubic_spline(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    n = len(x)
    if n < 4:
        return np.interp(query, x, y)
    h = np.diff(x)
    al = np.zeros(n)
    for i in range(1, n - 1):
        al[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1])
    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (al[i] - h[i - 1] * z[i - 1]) / l[i]
    c = np.zeros(n)
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = ((y[j + 1] - y[j]) / h[j]) - (h[j] * (c[j + 1] + 2 * c[j]) / 3)
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])
    a = y[:-1]

    def evaluate(xp: float) -> float:
        xp = min(max(x[0], xp), x[-1])
        idx = np.searchsorted(x, xp) - 1
        idx = max(0, min(idx, n - 2))
        dx = xp - x[idx]
        return a[idx] + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx

    return np.array([evaluate(val) for val in query])


def build_validation_curve(frame: pd.DataFrame, column: str, resolution: int = 400):
    subset = frame[["step", column]].dropna()
    if subset.empty:
        return None
    steps = subset["step"].to_numpy()
    values = subset[column].to_numpy()
    if len(steps) == 1:
        return steps, values, steps, values
    new_steps = np.linspace(steps.min(), steps.max(), resolution)
    new_values = compute_cubic_spline(steps, values, new_steps)
    return new_steps, new_values, steps, values


def plot_metrics(frame: pd.DataFrame, title: str, output_path: Path, smooth_window: int) -> None:
    fig, ax = plt.subplots()
    plotted = 0

    if frame["training_loss"].notna().any():
        values = frame["training_loss"]
        window = max(1, smooth_window)
        if window > 1:
            values = values.rolling(window=window, min_periods=1).mean()
        ax.plot(frame["step"], values, label="training", linewidth=1.2, color="#1f77b4")
        plotted += 1

    for column, label, color in [
        ("validation_loss", "validation", "#ff7f0e"),
    ]:
        curve = build_validation_curve(frame, column)
        if curve is None:
            continue
        smooth_steps, smooth_values, raw_steps, raw_values = curve
        ax.plot(smooth_steps, smooth_values, label=f"{label} (spline)", linewidth=1.6, color=color)
        ax.scatter(raw_steps, raw_values, color=color, s=15, alpha=0.6)
        plotted += 1

    full_val = frame["full_validation_loss"].dropna()
    if not full_val.empty:
        final_full = full_val.iloc[-1]
        ax.text(
            0.98,
            0.02,
            f"Final validation loss: {final_full:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    if plotted:
        ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def friendly_model_name(model_name: str) -> str:
    text = model_name or "Fine-tuned model"
    lowered = text.lower()
    if "gpt-4o" in lowered:
        return "GPT-4o FT"
    if "gpt-4.1" in lowered:
        return "GPT-4.1 FT"
    if "gpt-3.5" in lowered:
        return "GPT-3.5 FT"
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs-file",
        type=Path,
        default=Path("eval/learning-curve/finetune_jobs.jsonl"),
        help="JSONL file containing job metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/learning-curve/curves"),
        help="Directory where CSV/PNG files are saved.",
    )
    parser.add_argument(
        "--env-files",
        nargs="*",
        default=[".env", "eval/.env"],
        help=".env files to load before calling the API.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Rolling-window size (in steps) for smoothing the plotted curves.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", MPLCONFIG_DEFAULT)
    load_env(args.env_files)
    jobs = read_jobs(args.jobs_file)
    if not jobs:
        raise SystemExit(f"No jobs found in {args.jobs_file}")

    client = OpenAI()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        job_id = job["job_id"]
        print(f"Processing {job_id}...")
        try:
            job_info = client.fine_tuning.jobs.retrieve(job_id)
        except OpenAIError as exc:
            print(f"  Failed to retrieve job {job_id}: {exc}")
            continue

        events = list_events(client, job_id)
        metrics = extract_metrics(events)
        if metrics.empty:
            print(f"  No metrics available for {job_id}")
            continue

        csv_path = args.output_dir / f"{job_id}_loss.csv"
        metrics.to_csv(csv_path, index=False)
        png_path = args.output_dir / f"{job_id}_loss.png"
        model_name = job_info.fine_tuned_model or "Fine-tuned model"
        size = job.get("size", "?")
        title = f"{friendly_model_name(model_name)} ({size} examples)"
        plot_metrics(metrics, title, png_path, args.smooth_window)
        print(f"  Saved {csv_path} and {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

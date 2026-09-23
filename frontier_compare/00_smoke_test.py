#!/usr/bin/env python3
"""Smoke test: can each model read text, tables, AND figures from a PDF through our request path?

Builds a synthetic PDF whose facts are split across three channels:
  text   - a sentence in the body text layer
  table  - a ruled table drawn as real (extractable) text
  figure - a raster bar chart; its values and colours exist only as pixels, so a text/OCR
           layer alone cannot answer the figure questions
Then sends it with the same content builder used by 02_query_models.py and scores the answers.
Optionally (--real-pmid) also sends a real eval paper with open-ended probes for manual review.

  python frontier_compare/00_smoke_test.py --model gpt6-astra kimi-k3 --real-pmid 40400229
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import sys
import time
from pathlib import Path

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pymupdf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier_compare import config  # noqa: E402

qm = importlib.import_module("frontier_compare.02_query_models")

SMOKE_DIR = config.FC_DIR / "data/smoke"
EXPECTED = {
    "text_code": "ZEBRA-4471",
    "table_kisumu_n": "73",
    "table_kisumu_year": "2019",
    "figure_tallest_label": "M184V",
    "figure_tallest_value": "61",
    "figure_tallest_color": "green",
    "figure_code": "ORCHID-92",
}
PROBE = """Answer from the attached document only. Return ONLY a JSON object with these keys:
"text_code": the sentinel cohort identifier stated in the body text,
"table_kisumu_n": the number of participants (n) for the Kisumu row of Table 1,
"table_kisumu_year": the sampling year for the Kisumu row of Table 1,
"figure_tallest_label": the mutation label of the tallest bar in Figure 1,
"figure_tallest_value": the percentage printed above the tallest bar in Figure 1 (number only),
"figure_tallest_color": the fill colour of the tallest bar in Figure 1 (one common colour word),
"figure_code": the code printed in the bottom-right corner inside the Figure 1 image,
"channels": for each key above, whether you read it from text, table, or image."""

REAL_PROBE = """Answer from the attached article only. Return ONLY a JSON object with keys:
"title": the article title,
"table1_caption": the caption of Table 1, verbatim,
"table1_first_row": the first data row of Table 1 exactly as printed (all cells),
"figure1_description": what Figure 1 shows: chart type, axes/labels, panels, and two specific values or labels visible in the figure,
"figure1_detail_not_in_caption": one concrete detail visible in Figure 1 that is NOT stated in its caption or the body text."""


def build_synthetic_pdf(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=150)
    labels, values, colors = ["K65R", "M184V", "K103N", "Y181C"], [38, 61, 22, 45], ["tab:blue", "tab:green", "tab:orange", "tab:purple"]
    ax.bar(labels, values, color=colors)
    for x, v in enumerate(values):
        ax.text(x, v + 1, f"{v}%", ha="center", fontsize=9)
    ax.set_ylim(0, 75)
    ax.set_ylabel("Prevalence")
    fig.text(0.98, 0.02, "ORCHID-92", ha="right", fontsize=7, color="gray")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)

    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for line in [
        "Drug resistance surveillance in a synthetic cohort (smoke-test document)",
        "",
        "Methods. Plasma samples were genotyped by Sanger sequencing of protease and reverse transcriptase.",
        "The sentinel cohort identifier is ZEBRA-4471. Results by site are shown in Table 1 and Figure 1.",
        "",
        "Table 1. Participants by site.",
    ]:
        page.insert_text((72, y), line, fontsize=10)
        y += 15
    rows = [("Site", "n", "Year"), ("Nairobi", "58", "2018"), ("Kisumu", "73", "2019"), ("Mombasa", "41", "2020")]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(72 + c * 110, y + r * 18, 182 + c * 110, y + (r + 1) * 18)
            page.draw_rect(rect, color=(0, 0, 0), width=0.5)
            page.insert_text((rect.x0 + 4, rect.y1 - 5), cell, fontsize=10)
    y += len(rows) * 18 + 20
    page.insert_image(pymupdf.Rect(72, y, 72 + 360, y + 230), stream=buf.getvalue())
    page.insert_text((72, y + 245), "Figure 1. Prevalence of selected mutations.", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)


def ask(spec: config.ModelSpec, pdf: Path, probe: str) -> tuple[dict | None, dict]:
    payload = qm.build_request(spec, "You extract information from scientific documents.",
                               qm.build_user_content(spec, pdf, probe))
    url, key_env = config.PROVIDER_ENDPOINTS[spec.provider]
    started = time.monotonic()
    resp = httpx.post(url, headers={"Authorization": f"Bearer {os.environ[key_env]}"}, json=payload, timeout=900)
    meta = {"status": resp.status_code, "latency_s": round(time.monotonic() - started, 1)}
    if resp.status_code >= 400:
        meta["error"] = resp.text[:1500]
        return None, meta
    body = resp.json()
    content = body["choices"][0]["message"].get("content") or ""
    meta.update(usage=body.get("usage"), served_model=body.get("model"), provider=body.get("provider"))
    parse = importlib.import_module("frontier_compare.03_parse_responses").parse_content
    parsed, status = parse(content)
    meta["json_status"] = status
    if parsed is None:
        meta["raw"] = content[:1500]
    return parsed, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", default=sorted(config.MODELS), choices=sorted(config.MODELS))
    parser.add_argument("--real-pmid", help="Also probe this staged eval PDF (manual review).")
    args = parser.parse_args()
    qm.load_env()

    pdf = SMOKE_DIR / "synthetic_probe.pdf"
    build_synthetic_pdf(pdf)
    report: dict = {}
    for key in args.model:
        spec = config.MODELS[key]
        answers, meta = ask(spec, pdf, PROBE)
        checks = {}
        for k, want in EXPECTED.items():
            got = str((answers or {}).get(k, ""))
            checks[k] = {"expected": want, "got": got, "pass": want.lower() in got.lower()}
        report[key] = {"meta": meta, "checks": checks, "channels": (answers or {}).get("channels")}
        passed = sum(c["pass"] for c in checks.values())
        print(f"\n=== {spec.label} ({spec.model_id}) synthetic: {passed}/{len(checks)} "
              f"[HTTP {meta['status']}, {meta['latency_s']}s, json={meta.get('json_status')}]")
        for k, c in checks.items():
            print(f"  {'PASS' if c['pass'] else 'FAIL'}  {k}: expected {c['expected']!r}, got {c['got']!r}")
        if "error" in meta or "raw" in meta:
            print("  ", meta.get("error") or meta.get("raw"))

        if args.real_pmid:
            real_answers, real_meta = ask(spec, qm.pdf_path(args.real_pmid), REAL_PROBE)
            report[key]["real"] = {"pmid": args.real_pmid, "meta": real_meta, "answers": real_answers}
            print(f"--- real PMID {args.real_pmid} [HTTP {real_meta['status']}, {real_meta['latency_s']}s]")
            print(json.dumps(real_answers, indent=2, ensure_ascii=False) if real_answers else real_meta)

    out = SMOKE_DIR / "smoke_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Inventory full-text PDFs for the 150 eval PMIDs and stage them into frontier_compare/pdfs/.

Lookup order per PMID:
  1. frontier_compare/pdfs/<PMID>.pdf (already staged)
  2. advanced-prompting/papers_2025_30/<PMID>/*.pdf or advanced-prompting/papers/<PMID>/*.pdf
     (files named "* copy.pdf" are ignored as duplicates)

Writes data/pdf_manifest.csv. PMIDs with status "missing" must be obtained manually
(publisher PDF of the version the human annotators used, including supplements if merged).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier_compare import config  # noqa: E402


def eval_pmids() -> pd.DataFrame:
    df = pd.read_excel(config.MERGED_PATH, dtype=str, usecols=["PMID"])
    pmids = df["PMID"].str.replace(r"\.0$", "", regex=True).str.strip().drop_duplicates()
    new30 = {p.name for p in config.NEW30_DIR.iterdir() if p.is_dir()}
    return pd.DataFrame(
        {"PMID": pmids, "Set": ["new30" if p in new30 else "original120" for p in pmids]}
    ).reset_index(drop=True)


def find_source(pmid: str) -> tuple[Path | None, int]:
    staged = config.PDF_DIR / f"{pmid}.pdf"
    if staged.exists():
        return staged, 1
    for base in (config.NEW30_DIR, config.ORIGINAL120_DIR):
        folder = base / pmid
        if not folder.is_dir():
            continue
        pdfs = sorted(p for p in folder.rglob("*.pdf") if not p.stem.endswith(" copy"))
        if pdfs:
            return pdfs[0], len(pdfs)
    return None, 0


def describe(path: Path) -> dict:
    data = path.read_bytes()
    try:
        pages = len(PdfReader(path).pages)
    except Exception:  # noqa: BLE001 - corrupt/encrypted PDFs are reported, not fatal
        pages = None
    return {
        "size_mb": round(len(data) / 1e6, 2),
        "pages": pages,
        "sha256": hashlib.sha256(data).hexdigest()[:16],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true", help="Copy found PDFs into frontier_compare/pdfs/.")
    args = parser.parse_args()

    rows = []
    for rec in eval_pmids().itertuples(index=False):
        src, n_candidates = find_source(rec.PMID)
        row = {"PMID": rec.PMID, "Set": rec.Set, "source": "", "candidates": n_candidates, "status": "missing"}
        if src is not None:
            row.update(source=str(src.relative_to(config.ROOT)), status="ok", **describe(src))
            if n_candidates > 1:
                row["status"] = "ok_multiple_candidates"
            if args.stage and src.parent != config.PDF_DIR:
                config.PDF_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, config.PDF_DIR / f"{rec.PMID}.pdf")
        rows.append(row)

    manifest = pd.DataFrame(rows)
    config.PDF_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(config.PDF_MANIFEST, index=False)
    print(manifest.groupby(["Set", "status"]).size().to_string())
    print(f"\nWrote {config.PDF_MANIFEST.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

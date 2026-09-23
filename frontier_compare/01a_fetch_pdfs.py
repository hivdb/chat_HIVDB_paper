#!/usr/bin/env python3
"""Fetch open-access full-text PDFs for eval PMIDs that have no staged PDF.

PMID -> PMCID/DOI via the NCBI ID converter, then try in order:
  1. PMC Open Access dataset on AWS S3 (s3://pmc-oa-opendata, latest article version's PDF)
  2. The publisher's full-text PDF link registered with Crossref (application/pdf)
The PMC/Europe PMC web PDF links sit behind bot challenges and are deliberately not used.
Only responses starting with %PDF are kept. Results go to data/fetch_log.csv; anything
not fetched is listed for manual download into frontier_compare/pdfs/<PMID>.pdf.
Then re-run 01_pdf_manifest.py to refresh the manifest.
"""

from __future__ import annotations

import asyncio
import io
import re
import sys
import tarfile
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontier_compare import config  # noqa: E402

IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
S3 = "https://pmc-oa-opendata.s3.amazonaws.com/"
CROSSREF = "https://api.crossref.org/works/"
HEADERS = {"User-Agent": "chat_HIVDB_paper frontier_compare (research; PMC open access)"}
NCBI_RATE = asyncio.Semaphore(3)  # NCBI E-utilities etiquette without an API key
LOG_PATH = config.FC_DIR / "data/fetch_log.csv"


def idconv(pmids: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(pmids), 150):
        r = httpx.get(IDCONV, params={"ids": ",".join(pmids[i : i + 150]), "format": "json",
                                      "tool": "chat_hivdb_frontier"}, headers=HEADERS, timeout=60, follow_redirects=True)
        r.raise_for_status()
        out.update({rec["requested-id"]: rec for rec in r.json()["records"]})
    return out


def pubmed_dois(pmids: list[str]) -> dict[str, str]:
    """DOIs from PubMed for PMIDs the ID converter could not map (no PMC record)."""
    if not pmids:
        return {}
    r = httpx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                  params={"db": "pubmed", "id": ",".join(pmids), "retmode": "json", "tool": "chat_hivdb_frontier"},
                  headers=HEADERS, timeout=60)
    r.raise_for_status()
    result = r.json()["result"]
    return {p: next((a["value"] for a in result.get(p, {}).get("articleids", []) if a["idtype"] == "doi"), "")
            for p in pmids}


def is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


async def get(client: httpx.AsyncClient, url: str, ncbi: bool = False, **kw) -> httpx.Response | None:
    try:
        if ncbi:
            async with NCBI_RATE:
                r = await client.get(url, **kw)
                await asyncio.sleep(0.35)
        else:
            r = await client.get(url, **kw)
        return r if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


async def from_s3(client: httpx.AsyncClient, pmcid: str) -> bytes | None:
    listing = await get(client, S3, params={"list-type": "2", "prefix": f"{pmcid}."})
    if listing is None:
        return None
    keys = [k for k in re.findall(r"<Key>([^<]+\.pdf)</Key>", listing.text) if k.split("/")[0] + ".pdf" == k.split("/")[-1]]
    if not keys:
        return None
    latest = max(keys, key=lambda k: int(k.split("/")[0].rsplit(".", 1)[1]))
    r = await get(client, S3 + latest)
    return r.content if r is not None else None


async def from_crossref(client: httpx.AsyncClient, doi: str) -> bytes | None:
    meta = await get(client, CROSSREF + doi)
    if meta is None:
        return None
    links = [l["URL"] for l in meta.json()["message"].get("link", []) if l.get("content-type") == "application/pdf"]
    for url in links:
        r = await get(client, url)
        if r is not None and is_pdf(r.content):
            return r.content
    return None


async def fetch_one(client: httpx.AsyncClient, pmid: str, rec: dict) -> dict:
    pmcid, doi = rec.get("pmcid"), rec.get("doi")
    row = {"PMID": pmid, "PMCID": pmcid or "", "DOI": doi or "", "status": "", "source": ""}
    attempts = []
    if pmcid:
        attempts.append(("pmc_s3", lambda: from_s3(client, pmcid)))
    if doi:
        attempts.append(("publisher_crossref", lambda: from_crossref(client, doi)))
    for name, attempt in attempts:
        result = await attempt()
        data = result.content if isinstance(result, httpx.Response) else result
        if data and is_pdf(data):
            (config.PDF_DIR / f"{pmid}.pdf").write_bytes(data)
            row.update(status="ok", source=name, size_mb=round(len(data) / 1e6, 2))
            return row
    row["status"] = "not_found" if attempts else "no_ids"
    return row


async def main_async(pmids: list[str]) -> pd.DataFrame:
    ids = idconv(pmids)
    for pmid, doi in pubmed_dois([p for p in pmids if not ids.get(p, {}).get("pmcid")]).items():
        if doi:
            ids.setdefault(pmid, {})["doi"] = doi
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(headers=HEADERS, timeout=120, follow_redirects=True,
                                 limits=httpx.Limits(max_connections=12)) as client:
        rows = await asyncio.gather(*(fetch_one(client, p, ids.get(p, {})) for p in pmids))
    return pd.DataFrame(rows)


def main() -> int:
    manifest = pd.read_csv(config.PDF_MANIFEST, dtype=str)
    pmids = [p for p in manifest["PMID"] if not (config.PDF_DIR / f"{p}.pdf").exists()]
    if not pmids:
        print("All PMIDs already have staged PDFs.")
        return 0
    log = asyncio.run(main_async(pmids))
    log.to_csv(LOG_PATH, index=False)
    print(log.groupby(["status", "source"]).size().to_string())
    missing = log[log["status"] != "ok"]
    if len(missing):
        print("\nNot fetched (download manually to frontier_compare/pdfs/<PMID>.pdf):")
        print(missing[["PMID", "PMCID", "DOI", "status"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

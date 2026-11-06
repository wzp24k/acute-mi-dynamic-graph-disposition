"""Minimal PubMed search fallback when the academic-search MCP is unavailable."""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request


BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _get_json(endpoint: str, params: dict) -> dict:
    url = f"{BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--retmax", type=int, default=10)
    args = parser.parse_args()
    search = _get_json(
        "esearch.fcgi",
        {
            "db": "pubmed",
            "term": args.query,
            "retmode": "json",
            "retmax": args.retmax,
            "sort": "relevance",
        },
    )
    ids = search["esearchresult"].get("idlist", [])
    if not ids:
        print("No PubMed records found.")
        return
    time.sleep(0.4)
    summary = _get_json(
        "esummary.fcgi",
        {"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
    )
    for index, pmid in enumerate(ids, start=1):
        record = summary["result"].get(pmid, {})
        authors = ", ".join(
            author.get("name", "")
            for author in record.get("authors", [])[:4]
        )
        doi = next(
            (
                item.get("value", "")
                for item in record.get("articleids", [])
                if item.get("idtype") == "doi"
            ),
            "",
        )
        print(
            f"{index}. {record.get('title', '')}\n"
            f"   {authors}; {record.get('fulljournalname', '')} "
            f"{record.get('pubdate', '')}; PMID {pmid}; DOI {doi}"
        )


if __name__ == "__main__":
    main()

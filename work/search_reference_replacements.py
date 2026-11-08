"""Search Crossref for verifiable replacements for arXiv-only references."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


USER_AGENT = "CTOGT-reference-search/1.0 (mailto:local@example.org)"
QUERIES = {
    "vaswani2017attention": "transformer models review attention sequence modeling",
    "arjovsky2019irm": "invariant learning domain generalization survey",
    "sagawa2020groupdro": "distributionally robust optimization machine learning survey",
    "kidger2020ncde": "neural controlled differential equations irregular time series",
    "gu2023mamba": "state space models long sequence modeling survey",
}


def search(query: str, rows: int = 12) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "query.bibliographic": query,
            "rows": rows,
            "select": "DOI,title,subtitle,container-title,issued,type,author,volume,issue,page",
        }
    )
    url = f"https://api.crossref.org/works?{params}"
    for attempt in range(4):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.load(response)["message"]["items"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            time.sleep(1.5 * (attempt + 1))
    return []


def display(item: dict) -> dict:
    title_parts = [*item.get("title", []), *item.get("subtitle", [])]
    return {
        "doi": item.get("DOI", ""),
        "title": ": ".join(value for value in title_parts if value),
        "journal": (item.get("container-title") or [""])[0],
        "year": (
            item.get("issued", {}).get("date-parts", [[None]])[0][0]
        ),
        "type": item.get("type", ""),
    }


results = {}
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {
        executor.submit(search, query): key
        for key, query in QUERIES.items()
    }
    for future in as_completed(futures):
        key = futures[future]
        results[key] = [display(item) for item in future.result()]

for key in QUERIES:
    print(f"\n## {key}")
    for item in results[key]:
        print(
            f"{item['doi']}\t{item['year']}\t{item['type']}\t"
            f"{item['journal']}\t{item['title']}"
        )

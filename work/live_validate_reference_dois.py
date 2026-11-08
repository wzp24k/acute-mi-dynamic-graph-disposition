"""Live-check manuscript DOIs against Crossref metadata."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
REGISTRY = ROOT / "results" / "reference_registry_verified.json"
REPORT = ROOT / "results" / "reference_doi_live_validation.json"
USER_AGENT = "CTOGT-reference-audit/2.0 (mailto:local@example.org)"


def normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalise(left), normalise(right)).ratio()


def crossref_work(doi: str) -> tuple[dict | None, str | None]:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    error_message = None
    for attempt in range(4):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.load(response)["message"], None
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None, None
            if error.code not in {429, 500, 502, 503, 504}:
                return None, f"HTTP {error.code}"
            error_message = f"HTTP {error.code}"
        except (TimeoutError, urllib.error.URLError) as error:
            error_message = str(error)
        time.sleep(1.5 * (attempt + 1))
    return None, f"Crossref request failed after retries: {error_message}"


def validate_record(record: dict) -> dict:
    doi = record["doi"]
    expected_title = record["expected_title"]
    metadata, error_message = crossref_work(doi)
    if metadata is None:
        return {
            "key": record["key"],
            "doi": doi,
            "expected_title": expected_title,
            "crossref_exists": False,
            "live_doi_matches": False,
            "live_title": None,
            "live_type": None,
            "live_container": None,
            "live_year": None,
            "title_similarity": 0.0,
            "status": "ERROR" if error_message else "NOT_FOUND",
            "error": error_message,
        }

    live_doi = metadata.get("DOI", "")
    title_parts = [
        *metadata.get("title", []),
        *metadata.get("subtitle", []),
    ]
    live_title = ": ".join(
        value.strip().rstrip(".") for value in title_parts if value.strip()
    )
    return {
        "key": record["key"],
        "doi": doi,
        "expected_title": expected_title,
        "crossref_exists": True,
        "live_doi_matches": live_doi.lower() == doi.lower(),
        "live_title": live_title,
        "live_type": metadata.get("type"),
        "live_container": (metadata.get("container-title") or [""])[0],
        "live_year": (
            metadata.get("issued", {}).get("date-parts", [[None]])[0][0]
        ),
        "title_similarity": round(similarity(expected_title, live_title), 3),
        "status": "VERIFIED",
        "error": None,
    }


def main() -> None:
    records = json.loads(REGISTRY.read_text(encoding="utf-8"))
    results_by_key = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(validate_record, record): record
            for record in records
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results_by_key[result["key"]] = result
            print(
                f"{completed:02d}/{len(records)} {result['status']} "
                f"{result['key']} similarity={result['title_similarity']:.3f}",
                flush=True,
            )

    results = [results_by_key[record["key"]] for record in records]

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source": "Crossref REST API",
        "records": results,
        "summary": {
            "total": len(results),
            "verified": sum(result["status"] == "VERIFIED" for result in results),
            "not_found": sum(
                result["status"] == "NOT_FOUND" for result in results
            ),
            "errors": sum(result["status"] == "ERROR" for result in results),
            "journal_articles": sum(
                result["live_type"] == "journal-article" for result in results
            ),
            "non_journal": [
                {
                    "key": result["key"],
                    "doi": result["doi"],
                    "type": result["live_type"],
                }
                for result in results
                if result["live_type"] != "journal-article"
            ],
        },
    }
    REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(REPORT)


if __name__ == "__main__":
    main()

"""Write live Crossref journal-article checks back to the reference registry."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
REGISTRY = ROOT / "results" / "reference_registry_verified.json"
LIVE = ROOT / "results" / "reference_doi_live_validation.json"
CSV_PATH = ROOT / "results" / "reference_doi_verification.csv"

records = json.loads(REGISTRY.read_text(encoding="utf-8"))
report = json.loads(LIVE.read_text(encoding="utf-8"))
live_by_key = {record["key"]: record for record in report["records"]}

for record in records:
    live = live_by_key[record["key"]]
    verified = (
        live["status"] == "VERIFIED"
        and live["crossref_exists"]
        and live["live_doi_matches"]
        and live["live_type"] == "journal-article"
    )
    record["doi"] = live["doi"]
    record["doi_resolves"] = bool(live["crossref_exists"])
    record["metadata_verified"] = bool(
        live["crossref_exists"] and live["live_doi_matches"]
    )
    record["publication_type"] = live["live_type"]
    record["found_title"] = live["live_title"]
    record["title_similarity"] = live["title_similarity"]
    record["source"] = "Crossref"
    record["verified"] = verified
    record["verification_basis"] = (
        "Live Crossref journal-article record"
        if verified
        else "Live Crossref verification failed"
    )

REGISTRY.write_text(
    json.dumps(records, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "key",
            "doi",
            "publication_type",
            "verified",
            "verification_basis",
            "doi_resolves",
            "title_similarity",
            "found_title",
        ]
    )
    for record in records:
        writer.writerow(
            [
                record["key"],
                record["doi"],
                record["publication_type"],
                record["verified"],
                record["verification_basis"],
                record["doi_resolves"],
                record["title_similarity"],
                record["found_title"],
            ]
        )

invalid = [
    record["key"]
    for record in records
    if not record["verified"] or record["publication_type"] != "journal-article"
]
print(f"records={len(records)} invalid={invalid}")
if invalid:
    raise SystemExit(1)

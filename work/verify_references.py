"""Verify DOI metadata for the manuscript reference list."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REFERENCES = [
    {
        "key": "johnson2023mimiciv",
        "doi": "10.1038/s41597-022-01899-x",
        "fallback": "Johnson AEW, et al. MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data. 2023.",
    },
    {
        "key": "johnson2016mimiciii",
        "doi": "10.1038/sdata.2016.35",
        "fallback": "Johnson AEW, et al. MIMIC-III, a freely accessible critical care database. Scientific Data. 2016.",
    },
    {
        "key": "pollard2018eicu",
        "doi": "10.1038/sdata.2018.178",
        "fallback": "Pollard TJ, et al. The eICU Collaborative Research Database. Scientific Data. 2018.",
    },
    {
        "key": "collins2024tripodai",
        "doi": "10.1136/bmj-2023-078378",
        "fallback": "Collins GS, et al. TRIPOD+AI statement. BMJ. 2024.",
    },
    {
        "key": "vincent1996sofa",
        "doi": "10.1007/BF01709751",
        "fallback": "Vincent JL, et al. The SOFA score. Intensive Care Medicine. 1996.",
    },
    {
        "key": "che2018grud",
        "doi": "10.1038/s41598-018-24271-9",
        "fallback": "Che Z, et al. Recurrent neural networks for multivariate time series with missing values. Scientific Reports. 2018.",
    },
    {
        "key": "lim2021tft",
        "doi": "10.1016/j.ijforecast.2021.03.012",
        "fallback": "Lim B, et al. Temporal Fusion Transformers. International Journal of Forecasting. 2021.",
    },
    {
        "key": "vancalster2019calibration",
        "doi": "10.1186/s12916-019-1466-7",
        "fallback": "Van Calster B, et al. Calibration: the Achilles heel of predictive analytics. BMC Medicine. 2019.",
    },
    {
        "key": "vickers2006dca",
        "doi": "10.1177/0272989X06295361",
        "fallback": "Vickers AJ, Elkin EB. Decision curve analysis. Medical Decision Making. 2006.",
    },
    {
        "key": "thygesen2019mi",
        "doi": "10.1093/eurheartj/ehy462",
        "fallback": "Thygesen K, et al. Fourth universal definition of myocardial infarction. European Heart Journal. 2019.",
    },
    {
        "key": "sacco2013stroke",
        "doi": "10.1161/STR.0b013e318296aeca",
        "fallback": "Sacco RL, et al. An updated definition of stroke. Stroke. 2013.",
    },
    {
        "key": "shickel2018deepehr",
        "doi": "10.1109/JBHI.2017.2767063",
        "fallback": "Shickel B, et al. Deep EHR: a survey. IEEE Journal of Biomedical and Health Informatics. 2018.",
    },
    {
        "key": "rajkomar2018scalable",
        "doi": "10.1038/s41746-018-0029-1",
        "fallback": "Rajkomar A, et al. Scalable and accurate deep learning with electronic health records. npj Digital Medicine. 2018.",
    },
]


def fetch(doi: str) -> dict | None:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CTOGT research verification/1.0 (local academic project)"
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
        return payload["message"]
    except Exception as error:
        print(f"WARN {doi}: {error}")
        return None


def main() -> None:
    output = []
    for item in REFERENCES:
        metadata = fetch(item["doi"])
        if metadata is None:
            output.append({**item, "verified": False})
            continue
        authors = []
        for author in metadata.get("author", []):
            family = author.get("family", "")
            given = author.get("given", "")
            authors.append(f"{family} {given}".strip())
        output.append(
            {
                **item,
                "verified": True,
                "title": (metadata.get("title") or [""])[0],
                "journal": (metadata.get("container-title") or [""])[0],
                "year": (
                    metadata.get("published-print")
                    or metadata.get("published-online")
                    or metadata.get("issued")
                    or {}
                ).get("date-parts", [[None]])[0][0],
                "volume": metadata.get("volume", ""),
                "issue": metadata.get("issue", ""),
                "page": metadata.get("page", ""),
                "authors": authors,
            }
        )
        print(f"VERIFIED {item['key']}: {output[-1]['title']}")
        time.sleep(0.2)
    destination = ROOT / "data" / "references"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "references.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {destination / 'references.json'}")
    arxiv_ids = [
        "1706.03762",
        "1907.02893",
        "1911.08731",
        "2005.08926",
        "2312.00752",
    ]
    url = (
        "https://export.arxiv.org/api/query?id_list="
        + ",".join(arxiv_ids)
        + "&max_results=10"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CTOGT research verification/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        root = ET.fromstring(response.read())
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    arxiv_output = []
    for entry in root.findall("atom:entry", namespace):
        arxiv_output.append(
            {
                "id": entry.findtext("atom:id", default="", namespaces=namespace),
                "title": " ".join(
                    entry.findtext(
                        "atom:title", default="", namespaces=namespace
                    ).split()
                ),
                "published": entry.findtext(
                    "atom:published", default="", namespaces=namespace
                ),
                "authors": [
                    author.findtext(
                        "atom:name", default="", namespaces=namespace
                    )
                    for author in entry.findall(
                        "atom:author", namespace
                    )
                ],
            }
        )
    (destination / "arxiv_references.json").write_text(
        json.dumps(arxiv_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for item in arxiv_output:
        print(f"VERIFIED {item['id']}: {item['title']}")


if __name__ == "__main__":
    main()

"""Reviewer 1 audit for wording, numbers, citation order, and cross-references."""
from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
DOCX = ROOT / "1108manuscript.docx"
OUTPUT = ROOT / "results" / "reviewer1_consistency_audit.json"

document = Document(DOCX)
paragraphs = [
    " ".join(paragraph.text.split())
    for paragraph in document.paragraphs
    if paragraph.text.strip()
]
text = "\n".join(paragraphs)
start = paragraphs.index("1. Introduction")
end = paragraphs.index("7. References")
body = paragraphs[start:end]

required_claims = [
    "13,131 patients and 4,475 diagnosis-coded MACCE events (34.08%)",
    "3,470 patients and 409 events (11.79%)",
    "5,913 patients and 642 events (10.86%)",
    "across 160 sites, including 43 analytic sites",
    "The 2020-2022 holdout contained 2,456 patients and 1,072 events",
    "the locked model achieved 2020-2022 AUROC 0.75, versus 0.73",
    "45 strongly confirmed MIMIC-IV events",
    "1,265 events whose specific anchor interval overlapped the target window",
    "Retrained XGB+TS reached AUROC 0.79",
    "contained 76 events and yielded AUROC 0.75",
    "contained 119 overlap events and yielded AUROC 0.67",
    "Target support was 26.32% in the pilot",
]
stale_phrases = [
    "34.10%",
    "11.80%",
    "10.90%",
    "166 sites",
    "0.0098",
    "MIMIC-III temporal transport",
    "AUROC 0.80 for the MIMIC-IV anchor-overlap endpoint",
    "contained 76 events and yielded AUROC 0.76",
    "strongly tentatively supported",
    "tentatively supported timing strata",
    "Which met the 0.01 material criterion",
    "Was therefore not evaluated",
    "an conservative timing stratum",
    "Event window.",
    "require author review before submission",
    "-0.00",
]

citation_groups = re.findall(r"\[([0-9,\-]+)\]", text)
groups_ascending = all(
    [int(token.split("-", 1)[0]) for token in group.split(",")]
    == sorted(int(token.split("-", 1)[0]) for token in group.split(","))
    for group in citation_groups
)

first_positions = {}
for index, paragraph in enumerate(body):
    for match in re.finditer(r"\[([0-9,\-]+)\]", paragraph):
        for token in match.group(1).split(","):
            values = (
                range(
                    int(token.split("-", 1)[0]),
                    int(token.split("-", 1)[1]) + 1,
                )
                if "-" in token
                else [int(token)]
            )
            for value in values:
                first_positions.setdefault(value, index)
first_appearance_ascending = (
    sorted(first_positions, key=first_positions.get) == list(range(1, 46))
)

bookmarks = {
    element.get(qn("w:name"))
    for element in document.element.body.iter(qn("w:bookmarkStart"))
    if (element.get(qn("w:name")) or "").startswith("Ref")
}
anchors = [
    element.get(qn("w:anchor"))
    for element in document.element.body.iter(qn("w:hyperlink"))
    if element.get(qn("w:anchor"))
]

figure_positions = {
    f"Figure {index}": text.find(f"Figure {index}") for index in range(1, 10)
}
table_names = [
    "Table 1",
    "Table 2",
    "Table 3",
    "Table 4",
    "Table 5A",
    "Table 5B",
    "Table 6",
    "Table 7",
    "Table 8",
]
table_positions = {name: text.find(name) for name in table_names}

checks = {
    "required_claims_present": all(value in text for value in required_claims),
    "stale_or_inconsistent_phrases_absent": not any(
        value in text for value in stale_phrases
    ),
    "citation_groups_ascending": groups_ascending,
    "no_compressed_hyperlink_ranges": not bool(
        re.search(r"\[[0-9,\-]*\d-\d[0-9,\-]*\]", text)
    ),
    "first_citation_order_1_to_45": first_appearance_ascending,
    "all_citation_hyperlinks_resolve": (
        bookmarks == {f"Ref{index}" for index in range(1, 46)}
        and set(anchors) == bookmarks
    ),
    "all_figures_cited": all(position >= 0 for position in figure_positions.values()),
    "all_tables_cited": all(position >= 0 for position in table_positions.values()),
    "figure_number_order": list(figure_positions.values())
    == sorted(figure_positions.values()),
    "table_number_order": list(table_positions.values())
    == sorted(table_positions.values()),
    "component_overlap_note_present": (
        "Component event counts may overlap" in text
    ),
    "single_panel_figure1_caption": (
        "(A) Integrated cohort construction" in text
        and "(B) Recorded outcome components" not in text
    ),
    "no_fragmentary_sentences": not bool(
        re.search(r"(?:^|\n)(?:Which|Was|Were|Event window\.)\b", text)
    ),
}

report = {
    "checks": checks,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "citation_groups": citation_groups,
    "bookmark_count": len(bookmarks),
    "hyperlink_count": len(anchors),
    "figure_first_positions": figure_positions,
    "table_first_positions": table_positions,
}
OUTPUT.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(report, indent=2, ensure_ascii=False))

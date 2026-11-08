"""Reviewer 3 audit for clinical interpretation and translational boundaries."""
from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
DOCX = ROOT / "1108manuscript.docx"
OUTPUT = ROOT / "results" / "reviewer3_perspective_audit.json"

document = Document(DOCX)
paragraphs = [
    " ".join(paragraph.text.split())
    for paragraph in document.paragraphs
    if paragraph.text.strip()
]
text = "\n".join(paragraphs)
reference_start = paragraphs.index("7. References")
nonreference = paragraphs[:reference_start]
sentences = []
for value in nonreference:
    sentences.extend(
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", value)
        if item.strip()
    )
max_sentence_words = max((len(value.split()) for value in sentences), default=0)

bookmarks = {
    element.get(qn("w:name"))
    for element in document.element.body.iter(qn("w:bookmarkStart"))
    if (element.get(qn("w:name")) or "").startswith("Ref")
}
anchors = {
    element.get(qn("w:anchor"))
    for element in document.element.body.iter(qn("w:hyperlink"))
    if element.get(qn("w:anchor"))
}

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
    "no_direct_deployment_claim": (
        "They do not support direct clinical deployment" in text
    ),
    "thresholds_are_descriptive": (
        "not recommended clinical cutoffs" in text
        and "descriptive 10% threshold" in text
    ),
    "decision_curve_prevalence_scope": (
        "Decision-curve results are prevalence-dependent" in text
    ),
    "attention_not_causal": (
        "Attention is an internal model weight" in text
        and "not interpreted as a causal organ interaction" in text
    ),
    "negative_graph_finding_retained": (
        "did not improve materially over XGB+TS" in text
    ),
    "external_calibration_limitation_retained": (
        "calibration was poor until locally recalibrated" in text
    ),
    "holdout_case_mix_disclosed": (
        "43.65%, versus 34.08% in development" in text
    ),
    "multiplicity_limitation_disclosed": (
        "confidence intervals were not adjusted for multiplicity" in text
    ),
    "historical_transport_naming_clear": (
        "same-institution historical transport" in text
        and "MIMIC-III temporal transport" not in text
    ),
    "race_predictor_disclosed": (
        "Static predictors included age, sex, race indicators" in text
        and "Race indicators were not recorded in eICU" in text
    ),
    "fairness_limitation_disclosed": (
        "Fairness across race and ethnicity was not evaluated" in text
    ),
    "site_heterogeneity_limitation_disclosed": (
        "Site-level performance heterogeneity was not modeled" in text
    ),
    "all_figures_cited_in_order": all(
        position >= 0 for position in figure_positions.values()
    )
    and list(figure_positions.values()) == sorted(figure_positions.values()),
    "all_tables_cited_in_order": all(
        position >= 0 for position in table_positions.values()
    )
    and list(table_positions.values()) == sorted(table_positions.values()),
    "sentences_30_words_or_less": max_sentence_words <= 30,
    "all_references_have_internal_links": bookmarks == anchors,
}

report = {
    "checks": checks,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "max_sentence_words": max_sentence_words,
    "figure_first_positions": figure_positions,
    "table_first_positions": table_positions,
    "bookmark_count": len(bookmarks),
    "hyperlink_target_count": len(anchors),
}
OUTPUT.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(report, indent=2, ensure_ascii=False))

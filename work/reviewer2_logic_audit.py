"""Reviewer 2 audit for study logic, consistency, tables, and figures."""
from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
DOCX = ROOT / "1108manuscript.docx"
OUTPUT = ROOT / "results" / "reviewer2_logic_audit.json"

document = Document(DOCX)
paragraphs = [
    " ".join(paragraph.text.split())
    for paragraph in document.paragraphs
    if paragraph.text.strip()
]
text = "\n".join(paragraphs)
reference_start = paragraphs.index("7. References")
nonreference_texts = paragraphs[:reference_start]

sentences = []
for value in nonreference_texts:
    sentences.extend(
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", value)
        if item.strip()
    )
max_sentence_words = max((len(value.split()) for value in sentences), default=0)

cohort_claim = all(
    value in text
    for value in (
        "13,131 patients and 4,475 diagnosis-coded MACCE events",
        "3,470 patients and 409 events",
        "5,913 patients and 642 events",
        "across 160 sites",
        "43 analytic sites",
    )
)
site_consistency = (
    "166 sites" not in text
    and "covering 160 sites" in text
)
interval_logic = all(
    value in text
    for value in (
        "Frozen model predictions were applied to each endpoint without model retuning",
        "interval-specific XGB+TS model was trained on the interval endpoint in MIMIC-IV",
        "applied unchanged to external databases",
        "Retrained XGB+TS denotes interval-specific training in MIMIC-IV",
    )
)
interval_values = all(
    value in text
    for value in (
        "Retrained XGB+TS reached AUROC 0.79",
        "contained 76 events and yielded AUROC 0.75",
        "contained 119 overlap events and yielded AUROC 0.67",
        "temporal ensemble without interval-specific retraining produced MIMIC-IV overlap AUROC 0.77",
    )
)
figure6_scope = all(
    value in text
    for value in (
        "Net benefit with bootstrap intervals in expanded eICU",
        "Ten-percent threshold metrics in expanded eICU",
    )
)
metric_scope = (
    "Brier scores and 10% sensitivity, specificity, PPV, and NPV were bootstrapped"
    in text
)
threshold_logic = all(
    value in text
    for value in (
        "The rounded difference was 0.01",
        "the unrounded estimate remained just below the 0.01 material threshold",
        "Threshold status uses unrounded estimates",
    )
)
subgroup_completeness = "aged 75 years or older, AUROC was 0.75" in text
holdout_case_mix_disclosed = "43.65%, versus 34.08% in development" in text
overlap_explanation = "Component event counts may overlap" in text
table_na_explanation = "NA indicates" in text and any(
    cell.text.strip() == "NA"
    for table in document.tables
    for row in table.rows
    for cell in row.cells
)
cautious_interpretation = all(
    value in text
    for value in (
        "They do not support direct clinical deployment",
        "cannot establish treatment benefit or causal organ interactions",
    )
)

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

checks = {
    "cohort_counts_correct": cohort_claim,
    "site_counts_correct": site_consistency,
    "interval_analysis_defined_clearly": interval_logic,
    "interval_values_consistent": interval_values,
    "figure6_scope_explicit": figure6_scope,
    "bootstrap_metric_scope_consistent": metric_scope,
    "material_threshold_logic_consistent": threshold_logic,
    "age_75_subgroup_reported": subgroup_completeness,
    "holdout_case_mix_disclosed": holdout_case_mix_disclosed,
    "component_overlap_explained": overlap_explanation,
    "table_na_values_explained": table_na_explanation,
    "interpretation_remains_cautious": cautious_interpretation,
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

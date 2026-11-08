"""Audit the revised 1108manuscript.docx against paper-word requirements."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
MANUSCRIPT = ROOT / "1108manuscript.docx"
REGISTRY = ROOT / "results" / "reference_registry_verified.json"
LIVE_REFERENCE_AUDIT = ROOT / "results" / "reference_doi_live_validation.json"
R_FIGURES = ROOT / "results" / "figures_r"

document = Document(MANUSCRIPT)
paragraphs = [paragraph for paragraph in document.paragraphs if paragraph.text.strip()]
texts = [paragraph.text.strip() for paragraph in paragraphs]
text = "\n".join(texts)

start = texts.index("1. Introduction")
end = texts.index("7. References")
body_paragraphs = []
for paragraph in paragraphs[start:end]:
    value = paragraph.text.strip()
    if not value:
        continue
    if paragraph.paragraph_format.keep_with_next:
        continue
    if value.startswith(("Table ", "Figure ", "Interpretation:")):
        continue
    body_paragraphs.append(value)
sentences = []
for value in body_paragraphs:
    sentences.extend(
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", value)
        if item.strip()
    )

abstract_start = texts.index("Abstract")
keyword_index = next(index for index, value in enumerate(texts) if value.startswith("Keywords:"))
abstract = texts[abstract_start + 1:keyword_index]
abstract_word_count = len(" ".join(abstract).split())
keywords = texts[keyword_index].split(":", 1)[1]
keyword_count = len([value for value in keywords.split(";") if value.strip()])

body_table_text = "\n".join(
    cell.text
    for table in document.tables
    for row in table.rows
    for cell in row.cells
)
decimal_values = []
for value in re.findall(
    r"(?<![\d.])\d+\.\d+(?![\d.])",
    "\n".join(body_paragraphs) + "\n" + body_table_text,
):
    if value == "0.001":
        continue
    if len(value.split(".")[1]) > 2:
        decimal_values.append(value)

citation_tokens = re.findall(r"\[([0-9,\-]+)\]", text)
citation_numbers = []
first_position = {}
for match_index, token in enumerate(citation_tokens):
    for part in token.split(","):
        if "-" in part:
            left, right = [int(value) for value in part.split("-", 1)]
            values = list(range(left, right + 1))
        else:
            values = [int(part)]
        for value in values:
            citation_numbers.append(value)
            first_position.setdefault(value, match_index)

registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
live_references = json.loads(LIVE_REFERENCE_AUDIT.read_text(encoding="utf-8"))
reference_start = texts.index("7. References")
reference_entries = [
    value for value in texts[reference_start + 1:] if re.match(r"^\d+\.", value)
]
reference_text = "\n".join(reference_entries)
reference_bookmarks = {
    element.get(qn("w:name"))
    for element in document.element.body.iter(qn("w:bookmarkStart"))
    if (element.get(qn("w:name")) or "").startswith("Ref")
}
citation_anchors = [
    element.get(qn("w:anchor"))
    for element in document.element.body.iter(qn("w:hyperlink"))
    if element.get(qn("w:anchor"))
]
citation_groups = re.findall(r"\[([0-9,\-]+)\]", text)
citation_groups_ascending = all(
    [
        int(token.split("-", 1)[0])
        for token in group.split(",")
    ]
    == sorted(
        int(token.split("-", 1)[0])
        for token in group.split(",")
    )
    for group in citation_groups
)
figure_parts = [
    relation.target_part
    for relation in document.part.rels.values()
    if "image" in relation.reltype
]
r_hashes = {
    hashlib.sha256(path.read_bytes()).hexdigest(): path.name
    for path in R_FIGURES.glob("*.png")
}
embedded_names = sorted(
    r_hashes.get(hashlib.sha256(part.blob).hexdigest(), "unknown")
    for part in figure_parts
)

table_borders = []
for table in document.tables:
    tbl_borders = table._tbl.tblPr.find(qn("w:tblBorders"))
    table_borders.append(tbl_borders is not None)

caption_texts = [
    value
    for value in texts
    if re.match(r"^(Table \d+[A-Z]?\.|Figure \d+\.)", value)
]
table_captions = [
    value.split(".")[0]
    for value in caption_texts
    if value.startswith("Table ")
]
figure_captions = [
    value.split(".")[0]
    for value in caption_texts
    if value.startswith("Figure ")
]
body_text_without_captions = "\n".join(
    value
    for value in texts
    if not re.match(r"^(Table \d+[A-Z]?\.|Figure \d+\.)", value)
    and not value.startswith("Interpretation:")
)
chart_callouts = [
    "Table 1",
    "Table 2",
    "Table 3",
    "Table 4",
    "Table 5A",
    "Table 5B",
    "Table 6",
    "Table 7",
    "Table 8",
    *[f"Figure {index}" for index in range(1, 10)],
]

required_author = (
    "Zhipeng Wang1*, Luning Wang2*, Changsong Wang1, Xiyu Xu3, Pengli Zhai4, "
    "Zejun Liu1, Hui Feng1, Qian Hou1, Ming Guo1"
)
required_declarations = (
    "Data availability",
    "Code availability",
    "Ethics approval and consent to participate",
    "Patient and public involvement",
    "Authors' contributions",
    "Funding",
    "Declaration of generative AI and AI-assisted technologies in the writing process",
    "Consent for publication",
    "Clinical trial number",
    "Competing interests",
)

checks = {
    "page_letter": (
        abs(document.sections[0].page_width.inches - 8.5) < 1e-6
        and abs(document.sections[0].page_height.inches - 11) < 1e-6
    ),
    "margins_one_inch": all(
        abs(value.inches - 1.0) < 1e-6
        for value in (
            document.sections[0].left_margin,
            document.sections[0].right_margin,
            document.sections[0].top_margin,
            document.sections[0].bottom_margin,
        )
    ),
    "body_times_12_1_5": (
        document.styles["Normal"].font.name == "Times New Roman"
        and document.styles["Normal"].font.size.pt == 12
        and document.styles["Normal"].paragraph_format.line_spacing == 1.5
    ),
    "title_18_bold_center": (
        paragraphs[0].runs[0].font.size.pt == 18
        and paragraphs[0].runs[0].bold is True
        and str(paragraphs[0].alignment) == "CENTER (1)"
    ),
    "author_information_exact": required_author in text,
    "abstract_under_250_words": abstract_word_count <= 250,
    "abstract_four_labels": all(
        any(value.startswith(label) for value in abstract)
        for label in ("Background:", "Methods:", "Results:", "Conclusion:")
    ),
    "keywords_at_most_five": keyword_count <= 5,
    "sentence_length_under_31": max(
        (len(sentence.split()) for sentence in sentences),
        default=0,
    ) <= 30,
    "two_decimal_body_values": len(decimal_values) == 0,
    "no_collapse_triangles": not any(
        symbol in text for symbol in ("▶", "▼", "▷", "►", "◆")
    ),
    "nine_r_figures": len(figure_parts) == 9
    and all(name != "unknown" for name in embedded_names),
    "nine_three_line_tables": len(document.tables) == 9
    and all(table_borders),
    "table_numbering_sequential": table_captions
    == ["Table 1", "Table 2", "Table 3", "Table 4", "Table 5A",
        "Table 5B", "Table 6", "Table 7", "Table 8"],
    "figure_numbering_sequential": figure_captions
    == [f"Figure {index}" for index in range(1, 10)],
    "all_charts_cited_in_text": all(
        label in body_text_without_captions for label in chart_callouts
    ),
    "interpretations_present": sum(
        value.startswith("Interpretation:") for value in texts
    ) >= 17,
    "forty_five_verified_references": len(registry) == 45
    and all(record["verified"] for record in registry),
    "all_reference_entries_have_doi": (
        len(reference_entries) == 45
        and sum(
            bool(re.search(r"doi:10\.\S+", value, re.IGNORECASE))
            for value in reference_entries
        ) == 45
    ),
    "no_preprint_or_conference_references": not bool(
        re.search(
            r"\b(arxiv|medrxiv|biorxiv|proceedings|conference|symposium)\b",
            reference_text,
            re.IGNORECASE,
        )
    ),
    "all_citation_groups_ascending": citation_groups_ascending,
    "all_citation_hyperlinks_resolve": (
        reference_bookmarks == {f"Ref{index}" for index in range(1, 46)}
        and bool(citation_anchors)
        and set(citation_anchors).issubset(reference_bookmarks)
        and set(citation_anchors)
        == {f"Ref{index}" for index in range(1, 46)}
    ),
    "all_references_live_crossref_journals": (
        live_references["summary"]["verified"] == len(registry)
        and live_references["summary"]["journal_articles"] == len(registry)
        and not live_references["summary"]["non_journal"]
        and all(
            record.get("publication_type") == "journal-article"
            for record in registry
        )
    ),
    "all_references_cited": set(citation_numbers) == set(range(1, 46)),
    "citation_first_appearance_order": sorted(
        first_position,
        key=first_position.get,
    ) == list(range(1, 46)),
    "declarations_complete": all(
        heading in text for heading in required_declarations
    ),
    "public_repository_url": (
        "https://github.com/wzp24k/acute-mi-dynamic-graph-disposition" in text
    ),
}

report = {
    "checks": checks,
    "failures": [key for key, passed in checks.items() if not passed],
    "abstract_words": abstract_word_count,
    "keywords": keyword_count,
    "max_sentence_words": max(
        (len(sentence.split()) for sentence in sentences),
        default=0,
    ),
    "remaining_three_decimal_values": decimal_values,
    "embedded_figures": embedded_names,
    "table_count": len(document.tables),
    "reference_count": len(registry),
    "citation_numbers": sorted(set(citation_numbers)),
}
report["status"] = "PASS" if not report["failures"] else "FAIL"
(ROOT / "results" / "1108manuscript_audit.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(report, indent=2, ensure_ascii=False))

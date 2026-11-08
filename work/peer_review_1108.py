"""Run three deterministic reviewer passes over the revised 1108 manuscript."""
from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
document = Document(ROOT / "1108manuscript.docx")
text = "\n".join(paragraph.text for paragraph in document.paragraphs)
requirements = json.loads(
    (ROOT / "results" / "1108manuscript_audit.json").read_text(encoding="utf-8")
)
registry = json.loads(
    (ROOT / "results" / "reference_registry_verified.json").read_text(
        encoding="utf-8"
    )
)
live = json.loads(
    (ROOT / "results" / "reference_doi_live_validation.json").read_text(
        encoding="utf-8"
    )
)

reviewer1_checks = {
    "reference_count_40_to_50": 40 <= len(registry) <= 50,
    "all_doi_verified": all(record["verified"] for record in registry),
    "all_doi_live_crossref_verified": (
        live["summary"]["verified"] == len(registry)
        and live["summary"]["not_found"] == 0
        and live["summary"]["errors"] == 0
    ),
    "all_references_journal_articles": (
        live["summary"]["journal_articles"] == len(registry)
        and not live["summary"]["non_journal"]
        and all(
            record.get("publication_type") == "journal-article"
            for record in registry
        )
    ),
    "citation_numbers_complete": requirements["checks"]["all_references_cited"],
    "citation_first_appearance_ascending": requirements["checks"][
        "citation_first_appearance_order"
    ],
    "no_duplicate_doi": len({record["doi"].lower() for record in registry})
    == len(registry),
}

reviewer2_checks = {
    "cohort_counts_consistent": all(
        value in text
        for value in (
            "13,131 patients",
            "4,475 diagnosis-coded MACCE events",
            "3,470 patients",
            "409 events",
            "5,913 patients",
            "642 events",
        )
    ),
    "main_auroc_values_consistent": all(
        value in text for value in ("AUROC 0.76", "AUROC 0.75", "0.73")
    ),
    "two_decimal_metrics": requirements["checks"]["two_decimal_body_values"],
    "all_charts_cited": requirements["checks"]["all_charts_cited_in_text"],
    "tables_and_figures_sequential": (
        requirements["checks"]["table_numbering_sequential"]
        and requirements["checks"]["figure_numbering_sequential"]
    ),
    "p_value_rule_not_triggered": not bool(
        re.search(r"\bP\s*[=<>]\s*0\.\d{4,}", text)
    ),
}

reviewer3_checks = {
    "abstract_250_or_less": requirements["checks"]["abstract_under_250_words"],
    "abstract_four_sections": requirements["checks"]["abstract_four_labels"],
    "keywords_five_or_less": requirements["checks"]["keywords_at_most_five"],
    "sentence_length_30_or_less": requirements["checks"][
        "sentence_length_under_31"
    ],
    "format_requirements": all(
        requirements["checks"][key]
        for key in (
            "page_letter",
            "margins_one_inch",
            "body_times_12_1_5",
            "title_18_bold_center",
            "no_collapse_triangles",
        )
    ),
    "declarations_complete": requirements["checks"]["declarations_complete"],
}

report = {
    "reviewer_1_references": {
        "checks": reviewer1_checks,
        "status": "PASS" if all(reviewer1_checks.values()) else "FAIL",
    },
    "reviewer_2_logic_numbers_figures": {
        "checks": reviewer2_checks,
        "status": "PASS" if all(reviewer2_checks.values()) else "FAIL",
    },
    "reviewer_3_language_format": {
        "checks": reviewer3_checks,
        "status": "PASS" if all(reviewer3_checks.values()) else "FAIL",
    },
}
report["overall_status"] = (
    "PASS"
    if all(
        report[key]["status"] == "PASS"
        for key in (
            "reviewer_1_references",
            "reviewer_2_logic_numbers_figures",
            "reviewer_3_language_format",
        )
    )
    else "FAIL"
)
(ROOT / "results" / "1108_peer_review_audit.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(report, indent=2, ensure_ascii=False))

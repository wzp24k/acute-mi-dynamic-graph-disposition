from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path

import pandas as pd
from lxml import etree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = PROJECT_ROOT / "1104Manuscript.docx"
SUBGROUPS = PROJECT_ROOT / "results/tables/mimic4_subgroups.csv"
AUDIT_OUTPUT = PROJECT_ROOT / "results/audits/results_sections_split_20261106.json"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
W = f"{{{W_NS}}}"
QE = lambda tag: f"{{{W_NS}}}{tag}"


METHODS_2_6_HEADING = "2.6 Training and evaluation"
METHODS_2_6_ANCHOR = (
    "The primary metric was five-class one-versus-rest macro-AUC. Secondary "
    "metrics included accuracy, quadratic weighted kappa, ordinal C-index, "
    "ordinal mean absolute error, multiclass Brier score, per-class AUC, "
    "cumulative NRI/IDI, calibration slope and intercept, and decision curve "
    "analysis [18,19,20]."
)
METHODS_2_8_HEADING = "2.8 Regularization and sensitivity analyses"
RESULTS_3_7_HEADING = "3.7 Sensitivity and subgroup analyses"
RESULTS_3_7_BODY = (
    "All 16 prespecified sensitivity scenarios are reported in Table 16 and "
    "Figure 7B."
)
TABLE_16_CAPTION = "Table 16. Complete prespecified sensitivity analyses."
FIGURE_7_CAPTION = (
    "Figure 7. Prespecified subgroups and all 16 sensitivity analyses. "
    "Six-level and merged-label scenarios are annotated as separate scales "
    "and shown without a delta. Negative values are retained."
)
DISCUSSION_CALIBRATION = (
    "The calibration analysis resolves a practical question that "
    "discrimination alone cannot answer. Raw transported probabilities had "
    "slopes far from one in several settings, whereas grouped out-of-fold "
    "maps produced slopes near one and lower Brier scores."
)
LIMITATIONS_PARAGRAPH = (
    "First, the endpoint is care placement, not functional recovery or "
    "patient preference. Second, the primary analysis conditions on survival "
    "and continued ICU care at 24 hours. Third, the three databases differ "
    "in measurement and in the availability of home health, rehabilitation "
    "and hospice."
)


METHODS_2_6_SUBGROUP_PARAGRAPH = (
    "Prespecified subgroups were defined by sex, age, AMI subtype, primary "
    "AMI position, admission urgency and initial care unit. Discrimination "
    "was evaluated using the locked out-of-fold predictions. High-need "
    "calibration used the binary target y>=3. For each subgroup, we fitted "
    "a logistic calibration model on logit probabilities. Patient-level "
    "bootstrap resampling produced 95% confidence intervals for slope, "
    "intercept and Brier score. Subgroups were not refitted separately. "
    "Estimates required at least 30 patients, 10 events and 10 non-events."
)

RESULTS_3_7_PARAGRAPHS = [
    (
        "All 16 prespecified scenarios are reported in Table 16 and Figure 7B. "
        "Each estimate remains on its original outcome scale. Deltas are "
        "shown only when the outcome scale is unchanged. The 6-hour and "
        "72-hour landmark cohorts had lower macro-AUC than the 24-hour "
        "cohort. Estimates were 0.761 (delta -0.0173) and 0.751 "
        "(delta -0.0269), respectively."
    ),
    (
        "Primary-position AMI increased macro-AUC by 0.0140, whereas "
        "excluding type 2 MI increased it by 0.0066. Multiple imputation, "
        "troponin exclusion, treatment adjustment and against-advice "
        "exclusion changed estimates by at most 0.0009. The reversed-direction, "
        "time-permuted and generic-smoothness controls remained within 0.006 "
        "of their references. No scenario replaced the primary analysis "
        "after inspection."
    ),
]

RESULTS_3_8_PARAGRAPHS = [
    (
        "Table 17 reports subgroup discrimination, high-need calibration "
        "slopes and Brier scores. Fourteen prespecified subgroup strata were "
        "estimable. XGB-static+TS macro-AUC point estimates ranged from 0.729 "
        "in patients aged 80 years or older to 0.812 in STEMI. DG-GRU+TR "
        "macro-AUC ranged from 0.664 to 0.784. DG-GRU+TR minus XGB-static+TS "
        "was negative in every estimable subgroup. The confidence interval "
        "crossed zero only for Other or unspecified MI (n=193)."
    ),
    (
        "Overall high-need calibration was better for XGB-static+TS than for "
        "DG-GRU+TR. The calibration slopes were 1.188 (1.128 to 1.253) and "
        "0.638 (0.598 to 0.680), respectively. Intercepts were 0.003 (-0.052 "
        "to 0.062) and 0.006 (-0.048 to 0.066). Brier scores were 0.180 "
        "(0.175 to 0.184) and 0.192 (0.187 to 0.198)."
    ),
    (
        "XGB-static+TS slopes exceeded 1 in every estimable stratum, "
        "indicating under-confident probabilities. DG-GRU+TR slopes were "
        "below 1 in every estimable stratum, indicating over-confident "
        "probabilities. The largest DG-GRU+TR miscalibrations occurred for "
        "age >=80 (slope 0.494; intercept 0.450) and type 2 MI (slope 0.504). "
        "Calibration was closest to 1 in CCU/CVICU (DG-GRU+TR slope 0.983)."
    ),
    (
        "These patterns do not change the discrimination ranking. In "
        "CCU/CVICU, DG-GRU+TR had a near-ideal slope but lower macro-AUC than "
        "XGB-static+TS (0.721 vs 0.773). Subgroup estimates are descriptive "
        "because models were not refitted within strata."
    ),
]

TABLE_17_CAPTION = (
    "Table 17. Prespecified subgroup discrimination and high-need "
    "calibration. High need denotes outcome level 3 or higher. AUC denotes "
    "five-class macro-AUC. Slopes and Brier scores used locked out-of-fold "
    "predictions. The 2-patient non-emergency stratum was not estimable."
)

RESULTS_3_9_PARAGRAPHS = [
    (
        "The six-level outcome produced macro-AUC 0.776 on its own scale. "
        "Merged disposition labels produced 0.800 on a different four-class "
        "scale. These values neither replace nor directly subtract from the "
        "primary five-level estimate. Primary-position restriction and type 2 "
        "MI exclusion increased the five-level estimate by 0.0140 and 0.0066, "
        "respectively."
    ),
    (
        "In eICU, excluding the Other disposition category increased "
        "DG-GRU+TR macro-AUC from 0.729 to 0.735. Patient-safe "
        "hospital-cluster bootstrap gave 0.729 (0.714 to 0.744). This interval "
        "accounts for shared hospitals among patients. It describes precision "
        "of the frozen eICU transport estimate."
    ),
    (
        "None of these analyses replaced the prespecified primary estimand. "
        "They quantify sensitivity to outcome definition and evaluation unit, "
        "not a new primary comparison. Absolute discrimination therefore "
        "depends on outcome granularity, cohort restriction and clustering. "
        "Calibration varied by database and subgroup, so local recalibration "
        "remains necessary."
    ),
]

DISCUSSION_SUBGROUP_PARAGRAPH = (
    "Subgroup analysis extends this calibration evidence. XGB-static+TS "
    "discriminated better across estimable strata, but calibration slopes "
    "varied. DG-GRU+TR was well calibrated in CCU/CVICU but remained less "
    "discriminative. Subgroup-specific recalibration therefore requires "
    "prospective evaluation rather than assuming transportability."
)

LIMITATIONS_SUBGROUP_PARAGRAPH = (
    "Subgroup analyses reused locked out-of-fold predictions and were "
    "descriptive. They did not estimate subgroup-specific treatment effects "
    "or refit models within strata."
)

CURRENT_TEXT_REPAIRS = {
    (
        "Table 17 reports subgroup discrimination, high-need calibration "
        "slopes and Brier scores. Fourteen prespecified subgroup strata were "
        "estimable. XGB-static+TS point estimates ranged from 0.729 in "
        "patients aged 80 years or older to 0.812 in STEMI. DG-GRU+TR ranged "
        "from 0.664 to 0.784. DG-GRU+TR minus XGB-static+TS was negative in "
        "every subgroup. The confidence interval crossed zero only for Other "
        "or unspecified MI (n=193)."
    ): RESULTS_3_8_PARAGRAPHS[0],
    (
        "In eICU, excluding the Other disposition category increased "
        "DG-GRU+TR macro-AUC from 0.729 to 0.735. Patient-safe "
        "hospital-cluster bootstrap gave 0.729 (0.714 to 0.744). The "
        "hospital-cluster interval is wider than the patient-grouped interval. "
        "The primary direction remained unchanged."
    ): RESULTS_3_9_PARAGRAPHS[1],
}

NEW_FIGURE_7_CAPTION = (
    "Figure 7. Prespecified subgroup contrasts and all 16 sensitivity "
    "analyses. (a) Paired subgroup contrasts for DG-GRU+TR minus "
    "XGB-static+TS; negative values indicate lower DG-GRU+TR macro-AUC. "
    "(b) Sensitivity estimates; six-level and merged-label scenarios use "
    "separate scales and omit deltas. Negative values are retained."
)

TABLE_LABELS = {
    "overall": "Overall",
    "female": "Female",
    "male": "Male",
    "age_18_64": "Age 18-64",
    "age_65_79": "Age 65-79",
    "age_80_plus": "Age >=80",
    "STEMI": "STEMI",
    "NSTEMI": "NSTEMI",
    "type_2_MI": "Type 2 MI",
    "other_or_unspecified_MI": "Other/unspecified MI",
    "primary_position_AMI": "Primary-position AMI",
    "non_primary_position_AMI": "Non-primary-position AMI",
    "emergency_admission": "Emergency admission",
    "CCU_or_CVICU": "CCU/CVICU",
    "other_ICU": "Other ICU",
}

TABLE_ORDER = [
    "overall",
    "female",
    "male",
    "age_18_64",
    "age_65_79",
    "age_80_plus",
    "STEMI",
    "NSTEMI",
    "type_2_MI",
    "other_or_unspecified_MI",
    "primary_position_AMI",
    "non_primary_position_AMI",
    "emergency_admission",
    "CCU_or_CVICU",
    "other_ICU",
]


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(node.text or "" for node in paragraph.iter(QE("t")))


def set_paragraph_text(paragraph: etree._Element, text: str) -> None:
    text_nodes = list(paragraph.iter(QE("t")))
    if not text_nodes:
        raise RuntimeError(
            f"Paragraph has no text node: {paragraph_text(paragraph)!r}"
        )
    first = text_nodes[0]
    first.text = text
    if text[:1].isspace() or text[-1:].isspace():
        first.set(f"{{{XML_NS}}}space", "preserve")
    elif f"{{{XML_NS}}}space" in first.attrib:
        del first.attrib[f"{{{XML_NS}}}space"]
    for node in text_nodes[1:]:
        node.text = ""
        if f"{{{XML_NS}}}space" in node.attrib:
            del node.attrib[f"{{{XML_NS}}}space"]


def set_caption_text(paragraph: etree._Element, text: str) -> None:
    label_end = text.find(".")
    if label_end < 0:
        raise RuntimeError(f"Caption label not found: {text[:80]!r}")
    label = text[: label_end + 1]
    body = text[label_end + 1 :]
    text_nodes = list(paragraph.iter(QE("t")))
    if len(text_nodes) < 2:
        raise RuntimeError("Caption template requires label and body runs")
    assigned = [label, body, *([""] * (len(text_nodes) - 2))]
    for node, value in zip(text_nodes, assigned):
        node.text = value
        if value[:1].isspace() or value[-1:].isspace():
            node.set(f"{{{XML_NS}}}space", "preserve")
        elif f"{{{XML_NS}}}space" in node.attrib:
            del node.attrib[f"{{{XML_NS}}}space"]


def clone_paragraph(template: etree._Element, text: str) -> etree._Element:
    cloned = copy.deepcopy(template)
    set_paragraph_text(cloned, text)
    return cloned


def clone_caption(template: etree._Element, text: str) -> etree._Element:
    cloned = copy.deepcopy(template)
    set_caption_text(cloned, text)
    return cloned


def find_exact_paragraph(
    paragraphs: list[etree._Element], text: str
) -> etree._Element:
    matches = [p for p in paragraphs if paragraph_text(p) == text]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one paragraph for {text[:80]!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def find_prefix_paragraph(
    paragraphs: list[etree._Element], prefix: str
) -> etree._Element:
    matches = [p for p in paragraphs if paragraph_text(p).startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one paragraph for prefix {prefix[:80]!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def insert_after(
    body: etree._Element,
    anchor: etree._Element,
    new_elements: list[etree._Element],
) -> None:
    index = body.index(anchor)
    for offset, element in enumerate(new_elements, start=1):
        body.insert(index + offset, element)


def set_cell_text(cell: etree._Element, text: str) -> None:
    paragraphs = cell.findall(QE("p"))
    if not paragraphs:
        raise RuntimeError("Table cell has no paragraph")
    paragraph = paragraphs[0]
    for extra in paragraphs[1:]:
        cell.remove(extra)

    runs = paragraph.findall(QE("r"))
    if not runs:
        run = etree.SubElement(paragraph, QE("r"))
        etree.SubElement(run, QE("t"))
        runs = [run]
    for extra in runs[1:]:
        paragraph.remove(extra)

    text_nodes = list(runs[0].iter(QE("t")))
    if not text_nodes:
        text_node = etree.SubElement(runs[0], QE("t"))
        text_nodes = [text_node]
    text_nodes[0].text = text
    for node in text_nodes[1:]:
        node.text = ""


def set_table_widths(table: etree._Element) -> None:
    widths = [2000, 600, 1850, 1650, 1650, 1610]
    grid = table.find(QE("tblGrid"))
    if grid is None:
        raise RuntimeError("Table grid missing from template")
    columns = grid.findall(QE("gridCol"))
    if len(columns) != len(widths):
        raise RuntimeError("Template must have six table columns")
    for column, width in zip(columns, widths):
        column.set(QE("w"), str(width))

    for row in table.findall(QE("tr")):
        cells = row.findall(QE("tc"))
        if len(cells) != len(widths):
            raise RuntimeError("Template row must have six cells")
        for cell, width in zip(cells, widths):
            cell_width = cell.find(f"{QE('tcPr')}/{QE('tcW')}")
            if cell_width is None:
                cell_pr = cell.find(QE("tcPr"))
                if cell_pr is None:
                    cell_pr = etree.SubElement(cell, QE("tcPr"))
                cell_width = etree.SubElement(cell_pr, QE("tcW"))
            cell_width.set(QE("w"), str(width))
            cell_width.set(QE("type"), "dxa")


def format_ci(value: object, low: object, high: object, digits: int = 3) -> str:
    return (
        f"{float(value):.{digits}f} "
        f"({float(low):.{digits}f} to {float(high):.{digits}f})"
    )


def build_table_rows() -> list[list[str]]:
    frame = pd.read_csv(SUBGROUPS).set_index("subgroup")
    rows = [["Subgroup", "n", "DG - XGB AUC (95% CI)",
             "XGB slope (95% CI)", "DG slope (95% CI)", "Brier XGB / DG"]]
    for subgroup in TABLE_ORDER:
        row = frame.loc[subgroup]
        delta = ast.literal_eval(str(row["dggru_tr_minus_xgb"]))
        if not row["xgb_static_ts_high_need_estimable"]:
            raise RuntimeError(f"Estimable subgroup expected: {subgroup}")
        rows.append(
            [
                TABLE_LABELS[subgroup],
                f"{int(row['n']):,}",
                format_ci(delta["delta"], delta["ci_low"], delta["ci_high"], 3),
                format_ci(
                    row["xgb_static_ts_high_need_calibration_slope"],
                    row["xgb_static_ts_high_need_calibration_slope_ci_low"],
                    row["xgb_static_ts_high_need_calibration_slope_ci_high"],
                    3,
                ),
                format_ci(
                    row["dggru_tr_high_need_calibration_slope"],
                    row["dggru_tr_high_need_calibration_slope_ci_low"],
                    row["dggru_tr_high_need_calibration_slope_ci_high"],
                    3,
                ),
                (
                    f"{float(row['xgb_static_ts_high_need_brier']):.3f} / "
                    f"{float(row['dggru_tr_high_need_brier']):.3f}"
                ),
            ]
        )
    non_emergency = frame.loc["non_emergency_admission"]
    if bool(non_emergency["xgb_static_ts_high_need_estimable"]):
        raise RuntimeError("Non-emergency stratum should not be estimable")
    return rows


def build_table_17(
    template: etree._Element, rows: list[list[str]]
) -> etree._Element:
    table = copy.deepcopy(template)
    table_rows = table.findall(QE("tr"))
    if len(table_rows) < 2:
        raise RuntimeError("Table template has too few rows")
    header = table_rows[0]
    body_template = copy.deepcopy(table_rows[1])
    for row in table_rows[1:]:
        table.remove(row)

    header_cells = header.findall(QE("tc"))
    if len(header_cells) != 6:
        raise RuntimeError("Table template header must have six cells")
    for cell, text in zip(header_cells, rows[0]):
        set_cell_text(cell, text)

    for data_row in rows[1:]:
        row = copy.deepcopy(body_template)
        cells = row.findall(QE("tc"))
        if len(cells) != 6:
            raise RuntimeError("Table template body row must have six cells")
        for cell, text in zip(cells, data_row):
            set_cell_text(cell, text)
        table.append(row)

    set_table_widths(table)
    return table


def sentence_audit(texts: list[str]) -> list[dict[str, object]]:
    results = []
    for text in texts:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
            if sentence.strip()
        ]
        for sentence in sentences:
            words = re.findall(r"\b[\w<>=-]+(?:[-'][\w<>=-]+)*\b", sentence)
            results.append(
                {
                    "sentence": sentence,
                    "word_count": len(words),
                    "passes_30_word_limit": len(words) <= 30,
                }
            )
    failures = [
        item for item in results if not item["passes_30_word_limit"]
    ]
    if failures:
        raise RuntimeError(f"Sentences exceed 30 words: {failures}")
    return results


def document_xml_sha256(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return hashlib.sha256(archive.read("word/document.xml")).hexdigest()


def update_docx(path: Path) -> dict[str, object]:
    audit_texts = [
        METHODS_2_6_SUBGROUP_PARAGRAPH,
        *RESULTS_3_7_PARAGRAPHS,
        *RESULTS_3_8_PARAGRAPHS,
        TABLE_17_CAPTION,
        *RESULTS_3_9_PARAGRAPHS,
        DISCUSSION_SUBGROUP_PARAGRAPH,
        LIMITATIONS_SUBGROUP_PARAGRAPH,
        NEW_FIGURE_7_CAPTION,
    ]
    sentence_results = sentence_audit(audit_texts)

    with zipfile.ZipFile(path, "r") as archive:
        document_xml = archive.read("word/document.xml")
        root = etree.fromstring(document_xml)
        body = root.find(QE("body"))
        if body is None:
            raise RuntimeError("Word document body not found")

        paragraphs = [child for child in body if child.tag == QE("p")]
        funding_before = next(
            paragraph_text(p)
            for p in paragraphs
            if paragraph_text(p).startswith("This work was supported by")
        )

        methods_26_heading = find_exact_paragraph(
            paragraphs, METHODS_2_6_HEADING
        )
        methods_26_anchor = find_prefix_paragraph(
            paragraphs, METHODS_2_6_ANCHOR
        )
        methods_body_template = methods_26_anchor

        methods_28_heading = find_exact_paragraph(
            paragraphs, METHODS_2_8_HEADING
        )
        results_37_heading = find_exact_paragraph(
            paragraphs, RESULTS_3_7_HEADING
        )
        results_37_body = find_prefix_paragraph(
            paragraphs, RESULTS_3_7_BODY
        )
        table_16_caption = find_exact_paragraph(
            paragraphs, TABLE_16_CAPTION
        )
        figure_7_caption = find_exact_paragraph(
            paragraphs, FIGURE_7_CAPTION
        )
        discussion_anchor = find_prefix_paragraph(
            paragraphs, DISCUSSION_CALIBRATION
        )
        limitations_anchor = find_prefix_paragraph(
            paragraphs, LIMITATIONS_PARAGRAPH
        )

        heading_template = results_37_heading
        caption_template = table_16_caption

        set_paragraph_text(
            methods_28_heading,
            "2.8 Regularization, sensitivity, and subgroup analyses",
        )
        insert_after(
            body,
            methods_26_anchor,
            [
                clone_paragraph(
                    methods_body_template, METHODS_2_6_SUBGROUP_PARAGRAPH
                )
            ],
        )

        set_paragraph_text(
            results_37_heading, "3.7 Prespecified sensitivity analyses"
        )
        set_paragraph_text(results_37_body, RESULTS_3_7_PARAGRAPHS[0])
        insert_after(
            body,
            results_37_body,
            [
                clone_paragraph(methods_body_template, text)
                for text in RESULTS_3_7_PARAGRAPHS[1:]
            ],
        )

        table_16 = table_16_caption.getnext()
        if table_16 is None or table_16.tag != QE("tbl"):
            raise RuntimeError("Table 16 was not found after its caption")
        table_15_caption = find_exact_paragraph(
            paragraphs,
            "Table 15. Common three-class evaluation across databases "
            "(pragmatic harmonization).",
        )
        table_15 = table_15_caption.getnext()
        if table_15 is None or table_15.tag != QE("tbl"):
            raise RuntimeError("Table 15 was not found after its caption")

        image_paragraph = figure_7_caption.getprevious()
        if image_paragraph is None or image_paragraph.tag != QE("p"):
            raise RuntimeError("Figure 7 image paragraph was not found")

        table_rows = build_table_rows()
        if len(table_rows) != 16:
            raise RuntimeError("Table 17 should have a header and 15 data rows")

        new_elements = [
            clone_paragraph(heading_template, "3.8 Subgroup discrimination and calibration"),
            *[
                clone_paragraph(methods_body_template, text)
                for text in RESULTS_3_8_PARAGRAPHS
            ],
            clone_caption(caption_template, TABLE_17_CAPTION),
            build_table_17(table_15, table_rows),
            clone_paragraph(
                heading_template,
                "3.9 Robustness to outcome definition and evaluation unit",
            ),
            *[
                clone_paragraph(methods_body_template, text)
                for text in RESULTS_3_9_PARAGRAPHS
            ],
        ]
        for element in new_elements:
            image_paragraph.addprevious(element)

        set_caption_text(figure_7_caption, NEW_FIGURE_7_CAPTION)
        insert_after(
            body,
            discussion_anchor,
            [
                clone_paragraph(
                    methods_body_template, DISCUSSION_SUBGROUP_PARAGRAPH
                )
            ],
        )
        insert_after(
            body,
            limitations_anchor,
            [
                clone_paragraph(
                    methods_body_template, LIMITATIONS_SUBGROUP_PARAGRAPH
                )
            ],
        )

        paragraphs_after = [child for child in body if child.tag == QE("p")]
        funding_after = next(
            paragraph_text(p)
            for p in paragraphs_after
            if paragraph_text(p).startswith("This work was supported by")
        )
        if funding_after != funding_before:
            raise RuntimeError("Funding paragraph changed")

        table_count = sum(child.tag == QE("tbl") for child in body)
        if table_count != 17:
            raise RuntimeError(f"Expected 17 tables, found {table_count}")

        updated_xml = etree.tostring(
            root,
            xml_declaration=True,
            encoding="UTF-8",
            standalone=True,
        )

    tmp_path = path.with_name(f"{path.stem}.tmp_results_split{path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with zipfile.ZipFile(path, "r") as src_archive, zipfile.ZipFile(
            tmp_path, "w"
        ) as dst_archive:
            for item in src_archive.infolist():
                payload = (
                    updated_xml
                    if item.filename == "word/document.xml"
                    else src_archive.read(item.filename)
                )
                info = zipfile.ZipInfo(item.filename, date_time=item.date_time)
                info.compress_type = item.compress_type
                info.external_attr = item.external_attr
                info.create_system = item.create_system
                dst_archive.writestr(info, payload)

        with zipfile.ZipFile(tmp_path, "r") as check_archive:
            check_root = etree.fromstring(
                check_archive.read("word/document.xml")
            )
            check_body = check_root.find(QE("body"))
            check_paragraphs = [
                child for child in check_body if child.tag == QE("p")
            ]
            check_texts = [paragraph_text(p) for p in check_paragraphs]
            required = [
                "3.7 Prespecified sensitivity analyses",
                "3.8 Subgroup discrimination and calibration",
                "3.9 Robustness to outcome definition and evaluation unit",
                TABLE_17_CAPTION,
                NEW_FIGURE_7_CAPTION,
            ]
            for text in required:
                if text not in check_texts:
                    raise RuntimeError(
                        f"Required paragraph missing after write: {text[:80]!r}"
                    )
            check_table_count = sum(
                child.tag == QE("tbl") for child in check_body
            )
            if check_table_count != 17:
                raise RuntimeError("Table count changed after write")

        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    result = {
        "document": str(path),
        "document_xml_sha256": document_xml_sha256(path),
        "table_count": 17,
        "results_sections": ["3.7", "3.8", "3.9"],
        "table_17_rows": 16,
        "non_emergency_calibration_estimable": False,
        "funding_preserved": True,
        "sentence_count_audited": len(sentence_results),
        "maximum_sentence_words": max(
            int(item["word_count"]) for item in sentence_results
        ),
        "sentence_length_failures": 0,
        "sentence_audit": sentence_results,
    }
    AUDIT_OUTPUT.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def repair_caption_runs(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path, "r") as archive:
        root = etree.fromstring(archive.read("word/document.xml"))
    body = root.find(QE("body"))
    if body is None:
        raise RuntimeError("Word document body not found")

    repaired = []
    replaced = []
    for paragraph in [child for child in body if child.tag == QE("p")]:
        text = paragraph_text(paragraph)
        if text in CURRENT_TEXT_REPAIRS:
            set_paragraph_text(paragraph, CURRENT_TEXT_REPAIRS[text])
            replaced.append(text[:80])
        if text.startswith("Table 17."):
            set_caption_text(paragraph, text)
            repaired.append("Table 17")
        elif text.startswith("Figure 7."):
            set_caption_text(paragraph, text)
            repaired.append("Figure 7")

    if repaired != ["Table 17", "Figure 7"]:
        raise RuntimeError(f"Unexpected caption matches: {repaired}")

    updated_xml = etree.tostring(
        root,
        xml_declaration=True,
        encoding="UTF-8",
        standalone=True,
    )
    tmp_path = path.with_name(f"{path.stem}.tmp_caption_repair{path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with zipfile.ZipFile(path, "r") as src_archive, zipfile.ZipFile(
            tmp_path, "w"
        ) as dst_archive:
            for item in src_archive.infolist():
                payload = (
                    updated_xml
                    if item.filename == "word/document.xml"
                    else src_archive.read(item.filename)
                )
                info = zipfile.ZipInfo(item.filename, date_time=item.date_time)
                info.compress_type = item.compress_type
                info.external_attr = item.external_attr
                info.create_system = item.create_system
                dst_archive.writestr(info, payload)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {
        "document": str(path),
        "document_xml_sha256": document_xml_sha256(path),
        "captions_repaired": repaired,
        "paragraphs_replaced": replaced,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-captions", action="store_true")
    args = parser.parse_args()
    if args.repair_captions:
        result = repair_caption_runs(MANUSCRIPT.resolve())
    else:
        result = update_docx(MANUSCRIPT.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Build a manuscript DOCX that follows the 1.docx template formatting."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from .common import FIGURES, TABLES
from .make_manuscript import (
    TITLE,
    build_references,
    manuscript_metrics,
    paragraph_texts,
)


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
TEMPLATE = ROOT / "1.docx"
OUTPUT = ROOT / "CTOGT_MACCE_manuscript_1format.docx"
MIRROR = Path(
    r"C:\Users\Administrator\Documents\Codex\2026-11-07"
    r"\invariant-continuous-time-organ-graph-learning"
    r"\CTOGT_MACCE_manuscript_1format.docx"
)


def clear_body(document: Document) -> None:
    body = document._element.body
    for child in list(body):
        if child.tag == qn("w:sectPr"):
            continue
        body.remove(child)


def set_run_font(run, size: float | None = None, bold: bool | None = None) -> None:
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Times New Roman")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def add_body(document: Document, text: str):
    paragraph = document.add_paragraph(style="Normal")
    run = paragraph.add_run(text)
    set_run_font(run)
    return paragraph


def add_main_heading(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="Normal")
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    set_run_font(run, 14, True)


def add_subheading(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="Normal")
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run(text)
    set_run_font(run, 12, True)


def add_labeled_abstract(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="Normal")
    if ":" in text:
        label, body = text.split(":", 1)
        label = "Conclusion" if label == "Conclusions" else label
        label_run = paragraph.add_run(f"{label}:")
        set_run_font(label_run, bold=True)
        body_run = paragraph.add_run(body)
        set_run_font(body_run)
    else:
        run = paragraph.add_run(text)
        set_run_font(run)


def split_label(text: str) -> tuple[str, str]:
    match = re.match(r"^([A-Z][^.]{2,110})\.\s+(.*)$", text)
    if not match:
        return "", text
    return match.group(1).strip(), match.group(2).strip()


def set_cell_border(cell, *, bottom: bool = False) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    edge = OxmlElement("w:bottom")
    edge.set(qn("w:val"), "single" if bottom else "none")
    edge.set(qn("w:sz"), "12" if bottom else "0")
    edge.set(qn("w:space"), "0")
    edge.set(qn("w:color"), "000000")
    borders.append(edge)


def set_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge_name, size, value in (
        ("top", "12", "single"),
        ("left", "0", "none"),
        ("bottom", "12", "single"),
        ("right", "0", "none"),
        ("insideH", "0", "none"),
        ("insideV", "0", "none"),
    ):
        edge = OxmlElement(f"w:{edge_name}")
        edge.set(qn("w:val"), value)
        edge.set(qn("w:sz"), size)
        edge.set(qn("w:space"), "0")
        edge.set(qn("w:color"), "000000")
        borders.append(edge)
    tbl_pr.append(borders)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)


def add_caption(document: Document, text: str, *, bold_all: bool) -> None:
    paragraph = document.add_paragraph(style="Normal")
    paragraph.paragraph_format.keep_with_next = True
    if bold_all:
        run = paragraph.add_run(text)
        set_run_font(run, 12, True)
        return
    match = re.match(r"^(Figure \d+\.)\s*(.*)$", text)
    if match:
        prefix = paragraph.add_run(match.group(1) + " ")
        set_run_font(prefix, 12, True)
        body = paragraph.add_run(match.group(2))
        set_run_font(body, 12)
    else:
        run = paragraph.add_run(text)
        set_run_font(run, 12, True)


def add_footnote(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(style="Normal")
    run = paragraph.add_run(text)
    set_run_font(run, 10)


def add_figure(document: Document, filename: str, caption: str) -> None:
    path = FIGURES / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    paragraph = document.add_paragraph(style="Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Inches(6.4))
    add_caption(document, caption, bold_all=False)


def add_data_table(
    document: Document,
    frame: pd.DataFrame,
    widths: list[float],
) -> None:
    def format_value(value) -> str:
        if isinstance(value, float) and pd.notna(value):
            text = f"{value:.2f}"
            return "0.00" if text == "-0.00" else text
        text = "" if pd.isna(value) else str(value)
        text = re.sub(
            r"(?<!\d)(\d+\.\d{3,})(?!\d)",
            lambda match: (
                match.group(1)
                if abs(float(match.group(1)) - 0.001) < 1e-9
                else f"{float(match.group(1)):.2f}"
            ),
            text,
        )
        text = re.sub(
            r"(?<!\d)(\d+\.\d)%",
            lambda match: f"{float(match.group(1)):.2f}%",
            text,
        )
        text = re.sub(r"(?<!\d)-0\.00(?!\d)", "0.00", text)
        return text

    table = document.add_table(rows=1, cols=len(frame.columns))
    table.style = "Normal Table"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header = table.rows[0]
    tr_pr = header._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tr_pr.append(tbl_header)
    for index, column in enumerate(frame.columns):
        cell = header.cells[index]
        cell.width = Inches(widths[index])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_border(cell, bottom=True)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_before = Pt(2)
        paragraph.paragraph_format.space_after = Pt(2)
        run = paragraph.add_run(str(column))
        set_run_font(run, 9, True)
    for values in frame.itertuples(index=False):
        row = table.add_row()
        for index, value in enumerate(values):
            cell = row.cells[index]
            cell.width = Inches(widths[index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.LEFT
                if index == 0
                else WD_ALIGN_PARAGRAPH.CENTER
            )
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.paragraph_format.space_before = Pt(2)
            paragraph.paragraph_format.space_after = Pt(2)
            run = paragraph.add_run(format_value(value))
            set_run_font(run, 9)


def select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    return frame.loc[:, available].fillna("NA")


def table1(document: Document) -> None:
    frame = select_columns(
        pd.read_csv(TABLES / "table1_baseline.csv"),
        [
            "Characteristic",
            "MIMIC-IV development",
            "MIMIC-III temporal transport",
            "eICU expanded CAD transport",
        ],
    )
    add_caption(
        document,
        "Table 1. Cohort characteristics in the development and transport cohorts.",
        bold_all=True,
    )
    add_data_table(document, frame, [2.0, 1.5, 1.5, 1.5])
    add_footnote(
        document,
        "Values are mean (standard deviation), median, or percentages. "
        "NR, not recorded or not estimable in the source extract. Component "
        "event counts may overlap, so the MACCE composite is not their sum.",
    )


def table2(document: Document) -> None:
    frame = select_columns(
        pd.read_csv(TABLES / "table2_development_performance.csv"),
        [
            "Model",
            "AUROC (95% CI)",
            "AUPRC",
            "Brier",
            "Calibration slope",
            "ECE",
        ],
    )
    add_caption(
        document,
        "Table 2. Development performance at the 24-hour landmark.",
        bold_all=True,
    )
    add_data_table(document, frame, [2.2, 1.5, 0.7, 0.7, 0.8, 0.6])
    add_footnote(
        document,
        "Intervals are patient-level bootstrap 95% confidence intervals. "
        "AUROC, area under the receiver operating characteristic curve; "
        "AUPRC, area under the precision-recall curve; ECE, expected calibration error.",
    )


def table3(document: Document) -> None:
    frame = select_columns(
        pd.read_csv(TABLES / "table3_transport.csv"),
        [
            "Cohort",
            "Model",
            "AUROC raw",
            "AUROC recalibrated",
            "AUPRC raw",
            "Brier recalibrated",
            "ECE recalibrated",
        ],
    )
    add_caption(
        document,
        "Table 4. Transport without refitting and after local recalibration.",
        bold_all=True,
    )
    add_data_table(document, frame, [1.0, 2.0, 0.65, 0.8, 0.65, 0.65, 0.6])
    add_footnote(
        document,
        "Raw estimates use the pipeline fitted in MIMIC-IV. Recalibrated "
        "estimates use cross-fitted Platt scaling within each external database. "
        "NA indicates that a recalibrated estimate was not generated for the "
        "locked fusion.",
    )


def table4(document: Document) -> None:
    frame = select_columns(
        pd.read_csv(TABLES / "table4_interval_endpoint.csv"),
        [
            "Cohort",
            "Interval endpoint",
            "Events",
            "Retrained XGB+TS AUROC",
            "Existing temporal ensemble AUROC",
        ],
    )
    add_caption(
        document,
        "Table 6. Interval-endpoint sensitivity.",
        bold_all=True,
    )
    add_data_table(document, frame, [1.15, 2.2, 0.65, 1.05, 1.25])
    add_footnote(
        document,
        "Fully confirmed events require the complete candidate interval to lie "
        "within ICU hours 24-168. Anchor overlap is a sensitivity phenotype and "
        "is not an adjudicated clinical onset time. Retrained XGB+TS denotes "
        "interval-specific training in MIMIC-IV with unchanged external "
        "application; the temporal ensemble was applied without retraining.",
    )


def table5(document: Document) -> None:
    frame = select_columns(
        pd.read_csv(TABLES / "table5_validation.csv"),
        [
            "Validation",
            "Test set",
            "XGB+TS AUROC",
            "Temporal ensemble AUROC",
            "Difference (95% CI)",
            "0.01 threshold met",
        ],
    )
    add_caption(
        document,
        "Table 3. Locked temporal, nested, and internal-era validation.",
        bold_all=True,
    )
    add_data_table(document, frame, [1.1, 2.3, 0.6, 0.75, 1.1, 0.65])
    add_footnote(
        document,
        "Differences use patient-level paired bootstrap intervals. The "
        "2020-2022 era had been inspected in an earlier analysis snapshot; the "
        "second-stage model was nevertheless selected only on earlier eras. "
        "Threshold status uses unrounded estimates; a difference displayed as "
        "0.01 may therefore remain below the exact 0.01 margin.",
    )


def table6(document: Document) -> None:
    frame = select_columns(
        pd.read_csv(TABLES / "table6_manual_review.csv"),
        ["Review result", "Records"],
    )
    add_caption(
        document,
        "Table 7. Structured endpoint chart-review pilot.",
        bold_all=True,
    )
    add_data_table(document, frame, [4.8, 1.2])
    add_footnote(
        document,
        "The 38-record pilot was AI-assisted and is not independent clinical "
        "adjudication. Human PPV, timing error, and Cohen kappa remain pending.",
    )


def table7(document: Document) -> None:
    frame = pd.read_csv(TABLES / "table7_clinical_utility.csv")
    calibration = select_columns(
        frame,
        [
            "Cohort",
            "Model",
            "Probability",
            "AUROC",
            "Brier (95% CI)",
            "Calibration slope",
            "Calibration intercept",
            "ECE",
        ],
    )
    operating = select_columns(
        frame,
        [
            "Cohort",
            "Model",
            "Probability",
            "Sensitivity @10%",
            "Specificity @10%",
            "PPV @10%",
            "NPV @10%",
        ],
    )
    add_caption(
        document,
        "Table 5A. Calibration and probability error before and after local recalibration.",
        bold_all=True,
    )
    add_data_table(
        document,
        calibration,
        [0.85, 1.55, 0.7, 0.5, 1.0, 0.55, 0.65, 0.45],
    )
    add_caption(
        document,
        "Table 5B. Fixed 10% operating-threshold classification metrics.",
        bold_all=True,
    )
    add_data_table(
        document,
        operating,
        [1.0, 2.0, 0.8, 0.7, 0.7, 0.55, 0.55],
    )
    add_footnote(
        document,
        "Brier intervals and threshold metrics are bootstrap estimates. "
        "PPV and NPV depend on cohort prevalence. Calibration-minus-raw net-benefit "
        "differences are reported in the Results and Figure 6. NA indicates a "
        "non-estimable or non-generated value.",
    )


def table8(document: Document) -> None:
    frame = select_columns(
        pd.read_csv(TABLES / "table8_interval_threshold_sensitivity.csv"),
        [
            "Assumption",
            "Value",
            "Cohort",
            "Coded events",
            "Fully confirmed",
            "Anchor-overlap",
            "XGB+TS AUROC",
            "Temporal ensemble AUROC",
        ],
    )
    add_caption(
        document,
        "Table 8. Interval-rule threshold sensitivity.",
        bold_all=True,
    )
    add_data_table(
        document,
        frame,
        [1.55, 0.5, 0.8, 0.65, 0.65, 0.75, 0.7, 0.9],
    )
    add_footnote(
        document,
        "Frozen model predictions were applied without retuning. Values show "
        "event counts and discrimination under alternative interval rules.",
    )


def add_declaration(document: Document, heading: str, body: str) -> None:
    add_subheading(document, heading)
    add_body(document, body)


def build() -> Path:
    document = Document(TEMPLATE)
    clear_body(document)
    metrics = manuscript_metrics()
    references = build_references()
    text = paragraph_texts(metrics, references)

    title = document.add_paragraph(style="Normal")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(TITLE)
    set_run_font(run, 18, True)

    add_body(document, "Authors and affiliations to be finalized.")
    add_body(document, "Corresponding author: to be finalized.")

    add_main_heading(document, "Abstract")
    for paragraph in text["abstract"]:
        add_labeled_abstract(document, paragraph)
    keyword = document.add_paragraph(style="Normal")
    keyword_label = keyword.add_run("Keywords:")
    set_run_font(keyword_label, bold=True)
    keyword_body = keyword.add_run(
        " Coronary atherosclerosis; Major adverse cardiac and cerebrovascular "
        "events; Continuous-time modeling; Organ graph; Temporal ensemble; "
        "External validation"
    )
    set_run_font(keyword_body)

    add_main_heading(document, "1. Introduction")
    for paragraph in text["introduction"]:
        add_body(document, paragraph)

    add_main_heading(document, "2. Materials and Methods")
    method_insertions = {
        "Study design and data sources": [
            (
                "figure",
                "figure1_study_design.png",
                "Figure 1. Study flow, 24-hour landmark, and three-database validation.",
            )
        ],
        "CT-OGT architecture": [
            (
                "figure",
                "figure2_model_architecture.png",
                "Figure 2. CT-OGT architecture with continuous-time cues, dynamic organ graph, and temporal encoder.",
            )
        ],
    }
    for index, paragraph in enumerate(text["methods"], start=1):
        label, body = split_label(paragraph)
        heading = f"2.{index} {label}" if label else f"2.{index} Method element"
        add_subheading(document, heading)
        add_body(document, body)
        for insertion in method_insertions.get(label, []):
            add_figure(document, insertion[1], insertion[2])

    add_main_heading(document, "3. Results")
    result_insertions = {
        "Cohort": [("table", table1)],
        "Temporal information and benchmark performance": [
            ("table", table2),
            (
                "figure",
                "figure3_performance_ablation.png",
                "Figure 3. Development performance and mechanism controls.",
            ),
        ],
        "Nested selection and clinical utility": [("table", table5)],
        "Transport": [("table", table3)],
        "Calibration and clinical utility": [
            ("table", table7),
            (
                "figure",
                "figure4_calibration_dca.png",
                "Figure 4. External calibration and decision-curve analysis.",
            ),
            (
                "figure",
                "figure6_net_benefit_ci.png",
                "Figure 5. Net benefit with bootstrap confidence intervals before and after local recalibration.",
            ),
        ],
        "Endpoint sensitivity": [("table", table4)],
        "Endpoint chart-review pilot": [("table", table6)],
        "Interval-rule sensitivity": [
            ("table", table8),
            (
                "figure",
                "figure7_interval_threshold_sensitivity.png",
                "Figure 6. Sensitivity of interval endpoints to troponin thresholds and eICU documentation lag.",
            ),
        ],
        "Explainability": [
            (
                "figure",
                "figure5_attention.png",
                "Figure 7. Organ and temporal attention in the full model.",
            )
        ],
    }
    for index, paragraph in enumerate(text["results"], start=1):
        label, body = split_label(paragraph)
        body = body.replace("Figures 4 and 6", "Figures 4 and 5")
        heading = f"3.{index} {label}" if label else f"3.{index} Result element"
        add_subheading(document, heading)
        add_body(document, body)
        for kind, *payload in result_insertions.get(label, []):
            if kind == "table":
                payload[0](document)
            else:
                add_figure(document, payload[0], payload[1])

    add_main_heading(document, "4. Discussion")
    discussion_labels = {
        "Principal findings": "4.1 Principal findings",
        "Clinical interpretation": "4.2 Clinical interpretation",
        "Why calibration dominates transport": "4.3 Transport and calibration",
        "Graph learning and negative controls": "4.4 Graph learning and negative controls",
        "Endpoint ascertainment is the main scientific bottleneck": (
            "4.5 Endpoint ascertainment"
        ),
        "Implications for model development": "4.6 Implications for model development",
    }
    for paragraph in text["discussion"]:
        label, body = split_label(paragraph)
        heading = discussion_labels.get(label, f"4.7 {label or 'Discussion'}")
        add_subheading(document, heading)
        add_body(document, body)
    add_subheading(document, "4.7 Limitations")
    for paragraph in text["limitations"]:
        add_body(document, paragraph)

    add_main_heading(document, "5. Conclusion")
    for paragraph in text["conclusion"]:
        add_body(document, paragraph)

    add_main_heading(document, "6. Declarations")
    add_declaration(
        document,
        "Data availability",
        "MIMIC-IV, MIMIC-III, and eICU-CRD are available through PhysioNet "
        "under their respective credentialed data-use agreements. Raw patient "
        "data are not redistributed.",
    )
    add_declaration(
        document,
        "Code availability",
        "The analysis code, endpoint audits, model configurations, fold-level "
        "predictions, result tables, and figure-generation scripts are preserved "
        "in the study repository.",
    )
    add_declaration(
        document,
        "Reporting guidelines",
        "The study is reported in accordance with TRIPOD+AI guidance for "
        "prediction models that use machine learning.",
    )
    add_declaration(
        document,
        "Ethics approval and consent to participate",
        "This secondary analysis used de-identified data collected under the "
        "original database approvals and does not involve a new patient-facing "
        "intervention.",
    )
    add_declaration(
        document,
        "Authors' contributions",
        "Author contributions are to be finalized.",
    )
    add_declaration(
        document,
        "Funding",
        "Funding information is to be finalized.",
    )
    add_declaration(
        document,
        "Declaration of competing interest",
        "Conflict-of-interest statements are to be finalized by the authors.",
    )
    add_declaration(
        document,
        "Declaration of generative AI and AI-assisted technologies",
        "OpenAI Codex was used to assist with code implementation, statistical "
        "workflow execution, and language editing. The authors remain responsible "
        "for the scientific interpretation and final manuscript.",
    )
    add_declaration(document, "Consent for publication", "Not applicable.")

    add_main_heading(document, "7. References")
    for index, record in enumerate(references, start=1):
        paragraph = document.add_paragraph(style="RefEntry")
        run = paragraph.add_run(f"{index}. {record['citation']}")
        set_run_font(run)

    document.core_properties.title = TITLE
    document.core_properties.subject = (
        "Diagnosis-coded MACCE prediction with continuous-time organ graph "
        "learning and complementary temporal ensembling"
    )
    document.core_properties.author = "To be finalized"
    document.core_properties.keywords = (
        "MACCE; coronary atherosclerosis; continuous-time; organ graph; "
        "temporal ensemble; external validation"
    )
    document.save(OUTPUT)
    MIRROR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT, MIRROR)
    return OUTPUT


if __name__ == "__main__":
    print(build())

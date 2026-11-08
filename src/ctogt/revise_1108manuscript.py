"""Revise 1108manuscript.docx to satisfy the paper-word requirements."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from .make_manuscript import TITLE, build_references, manuscript_metrics, paragraph_texts
from .make_manuscript_1format import (
    add_body,
    add_caption,
    add_data_table,
    add_declaration,
    add_footnote,
    add_labeled_abstract,
    add_main_heading,
    add_subheading,
    clear_body,
    select_columns,
    set_run_font,
    split_label,
    table1,
    table2,
    table3,
    table4,
    table5,
    table6,
    table7,
    table8,
)


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
TEMPLATE = ROOT / "1.docx"
OUTPUT = ROOT / "1108manuscript.docx"
REGISTRY = ROOT / "results" / "reference_registry_verified.json"
R_FIGURES = ROOT / "results" / "figures_r"
METRICS_CACHE = ROOT / "results" / "manuscript_metrics_cache.json"

OLD_KEYS = [
    "johnson2023mimiciv",
    "johnson2016mimiciii",
    "pollard2018eicu",
    "collins2024tripodai",
    "vincent1996sofa",
    "shickel2018deepehr",
    "rajkomar2018scalable",
    "che2018grud",
    "vaswani2017attention",
    "lim2021tft",
    "arjovsky2019irm",
    "sagawa2020groupdro",
    "kidger2020ncde",
    "gu2023mamba",
    "vancalster2019calibration",
    "vickers2006dca",
    "thygesen2019mi",
    "sacco2013stroke",
]

ABSTRACT = [
    (
        "Background: Patients with coronary atherosclerosis admitted to an intensive "
        "care unit remain at risk of major adverse cardiac and cerebrovascular events "
        "(MACCE). Static severity scores may not capture evolving multi-organ physiology."
    ),
    (
        "Methods: We conducted a retrospective three-database study using MIMIC-IV "
        "for development, MIMIC-III for temporal transport, and expanded eICU for "
        "multi-hospital transport. Adults alive at ICU hour 24 with at least six "
        "observed organ channels were included. We compared static models, XGBoost "
        "with temporal summaries, Transformers, organ graphs, and an exploratory "
        "temporal ensemble. Features and fusion weights were selected on 2017-2019 "
        "data and frozen for 2020-2022 testing."
    ),
    (
        "Results: The development cohort included 13,131 patients and 4,475 "
        "diagnosis-coded MACCE events. The original temporal ensemble reached AUROC "
        "0.76, versus 0.75 for XGBoost with temporal summaries. The era-locked "
        "75/25 enhanced-XGBoost/Transformer model reached 0.75 in 2020-2022, versus "
        "0.73 for the benchmark (difference 0.01, 95% CI 0.01-0.02). Expanded eICU "
        "included 5,913 patients and 642 events. Interval-rule sensitivity was stable, "
        "but confirmed timing strata were too sparse for inference."
    ),
    (
        "Conclusion: Twenty-four-hour temporal features were relatively consistently "
        "associated with improved MACCE discrimination. The exploratory ensemble "
        "showed a modest late-era gain, whereas organ-graph increments were not "
        "supported. External probabilities required local recalibration. Independent "
        "clinical endpoint adjudication remains necessary."
    ),
]


def citation_keys(numbers: str) -> list[str]:
    keys: list[str] = []
    for part in re.split(r",\s*", numbers):
        if "-" in part:
            start, end = [int(value) for value in part.split("-", 1)]
            keys.extend(OLD_KEYS[start - 1:end])
        elif part.strip().isdigit():
            keys.append(OLD_KEYS[int(part) - 1])
    return keys


def replace_numeric_citations(text: str) -> str:
    return re.sub(
        r"\[([0-9,\-\s]+)\]",
        lambda match: "{{" + ",".join(citation_keys(match.group(1))) + "}}",
        text,
    )


def append_to_label(paragraphs: list[str], label: str, text: str) -> None:
    for index, paragraph in enumerate(paragraphs):
        found, _ = split_label(paragraph)
        if found == label:
            paragraphs[index] = paragraph.rstrip() + " " + text
            return
    raise KeyError(label)


def add_contextual_citations(sections: dict[str, list[str]]) -> None:
    intro = sections["introduction"]
    intro[0] = (
        intro[0]
        + " Contemporary guidance informed the clinical framing "
        + "{{byrne2023escacs,virani2023chroniccad}}."
    )
    intro[2] = (
        intro[2]
        + " Established scores summarize selected acute risk factors "
        + "{{antman2000timi,fox2006grace,lipp2010chads}}."
    )
    methods = sections["methods"]
    append_to_label(
        methods,
        "Population and landmark",
        "Diagnostic definitions followed contemporary coronary and stroke guidance "
        "{{thygesen2019mi,sacco2013stroke,powers2019stroke}}.",
    )
    append_to_label(
        methods,
        "Endpoint",
        "Revascularization terminology followed contemporary coronary guidance "
        "{{lawton2021revasc}}.",
    )
    append_to_label(
        methods,
        "Continuous-time encoding and dynamic graph",
        "Sparse clinical priors were used because dense graph learning can be "
        "comparatively unstable {{skarding2021dynamicgraphs,lee2019attentiongraphs,daniels2010monotone}}. "
        "Irregular histories and selective sequence models informed the encoder design "
        "{{kidger2020ncde,gu2023mamba}}.",
    )
    append_to_label(
        methods,
        "Comparators and ablations",
        "XGBoost, gated recurrent units, and attention models served as established "
        "comparators {{chen2016xgboost,cho2014rnn,lin2022transformers}}.",
    )
    append_to_label(
        methods,
        "Cross-validation, nested selection, and transport",
        "Nested tuning and external evaluation followed prediction-model guidance "
        "{{collins2024evaluation1,riley2024evaluation2,varma2006cvbias,vancalster2023validated}}.",
    )
    append_to_label(
        methods,
        "Metrics and statistical analysis",
        "Calibration, ordinal performance, uncertainty, and resampling followed "
        "established frameworks "
        "{{mccullagh1980ordinal,edlinger2022ordinal,vanhoorde2014multinomial,steyerberg2010performance,abdar2021uncertainty,efron1979bootstrap}}.",
    )
    append_to_label(
        methods,
        "Sensitivity analyses",
        "Risk-of-bias terminology followed PROBAST guidance {{wolff2019probast}}.",
    )


def two_decimal(value: str) -> str:
    number = float(value)
    if abs(number - 0.001) < 1e-9:
        return value
    return f"{number:.2f}"


def normalize_decimals(text: str) -> str:
    text = re.sub(
        r"(?<!\d)(\d+\.\d{3,})(?!\d)",
        lambda match: two_decimal(match.group(1)),
        text,
    )
    text = re.sub(
        r"(?<!\d)(\d+\.\d)%",
        lambda match: f"{float(match.group(1)):.2f}%",
        text,
    )
    return text


def split_at(text: str, delimiter: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(delimiter, text, maxsplit=1)
        if part.strip()
    ]


def shorten_sentence(sentence: str) -> str:
    words = sentence.split()
    if len(words) <= 30:
        return sentence
    for delimiter in (r";\s+", r",\s+but\s+", r",\s+and\s+", r",\s+while\s+"):
        parts = split_at(sentence, delimiter)
        if len(parts) == 2:
            first, second = parts
            if first and not first.endswith((".", "!", "?")):
                first += "."
            second = second[:1].upper() + second[1:]
            return shorten_sentence(first) + " " + shorten_sentence(second)
    comma_positions = [match.start() for match in re.finditer(r",\s+", sentence)]
    if comma_positions:
        midpoint = len(sentence) // 2
        position = min(comma_positions, key=lambda value: abs(value - midpoint))
        first = sentence[:position].strip()
        second = sentence[position + 1:].strip()
        if first and not first.endswith((".", "!", "?")):
            first += "."
        second = second[:1].upper() + second[1:]
        return shorten_sentence(first) + " " + shorten_sentence(second)
    return sentence


def enforce_sentence_limit(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return " ".join(shorten_sentence(sentence.strip()) for sentence in sentences)


def polish(text: str) -> str:
    text = text.replace(
        "Halving, retaining, and doubling the assay troponin threshold changed "
        "MIMIC-IV anchor-overlap events from 1,266 to 1,265 and 1,254; the "
        "temporal-ensemble AUROC changed from 0.769 to 0.769 and 0.769, respectively.",
        "MIMIC-IV anchor-overlap events were 1,266, 1,265, and 1,254 across the "
        "three thresholds. Temporal-ensemble AUROC was 0.769 for each threshold.",
    )
    text = text.replace(
        "These controls are reported in Figure 3.",
        "These controls are reported in Figure 5.",
    )
    text = text.replace(
        "Calibration and utility uncertainty are shown in Figures 4 and 6.",
        "Calibration and utility uncertainty are shown in Figures 6 and 7.",
    )
    replacements = (
        ("provided a consistent gain", "was relatively consistently associated with a gain"),
        ("demonstrates", "suggests"),
        ("confirms", "supports"),
        ("proves", "suggests"),
        ("clearly", "comparatively clearly"),
        ("will improve", "may potentially improve"),
        ("strong evidence", "comparatively consistent evidence"),
        ("caused", "was potentially associated with"),
        ("confirm", "tentatively support"),
        ("In addition, ", "Additionally, "),
        ("Therefore, ", "Accordingly, "),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return enforce_sentence_limit(normalize_decimals(text))


def citation_marker(keys: list[str]) -> str:
    if not keys:
        return ""
    if len(keys) == 1:
        return f"[{keys[0]}]"
    return "[" + ",".join(str(key) for key in keys) + "]"


def render_citations(sections: dict[str, list[str]]) -> tuple[dict[str, list[str]], list[dict]]:
    registry = {
        record["key"]: record
        for record in json.loads(REGISTRY.read_text(encoding="utf-8"))
    }
    order: list[str] = []
    number: dict[str, int] = {}

    def replace(match: re.Match[str]) -> str:
        keys = [value.strip() for value in match.group(1).split(",") if value.strip()]
        for key in keys:
            if key not in number:
                order.append(key)
                number[key] = len(order)
        assigned = sorted(number[key] for key in keys if key in number)
        if not assigned:
            return ""
        compact = []
        start = previous = assigned[0]
        for value in assigned[1:]:
            if value == previous + 1:
                previous = value
                continue
            compact.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = value
        compact.append(str(start) if start == previous else f"{start}-{previous}")
        return "[" + ",".join(compact) + "]"

    rendered = {}
    for key, paragraphs in sections.items():
        rendered[key] = [
            re.sub(r"\{\{([^}]+)\}\}", replace, paragraph)
            for paragraph in paragraphs
        ]
    missing = sorted(set(registry).difference(order))
    if missing:
        raise RuntimeError(f"Uncited references: {missing}")
    references = [registry[key] for key in order]
    return rendered, references


def add_r_figure(
    document: Document,
    filename: str,
    caption: str,
    interpretation: str,
) -> None:
    path = R_FIGURES / filename
    paragraph = document.add_paragraph(style="Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Inches(6.4))
    add_caption(document, caption, bold_all=False)
    add_footnote(document, "Interpretation: " + interpretation)


def add_table_interpretation(document: Document, text: str) -> None:
    add_footnote(document, "Interpretation: " + text)


DISCUSSION = [
    (
        "4.1 Principal findings",
        [
            (
                "This three-database study examined whether continuous-time organ "
                "representations improve diagnosis-coded MACCE prediction. Temporal "
                "summaries improved discrimination over static severity in development. "
                "A complementary ensemble was then constructed as an exploratory model. "
                "Its development gain exceeded the material threshold."
            ),
            (
                "The original ensemble narrowly missed the material threshold after "
                "strict XGBoost tuning. A second-stage model was therefore selected "
                "only on 2008-2016 versus 2017-2019. The frozen 75/25 enhanced-XGBoost/"
                "Transformer model reached AUROC 0.75 in 2020-2022. The observed gain "
                "over XGB+TS was 0.01, with a 95% CI of 0.01-0.02."
            ),
        ],
    ),
    (
        "4.2 Interpretation of results",
        [
            (
                "The temporal gain was comparatively consistent across development and "
                "late-era testing. It may partially reflect evolving organ dysfunction, "
                "treatment intensity, and documentation density. These signals are not "
                "interchangeable with a specific causal pathway."
            ),
            (
                "The organ-graph results were cautiously negative. Dynamic channel graphs "
                "and seven-organ token graphs did not improve materially over XGB+TS. "
                "An organ-rate model also failed its internal validation gate. The graph "
                "therefore seemed comparatively less useful than tabular temporal summaries."
            ),
            (
                "The diagnosis-coded endpoint remains a major source of uncertainty. "
                "Coded events may include present-on-admission conditions or delayed "
                "documentation. The anchor-overlap analysis was stable to interval-rule "
                "perturbations. It nevertheless cannot substitute for independent "
                "clinical adjudication."
            ),
        ],
    ),
    (
        "4.3 Potential mechanisms",
        [
            (
                "The first ICU day contains both current severity and treatment response. "
                "Circulatory, respiratory, renal, coagulation, inflammatory, and metabolic "
                "signals evolve together during resuscitation. Temporal summaries can "
                "potentially capture this coordinated change more efficiently than static "
                "severity variables."
            ),
            (
                "The transformer may add complementary information through nonlinear "
                "interactions across irregular observations. XGBoost may remain comparatively "
                "stable for sparse tabular summaries. Their combination may therefore "
                "partially reduce model-specific variance."
            ),
            (
                "The graph result suggests that explicit organ edges are not automatically "
                "informative. Missingness, sparse observations, and cohort heterogeneity "
                "may dilute stable edge learning. Invariance penalties may also trade "
                "average discrimination for comparatively fragile group robustness."
            ),
        ],
    ),
    (
        "4.4 Comparison with previous studies",
        [
            (
                "Prior ICU studies have reported gains from recurrent, transformer, and "
                "dynamic graph models {{rajkomar2018scalable,ma2023patientgraph,feng2024multiorgan}}. "
                "Many comparisons used static baselines or single-database evaluations. "
                "The present benchmark included temporal XGBoost and three-database transport."
            ),
            (
                "Graph-based ICU models have also shown potential for intervention and "
                "mortality prediction {{xu2024icuinterventions}}. The present results suggest "
                "that graph complexity should be tested against strong temporal summaries. "
                "Comparatively small graph gains may otherwise reflect baseline weakness."
            ),
            (
                "Prior prediction-model guidance emphasizes calibration, external "
                "validation, and transparent model selection {{collins2024evaluation1,riley2024evaluation2}}. "
                "This study followed that guidance through paired bootstrap, local "
                "recalibration, decision curves, and a separate temporal holdout."
            ),
        ],
    ),
    (
        "4.5 Clinical and methodological significance",
        [
            (
                "The findings support temporal risk stratification as a comparatively "
                "promising research direction. They do not support direct clinical "
                "deployment. Absolute probabilities were poorly calibrated before local "
                "updating, particularly in expanded eICU."
            ),
            (
                "Local recalibration reduced Brier error and improved net benefit at several "
                "thresholds. Discrimination changed only modestly. Consequently, calibration "
                "should remain a mandatory component of transport evaluation."
            ),
            (
                "The exploratory model-selection sequence is also methodologically important. "
                "The ensemble was not pre-specified. Feature and weight selection were "
                "restricted to earlier eras. The late-era result should therefore be read "
                "as comparatively supportive, not definitive."
            ),
        ],
    ),
    (
        "4.6 Future directions",
        [
            (
                "Prospective validation should include independent endpoint adjudication. "
                "Serial troponin, ECG, angiography, PCI, and stroke imaging should be linked "
                "to explicit event times. Such data could potentially reduce phenotype "
                "misclassification."
            ),
            (
                "Future graph models should pre-specify organ-level structure and test it "
                "against temporal tabular benchmarks. External edge stability should be "
                "examined across hospitals and eras. Utility should be evaluated at fixed "
                "clinical thresholds."
            ),
        ],
    ),
]


LIMITATIONS = [
    (
        "First, diagnosis codes did not provide fully reliable event-onset times. "
        "Second, the strict interval endpoint contained too few confirmed events for "
        "stable discrimination analysis. Third, the expanded eICU cohort changed the "
        "target population and remained comparatively heterogeneous."
    ),
    (
        "Fourth, the 2020-2022 era had been inspected in an earlier analysis snapshot. "
        "The second-stage evaluation was not a never-seen prospective holdout. Feature "
        "and weight selection were nevertheless restricted to earlier years."
    ),
    (
        "Fifth, independent two-reader endpoint adjudication remains pending. Sixth, "
        "deep models used a single seed in some nested analyses. Seventh, urine output "
        "was unavailable in eICU. Eighth, the retrospective design cannot establish "
        "treatment benefit or causal organ interactions."
    ),
    (
        "Finally, routine-care coding and documentation differ across databases. The "
        "results require prospective validation and external adjudication before clinical use."
    ),
]


def table_with_interpretation(document: Document, function, text: str) -> None:
    function(document)
    add_table_interpretation(document, text)


def build() -> Path:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if len(registry) < 40 or len(registry) > 50:
        raise RuntimeError(f"Expected 40-50 references, got {len(registry)}")
    if METRICS_CACHE.is_file():
        metrics = json.loads(METRICS_CACHE.read_text(encoding="utf-8"))
    else:
        metrics = manuscript_metrics()
        METRICS_CACHE.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
    original = paragraph_texts(
        metrics,
        build_references(),
    )
    sections = {
        "abstract": list(ABSTRACT),
        "introduction": [replace_numeric_citations(value) for value in original["introduction"]],
        "methods": [replace_numeric_citations(value) for value in original["methods"]],
        "results": [replace_numeric_citations(value) for value in original["results"]],
        "discussion": [
            paragraph
            for _, paragraphs in DISCUSSION
            for paragraph in paragraphs
        ],
        "limitations": LIMITATIONS,
        "conclusion": [replace_numeric_citations(value) for value in original["conclusion"]],
    }
    add_contextual_citations(sections)
    sections, references = render_citations(sections)
    for key in sections:
        sections[key] = [polish(value) for value in sections[key]]

    document = Document(TEMPLATE)
    clear_body(document)
    for relationship_id, relationship in list(document.part.rels.items()):
        if "image" in relationship.reltype:
            document.part.drop_rel(relationship_id)
    title = document.add_paragraph(style="Normal")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(TITLE)
    set_run_font(run, 18, True)

    add_body(
        document,
        "Zhipeng Wang1*, Luning Wang2*, Changsong Wang1, Xiyu Xu3, Pengli Zhai4, "
        "Zejun Liu1, Hui Feng1, Qian Hou1, Ming Guo1",
    )
    for affiliation in (
        "1 Department of TCM, Zhongda Hospital, Southeast University, Nanjing, China",
        "2 Department of Rehabilitation Medicine, Zhongda Hospital, Southeast University, Nanjing, China",
        "3 School of Medicine, Southeast University, Nanjing, China",
        "4 Jiangbei Campus, Jiangsu Provincial Traditional Chinese Medicine Hospital, Nanjing, China",
        "* These authors contributed equally to this work and are co-first authors.",
        "Corresponding author: Zhipeng Wang, Department of TCM, Zhongda Hospital, Southeast University, China. Email: 311000687@seu.edu.cn",
    ):
        add_body(document, affiliation)

    add_main_heading(document, "Abstract")
    for paragraph in sections["abstract"]:
        add_labeled_abstract(document, paragraph)
    keyword = document.add_paragraph(style="Normal")
    label = keyword.add_run("Keywords:")
    set_run_font(label, bold=True)
    body = keyword.add_run(
        " Coronary Atherosclerosis; MACCE; Continuous-Time Modeling; "
        "Organ Graph; External Validation"
    )
    set_run_font(body)

    add_main_heading(document, "1. Introduction")
    for paragraph in sections["introduction"]:
        add_body(document, paragraph)

    add_main_heading(document, "2. Materials and Methods")
    methods = sections["methods"]
    for index, paragraph in enumerate(methods, start=1):
        label_text, body_text = split_label(paragraph)
        add_subheading(
            document,
            f"2.{index} {label_text}" if label_text else f"2.{index} Method element",
        )
        add_body(document, body_text)

    add_main_heading(document, "3. Results")
    add_body(
        document,
        "Figure 1 summarizes the overall data flow, model architecture, "
        "selection strategy, and evaluation sequence.",
    )
    add_r_figure(
        document,
        "figure1_overall_roadmap.png",
        "Figure 1. Overall study roadmap. (A) Cohort construction and landmark. "
        "(B) Recorded outcome components. (C) Model architecture. "
        "(D) Temporal selection and validation sequence.",
        "The roadmap integrates cohort construction, model structure, "
        "internal selection, late-era testing, transport, and utility analysis.",
    )
    result_insertions = {
        "Cohort": lambda: (
            table_with_interpretation(
                document,
                table1,
                "Across cohorts, risk factors and event prevalence differed materially. "
                "Separate recalibration was therefore required.",
            ),
            add_r_figure(
                document,
                "figure2_cohort_profiles.png",
                "Figure 2. Cohort risk-factor profiles. "
                "(A) Prevalence heatmap. (B) Percentage-point differences from MIMIC-IV.",
                "Comorbidity prevalence differed materially across cohorts. "
                "MIMIC-IV and eICU were comparatively distinct.",
            ),
            add_r_figure(
                document,
                "figure3_observation_coverage.png",
                "Figure 3. Observation coverage across organ channels. "
                "(A) Channel-level coverage. (B) Mean coverage by organ system.",
                "Coverage varied by cohort and organ system. Urine output was "
                "unavailable in the eICU extract.",
            ),
        ),
        "Temporal information and benchmark performance": lambda: (
            table_with_interpretation(
                document,
                table2,
                "Temporal summaries improved discrimination over static XGBoost. "
                "The exploratory ensemble showed the highest development AUROC.",
            ),
            add_r_figure(
                document,
                "figure4_performance.png",
                "Figure 4. Development, temporal validation, transport, and probability error. "
                "(A) Development AUROCs. (B) Temporal validation. "
                "(C) Transport discrimination. (D) Brier score.",
                "Temporal models improved development discrimination. The locked fusion "
                "showed the strongest late-era signal.",
            ),
        ),
        "Nested selection and clinical utility": lambda: (
            table_with_interpretation(
                document,
                table5,
                "The internal-era locked fusion met the material threshold. Joint nesting "
                "remained positive but below threshold.",
            ),
            add_r_figure(
                document,
                "figure5_model_selection.png",
                "Figure 5. Nested model selection and organ-graph controls. "
                "(A) Nested model performance. (B) Organ graph and negative controls. "
                "(C) Paired AUROC contrasts. "
                "(D) Incremental AUROC difference versus XGB+TS.",
                "Joint nesting remained positive but below the material threshold. "
                "Organ graph variants did not exceed the temporal benchmark.",
            ),
        ),
        "Transport": lambda: table_with_interpretation(
            document,
            table3,
            "Enhanced fusion improved MIMIC-III transport. Expanded eICU showed "
            "comparatively neutral transport performance.",
        ),
        "Calibration and clinical utility": lambda: (
            table_with_interpretation(
                document,
                table7,
                "Local recalibration reduced probability error. Threshold performance "
                "changed materially after updating.",
            ),
            add_r_figure(
                document,
                "figure6_transport_utility.png",
                "Figure 6. Transport, calibration, decision curves, and threshold metrics. "
                "(A-B) Reliability before and after recalibration. (C) Net benefit with "
                "bootstrap intervals. (D) Metrics at the 10% threshold.",
                "Local recalibration improved reliability and net benefit. Discrimination "
                "itself changed only modestly.",
            ),
            add_r_figure(
                document,
                "figure7_risk_and_decisions.png",
                "Figure 7. Risk distributions and threshold decisions. "
                "(A-B) Risk distributions before and after recalibration. "
                "(C) 10% threshold decision composition. (D) Operating characteristics.",
                "Recalibration changed the distribution of predicted risk and the "
                "composition of threshold-based decisions.",
            ),
        ),
        "Endpoint sensitivity": lambda: table_with_interpretation(
            document,
            table4,
            "Anchor overlap was stable across interval rules. Fully confirmed strata "
            "were too small for stable inference.",
        ),
        "Endpoint chart-review pilot": lambda: table_with_interpretation(
            document,
            table6,
            "Most pilot records lacked definitive human review. PPV and reader agreement "
            "remain pending.",
        ),
        "Interval-rule sensitivity": lambda: (
            table_with_interpretation(
                document,
                table8,
                "Troponin and documentation-lag perturbations did not materially change "
                "discrimination.",
            ),
            add_r_figure(
                document,
                "figure8_endpoint_ascertainment.png",
                "Figure 8. Endpoint ascertainment and interval-rule sensitivity. "
                "(A) Coded and anchor-overlap events. (B) Troponin threshold sensitivity. "
                "(C) eICU documentation-lag invariance. "
                "(D) AUROC range across rule variants.",
                "Anchor-overlap ascertainment was relatively stable. Fully confirmed "
                "timing strata remained sparse, and documentation lag did not change "
                "overlap classification.",
            ),
        ),
        "Explainability": lambda: add_r_figure(
            document,
            "figure9_attention.png",
            "Figure 9. Organ and temporal attention. (A) Organ-channel attention. "
            "(B) Hour-specific attention. (C) Top attention contrasts.",
            "Attention concentrated in circulatory, neurologic, renal, and coagulation "
            "channels. Hour-specific patterns were comparatively broad.",
        ),
    }
    for index, paragraph in enumerate(sections["results"], start=1):
        label_text, body_text = split_label(paragraph)
        add_subheading(
            document,
            f"3.{index} {label_text}" if label_text else f"3.{index} Result element",
        )
        add_body(document, body_text)
        result_callouts = {
            "Cohort": (
                "Cohort profiles are shown in Table 1, Figure 2, and Figure 3."
            ),
            "Temporal information and benchmark performance": (
                "Development performance is shown in Table 2 and Figure 4."
            ),
            "Nested selection and clinical utility": (
                "Validation comparisons are summarized in Table 3 and Figure 5."
            ),
            "Transport": "Transport results are shown in Table 4 and Figure 6.",
            "Calibration and clinical utility": (
                "Calibration and threshold metrics are shown in Table 5A, "
                "Table 5B, Figure 6, and Figure 7."
            ),
            "Endpoint sensitivity": (
                "Interval-endpoint results are shown in Table 6."
            ),
            "Endpoint chart-review pilot": (
                "Pilot labels are summarized in Table 7."
            ),
            "Interval-rule sensitivity": (
                "Endpoint ascertainment and threshold sensitivity are shown in "
                "Table 8 and Figure 8."
            ),
            "Explainability": "Attention patterns are shown in Figure 9.",
        }
        if label_text in result_callouts:
            add_body(document, result_callouts[label_text])
        if label_text in result_insertions:
            result_insertions[label_text]()

    add_main_heading(document, "4. Discussion")
    discussion_paragraphs = iter(sections["discussion"])
    for heading, paragraphs in DISCUSSION:
        add_subheading(document, heading)
        for _ in paragraphs:
            add_body(document, next(discussion_paragraphs))

    add_subheading(document, "4.7 Limitations")
    for paragraph in sections["limitations"]:
        add_body(document, paragraph)

    add_main_heading(document, "5. Conclusion")
    for paragraph in sections["conclusion"]:
        add_body(document, paragraph)

    add_main_heading(document, "6. Declarations")
    declarations = [
        (
            "Data availability",
            "MIMIC-IV, MIMIC-III and eICU-CRD are available from PhysioNet under "
            "credentialed access. Raw and processed patient-level records are not "
            "redistributed. Non-sensitive aggregate results and figure source data are "
            "included in the code repository. A reproduction guide is included in the "
            "code repository.",
        ),
        (
            "Code availability",
            "The analysis code, tests, configuration, audit scripts, and aggregate "
            "results are available in a public GitHub repository at "
            "https://github.com/wzp24k/acute-mi-dynamic-graph-disposition. The repository "
            "excludes credentialed patient-level records and prediction caches.",
        ),
        (
            "Ethics approval and consent to participate",
            "MIMIC-IV, MIMIC-III and eICU-CRD were collected under the ethics approvals "
            "of their originating institutions. They are distributed by PhysioNet under "
            "credentialed data-use agreements. The present secondary analysis used only "
            "de-identified records and involved no new intervention or patient contact. "
            "Under the PhysioNet data-use agreements, no additional ethics approval or "
            "informed consent was required.",
        ),
        (
            "Patient and public involvement",
            "Patients and members of the public were not involved in the design, conduct, "
            "interpretation or writing of this secondary database analysis.",
        ),
        (
            "Authors' contributions",
            "Z.W. and L.W. contributed equally and are co-first authors. Z.W. constructed "
            "the cohorts, implemented the models, performed the statistical analyses, "
            "supervised the study, obtained the data use agreements and revised the "
            "manuscript. L.W. conceived the study, designed the analysis and drafted the "
            "manuscript. C.W. revised the manuscript critically for clinical content. "
            "X.X. contributed to data curation and reviewed the manuscript. P.Z. contributed "
            "to the clinical interpretation of the endpoint and to the discussion. H.F. "
            "reviewed the cohort definitions and the measurement audits. Z.L. contributed "
            "to data curation and reviewed the manuscript. Q.H. prepared the figures and "
            "tables. M.G. verified the reproducibility of the analysis pipeline. All "
            "authors read and approved the final manuscript.",
        ),
        (
            "Funding",
            "This work was supported by the Higher Education Teaching Reform Research "
            "Project of Xuzhou Medical University, XYJG042. Support also came from the "
            "Special Project on Teaching Reform of Artificial Intelligence General "
            "Literacy Education in Undergraduate Universities in Jiangsu Province, "
            "2026TSY013.",
        ),
        (
            "Declaration of generative AI and AI-assisted technologies in the writing process",
            "During this work, the authors used OpenAI Codex for code development and "
            "analysis documentation. Codex also supported language editing and reference "
            "consistency checks. "
            "No patient-level records were provided to the AI service. The authors "
            "reviewed and edited all output and take full responsibility for the publication.",
        ),
        ("Consent for publication", "Not applicable."),
        ("Clinical trial number", "Not applicable."),
        ("Competing interests", "The authors declare no competing interests."),
    ]
    for heading, text in declarations:
        add_declaration(document, heading, text)

    add_main_heading(document, "7. References")
    for index, record in enumerate(references, start=1):
        paragraph = document.add_paragraph(style="RefEntry")
        run = paragraph.add_run(f"{index}. {record['citation']}")
        set_run_font(run)

    document.core_properties.title = TITLE
    document.core_properties.subject = (
        "Diagnosis-coded MACCE prediction with continuous-time organ graph learning "
        "and exploratory temporal ensembling"
    )
    document.core_properties.author = (
        "Zhipeng Wang; Luning Wang; Changsong Wang; Xiyu Xu; Pengli Zhai; "
        "Zejun Liu; Hui Feng; Qian Hou; Ming Guo"
    )
    document.core_properties.keywords = (
        "Coronary Atherosclerosis; MACCE; Continuous-Time Modeling; Organ Graph; "
        "External Validation"
    )
    document.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())

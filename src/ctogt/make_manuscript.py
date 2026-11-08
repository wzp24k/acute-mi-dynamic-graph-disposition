"""Assemble the main manuscript, supplement, and reviewer-facing output files."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from sklearn.metrics import roc_auc_score

from .common import (
    FIGURES,
    OUTPUTS,
    PREDICTIONS,
    PROCESSED,
    RESULTS,
    TABLES,
    ensure_dirs,
)
from .make_tables_figures import (
    load_development_metrics,
    load_external_metrics,
    main as render_tables_figures,
)


TITLE = (
    "Invariant Continuous-Time Organ Graph Learning with Complementary "
    "Temporal Ensembling for Diagnosis-Coded MACCE in Coronary "
    "Atherosclerosis: A Three-Database Transport Study"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_auc_ci(
    labels: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    replicates: int = 1000,
    seed: int = 7,
) -> tuple[float, float, float]:
    point = roc_auc_score(labels, probability_a) - roc_auc_score(
        labels, probability_b
    )
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        index = rng.integers(0, len(labels), len(labels))
        sampled = labels[index]
        if len(np.unique(sampled)) < 2:
            continue
        values.append(
            roc_auc_score(sampled, probability_a[index])
            - roc_auc_score(sampled, probability_b[index])
        )
    return (
        float(point),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def manuscript_metrics() -> dict:
    development, aggregate = load_development_metrics()
    development_records = development.set_index("model").to_dict(orient="index")
    external = load_external_metrics()
    external_records = {
        cohort: frame.set_index("model").to_dict(orient="index")
        for cohort, frame in external.items()
    }
    sensitivity = (
        load_json(RESULTS / "sensitivity_analysis.json")
        if (RESULTS / "sensitivity_analysis.json").is_file()
        else {"analyses": {}}
    )
    ensemble_validation = (
        load_json(RESULTS / "ensemble_validation.json")
        if (RESULTS / "ensemble_validation.json").is_file()
        else {"external": {}}
    )
    contrasts = {}
    predictions: dict[str, np.ndarray] = {}
    labels = None
    for model in development_records:
        path = PREDICTIONS / f"mimic4_cv_{model}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=True) as data:
            current_labels = data["labels"].astype(int)
            if labels is None:
                labels = current_labels
            if np.array_equal(labels, current_labels):
                predictions[model] = data["probability"].astype(float)
    if labels is not None:
        pairs = (
            ("xgb_static_ts", "xgb_static"),
            ("transformer", "xgb_static_ts"),
            ("temporal_ensemble", "xgb_static_ts"),
            ("temporal_ensemble", "transformer"),
            ("ctogt", "xgb_static_ts"),
            ("ctogt", "ct_graph"),
            ("ct_graph", "ct_fixed"),
            ("ct_reverse", "ctogt"),
            ("ct_permute", "ctogt"),
            ("ct_random", "ctogt"),
            ("ct_no_time", "ctogt"),
        )
        for left, right in pairs:
            if left in predictions and right in predictions:
                estimate, low, high = paired_auc_ci(
                    labels,
                    predictions[left],
                    predictions[right],
                    seed=7 + len(contrasts),
                )
                contrasts[f"{left}_minus_{right}"] = {
                    "estimate": estimate,
                    "ci": [low, high],
                }
    return {
        "development": development_records,
        "external": external_records,
        "sensitivity": sensitivity,
        "contrasts": contrasts,
        "aggregate": aggregate,
        "ensemble_validation": ensemble_validation,
    }


def value(metrics: dict, model: str, field: str, default=float("nan")) -> float:
    return float(metrics.get(model, {}).get(field, default))


def fmt(value_: float, digits: int = 3) -> str:
    if value_ is None or not np.isfinite(value_):
        return "NA"
    return f"{value_:.{digits}f}"


def build_references() -> list[dict]:
    references = load_json(
        Path(__file__).resolve().parents[2]
        / "data"
        / "references"
        / "references.json"
    )
    lookup = {record["key"]: record for record in references}
    ordered = [
        {
            "key": "johnson2023mimiciv",
            "citation": (
                "Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely "
                "accessible electronic health record dataset. Sci Data. "
                "2023;10:1. doi:10.1038/s41597-022-01899-x"
            ),
        },
        {
            "key": "johnson2016mimiciii",
            "citation": (
                "Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely "
                "accessible critical care database. Sci Data. 2016;3:160035. "
                "doi:10.1038/sdata.2016.35"
            ),
        },
        {
            "key": "pollard2018eicu",
            "citation": (
                "Pollard TJ, Johnson AEW, Raffa JD, Celi LA, Mark RG, Badawi O. "
                "The eICU Collaborative Research Database, a freely available "
                "multi-center database for critical care research. Sci Data. "
                "2018;5:180178. doi:10.1038/sdata.2018.178"
            ),
        },
        {
            "key": "collins2024tripodai",
            "citation": (
                "Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: "
                "updated guidance for reporting clinical prediction models that "
                "use regression or machine learning methods. BMJ. "
                "2024;385:e078378. doi:10.1136/bmj-2023-078378"
            ),
        },
        {
            "key": "vincent1996sofa",
            "citation": (
                "Vincent JL, Moreno R, Takala J, et al. The SOFA score to "
                "describe organ dysfunction/failure. Intensive Care Med. "
                "1996;22:707-710. doi:10.1007/BF01709751"
            ),
        },
        {
            "key": "shickel2018deepehr",
            "citation": (
                "Shickel B, Tighe PJ, Bihorac A, Rashidi P. Deep EHR: a survey "
                "of recent advances in deep learning techniques for electronic "
                "health record analysis. IEEE J Biomed Health Inform. "
                "2018;22:1589-1604. doi:10.1109/JBHI.2017.2767063"
            ),
        },
        {
            "key": "rajkomar2018scalable",
            "citation": (
                "Rajkomar A, Oren E, Chen K, et al. Scalable and accurate deep "
                "learning with electronic health records. npj Digit Med. "
                "2018;1:18. doi:10.1038/s41746-018-0029-1"
            ),
        },
        {
            "key": "che2018grud",
            "citation": (
                "Che Z, Purushotham S, Cho K, Sontag D, Liu Y. Recurrent neural "
                "networks for multivariate time series with missing values. Sci "
                "Rep. 2018;8:6085. doi:10.1038/s41598-018-24271-9"
            ),
        },
        {
            "key": "vaswani2017attention",
            "citation": (
                "Mohamed A, AlAleeli R, Shaalan K. Advancing Predictive "
                "Healthcare: A Systematic Review of Transformer Models in "
                "Electronic Health Records. Computers. 2025;14(4):148. "
                "doi:10.3390/computers14040148"
            ),
        },
        {
            "key": "lim2021tft",
            "citation": (
                "Lim B, Arik SO, Loeff N, Pfister T. Temporal Fusion "
                "Transformers for interpretable multi-horizon time series "
                "forecasting. Int J Forecast. 2021;37:1748-1764. "
                "doi:10.1016/j.ijforecast.2021.03.012"
            ),
        },
        {
            "key": "arjovsky2019irm",
            "citation": (
                "Jia C, Zhang Y. Meta-learning the invariant representation for "
                "domain generalization. Machine Learning. 2024;113(4):"
                "1661-1681. doi:10.1007/s10994-022-06256-y"
            ),
        },
        {
            "key": "sagawa2020groupdro",
            "citation": (
                "Duchi JC, Namkoong H. Learning models with uniform performance "
                "via distributionally robust optimization. The Annals of "
                "Statistics. 2021;49(3):1378-1406. doi:10.1214/20-aos2004"
            ),
        },
        {
            "key": "kidger2020ncde",
            "citation": (
                "Goyal P, Benner P. Neural ordinary differential equations with "
                "irregular and noisy data. Royal Society Open Science. "
                "2023;10(7):221475. doi:10.1098/rsos.221475"
            ),
        },
        {
            "key": "gu2023mamba",
            "citation": (
                "Patro BN, Agneeswaran VS. Mamba-360: Survey of state space "
                "models as transformer alternative for long sequence modelling: "
                "Methods, Applications, and Challenges. Engineering Applications "
                "of Artificial Intelligence. 2025;159:111279. "
                "doi:10.1016/j.engappai.2025.111279"
            ),
        },
        {
            "key": "vancalster2019calibration",
            "citation": (
                "Van Calster B, McLernon DJ, van Smeden M, Wynants L, "
                "Steyerberg EW. Calibration: the Achilles heel of predictive "
                "analytics. BMC Med. 2019;17:230. "
                "doi:10.1186/s12916-019-1466-7"
            ),
        },
        {
            "key": "vickers2006dca",
            "citation": (
                "Vickers AJ, Elkin EB. Decision curve analysis: a novel method "
                "for evaluating prediction models. Med Decis Making. "
                "2006;26:565-574. doi:10.1177/0272989X06295361"
            ),
        },
        {
            "key": "thygesen2019mi",
            "citation": (
                "Thygesen K, Alpert JS, Jaffe AS, et al. Fourth universal "
                "definition of myocardial infarction (2018). Eur Heart J. "
                "2019;40:237-269. doi:10.1093/eurheartj/ehy462"
            ),
        },
        {
            "key": "sacco2013stroke",
            "citation": (
                "Sacco RL, Kasner SE, Broderick JP, et al. An updated "
                "definition of stroke for the 21st century. Stroke. "
                "2013;44:2064-2089. doi:10.1161/STR.0b013e318296aeca"
            ),
        },
    ]
    for record in ordered:
        if record["key"] in lookup:
            record["doi_verified"] = bool(lookup[record["key"]]["verified"])
    return ordered


def reference_numbers(references: list[dict]) -> dict[str, int]:
    return {record["key"]: index + 1 for index, record in enumerate(references)}


def markdown_table(path: Path) -> str:
    if not path.is_file():
        return ""
    frame = pd.read_csv(path)
    return frame.to_markdown(index=False)


def paragraph_texts(metrics: dict, references: list[dict]) -> dict[str, list[str]]:
    ref = reference_numbers(references)
    dev = metrics["development"]
    ext = metrics["external"]
    sensitivity = metrics["sensitivity"].get("analyses", {})
    ensemble_validation = metrics.get("ensemble_validation", {}).get(
        "external", {}
    )
    interval_model = (
        load_json(RESULTS / "interval_model_analysis.json")
        if (RESULTS / "interval_model_analysis.json").is_file()
        else {"mimic4": {}, "external": {}}
    )
    interval_prediction = (
        load_json(RESULTS / "interval_endpoint_analysis.json")
        if (RESULTS / "interval_endpoint_analysis.json").is_file()
        else {"cohorts": {}}
    )
    temporal_holdout = (
        load_json(RESULTS / "temporal_holdout_validation.json")
        if (RESULTS / "temporal_holdout_validation.json").is_file()
        else {}
    )
    nested_selection = (
        load_json(RESULTS / "nested_selection_and_utility.json")
        if (RESULTS / "nested_selection_and_utility.json").is_file()
        else {}
    )
    manual_review = (
        load_json(RESULTS / "manual_endpoint_review_pilot.json")
        if (RESULTS / "manual_endpoint_review_pilot.json").is_file()
        else {}
    )
    manual_review_score = (
        load_json(RESULTS / "manual_endpoint_review_scored.json")
        if (RESULTS / "manual_endpoint_review_scored.json").is_file()
        else {}
    )
    strong_holdout = (
        load_json(RESULTS / "strong_temporal_holdout_fusion.json")
        if (RESULTS / "strong_temporal_holdout_fusion.json").is_file()
        else {}
    )
    strong_xgb_holdout = (
        load_json(RESULTS / "strong_temporal_holdout.json")
        if (RESULTS / "strong_temporal_holdout.json").is_file()
        else {}
    )
    strong_tabular = (
        load_json(RESULTS / "strong_tabular_candidates.json")
        if (RESULTS / "strong_tabular_candidates.json").is_file()
        else {}
    )
    strong_triple = (
        load_json(RESULTS / "strong_fusion_triple.json")
        if (RESULTS / "strong_fusion_triple.json").is_file()
        else {}
    )
    interval_sensitivity = (
        load_json(RESULTS / "interval_threshold_sensitivity.json")
        if (RESULTS / "interval_threshold_sensitivity.json").is_file()
        else {}
    )
    clinical_utility = (
        load_json(RESULTS / "clinical_utility_analysis.json")
        if (RESULTS / "clinical_utility_analysis.json").is_file()
        else {}
    )
    organ_dynamics = (
        load_json(RESULTS / "organ_dynamics_holdout.json")
        if (RESULTS / "organ_dynamics_holdout.json").is_file()
        else {}
    )
    eicu_site_holdout = (
        load_json(RESULTS / "eicu_site_holdout.json")
        if (RESULTS / "eicu_site_holdout.json").is_file()
        else {}
    )
    strong_external = (
        load_json(RESULTS / "strong_external_transport.json")
        if (RESULTS / "strong_external_transport.json").is_file()
        else {}
    )
    nested_transformer = (
        load_json(RESULTS / "nested_transformer_selection.json")
        if (RESULTS / "nested_transformer_selection.json").is_file()
        else {}
    )
    eicu_anycad_record = (
        load_json(RESULTS / "metrics_external_eicu_anycad_xgb_static_ts.json")
        if (RESULTS / "metrics_external_eicu_anycad_xgb_static_ts.json").is_file()
        else {}
    )
    eicu_anycad_n = int(eicu_anycad_record.get("n", 0))
    eicu_anycad_events = int(eicu_anycad_record.get("events", 0))
    contrasts = metrics["contrasts"]

    xgb_static = dev.get("xgb_static", {})
    xgb_ts = dev.get("xgb_static_ts", {})
    ctogt = dev.get("ctogt", {})
    ensemble = dev.get("temporal_ensemble", {})
    transformer = dev.get("transformer", {})
    fixed = dev.get("ct_fixed", {})
    graph = dev.get("ct_graph", {})
    organ_graph = dev.get("organ_graph", {})
    reverse = dev.get("ct_reverse", {})
    permute = dev.get("ct_permute", {})
    random_graph = dev.get("ct_random", {})
    no_time = dev.get("ct_no_time", {})
    best_dev_model = max(
        dev, key=lambda key: value(dev, key, "auroc"), default="xgb_static_ts"
    ) if dev else "xgb_static_ts"
    best_dev_label = dev.get(best_dev_model, {}).get("label", best_dev_model)
    full_completed = "ctogt" in dev
    organ_model = "ctogt" if full_completed else "ct_graph"
    organ_label = (
        "full CT-OGT"
        if full_completed
        else "continuous-time dynamic organ graph without the incomplete invariance arm"
    )
    contrast_ts = contrasts.get(f"{organ_model}_minus_xgb_static_ts", {})
    contrast_ensemble = contrasts.get(
        "temporal_ensemble_minus_xgb_static_ts", {}
    )
    contrast_h1 = contrasts.get("xgb_static_ts_minus_xgb_static", {})
    contrast_transformer = contrasts.get(
        "transformer_minus_xgb_static_ts", {}
    )
    h2_supported = (
        contrast_ensemble.get("estimate", -1) >= 0.01
        and contrast_ensemble.get("ci", [1, 1])[0] > 0
    )
    organ_supported = (
        contrast_ts.get("estimate", -1) >= 0.01
        and contrast_ts.get("ci", [1, 1])[0] > 0
    )
    def completed_metric(model: str, field: str = "auroc") -> str:
        return fmt(value(dev, model, field)) if model in dev else "not completed"

    ext_summary = {}
    for cohort_name, frame in ext.items():
        if not frame:
            continue
        best = max(frame, key=lambda key: frame[key].get("auroc_raw", -1))
        xgb_external = frame.get("xgb_static_ts", {})
        transformer_external = frame.get("transformer", {})
        ctogt_external = frame.get("ctogt", {})
        ensemble_external = frame.get("temporal_ensemble", {})
        ext_summary[cohort_name] = {
            "best": best,
            "best_auroc": frame[best].get("auroc_raw", float("nan")),
            "xgb_auroc": xgb_external.get("auroc_raw", float("nan")),
            "transformer_auroc": transformer_external.get(
                "auroc_raw", float("nan")
            ),
            "ctogt_auroc": ctogt_external.get("auroc_raw", float("nan")),
            "xgb_ece": xgb_external.get("ece_raw", float("nan")),
            "xgb_ece_recal": xgb_external.get("ece_recal", float("nan")),
            "ensemble_auroc": ensemble_external.get(
                "auroc_raw", float("nan")
            ),
            "ensemble_ece": ensemble_external.get(
                "ece_raw", float("nan")
            ),
            "ensemble_ece_recal": ensemble_external.get(
                "ece_recal", float("nan")
            ),
        }
    best_mimic3_external = str(
        ext_summary.get("mimic3", {}).get("best", "NA")
    )
    if best_mimic3_external == "gru":
        best_mimic3_external = "GRU"

    abstract = [
        (
            "Background: Patients with coronary atherosclerosis admitted to an "
            "intensive care unit (ICU) remain at risk of major adverse cardiac "
            "and cerebrovascular events (MACCE), yet static severity scores do "
            "not represent evolving multi-organ physiology. We tested whether a "
            "continuous-time organ graph or a complementary temporal ensemble "
            "improves diagnosis-coded MACCE prediction beyond a strong XGBoost "
            "model using 24-hour temporal summaries (XGB+TS)."
        ),
        (
            f"Methods: We performed a retrospective three-database study using "
            f"MIMIC-IV for development, MIMIC-III for temporal transport, and "
            f"eICU-CRD for multi-hospital transport, expanding eICU to coronary "
            f"atherosclerosis diagnoses recorded in any diagnosis position. "
            f"Adults alive at the 24-hour landmark with at least six observed "
            f"organ channels were included. The primary phenotype combined "
            f"diagnosis-coded acute myocardial infarction or stroke with death "
            f"and revascularization recorded in the 7-day window. CT-OGT encoded "
            f"elapsed observation time, constructed a prior-constrained dynamic "
            f"organ graph, and applied environment-balanced and graph-sparsity "
            f"penalties. After the organ graph failed to improve on XGB+TS, an "
            f"exploratory complementary ensemble was formed and then frozen. "
            f"Feature and weight selection used only 2017-2019; the locked model "
            f"was evaluated in 2020-2022. Models were compared by five-fold cross-validation, "
            f"nested selection, paired bootstrap, calibration, and decision-"
            f"curve analysis."
        ),
        (
            f"Results: The development cohort included 13,131 patients and "
            f"4,475 MACCE events. MIMIC-III included 3,470 patients and 409 "
            f"events; the expanded eICU cohort included {eicu_anycad_n:,} "
            f"patients and {eicu_anycad_events:,} events across 166 sites. "
            f"XGB+TS achieved "
            f"a development AUROC of {fmt(value(dev, 'xgb_static_ts', 'auroc'))}, "
            f"compared with {fmt(value(dev, 'xgb_static', 'auroc'))} for static "
            f"XGBoost. The original temporal ensemble reached "
            f"{fmt(value(dev, 'temporal_ensemble', 'auroc'))}, a paired gain of "
            f"{fmt(contrast_ensemble.get('estimate'), 4)} "
            f"(95% CI {fmt(contrast_ensemble.get('ci', [float('nan')])[0])} to "
            f"{fmt(contrast_ensemble.get('ci', [float('nan'), float('nan')])[1])}) "
            f"over XGB+TS. It retained a gain of "
            f"{fmt(temporal_holdout.get('ensemble_minus_xgb_static_ts', {}).get('estimate'), 4)} "
            f"in the locked temporal holdout and "
            f"{fmt(nested_selection.get('contrasts', {}).get('nested_ensemble_minus_standard_xgb', {}).get('estimate'), 4)} "
            f"after nested XGBoost tuning; the latter narrowly missed the 0.01 "
            f"material threshold. The seven-organ token graph reached "
            f"{fmt(value(dev, 'organ_graph', 'auroc'))} and did not improve on "
            f"XGB+TS. "
            f"An enhanced organ-dynamics XGBoost plus Transformer model was "
            f"selected on 2017-2019 and evaluated in 2020-2022 at AUROC "
            f"{fmt(strong_holdout.get('final_holdout', {}).get('locked_fusion', {}).get('auroc'))}, "
            f"compared with "
            f"{fmt(strong_holdout.get('final_holdout', {}).get('baseline_xgb', {}).get('auroc'))}, "
            f"a difference of "
            f"{fmt(strong_holdout.get('final_holdout', {}).get('locked_fusion_minus_baseline_xgb', {}).get('difference'), 4)} "
            f"(95% CI "
            f"{fmt(strong_holdout.get('final_holdout', {}).get('locked_fusion_minus_baseline_xgb', {}).get('ci', [float('nan')])[0], 4)} "
            f"to "
            f"{fmt(strong_holdout.get('final_holdout', {}).get('locked_fusion_minus_baseline_xgb', {}).get('ci', [float('nan'), float('nan')])[1], 4)}). "
            f"The highest development AUROC was observed for {best_dev_label} "
            f"({fmt(value(dev, best_dev_model, 'auroc'))}). Direct transport "
            f"AUROCs for the temporal ensemble were "
            f"{fmt(ext_summary.get('mimic3', {}).get('ensemble_auroc'))} in "
            f"MIMIC-III and "
            f"{fmt(ext_summary.get('eicu_anycad', {}).get('ensemble_auroc'))} "
            f"in expanded eICU, "
            f"compared with "
            f"{fmt(ext_summary.get('mimic3', {}).get('xgb_auroc'))} and "
            f"{fmt(ext_summary.get('eicu_anycad', {}).get('xgb_auroc'))} for XGB+TS "
            f"alone. The enhanced locked 75/25 fusion reached AUROC "
            f"{fmt(strong_external.get('models', {}).get('mimic3', {}).get('locked_75_25_fusion', {}).get('auroc'))} "
            f"in MIMIC-III and "
            f"{fmt(strong_external.get('models', {}).get('eicu_anycad', {}).get('locked_75_25_fusion', {}).get('auroc'))} "
            f"in expanded eICU. An interval-censored sensitivity endpoint "
            f"identified 45 strongly confirmed and 1,265 anchor-overlap MIMIC-IV events; "
            f"the confirmed stratum was too small for reliable discrimination, "
            f"whereas retrained XGB+TS achieved AUROC "
            f"{fmt(interval_model.get('mimic4', {}).get('macce_interval_overlap', {}).get('auroc'))} "
            f"for the overlap endpoint. "
            f"Transport discrimination was preserved, but calibration was poor "
            f"before local recalibration."
        ),
        (
            "Conclusions: Twenty-four-hour temporal information provided a "
            "consistent gain over static severity. The complementary temporal "
            f"ensemble improved discrimination in development and in a locked "
            f"late-era evaluation. The original pair was marginal after strict "
            f"nested XGBoost tuning, whereas the second-stage era-locked model "
            f"improved AUROC by "
            f"{fmt(strong_holdout.get('final_holdout', {}).get('locked_fusion_minus_baseline_xgb', {}).get('difference'), 4)} "
            f"in 2020-2022. This model was exploratory rather than pre-specified, "
            f"but its feature set and weights were frozen before the final era "
            f"test. "
            "Neither the original CT-OGT, seven-organ token graph nor organ-rate "
            "graph improved materially over XGB+TS."
            + " Transport discrimination was useful, but probability estimates "
            "required local recalibration before clinical interpretation."
        ),
    ]

    introduction = [
        (
            "Coronary atherosclerosis is frequently present in ICU populations, "
            "but the early post-admission risk of recurrent ischemia, stroke, "
            "urgent revascularization, and death reflects more than the extent "
            "of coronary disease. Circulation, gas exchange, renal clearance, "
            "coagulation, inflammation, metabolism, and neurologic status "
            "evolve together in the first ICU day. A risk model anchored at 24 "
            "hours should therefore integrate both current severity and the "
            "direction of multi-organ change."
        ),
        (
            "The sequential organ failure assessment (SOFA) and related severity "
            "scores summarize cross-sectional organ dysfunction and have shaped "
            "critical-care risk stratification [5]. Electronic health record "
            "studies have since shown that recurrent, transformer-based, and "
            "irregularly sampled sequence models can capture temporal risk "
            "signals that are absent from static scores [6-10]. Most such models "
            "still compress organs into a flat feature vector. They may learn "
            "that variables co-vary, but they do not explicitly represent which "
            "organ systems are coupled, whether that coupling changes over "
            "time, or whether the learned relationships transport across "
            "hospitals."
        ),
        (
            "We addressed these gaps with four linked questions. First, "
            "does a 24-hour temporal representation improve discrimination over "
            "static risk? Second, does a complementary tabular-sequence "
            "ensemble improve over XGBoost using the same temporal summaries? "
            "Third, does an organ graph add value beyond XGB+TS? "
            "Fourth, do the findings persist under an interval-censored event-"
            "time sensitivity endpoint? Each mechanism "
            "was paired with structural ablations and negative controls so that "
            "a negative finding would remain interpretable. The analysis "
            "followed TRIPOD+AI reporting principles [4]."
        ),
    ]

    methods = [
        (
            "Study design and data sources. This retrospective prediction study "
            "used MIMIC-IV v3.1 for development [1], MIMIC-III v1.4 for "
            "same-institution temporal transport [2], and eICU-CRD v2.0 for "
            "multi-hospital external transport [3]. The primary eICU transport "
            "cohort allowed a coronary atherosclerosis diagnosis in any "
            "diagnosis position; a principal-diagnosis eICU cohort was retained "
            "as a sensitivity analysis. All databases contain "
            "de-identified routine-care data available through credentialed "
            "PhysioNet access. The study used the first eligible ICU stay per "
            "patient. MIMIC-IV is a contemporary tertiary academic center; "
            "MIMIC-III spans an earlier era at the same institution; eICU "
            "contains ICUs from multiple hospitals."
        ),
        (
            "Population and landmark. Eligible patients were at least 18 years "
            "old and had a diagnosis code for coronary atherosclerosis "
            "(ICD-10 I25.x or ICD-9 414.x). Development and MIMIC-III required "
            "a principal diagnosis position; expanded eICU allowed any "
            "diagnosis position. The first eligible "
            "ICU stay was retained. Patients had to survive to ICU hour 24, "
            "remain under observation at that landmark, and have at least six "
            "observed organ channels in the first 24 hours. Only measurements "
            "at or before hour 24 were used for prediction."
        ),
        (
            "Endpoint. The composite included all-cause death, acute myocardial "
            "infarction, ischemic or hemorrhagic stroke, and urgent or "
            "unplanned revascularization. The intended target window extended "
            "from ICU hour 24 through day 7. Acute myocardial infarction and stroke were "
            "defined using established clinical concepts [17,18]. In MIMIC, "
            "death time was exact, but diagnosis tables did not provide event "
            "onset times and MIMIC-III procedure records did not provide "
            "procedure dates. The main endpoint is therefore explicitly a "
            "diagnosis-coded 7-day MACCE phenotype, not a fully timestamped "
            "adjudicated endpoint. A timestamped sensitivity endpoint included "
            "death and revascularization in MIMIC and diagnosis-offset events in "
            "eICU. This distinction is reported wherever performance is "
            "interpreted. We additionally constructed an interval-censored "
            "endpoint. Death used an exact interval, MIMIC-IV revascularization "
            "used the procedure day, radiology or timed notes used a 24-hour "
            "pre-documentation interval, and troponin elevation used a six-hour "
            "pre-measurement interval. eICU diagnosis offsets used a six-hour "
            "documentation-lag interval. The primary interval analysis used "
            "events with a specific clinical or procedure time anchor whose "
            "interval overlapped ICU hours 24 through 168. A stricter "
            "full-interval-inside-window definition was prespecified as an "
            "upper-confidence stratum but was not used for model inference when "
            "it contained fewer than 150 events. Coded events without a "
            "specific anchor were not counted as anchor-overlap events."
        ),
        (
            "Organ representation. Seven organ systems were represented by up to "
            "20 channels: neurologic (Glasgow Coma Scale, pupil reactivity), "
            "circulatory (heart rate, blood pressures, lactate), respiratory "
            "(respiratory rate, oxygen saturation, inspired oxygen), renal "
            "(creatinine, urine output), coagulation (platelets, INR, aPTT), "
            "inflammation (white-cell count, temperature), and metabolic "
            "(glucose, pH, bicarbonate). Urine output was unavailable in the "
            "eICU extract and was treated as missing rather than imputed as "
            "normal. Values were aggregated within clock hour, filtered with "
            "prespecified physiologic ranges, standardized using development-"
            "fold statistics, and accompanied by explicit observation masks."
        ),
        (
            "Continuous-time encoding and dynamic graph. For every organ channel "
            "and hour, the model represented the standardized value, the "
            "observation indicator, the elapsed time since the last observation, "
            "absolute time, and their interaction. Time2Vec-style temporal "
            "features were added to the sequence representation. Organ nodes "
            "were connected by a sparse prior based on clinical adjacency: "
            "channels from the same organ were linked densely, and selected "
            "cross-organ links encoded circulation-neurology, circulation-renal, "
            "respiratory-neurology, renal-metabolic, coagulation-inflammation, "
            "and related pathways. Dynamic edges were generated from observed "
            "node states at each hour, masked when a channel was unobserved, "
            "and regularized toward sparsity."
        ),
        (
            "CT-OGT architecture. The model contained an observation-aware organ "
            "graph block, masked organ attention, a Transformer temporal encoder, "
            "a static-context encoder, and a risk trajectory head [9]. The final "
            "risk combined the temporally attended representation with static "
            "demographic and comorbidity information. The training objective "
            "combined binary cross-entropy, temporal smoothness, drift, graph "
            "sparsity, an environment-balanced risk penalty, and a gradient-"
            "reversed environment classification loss inspired by invariant risk "
            "minimization and group distributionally robust optimization [11,12]. "
            "ICU care unit crossed with admission era defined development "
            "environments; hospital identifiers defined external environments "
            "but were never used as direct predictors."
        ),
        (
            "Organ-token redesign. Because channel-level graph learning may "
            "dilute organ-level coupling, an exploratory seven-node model "
            "aggregated the 20 channels into neurologic, circulatory, "
            "respiratory, renal, coagulation, inflammation, and metabolic "
            "tokens. Each token carried its observed-channel mask, elapsed-time "
            "context, static context, Time2Vec and Transformer time encodings, "
            "and a dynamic organ adjacency. This redesign was compared with "
            "XGB+TS and the original dynamic graph. It was not substituted for "
            "XGB+TS after seeing its development performance."
        ),
        (
            "Exploratory complementary temporal ensemble. The initial mechanistic hypothesis "
            "concerned the organ graph. After the graph failed to improve on "
            "XGB+TS, complementary tabular and temporal models were treated as "
            "exploratory. The original equal-weight ensemble was frozen before "
            "the first 2020-2022 temporal evaluation. A second-stage model added "
            "organ-level adverse slopes, recovery directions, ranges, clinical "
            "ratios, and observed-organ fractions. Its feature set, XGBoost "
            "hyperparameters, and fusion weight were selected only on the "
            "2017-2019 inner temporal validation era, then locked before the "
            "2020-2022 test. Components were "
            "trained within the same development folds and seed protocol. No "
            "external labels were used to choose weights. The ensemble was "
            "compared with XGB+TS by patient-level paired bootstrap and was then "
            "transported unchanged to MIMIC-III and eICU before local "
            "recalibration."
        ),
        (
            "Comparators and ablations. Comparators included static logistic "
            "regression, static XGBoost, XGBoost with 24-hour temporal summaries, "
            "GRU, Transformer, and a TFT-lite model [8-10]. The primary strong "
            "benchmark was XGB+TS because it uses the same temporal information "
            "as the deep models while remaining tabular and clinically "
            "auditable. Ablations included a fixed prior graph, a dynamic graph "
            "without environment regularization, CT-OGT without continuous-time "
            "encoding, a random graph, reversed time, and permuted hour order. "
            "Reversed and permuted sequences were used as negative controls; "
            "performance inexplicably preserved under these controls would "
            "indicate reliance on non-temporal shortcuts."
        ),
        (
            "Cross-validation, nested selection, and transport. Development used "
            "stratified five-fold cross-validation. For nested selection, XGBoost "
            "hyperparameters were chosen by three-fold inner cross-validation "
            "within each outer training fold; the Transformer component was "
            "unchanged. Environment groups defined by the MIMIC-IV anchor-year "
            "period were used to construct a locked 2020-2022 temporal holdout "
            "(1,072 events) after all 2008-2019 development-fold predictions were "
            "fixed. Tabular baselines were repeated across "
            "three seeds. The Transformer, dynamic graph, and full CT-OGT were "
            "repeated across three seeds in five folds. One fixed seed was used "
            "for the remaining ablations and negative controls. Hyperparameters were prespecified and shared "
            "across comparators. External models were fit in MIMIC-IV without "
            "external labels and directly applied to MIMIC-III and eICU. "
            "Calibration was then estimated by cross-fitted Platt scaling within "
            "each external database, with hospital-stratified folds where "
            "multiple sites were available. Discrimination is reported before "
            "and after recalibration; calibration intercept, slope, Brier score, "
            "and expected calibration error are reported separately [15]."
        ),
        (
            "Metrics and statistical analysis. The primary metric was AUROC. "
            "Secondary metrics included AUPRC, Brier score, log loss, calibration "
            "slope and intercept, expected calibration error, top-decile "
            "sensitivity and specificity, and decision-curve net benefit [16]. "
            "Differences between paired models used patient-level bootstrap "
            "intervals. Net benefit was evaluated from threshold probabilities "
            "0.05 through 0.30 with bootstrap confidence intervals. Brier score "
            "and sensitivity, specificity, PPV, and NPV at 0.10 and 0.15 were "
            "also bootstrapped before and after local recalibration. The material "
            "AUROC threshold was 0.01. A "
            "model was considered to have incremental value only if the point "
            "difference exceeded this threshold and the paired confidence "
            "interval excluded zero."
        ),
        (
            "Sensitivity analyses. The diagnosis-coded primary phenotype was "
            "compared with a timestamped endpoint in MIMIC-IV. Prespecified "
            "population sensitivities included patients without prior myocardial "
            "infarction or stroke, prior PCI, prior CABG, and age at least 75 "
            "years. An ACS-based subgroup was excluded from inferential "
            "interpretation because acute MI codes overlapped the outcome "
            "definition. The original principal-diagnosis eICU cohort and the "
            "expanded any-position eICU cohort were analyzed separately. A "
            "structured chart-review pilot sampled diagnosis-coded events across "
            "the target window, pre-landmark period, post-horizon period, and "
            "historical or non-event documentation. The available eICU extract "
            "contained no clinical notes, so eICU records were flagged as "
            "unreviewable rather than adjudicated."
        ),
        (
            "Interval-rule sensitivity. The main interval endpoint remained a "
            "sensitivity analysis. Troponin positivity was recalculated at "
            "0.5, 1.0, and 2.0 times the assay cut-point, and eICU diagnosis "
            "documentation lag was evaluated at 0, 3, 6, and 12 hours. Frozen "
            "model predictions were applied to each endpoint without model "
            "retuning, so changes reflect endpoint rules rather than new labels "
            "used to fit the model."
        ),
        (
            "Structured endpoint adjudication. The 120-record review pack was "
            "expanded into a two-reader schema with separate fields for serial "
            "troponin change, ECG evidence, angiography or PCI, acute stroke "
            "imaging, focal neurologic deficit, clinical-record support, and "
            "event window. One hundred MIMIC records are reviewable; 20 eICU "
            "records lack source text. PPV, timing misclassification, and Cohen "
            "kappa are intentionally not estimated until both independent "
            "clinical readers complete the form."
        ),
        (
            "Transparency and reproducibility. The analysis code stores cohort "
            "flow, endpoints, sequence tensors, fold predictions, attention "
            "weights, and environment versions. The study is observational and "
            "does not estimate causal treatment effects. Public-data analyses "
            "were conducted under the databases' credentialed data-use "
            "agreements. Large language-model tools assisted with code "
            "implementation and language editing; statistical outputs were "
            "generated from the local data pipeline and require author review "
            "before submission."
        ),
    ]

    results = [
        (
            f"Cohort. After applying the 24-hour landmark and channel threshold, "
            f"the development cohort contained 13,131 patients and 4,475 "
            f"diagnosis-coded MACCE events (34.1%). MIMIC-III contained 3,470 "
            f"patients and 409 events (11.8%). The expanded eICU cohort "
            f"contained {eicu_anycad_n:,} patients and {eicu_anycad_events:,} "
            f"events (10.9%) across 166 sites, including 43 analytic sites with at least "
            f"30 patients. The original principal-diagnosis eICU cohort "
            f"(n=903; 57 events) was retained for sensitivity analysis. Median "
            f"observed organ-channel counts were 19 in MIMIC-IV, 17 in "
            f"MIMIC-III, 13 in principal-diagnosis eICU, and 11 in expanded "
            f"eICU. Baseline characteristics and component event counts are "
            f"shown in Table 1."
        ),
        (
            f"Temporal information and benchmark performance. Static logistic "
            f"regression achieved an AUROC of {fmt(value(dev, 'logit_static', 'auroc'))}, "
            f"and static XGBoost achieved {fmt(value(dev, 'xgb_static', 'auroc'))}. "
            f"Adding temporal summaries increased XGBoost to "
            f"{fmt(value(dev, 'xgb_static_ts', 'auroc'))} (AUPRC "
            f"{fmt(value(dev, 'xgb_static_ts', 'auprc'))}; paired AUROC gain "
            f"{fmt(contrast_h1.get('estimate'))}, 95% CI "
            f"{fmt(contrast_h1.get('ci', [float('nan')])[0])} to "
            f"{fmt(contrast_h1.get('ci', [float('nan'), float('nan')])[1])}). "
            f"GRU, Transformer, and "
            f"TFT-lite achieved AUROCs of {fmt(value(dev, 'gru', 'auroc'))}, "
            f"{fmt(value(dev, 'transformer', 'auroc'))}, and "
            f"{fmt(value(dev, 'tft', 'auroc'))}, respectively. The three-seed "
            f"Transformer slightly exceeded XGB+TS (difference "
            f"{fmt(contrast_transformer.get('estimate'))}, 95% CI "
            f"{fmt(contrast_transformer.get('ci', [float('nan')])[0])} to "
            f"{fmt(contrast_transformer.get('ci', [float('nan'), float('nan')])[1])}), "
            f"but did not reach the material threshold. The locked equal-weight "
            f"ensemble of XGB+TS and Transformer achieved "
            f"{fmt(value(dev, 'temporal_ensemble', 'auroc'))}, a paired gain of "
            f"{fmt(contrast_ensemble.get('estimate'))} "
            f"(95% CI {fmt(contrast_ensemble.get('ci', [float('nan')])[0])} to "
            f"{fmt(contrast_ensemble.get('ci', [float('nan'), float('nan')])[1])}), "
            f"which met the 0.01 material criterion in the original five-fold "
            f"analysis."
        ),
        (
            f"Locked temporal holdout. MIMIC-IV anchor-year periods were "
            f"used only for evaluation after the 2008-2019 development "
            f"predictions were fixed. The 2020-2022 holdout contained "
            f"{int(temporal_holdout.get('split', {}).get('test_n', 0)):,} "
            f"patients and "
            f"{int(temporal_holdout.get('split', {}).get('test_events', 0)):,} "
            f"events. XGB+TS achieved AUROC "
            f"{fmt(temporal_holdout.get('models', {}).get('xgb_static_ts', {}).get('metrics', {}).get('auroc'))}, "
            f"the Transformer "
            f"{fmt(temporal_holdout.get('models', {}).get('transformer', {}).get('metrics', {}).get('auroc'))}, "
            f"and the temporal ensemble "
            f"{fmt(temporal_holdout.get('models', {}).get('temporal_ensemble', {}).get('metrics', {}).get('auroc'))}. "
            f"The ensemble-minus-XGB+TS difference was "
            f"{fmt(temporal_holdout.get('ensemble_minus_xgb_static_ts', {}).get('estimate'))} "
            f"(95% CI "
            f"{fmt(temporal_holdout.get('ensemble_minus_xgb_static_ts', {}).get('ci', [float('nan')])[0])} "
            f"to "
            f"{fmt(temporal_holdout.get('ensemble_minus_xgb_static_ts', {}).get('ci', [float('nan'), float('nan')])[1])}), "
            f"supporting temporal complementarity in a held-out era."
        ),
        (
            f"Second-stage era-locked strengthening. Within the pre-2020 data, "
            f"2008-2016 was used for fitting and 2017-2019 for model selection. "
            f"Adding organ adverse-slope, recovery-direction, range, ratio, and "
            f"observation-fraction features improved internal validation AUROC "
            f"from "
            f"{fmt(strong_xgb_holdout.get('candidates', [{}])[0].get('auroc'))} "
            f"for the base candidate to "
            f"{fmt(strong_xgb_holdout.get('candidates', [{}])[-1].get('auroc'))}; "
            f"the enhanced XGBoost with deeper trees was selected. Fusion-weight "
            f"selection on 2017-2019 chose "
            f"{100 * float(strong_holdout.get('selection', {}).get('locked_xgb_weight', 0)):.0f}% "
            f"enhanced XGBoost and "
            f"{100 * (1 - float(strong_holdout.get('selection', {}).get('locked_xgb_weight', 0))):.0f}% "
            f"Transformer. After refitting on all 2008-2019 data, the locked "
            f"model achieved 2020-2022 AUROC "
            f"{fmt(strong_holdout.get('final_holdout', {}).get('locked_fusion', {}).get('auroc'))}, "
            f"versus "
            f"{fmt(strong_holdout.get('final_holdout', {}).get('baseline_xgb', {}).get('auroc'))} "
            f"for the original XGB+TS. The paired difference was "
            f"{fmt(strong_holdout.get('final_holdout', {}).get('locked_fusion_minus_baseline_xgb', {}).get('difference'), 4)} "
            f"(95% CI "
            f"{fmt(strong_holdout.get('final_holdout', {}).get('locked_fusion_minus_baseline_xgb', {}).get('ci', [float('nan')])[0], 4)} "
            f"to "
            f"{fmt(strong_holdout.get('final_holdout', {}).get('locked_fusion_minus_baseline_xgb', {}).get('ci', [float('nan'), float('nan')])[1], 4)}). "
            f"A validation-selected three-model blend that added ExtraTrees "
            f"reached "
            f"{fmt(strong_triple.get('final_holdout', {}).get('locked_triple_fusion', {}).get('auroc'))} "
            f"and did not improve the locked pair "
            f"({fmt(strong_triple.get('final_holdout', {}).get('locked_triple_minus_previous_pair', {}).get('difference'), 4)}; "
            f"95% CI "
            f"{fmt(strong_triple.get('final_holdout', {}).get('locked_triple_minus_previous_pair', {}).get('ci', [float('nan')])[0], 4)} "
            f"to "
            f"{fmt(strong_triple.get('final_holdout', {}).get('locked_triple_minus_previous_pair', {}).get('ci', [float('nan'), float('nan')])[1], 4)})."
        ),
        (
            f"Nested selection and clinical utility. Inner three-fold XGBoost "
            f"tuning produced an outer-fold AUROC of "
            f"{fmt(nested_selection.get('development', {}).get('nested_xgb_ts', {}).get('metrics', {}).get('auroc'))}, "
            f"compared with "
            f"{fmt(nested_selection.get('development', {}).get('standard_xgb_ts', {}).get('metrics', {}).get('auroc'))} "
            f"for the original XGB+TS. The nested temporal ensemble achieved "
            f"{fmt(nested_selection.get('development', {}).get('nested_temporal_ensemble', {}).get('metrics', {}).get('auroc'))}; "
            f"its gain over original XGB+TS was "
            f"{fmt(nested_selection.get('contrasts', {}).get('nested_ensemble_minus_standard_xgb', {}).get('estimate'), 4)} "
            f"(95% CI "
            f"{fmt(nested_selection.get('contrasts', {}).get('nested_ensemble_minus_standard_xgb', {}).get('ci', [float('nan')])[0], 4)} "
            f"to "
            f"{fmt(nested_selection.get('contrasts', {}).get('nested_ensemble_minus_standard_xgb', {}).get('ci', [float('nan'), float('nan')])[1], 4)}). "
            f"This interval excluded zero but the point estimate was 0.000208 "
            f"below the 0.01 material threshold. Bootstrap net-benefit "
            f"intervals were positive at lower thresholds but crossed zero or "
            f"became negative at higher thresholds, so clinical utility is "
            f"decision-threshold dependent. A stricter joint nesting procedure "
            f"also selected Transformer hyperparameters inside each outer fold. "
            f"The joint nested ensemble achieved AUROC "
            f"{fmt(nested_transformer.get('development', {}).get('joint_nested_temporal_ensemble', {}).get('metrics', {}).get('auroc'))}, "
            f"a gain of "
            f"{fmt(nested_transformer.get('contrasts', {}).get('joint_nested_ensemble_minus_standard_xgb', {}).get('estimate'), 4)} "
            f"(95% CI "
            f"{fmt(nested_transformer.get('contrasts', {}).get('joint_nested_ensemble_minus_standard_xgb', {}).get('ci', [float('nan')])[0], 4)} "
            f"to "
            f"{fmt(nested_transformer.get('contrasts', {}).get('joint_nested_ensemble_minus_standard_xgb', {}).get('ci', [float('nan'), float('nan')])[1], 4)}), "
            f"which was positive but below the material threshold."
        ),
        (
            f"Organ graph and invariance. The fixed graph achieved an AUROC of "
            f"{fmt(value(dev, 'ct_fixed', 'auroc'))}; the dynamic graph achieved "
            f"{fmt(value(dev, 'ct_graph', 'auroc'))}. "
            + (
                f"Full CT-OGT achieved {fmt(value(dev, 'ctogt', 'auroc'))}. "
                if full_completed
                else "The environment-invariant CT-OGT development run was "
                "stopped before all folds completed, so its out-of-fold "
                "predictions are not treated as a study result. "
            )
            + f"The paired difference between {organ_label} and XGB+TS was "
            f"{fmt(contrast_ts.get('estimate'))} "
            f"(95% CI "
            f"{fmt(contrast_ts.get('ci', [float('nan')])[0])} "
            f"to "
            f"{fmt(contrast_ts.get('ci', [float('nan'), float('nan')])[1])}). "
            f"The comparison did "
            + ("meet " if organ_supported else "not meet ")
            + "the 0.01 material threshold. The redesigned seven-organ token "
            f"graph achieved {fmt(value(dev, 'organ_graph', 'auroc'))}, below "
            f"XGB+TS ({fmt(value(dev, 'xgb_static_ts', 'auroc'))}) and the "
            f"original dynamic graph ({fmt(value(dev, 'ct_graph', 'auroc'))}); "
            f"the redesign therefore did not rescue the organ-graph hypothesis. "
            f"A further seven-organ model with explicit organ change rates and "
            f"elapsed-time tokens reached AUROC "
            f"{fmt(organ_dynamics.get('validation', {}).get('auroc'))} in the "
            f"2017-2019 internal era, below the pre-set "
            f"{fmt(organ_dynamics.get('gate_auroc'))} gate, and was therefore "
            f"not evaluated in 2020-2022. "
            f"The highest "
            f"development AUROC was observed for {best_dev_label} "
            f"({fmt(value(dev, best_dev_model, 'auroc'))})."
        ),
        (
            f"Negative controls. Reversed time achieved an AUROC of "
            f"{completed_metric('ct_reverse')}, permuted hour order "
            f"{completed_metric('ct_permute')}, and the random graph "
            f"{completed_metric('ct_random')}. Removing continuous-time "
            f"encoding yielded {completed_metric('ct_no_time')}. These "
            f"controls are reported in Figure 3. A control that approaches the "
            f"full model indicates that the corresponding mechanism is not "
            f"uniquely responsible for prediction and should not be presented "
            f"as a causal pathway."
        ),
        (
            f"Transport. In MIMIC-III, the strongest of the original "
            f"transported models was "
            f"{best_mimic3_external} with an AUROC of "
            f"{fmt(ext_summary.get('mimic3', {}).get('best_auroc'))}. XGB+TS "
            f"achieved {fmt(ext_summary.get('mimic3', {}).get('xgb_auroc'))}, "
            f"the temporal ensemble achieved "
            f"{fmt(ext_summary.get('mimic3', {}).get('ensemble_auroc'))}, and "
            f"CT-OGT achieved "
            f"{fmt(ext_summary.get('mimic3', {}).get('ctogt_auroc'))}. In the "
            f"expanded eICU cohort, XGB+TS achieved "
            f"{fmt(ext_summary.get('eicu_anycad', {}).get('xgb_auroc'))}, the "
            f"Transformer achieved "
            f"{fmt(ext_summary.get('eicu_anycad', {}).get('transformer_auroc'))}, and "
            f"the temporal ensemble achieved "
            f"{fmt(ext_summary.get('eicu_anycad', {}).get('ensemble_auroc'))}. "
            f"The original principal-diagnosis eICU cohort yielded temporal-"
            f"ensemble AUROC "
            f"{fmt(ext_summary.get('eicu', {}).get('ensemble_auroc'))}. "
            f"The paired ensemble-minus-XGB+TS differences were "
            f"{fmt(ensemble_validation.get('mimic3', {}).get('mean_xgb_transformer_vs_xgb', {}).get('difference'))} "
            f"(95% CI "
            f"{fmt(ensemble_validation.get('mimic3', {}).get('mean_xgb_transformer_vs_xgb', {}).get('ci', [float('nan')])[0])} to "
            f"{fmt(ensemble_validation.get('mimic3', {}).get('mean_xgb_transformer_vs_xgb', {}).get('ci', [float('nan'), float('nan')])[1])}) "
            f"in MIMIC-III and "
            f"{fmt(ensemble_validation.get('eicu_anycad', {}).get('mean_xgb_transformer_vs_xgb', {}).get('difference'))} "
            f"(95% CI "
            f"{fmt(ensemble_validation.get('eicu_anycad', {}).get('mean_xgb_transformer_vs_xgb', {}).get('ci', [float('nan')])[0])} to "
            f"{fmt(ensemble_validation.get('eicu_anycad', {}).get('mean_xgb_transformer_vs_xgb', {}).get('ci', [float('nan'), float('nan')])[1])}) "
            f"in expanded eICU. Discrimination therefore transported better "
            f"than calibration. Replacing the original XGBoost component with "
            f"the organ-dynamics XGBoost increased MIMIC-III AUROC to "
            f"{fmt(strong_external.get('models', {}).get('mimic3', {}).get('strong_xgb', {}).get('unrecalibrated', {}).get('auroc'))} "
            f"for XGB alone and "
            f"{fmt(strong_external.get('models', {}).get('mimic3', {}).get('locked_75_25_fusion', {}).get('auroc'))} "
            f"for the fixed 75/25 fusion; the paired gain over the original "
            f"MIMIC-III fusion was "
            f"{fmt(strong_external.get('models', {}).get('mimic3', {}).get('contrasts', {}).get('locked_fusion_minus_original_fusion', {}).get('difference'), 4)} "
            f"(95% CI "
            f"{fmt(strong_external.get('models', {}).get('mimic3', {}).get('contrasts', {}).get('locked_fusion_minus_original_fusion', {}).get('ci', [float('nan')])[0], 4)} "
            f"to "
            f"{fmt(strong_external.get('models', {}).get('mimic3', {}).get('contrasts', {}).get('locked_fusion_minus_original_fusion', {}).get('ci', [float('nan'), float('nan')])[1], 4)}). "
            f"Expanded eICU AUROC was "
            f"{fmt(strong_external.get('models', {}).get('eicu_anycad', {}).get('locked_75_25_fusion', {}).get('auroc'))}, "
            f"similar to the original temporal ensemble "
            f"(difference "
            f"{fmt(strong_external.get('models', {}).get('eicu_anycad', {}).get('contrasts', {}).get('locked_fusion_minus_original_fusion', {}).get('difference'), 4)}; "
            f"95% CI "
            f"{fmt(strong_external.get('models', {}).get('eicu_anycad', {}).get('contrasts', {}).get('locked_fusion_minus_original_fusion', {}).get('ci', [float('nan')])[0], 4)} "
            f"to "
            f"{fmt(strong_external.get('models', {}).get('eicu_anycad', {}).get('contrasts', {}).get('locked_fusion_minus_original_fusion', {}).get('ci', [float('nan'), float('nan')])[1], 4)})."
        ),
        (
            f"Hospital-held-out eICU analysis. To separate database transport "
            f"from local cross-hospital generalization, the enhanced XGBoost "
            f"was trained and tested with complete hospital separation in "
            f"expanded eICU. Across five folds covering 160 sites, AUROC was "
            f"{fmt(eicu_site_holdout.get('metrics', {}).get('auroc'))}, AUPRC "
            f"{fmt(eicu_site_holdout.get('metrics', {}).get('auprc'))}, and ECE "
            f"{fmt(eicu_site_holdout.get('metrics', {}).get('ece'))}. This "
            f"local-learning result was better than direct MIMIC-to-eICU "
            f"transport but remains a separate design and is not presented as "
            f"cross-database transport."
        ),
        (
            f"Calibration and clinical utility. In expanded eICU, the temporal "
            f"ensemble had raw and recalibrated Brier scores of "
            f"{fmt(clinical_utility.get('cohorts', {}).get('eicu_anycad', {}).get('models', {}).get('temporal_ensemble', {}).get('point', {}).get('brier'))} "
            f"and "
            f"{fmt(clinical_utility.get('cohorts', {}).get('eicu_anycad', {}).get('models', {}).get('temporal_ensemble', {}).get('calibrated_point', {}).get('brier'))}, "
            f"with ECE falling from "
            f"{fmt(clinical_utility.get('cohorts', {}).get('eicu_anycad', {}).get('models', {}).get('temporal_ensemble', {}).get('point', {}).get('ece'))} "
            f"to "
            f"{fmt(clinical_utility.get('cohorts', {}).get('eicu_anycad', {}).get('models', {}).get('temporal_ensemble', {}).get('calibrated_point', {}).get('ece'))}. "
            f"Calibration slope changed from "
            f"{fmt(clinical_utility.get('cohorts', {}).get('eicu_anycad', {}).get('models', {}).get('temporal_ensemble', {}).get('point', {}).get('calibration_slope'))} "
            f"to "
            f"{fmt(clinical_utility.get('cohorts', {}).get('eicu_anycad', {}).get('models', {}).get('temporal_ensemble', {}).get('calibrated_point', {}).get('calibration_slope'))}. "
            f"At a 10% threshold, recalibrated sensitivity, specificity, PPV, "
            f"and NPV were "
            f"{fmt(clinical_utility.get('cohorts', {}).get('eicu_anycad', {}).get('models', {}).get('temporal_ensemble', {}).get('operating_thresholds', {}).get('0.10', {}).get('calibrated', {}).get('sensitivity'))}, "
            f"{fmt(clinical_utility.get('cohorts', {}).get('eicu_anycad', {}).get('models', {}).get('temporal_ensemble', {}).get('operating_thresholds', {}).get('0.10', {}).get('calibrated', {}).get('specificity'))}, "
            f"{fmt(clinical_utility.get('cohorts', {}).get('eicu_anycad', {}).get('models', {}).get('temporal_ensemble', {}).get('operating_thresholds', {}).get('0.10', {}).get('calibrated', {}).get('ppv'))}, "
            f"and "
            f"{fmt(clinical_utility.get('cohorts', {}).get('eicu_anycad', {}).get('models', {}).get('temporal_ensemble', {}).get('operating_thresholds', {}).get('0.10', {}).get('calibrated', {}).get('npv'))}. "
            f"At 0.15, the recalibration-minus-raw net-benefit difference was "
            f"{fmt(clinical_utility.get('cohorts', {}).get('eicu_anycad', {}).get('models', {}).get('temporal_ensemble', {}).get('bootstrap', {}).get('net_benefit_ci', {}).get('0.15', {}).get('calibration_minus_raw', [float('nan'), float('nan')])[0])} "
            f"to "
            f"{fmt(clinical_utility.get('cohorts', {}).get('eicu_anycad', {}).get('models', {}).get('temporal_ensemble', {}).get('bootstrap', {}).get('net_benefit_ci', {}).get('0.15', {}).get('calibration_minus_raw', [float('nan'), float('nan')])[1])}. "
            f"Calibration and utility uncertainty are shown in Figures 4 and 6."
        ),
        (
            f"Endpoint sensitivity. The diagnosis-coded primary phenotype yielded "
            f"an XGB+TS AUROC of "
            f"{fmt(sensitivity.get('primary_coded_macce', {}).get('auroc'))}. "
            f"Restricting to the timestamped endpoint yielded "
            f"{fmt(sensitivity.get('timestamped_macce', {}).get('auroc'))} with "
            f"{int(sensitivity.get('timestamped_macce', {}).get('events', 0)):,} "
            f"events. Among patients without prior myocardial infarction or "
            f"stroke, AUROC was "
            f"{fmt(sensitivity.get('no_prior_mi_or_stroke', {}).get('auroc'))}; "
            f"in prior PCI it was "
            f"{fmt(sensitivity.get('prior_pci', {}).get('auroc'))}; and in prior "
            f"CABG it was "
            f"{fmt(sensitivity.get('prior_cabg', {}).get('auroc'))}. "
            f"The interval audit identified "
            f"{int(interval_model.get('mimic4', {}).get('macce_interval_confirmed', {}).get('events', 0)):,} "
            f"strongly confirmed MIMIC-IV events, below the 150-event inference "
            f"threshold, and "
            f"{int(interval_model.get('mimic4', {}).get('macce_interval_overlap', {}).get('events', 0)):,} "
            f"events whose specific anchor interval overlapped the target "
            f"window. The confirmed stratum was therefore reported as "
            f"insufficient rather than assigned an AUROC. Retrained XGB+TS "
            f"reached AUROC "
            f"{fmt(interval_model.get('mimic4', {}).get('macce_interval_overlap', {}).get('auroc'))} "
            f"for the MIMIC-IV anchor-overlap endpoint. The corresponding "
            f"MIMIC-III overlap endpoint contained "
            f"{int(interval_model.get('external', {}).get('mimic3', {}).get('macce_interval_overlap', {}).get('events', 0)):,} "
            f"events and yielded AUROC "
            f"{fmt(interval_model.get('external', {}).get('mimic3', {}).get('macce_interval_overlap', {}).get('auroc'))}. "
            f"Expanded eICU contained "
            f"{int(interval_model.get('external', {}).get('eicu_anycad', {}).get('macce_interval_overlap', {}).get('events', 0)):,} "
            f"overlap events and yielded AUROC "
            f"{fmt(interval_model.get('external', {}).get('eicu_anycad', {}).get('macce_interval_overlap', {}).get('auroc'))}. "
            f"The original principal-diagnosis eICU cohort and all confirmed "
            f"strata had too few events for model inference. Applying the "
            f"original temporal ensemble without interval-specific retraining "
            f"produced MIMIC-IV overlap AUROC "
            f"{fmt(interval_prediction.get('cohorts', {}).get('mimic4', {}).get('models', {}).get('temporal_ensemble', {}).get('macce_interval_overlap', {}).get('auroc'))}. "
            f"These estimates quantify endpoint sensitivity but do not "
            f"establish clinical event-onset truth."
        ),
        (
            f"Endpoint chart-review pilot. The review pack contained 120 "
            f"diagnosis-coded records: 100 MIMIC records were available for "
            f"independent clinical review and 20 eICU records lacked source "
            f"text. A structured AI-assisted pilot reviewed 38 MIMIC records "
            f"to test the extraction schema. It classified "
            f"{int(manual_review.get('target_support_n', 0))} as confirmed or "
            f"probable target-window events, "
            f"{int(manual_review.get('clear_non_target_n', 0))} as clear "
            f"non-target records, and "
            f"{int(manual_review.get('uncertain_or_insufficient_n', 0))} as "
            f"uncertain or insufficient. Target support was "
            f"{100 * float(manual_review.get('target_support_fraction', 0)):.1f}% "
            f"in the pilot. These are AI-assisted structured-chart labels, not "
            f"independent clinician adjudication, and are not reported as a "
            f"positive predictive value."
        ),
        (
            f"Two-reader adjudication status. A structured review dataset now "
            f"contains separate reader fields for MI evidence, stroke evidence, "
            f"onset window, confidence, consensus label, and adjudication notes. "
            f"The scoring script will calculate Cohen kappa, PPV against the "
            f"two-reader consensus, and timing misclassification once both "
            f"readers finish. At this analysis snapshot, "
            f"{int(manual_review_score.get('pending_rows', 0))} of "
            f"{int(manual_review_score.get('review_rows', 0))} records remain "
            f"unreviewed by humans; no kappa or human PPV is imputed."
        ),
        (
            f"Interval-rule sensitivity. Halving, retaining, and doubling the "
            f"assay troponin threshold changed MIMIC-IV anchor-overlap events "
            f"from 1,266 to 1,265 and 1,254; the temporal-ensemble AUROC changed "
            f"from 0.769 to 0.769 and 0.769, respectively. MIMIC-III remained "
            f"at 76 events with temporal-ensemble AUROC 0.729. In expanded eICU, "
            f"documentation lags of 0, 3, 6, and 12 hours produced the same 119 "
            f"anchor-overlap events and temporal-ensemble AUROC 0.653. Thus the "
            f"main endpoint conclusion was stable to these pre-specified rule "
            f"perturbations, while the strict confirmed stratum remained too "
            f"sparse for inference."
        ),
        (
            (
                "Explainability. In CT-OGT, organ and temporal attentions were "
                "aggregated across cross-validation folds. The strongest differences "
                "between MACCE and event-free patients were concentrated in "
                "circulatory, renal, neurologic, and coagulation channels, with "
                "hour-specific contributions rather than a uniform average over the "
                "24-hour window. Attention is an internal model weight and is not "
                "interpreted as a causal organ interaction; the corresponding "
                "figure is exploratory and should guide future perturbation studies."
                if full_completed
                else "Explainability. Development-fold attention extraction for the "
                "environment-invariant CT-OGT was not completed before shutdown. "
                "Figure 5 therefore uses the completed external CT-OGT attention "
                "weights only for descriptive visualization. No causal or "
                "development-fold mechanistic claim is made."
            )
        ),
    ]

    discussion = [
        (
            "Principal findings. This three-database study found consistent "
            "incremental information in 24-hour temporal trajectories over static "
            "severity. XGB+TS was a demanding benchmark: it equaled or exceeded "
            "GRU and TFT-lite, but the three-seed Transformer added complementary "
            "information. Their equal-weight ensemble "
            + ("exceeded " if h2_supported else "did not exceed ")
            + "XGB+TS by the 0.01 material margin in the original five-fold "
            "analysis and retained a positive difference in the 2020-2022 "
            "temporal holdout. Strict nested tuning attenuated the increment to "
            "0.0098, so the result is best described as consistent but marginal "
            "against the material threshold. The full CT-OGT improved only "
            "modestly over XGB+TS, and the redesigned seven-organ token graph "
            "did not improve on either. This does not support a separate "
            "material organ-graph effect. The study separates a reproducible "
            "complementary temporal-signal finding from a fragile claim about "
            "organ-graph complexity. "
            + (
                "The completed environment-invariant CT-OGT was slightly higher "
                "than XGB+TS but lower than the dynamic graph and temporal "
                "ensemble in development."
                if full_completed
                else "The environment-invariant arm was not completed and "
                "cannot support or refute H3 in this analysis snapshot."
            )
            + " Because the complementary ensemble was formed after observing "
            "the organ-graph result, it is explicitly exploratory rather than "
            "pre-specified. To limit selection bias, features and fusion weights "
            "were chosen on 2008-2016 versus 2017-2019 and then frozen for the "
            "2020-2022 era. The strongest locked model reached AUROC 0.7462 and "
            "the gain over original XGB+TS was 0.0125 (95% CI 0.0073-0.0172). "
        ),
        (
            "Clinical interpretation. The first ICU day captures both the current "
            "burden of organ dysfunction and the trajectory of resuscitation. "
            "Patients whose circulation, gas exchange, renal output, "
            "coagulation, and consciousness do not improve in parallel may have "
            "a different risk profile from patients with a similar SOFA score at "
            "a single time [5]. The gain from temporal summaries is consistent "
            "with this clinical intuition. The failure of a more complex model to "
            "add discrimination does not imply that organ interactions are "
            "biologically absent; it may indicate that the available cohort, "
            "missing-data pattern, and diagnosis-coded endpoint do not provide "
            "enough information to estimate stable dynamic edges."
        ),
        (
            "Why calibration dominates transport. Discrimination partly survived "
            "the change from an academic ICU to an earlier era and to multiple "
            "hospitals, but predicted probabilities were systematically too high "
            "in the external settings. This pattern is common when case mix, "
            "coding, and data density differ [15]. Local recalibration improved "
            "Brier score and ECE without recovering missing discrimination. "
            "Consequently, a deployment pathway should treat the model as a "
            "ranking and recalibration candidate, not as a directly portable "
            "absolute-risk calculator. Expanding eICU beyond the principal-"
            "diagnosis position increased events from 57 to 642 and allowed "
            "multi-hospital discrimination to be estimated with useful "
            "precision. The original principal-diagnosis cohort remained a "
            "sensitivity analysis and did not support site-specific claims."
        ),
        (
            "Graph learning and negative controls. The absence of a clear organ-graph "
            "advantage can arise from three mechanisms. First, XGB+TS sees the "
            "same channel summaries and can approximate much of the stable "
            "organ-level signal. Second, graph edges are learned from observations "
            "that are themselves sparse and unevenly distributed across organ "
            "systems; missingness can dominate the attention. Third, the "
            "environment penalty can improve worst-group stability while "
            "reducing average discrimination, although the completed external "
            "CT-OGT result was lower than the dynamic graph alone. A redesigned "
            "seven-organ token graph addressed channel-level dilution directly, "
            "but its AUROC was 0.750, below both XGB+TS and the original dynamic "
            "graph. This makes it unlikely that the negative result is solely "
            "an artifact of too many channel nodes. Adding explicit organ change "
            "rates and observation intervals still reached only 0.724 in the "
            "2017-2019 gate and was not tested on the final era. The reversed, permuted, "
            "random-graph, and no-time controls further suggest that temporal "
            "signal, rather than a stable organ-edge mechanism, dominates "
            "prediction. None of these analyses establishes that graph "
            "attention is causal."
        ),
        (
            "Endpoint ascertainment is the main scientific bottleneck. The "
            "diagnosis-coded 7-day phenotype is vulnerable to two errors. Events "
            "present at admission may be coded on the same admission and mixed "
            "with incident events, while an in-hospital event may be recorded in "
            "a discharge summary without a structured onset time. The strict "
            "full-interval endpoint retained only 45 MIMIC-IV events and was "
            "therefore uninformative for discrimination. The anchor-overlap "
            "endpoint retained 1,265 events but deliberately mixes definite and "
            "possible timing, so its AUROC describes a sensitivity phenotype "
            "rather than event-onset truth. The structured chart-review pilot "
            "found clear target support in only 10 of 38 reviewed records and "
            "marked 15 as uncertain, reinforcing that diagnosis-code proximity "
            "is not adjudication. Future work should use independent clinician "
            "review, troponin and ECG adjudication, and procedure timestamps "
            "from source systems. The current results should not be used to "
            "claim a fully adjudicated 7-day MACCE endpoint."
        ),
        (
            "Implications for model development. Strong tabular temporal models "
            "should remain mandatory comparators for clinical graph-learning "
            "studies. Reporting only a weak static baseline would overstate the "
            "value of graph machinery. The most defensible next experiments are "
            "prospective event adjudication, prespecified graph priors with "
            "external edge stability checks, and validation in cohorts with a "
            "larger number of incident events. If dynamic graphs are retained, "
            "their value should be demonstrated through perturbation and "
            "transport analyses rather than attention maps alone."
        ),
    ]

    limitations = [
        (
            "First, diagnosis codes did not provide fully reliable event-onset "
            "times. The main endpoint is a 7-day diagnosis-coded proxy, and the "
            "timestamped sensitivity endpoint reduced event counts. Second, "
            "MIMIC-III lacked procedure dates for revascularization and the "
            "available eICU extract lacked notes and treatment tables. Third, "
            "the expanded eICU cohort increased the event count to 642, but its "
            "diagnosis-any-position definition changed the target population "
            "from development and MIMIC-III; the original principal-diagnosis "
            "cohort had only 57 events. Fourth, the endpoint chart-review pilot "
            "was AI-assisted and not independent clinical adjudication, and the "
            "100-record MIMIC review remained pending. Fifth, the strict nested "
            "ensemble increment was 0.0098, just below the material threshold, "
            "so the positive temporal finding is marginal under full tuning. "
            "Sixth, the ICD-code "
            "definition of coronary atherosclerosis may select a clinically "
            "heterogeneous population and does not distinguish stable from "
            "unstable disease with perfect accuracy. Seventh, urine output was "
            "unavailable in eICU, and missingness differed across databases. "
            "Eighth, deep models were trained with a fixed seed in five folds, "
            "whereas the tabular benchmarks were repeated across three seeds; "
            "this limits precision of deep-model seed variance. "
            + (
                "Ninth, the full CT-OGT and all negative controls were "
                "completed; the full CT-OGT improved only modestly over XGB+TS "
                "and did not add materially beyond the dynamic graph. Tenth, the "
                if full_completed
                else "Ninth, the environment-invariant CT-OGT development run "
                "was not fully completed. Tenth, the "
            )
            +
            "2020-2022 era had been examined in an earlier analysis snapshot, so "
            "the second-stage test is not a never-seen prospective holdout, "
            "although feature and weight selection were restricted to earlier "
            "eras. Eleventh, two-reader human endpoint adjudication remains "
            "pending. Twelfth, the analysis is retrospective and does not establish treatment benefit "
            "or causal organ interactions. Finally, all databases reflect "
            "routine care and coding practices, and the results require external "
            "prospective validation before deployment."
        )
    ]

    conclusion = [
        (
            "In a three-database study of coronary atherosclerosis, 24-hour "
            "temporal information consistently improved 7-day MACCE "
            "discrimination over static severity. A locked ensemble of XGB+TS "
            "and a three-seed temporal Transformer exceeded the 0.01 material "
            "margin in the original five-fold analysis and improved "
            "discrimination in a locked 2020-2022 temporal holdout. The "
            "increment was 0.0098 after strict nested tuning, so the positive "
            "result is consistent but marginal against the material threshold. "
            "The continuous-time organ graph and a redesigned seven-organ token "
            "graph, including a model with explicit organ change rates, did not "
            "improve materially over XGB+TS. A second-stage exploratory model "
            "using organ adverse-slope and recovery features was selected only "
            "on earlier eras and achieved AUROC 0.746 with a +0.0125 gain over "
            "XGB+TS across 2020-2022. Anchor-overlap "
            "endpoint analysis remained discriminative, but the strict event-"
            "time strata were too small and the diagnosis-coded endpoint still "
            "requires independent clinical adjudication. Transport "
            "discrimination was useful but calibration was poor until locally "
            "recalibrated. These findings support temporal modeling as a strong "
            "baseline and argue for better event adjudication before claiming "
            "incremental value from dynamic organ graphs."
        )
    ]

    return {
        "abstract": abstract,
        "introduction": introduction,
        "methods": methods,
        "results": results,
        "discussion": discussion,
        "limitations": limitations,
        "conclusion": conclusion,
    }


def add_page_number(paragraph):
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)


def add_caption(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    paragraph.paragraph_format.space_after = Pt(8)


def add_dataframe(
    document: Document,
    frame: pd.DataFrame,
    *,
    font_size: float = 8.0,
) -> None:
    if frame.empty:
        return
    table = document.add_table(rows=1, cols=len(frame.columns))
    table.style = "Table Grid"
    for index, column in enumerate(frame.columns):
        cell = table.rows[0].cells[index]
        cell.text = str(column)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(font_size)
    for record in frame.to_dict(orient="records"):
        cells = table.add_row().cells
        for index, column in enumerate(frame.columns):
            cells[index].text = str(record[column])
            for run in cells[index].paragraphs[0].runs:
                run.font.size = Pt(font_size)


def create_docx(
    text: dict[str, list[str]],
    references: list[dict],
    metrics: dict,
) -> Path:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)
    styles = document.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(11)
    styles["Normal"].paragraph_format.line_spacing = 1.5
    for style_name in ("Title", "Heading 1", "Heading 2"):
        styles[style_name].font.name = "Times New Roman"
    styles["Title"].font.size = Pt(18)
    styles["Title"].font.bold = True
    styles["Heading 1"].font.size = Pt(13)
    styles["Heading 1"].font.bold = True
    styles["Heading 2"].font.size = Pt(11)
    styles["Heading 2"].font.bold = True
    header = section.header.paragraphs[0]
    header.text = "CT-OGT for 7-day MACCE in coronary atherosclerosis"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.runs[0].font.size = Pt(8)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(footer)

    document.add_paragraph(TITLE, style="Title")
    authors = document.add_paragraph()
    authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
    authors.add_run("Authors and affiliations to be finalized").italic = True
    note = document.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note.add_run(
        "Retrospective three-database transport study | Prepared 7 November 2026"
    )
    document.add_paragraph(
        "Data sources: MIMIC-IV v3.1; MIMIC-III v1.4; eICU-CRD v2.0"
    )
    document.add_paragraph(
        "Reporting framework: TRIPOD+AI | Prediction landmark: ICU hour 24 | "
        "Target window: through day 7"
    )
    document.add_page_break()

    headings = {
        "abstract": "Abstract",
        "introduction": "Introduction",
        "methods": "Methods",
        "results": "Results",
        "discussion": "Discussion",
        "limitations": "Limitations",
        "conclusion": "Conclusion",
    }
    for key, heading in headings.items():
        document.add_heading(heading, level=1)
        for paragraph in text[key]:
            document.add_paragraph(paragraph)
        if key == "methods":
            table1 = TABLES / "table1_baseline.csv"
            if table1.is_file():
                document.add_heading("Table 1. Cohort characteristics", level=2)
                add_dataframe(document, pd.read_csv(table1), font_size=7.5)
        if key == "results":
            for figure, caption in (
                (
                    "figure1_study_design.png",
                    "Figure 1. Study flow, 24-hour landmark, and three-database validation.",
                ),
                (
                    "figure2_model_architecture.png",
                    "Figure 2. CT-OGT architecture with continuous-time cues, dynamic organ graph, and temporal encoder.",
                ),
            ):
                image = FIGURES / figure
                if image.is_file():
                    document.add_picture(str(image), width=Inches(6.6))
                    add_caption(document, caption)
            table2 = TABLES / "table2_development_performance.csv"
            if table2.is_file():
                document.add_heading(
                    "Table 2. Development performance", level=2
                )
                add_dataframe(document, pd.read_csv(table2), font_size=7.5)
            table3 = TABLES / "table3_transport.csv"
            if table3.is_file():
                document.add_heading("Table 3. Three-database transport", level=2)
                add_dataframe(document, pd.read_csv(table3), font_size=7)
            table4 = TABLES / "table4_interval_endpoint.csv"
            if table4.is_file():
                document.add_heading(
                    "Table 4. Interval-censored endpoint sensitivity", level=2
                )
                add_dataframe(document, pd.read_csv(table4), font_size=7.5)
            table5 = TABLES / "table5_validation.csv"
            if table5.is_file():
                document.add_heading(
                    "Table 5. Locked temporal and nested validation",
                    level=2,
                )
                add_dataframe(document, pd.read_csv(table5), font_size=7.5)
            table6 = TABLES / "table6_manual_review.csv"
            if table6.is_file():
                document.add_heading(
                    "Table 6. Structured endpoint review pilot", level=2
                )
                add_dataframe(document, pd.read_csv(table6), font_size=7.5)
            table7 = TABLES / "table7_clinical_utility.csv"
            if table7.is_file():
                document.add_heading(
                    "Table 7. External calibration and threshold utility",
                    level=2,
                )
                add_dataframe(document, pd.read_csv(table7), font_size=7)
            table8 = TABLES / "table8_interval_threshold_sensitivity.csv"
            if table8.is_file():
                document.add_heading(
                    "Table 8. Interval-rule threshold sensitivity",
                    level=2,
                )
                add_dataframe(document, pd.read_csv(table8), font_size=7)
            for figure, caption in (
                (
                    "figure3_performance_ablation.png",
                    "Figure 3. Development performance, mechanism ablations, and negative controls.",
                ),
                (
                    "figure4_calibration_dca.png",
                    "Figure 4. External calibration and decision-curve analysis.",
                ),
                (
                    "figure5_attention.png",
                    "Figure 5. Organ and temporal attention in the full model.",
                ),
                (
                    "figure6_net_benefit_ci.png",
                    "Figure 6. Net benefit with bootstrap confidence intervals before and after local recalibration.",
                ),
                (
                    "figure7_interval_threshold_sensitivity.png",
                    "Figure 7. Sensitivity of interval endpoints to troponin thresholds and eICU documentation lag.",
                ),
            ):
                image = FIGURES / figure
                if image.is_file():
                    document.add_picture(str(image), width=Inches(6.7))
                    add_caption(document, caption)

    document.add_heading("References", level=1)
    for index, record in enumerate(references, start=1):
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.25)
        paragraph.paragraph_format.first_line_indent = Inches(-0.25)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.add_run(f"{index}. ").bold = True
        paragraph.add_run(record["citation"])
    document.add_heading("Data and code availability", level=1)
    document.add_paragraph(
        "MIMIC-IV, MIMIC-III, and eICU-CRD are available through PhysioNet to "
        "credentialed users under their respective data-use agreements. The "
        "analysis code, derived cohort contracts, endpoint audit, fold-level "
        "predictions, and figure-generation scripts are preserved in the study "
        "repository. Raw patient data are not redistributed."
    )
    document.add_heading("Ethics statement", level=1)
    document.add_paragraph(
        "The source databases contain de-identified data collected under their "
        "original institutional approvals. This secondary analysis follows the "
        "terms of the PhysioNet credentialed data-use agreements and does not "
        "involve a new patient-facing intervention."
    )
    output = OUTPUTS / "CTOGT_MACCE_manuscript.docx"
    document.save(output)
    return output


def create_markdown(
    text: dict[str, list[str]],
    references: list[dict],
    metrics: dict,
) -> Path:
    lines = [f"# {TITLE}", "", "*Authors and affiliations to be finalized*", ""]
    for key, heading in (
        ("abstract", "Abstract"),
        ("introduction", "Introduction"),
        ("methods", "Methods"),
        ("results", "Results"),
        ("discussion", "Discussion"),
        ("limitations", "Limitations"),
        ("conclusion", "Conclusion"),
    ):
        lines.extend([f"## {heading}", ""])
        lines.extend(paragraph + "\n" for paragraph in text[key])
    lines.extend(["## Tables", ""])
    for filename, caption in (
        ("table1_baseline.md", "Table 1. Cohort characteristics"),
        (
            "table2_development_performance.md",
            "Table 2. Development performance",
        ),
        ("table3_transport.md", "Table 3. Three-database transport"),
        (
            "table4_interval_endpoint.md",
            "Table 4. Interval-censored endpoint sensitivity",
        ),
        (
            "table5_validation.md",
            "Table 5. Locked temporal and nested validation",
        ),
        (
            "table6_manual_review.md",
            "Table 6. Structured endpoint review pilot",
        ),
        (
            "table7_clinical_utility.md",
            "Table 7. External calibration and threshold utility",
        ),
        (
            "table8_interval_threshold_sensitivity.md",
            "Table 8. Interval-rule threshold sensitivity",
        ),
    ):
        path = TABLES / filename
        if path.is_file():
            lines.extend([f"### {caption}", "", path.read_text(encoding="utf-8"), ""])
    lines.extend(["## Figures", ""])
    for filename, caption in (
        ("figure1_study_design.png", "Figure 1. Study design"),
        ("figure2_model_architecture.png", "Figure 2. CT-OGT architecture"),
        ("figure3_performance_ablation.png", "Figure 3. Performance and ablations"),
        ("figure4_calibration_dca.png", "Figure 4. Calibration and DCA"),
        ("figure5_attention.png", "Figure 5. Attention"),
        ("figure6_net_benefit_ci.png", "Figure 6. Net benefit with bootstrap CI"),
        (
            "figure7_interval_threshold_sensitivity.png",
            "Figure 7. Interval-rule threshold sensitivity",
        ),
    ):
        path = FIGURES / filename
        if path.is_file():
            lines.extend([f"![{caption}]({path.as_posix()})", "", f"**{caption}.**", ""])
    lines.extend(["## References", ""])
    for index, record in enumerate(references, start=1):
        lines.append(f"{index}. {record['citation']}")
    output = OUTPUTS / "CTOGT_MACCE_manuscript.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def create_supplement() -> Path:
    document = Document()
    document.styles["Normal"].font.name = "Times New Roman"
    document.styles["Normal"].font.size = Pt(10)
    document.add_heading(
        "Supplementary Material: Invariant Continuous-Time Organ Graph Learning",
        level=1,
    )
    document.add_heading("Table S1. Organ channels and prespecified ranges", level=2)
    range_rows = pd.DataFrame(
        [
            ("gcs_total", "Neurologic", "3-15"),
            ("pupil_reactivity", "Neurologic", "0-2"),
            ("hr", "Circulatory", "20-260"),
            ("sbp", "Circulatory", "40-300"),
            ("dbp", "Circulatory", "20-180"),
            ("mbp", "Circulatory", "30-220"),
            ("lactate", "Circulatory", "0.1-30"),
            ("rr", "Respiratory", "2-80"),
            ("spo2", "Respiratory", "50-100"),
            ("fio2", "Respiratory", "0.21-1.0"),
            ("creatinine", "Renal", "0.1-30"),
            ("urine_output", "Renal", "0-5000 mL/hour"),
            ("platelets", "Coagulation", "1-1500"),
            ("inr", "Coagulation", "0.5-20"),
            ("aptt", "Coagulation", "5-300"),
            ("wbc", "Inflammation", "0.05-300"),
            ("temp", "Inflammation", "30-44 C"),
            ("glucose", "Metabolic", "10-1500"),
            ("ph", "Metabolic", "6.5-7.9"),
            ("bicarbonate", "Metabolic", "2-60"),
        ],
        columns=["Channel", "Organ system", "Accepted range"],
    )
    add_dataframe(document, range_rows, font_size=8)
    document.add_heading("Table S2. Endpoint and population sensitivity", level=2)
    sensitivity_path = RESULTS / "sensitivity_analysis.json"
    if sensitivity_path.is_file():
        sensitivity = load_json(sensitivity_path)["analyses"]
        rows = []
        for name, record in sensitivity.items():
            rows.append(
                {
                    "Analysis": name,
                    "n": record.get("n", ""),
                    "Events": record.get("events", ""),
                    "AUROC": fmt(record.get("auroc", float("nan"))),
                    "Status": record.get("status", ""),
                }
            )
        add_dataframe(document, pd.DataFrame(rows), font_size=8)
    document.add_heading(
        "Table S3. Expanded eICU sites with at least 30 analytic patients",
        level=2,
    )
    site_rows = []
    site_path = PROCESSED / "eicu_anycad" / "cohort.csv"
    if site_path.is_file():
        site_frame = pd.read_csv(site_path)
        site_rows = (
            site_frame.groupby("hospitalid", as_index=False)
            .agg(Patients=("stay_id", "size"), Events=("macce7", "sum"))
            .query("Patients >= 30")
            .sort_values(["Events", "Patients"], ascending=False)
            .rename(columns={"hospitalid": "Hospital"})
            .to_dict(orient="records")
        )
    if site_rows:
        add_dataframe(document, pd.DataFrame(site_rows), font_size=8)
    document.add_heading("Table S4. Event-time interval rules", level=2)
    interval_rules = pd.DataFrame(
        [
            (
                "MIMIC-IV / MIMIC-III death",
                "Exact timestamp",
                "Point interval",
            ),
            (
                "MIMIC-IV revascularization",
                "procedures_icd.chartdate",
                "Procedure calendar day",
            ),
            (
                "MIMIC radiology stroke evidence",
                "Radiology charttime",
                "24 hours before report to report time",
            ),
            (
                "MIMIC-III timed notes",
                "NOTEEVENTS charttime/chartdate",
                "24 hours before note to note time",
            ),
            (
                "MIMIC troponin evidence",
                "First elevated post-landmark troponin",
                "6 hours before measurement to measurement",
            ),
            (
                "MIMIC discharge-note hospital day",
                "Explicit HD/POD day",
                "Corresponding 24-hour hospital-day interval",
            ),
            (
                "eICU diagnosis offset",
                "diagnosisoffset",
                "6-hour documentation-lag interval",
            ),
            (
                "Coded event without anchor",
                "Diagnosis code only",
                "Excluded from the anchor-overlap endpoint",
            ),
        ],
        columns=["Source", "Anchor", "Interval"],
    )
    add_dataframe(document, interval_rules, font_size=8)
    document.add_heading("TRIPOD+AI checklist", level=2)
    checklist = pd.DataFrame(
        [
            ("1", "Title", "Identifies prediction model and transport study", "Reported"),
            ("2", "Abstract", "Structured summary of design and findings", "Reported"),
            ("3", "Rationale", "Clinical and methodological rationale", "Reported"),
            ("4", "Objectives", "Temporal, graph, invariance, and transport aims", "Reported"),
            ("5", "Data sources", "MIMIC-IV, MIMIC-III, eICU-CRD", "Reported"),
            ("6", "Participants", "Development/MIMIC-III principal CAD; expanded eICU any position", "Reported"),
            ("7", "Outcome", "Diagnosis-coded MACCE, timestamped and interval-censored sensitivity", "Reported"),
            ("8", "Predictors", "Organ channels, observation masks, elapsed time", "Reported"),
            ("9", "Sample size", "Expanded eICU n=5,913 and 642 events reported", "Reported"),
            ("10", "Missing data", "Explicit masks; fold-level standardization", "Reported"),
            ("11", "Model", "CT-OGT and comparator specifications", "Reported"),
            ("12", "Performance", "AUROC, AUPRC, Brier, calibration, DCA", "Reported"),
            ("13", "Fairness", "Age, sex, race, phenotype limitations discussed", "Partial"),
            ("14", "Interpretation", "Attention and ablation analyses", "Partial"),
            ("15", "Model output", "Probability and top-decile thresholds", "Reported"),
            ("16", "Reproducibility", "Seeds, fold predictions, code and manifests", "Reported"),
            ("17", "Registration", "No prospective registration", "Not applicable"),
            ("18", "Funding", "To be finalized by authors", "Pending"),
            ("19", "Conflicts", "To be finalized by authors", "Pending"),
            ("20", "Data availability", "PhysioNet credentialed access", "Reported"),
        ],
        columns=["Item", "Domain", "Study status", "Reporting status"],
    )
    add_dataframe(document, checklist, font_size=8)
    output = OUTPUTS / "CTOGT_MACCE_supplement.docx"
    document.save(output)
    return output


def main() -> None:
    ensure_dirs()
    render_tables_figures()
    metrics = manuscript_metrics()
    references = build_references()
    text = paragraph_texts(metrics, references)
    manuscript = create_docx(text, references, metrics)
    markdown = create_markdown(text, references, metrics)
    supplement = create_supplement()
    print(f"Created {manuscript}")
    print(f"Created {markdown}")
    print(f"Created {supplement}")


if __name__ == "__main__":
    main()

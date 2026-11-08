"""Verify final deliverables, predictions, and figure non-blankness."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat
from docx import Document

from .common import FIGURES, OUTPUTS, PREDICTIONS, PROCESSED, RESULTS, ensure_dirs
from .train_evaluate import ALL_MODELS


def main() -> None:
    ensure_dirs()
    checks: dict[str, object] = {"checks": {}, "failures": []}

    def record(name: str, passed: bool, detail: object) -> None:
        checks["checks"][name] = {"passed": bool(passed), "detail": detail}
        if not passed:
            checks["failures"].append(name)

    missing_models = []
    finite_models = []
    for model in ALL_MODELS:
        path = PREDICTIONS / f"mimic4_cv_{model}.npz"
        if not path.is_file():
            missing_models.append(model)
            continue
        with np.load(path, allow_pickle=True) as data:
            probability = data["probability"].astype(float)
            finite = (
                np.isfinite(probability).all()
                and probability.min() >= 0
                and probability.max() <= 1
            )
            if not finite:
                finite_models.append(model)
    record("all_development_models_present", not missing_models, missing_models)
    record("all_development_probabilities_valid", not finite_models, finite_models)
    ensemble_path = PREDICTIONS / "mimic4_cv_temporal_ensemble.npz"
    record("temporal_ensemble_present", ensemble_path.is_file(), str(ensemble_path))

    external_models = {}
    minimum_external_models = {"mimic3": 10, "eicu": 10, "eicu_anycad": 3}
    for name, minimum in minimum_external_models.items():
        files = sorted(PREDICTIONS.glob(f"{name}_external_*.npz"))
        external_models[name] = [path.stem for path in files]
        record(
            f"{name}_external_predictions_present",
            len(files) >= minimum,
            len(files),
        )

    output_names = (
        "CTOGT_MACCE_manuscript.docx",
        "CTOGT_MACCE_manuscript.md",
        "CTOGT_MACCE_supplement.docx",
        "CTOGT_MACCE_执行摘要_中文.docx",
        "CTOGT_MACCE_执行摘要_中文.md",
        "POSITIVE_RESULT_SUMMARY.md",
        "TRAINING_STATUS.md",
        "CTOGT_MACCE_submission_bundle.zip",
        "SHA256SUMS.txt",
    )
    missing_outputs = [
        name for name in output_names if not (OUTPUTS / name).is_file()
    ]
    record("final_outputs_present", not missing_outputs, missing_outputs)

    docx_status = {}
    for name in (
        "CTOGT_MACCE_manuscript.docx",
        "CTOGT_MACCE_supplement.docx",
        "CTOGT_MACCE_执行摘要_中文.docx",
    ):
        path = OUTPUTS / name
        if path.is_file():
            document = Document(path)
            docx_status[name] = {
                "paragraphs": len(document.paragraphs),
                "tables": len(document.tables),
            }
        else:
            docx_status[name] = None
    record(
        "docx_files_readable",
        all(
            item is not None and item["paragraphs"] > 0
            for item in docx_status.values()
        ),
        docx_status,
    )

    figure_status = {}
    for path in sorted(FIGURES.glob("figure*.png")):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            stat = ImageStat.Stat(rgb)
            nonwhite = np.asarray(rgb).mean(axis=2) < 250
            figure_status[path.name] = {
                "size": list(image.size),
                "mean": [round(value, 2) for value in stat.mean],
                "nonwhite_fraction": float(nonwhite.mean()),
            }
    record("all_figures_present", len(figure_status) == 7, len(figure_status))
    record(
        "all_figures_nonblank",
        all(item["nonwhite_fraction"] > 0.01 for item in figure_status.values()),
        figure_status,
    )
    interval_files = (
        PROCESSED / "mimic4" / "interval_endpoint.csv",
        PROCESSED / "mimic3" / "interval_endpoint.csv",
        PROCESSED / "eicu" / "interval_endpoint.csv",
        PROCESSED / "eicu_anycad" / "interval_endpoint.csv",
        RESULTS / "interval_endpoint_summary.json",
        RESULTS / "interval_model_analysis.json",
    )
    missing_intervals = [
        str(path) for path in interval_files if not Path(path).is_file()
    ]
    record("interval_endpoint_outputs_present", not missing_intervals, missing_intervals)

    required_evidence = (
        RESULTS / "temporal_holdout_validation.json",
        RESULTS / "nested_selection_and_utility.json",
        RESULTS / "manual_endpoint_review.csv",
        RESULTS / "manual_endpoint_review_summary.json",
        RESULTS / "manual_endpoint_review_pilot.csv",
        RESULTS / "manual_endpoint_review_pilot.json",
        RESULTS / "endpoint_feasibility_eicu_anycad.json",
        RESULTS / "metrics_development_organ_graph.json",
        RESULTS / "metrics_external_eicu_anycad_xgb_static_ts.json",
        RESULTS / "metrics_external_eicu_anycad_transformer.json",
        RESULTS / "metrics_external_eicu_anycad_temporal_ensemble.json",
        RESULTS / "ensemble_validation.json",
        RESULTS / "clinical_utility_analysis.json",
        RESULTS / "clinical_utility_analysis.csv",
        RESULTS / "interval_threshold_sensitivity.json",
        RESULTS / "interval_threshold_sensitivity.csv",
        RESULTS / "strong_temporal_holdout.json",
        RESULTS / "strong_temporal_holdout_fusion.json",
        RESULTS / "strong_tabular_candidates.json",
        RESULTS / "strong_fusion_triple.json",
        RESULTS / "strong_external_transport.json",
        RESULTS / "eicu_site_holdout.json",
        RESULTS / "organ_dynamics_holdout.json",
        RESULTS / "manual_endpoint_review_structured.csv",
        RESULTS / "manual_endpoint_review_protocol.json",
        RESULTS / "manual_endpoint_review_scored.json",
        RESULTS / "nested_transformer_selection.json",
    )
    missing_evidence = [
        str(path) for path in required_evidence if not Path(path).is_file()
    ]
    record("strengthening_evidence_present", not missing_evidence, missing_evidence)

    manuscript = OUTPUTS / "CTOGT_MACCE_manuscript.md"
    word_count = len(manuscript.read_text(encoding="utf-8").split()) if manuscript.is_file() else 0
    record("manuscript_word_count_reasonable", word_count >= 2000, word_count)
    manuscript_text = (
        manuscript.read_text(encoding="utf-8") if manuscript.is_file() else ""
    )
    stale_phrases = [
        phrase
        for phrase in (
            "694 confirmed MIMIC-IV events",
            "The prespecified temporal ensemble",
            "prespecified equal-weight ensemble",
            "prespecified complementary temporal ensemble",
        )
        if phrase in manuscript_text
    ]
    record("no_stale_primary_claims", not stale_phrases, stale_phrases)
    checks["word_count"] = word_count
    checks["external_models"] = external_models
    checks["status"] = "PASS" if not checks["failures"] else "FAIL"
    (RESULTS / "final_qc.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(checks, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

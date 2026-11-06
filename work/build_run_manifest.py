"""Create a reproducibility manifest for the completed AMI analysis."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy
import pandas
import scipy
import sklearn
import statsmodels
import torch
import xgboost


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUTS = ROOT / "outputs"

KEY_INPUTS = [
    "configs/study.json",
    "data/processed/mimic4/cohort.csv",
    "data/processed/mimic4/hourly.csv",
    "data/processed/mimic4/labs_long.csv",
    "data/processed/mimic4/sequences.npz",
    "data/processed/mimic4/diagnosis_flags.csv",
    "data/processed/mimic4_6h/cohort.csv",
    "data/processed/mimic4_6h/hourly.csv",
    "data/processed/mimic4_6h/labs_long.csv",
    "data/processed/mimic4_6h/sequences.npz",
    "data/processed/mimic4_6h/audit.json",
    "data/processed/mimic4_72h/cohort.csv",
    "data/processed/mimic4_72h/hourly.csv",
    "data/processed/mimic4_72h/labs_long.csv",
    "data/processed/mimic4_72h/sequences.npz",
    "data/processed/mimic4_72h/audit.json",
    "data/processed/mimic3/cohort.csv",
    "data/processed/mimic3/sequences.npz",
    "data/processed/eicu/cohort.csv",
    "data/processed/eicu/sequences.npz",
]

ANALYSIS_COMMANDS = [
    "py -3.10 -m src.ami_dynamic.build_cohorts --datasets mimic4 mimic3 eicu",
    "py -3.10 work/derive_diagnosis_flags.py",
    "py -3.10 work/audit_processed_cohorts.py",
    (
        "py -3.10 -m src.ami_dynamic.train_evaluate --dataset mimic4 "
        "--folds 5 --seeds 1 7 42 123 2024 --epochs 30 "
        "--models xgb_static xgb_static_ts mlp_static mlp_static_ts "
        "gru grud transformer stgcn_lite dggru_static dggru dggru_tr "
        "dggru_ctr --lambda-curvature 0.01 --transport --leakage-audit "
        "--no-resume"
    ),
    "py -3.10 work/run_severity_benchmark.py --seeds 1 7 42 123 2024 --folds 5",
    (
        "py -3.10 -m src.ami_dynamic.train_evaluate --dataset eicu "
        "--folds 5 --seeds 1 7 42 123 2024 --epochs 30 "
        "--models xgb_static xgb_static_ts dggru_tr --no-resume"
    ),
    "py work/analyze_external.py",
    "py work/run_transport_sensitivity.py",
    "py work/run_subgroup_analysis.py",
    "py work/run_temporal_controls.py --seed 42",
    (
        "foreach ($seed in 1,7,42,123,2024) { "
        "py work/run_temporal_controls.py --seed $seed "
        "--skip-controls --lambdas 0.01 1.5 }"
    ),
    "py work/build_multiseed_temporal_summary.py",
    (
        "py work/run_cumulative_temporal_path.py --seed 42 "
        "--folds 5 --epochs 30"
    ),
    "py work/run_ordinal_metrics_external_dca.py --bootstrap-replicates 2000",
    "py work/run_external_calibration_common3.py --replicates 2000",
    "py work/run_sensitivity_models.py --analysis six_level --model xgb_static_ts",
    "py work/run_sensitivity_models.py --analysis primary_position --model xgb_static_ts",
    "py work/run_sensitivity_models.py --analysis exclude_type2 --model xgb_static_ts",
    "py work/run_sensitivity_models.py --analysis exclude_against_advice --model xgb_static_ts",
    "py work/run_sensitivity_models.py --analysis exclude_troponin --model xgb_static_ts",
    "py work/build_landmark_cohorts.py --landmarks 6 72",
    (
        "py -3.10 -m src.ami_dynamic.train_evaluate --dataset mimic4_6h "
        "--models xgb_static xgb_static_ts dggru dggru_tr "
        "--seeds 1 7 42 123 2024 --folds 5 --epochs 30 --no-resume"
    ),
    (
        "py -3.10 -m src.ami_dynamic.train_evaluate --dataset mimic4_72h "
        "--models xgb_static xgb_static_ts dggru dggru_tr "
        "--seeds 1 7 42 123 2024 --folds 5 --epochs 30 --no-resume"
    ),
    (
        "py work/run_additional_sensitivities.py --analyses "
        "pre24h_revascularization pre24h_vasopressor_mcs "
        "label_merge_rehab_facility --bootstrap 2000 --overwrite"
    ),
    (
        "py work/run_additional_sensitivities.py --analyses "
        "multiple_imputation --imputations 3 --mi-max-iter 1 "
        "--bootstrap 2000 --overwrite"
    ),
    "py work/run_feature_importance.py --repeats 5",
    "py work/audit_count_feature_ablation.py",
    "py work/audit_eicu_hospital_cluster_cv.py",
    "py work/build_site_heterogeneity_report.py",
    "py work/build_inclusion_comparison.py",
    "py work/make_tables.py",
    "py work/make_figures.py",
    "py work/build_manuscript.py",
    "py work/build_supplement.py",
    "py work/build_tripod_ai_checklist.py",
    "py work/build_1104_manuscript.py",
    "py work/audit_docx_delivery.py",
    "py work/audit_outline_consistency.py",
    "py work/audit_sign_conventions.py",
    "py work/audit_p1_reviewer_items.py",
    "py work/verify_prediction_cache.py --temporal-controls",
    "py work/build_run_manifest.py",
    "py work/build_release_archive.py",
    "py -m pytest -q",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hashed_files(paths: list[Path]) -> dict:
    rows = {}
    for path in paths:
        if path.is_file():
            rows[str(path.relative_to(ROOT))] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    prediction_files = sorted((RESULTS / "predictions").glob("*.npz"))
    metric_files = sorted(RESULTS.glob("metrics*.json"))
    audit_files = sorted((RESULTS / "audits").glob("*.json"))
    calibration_files = sorted((RESULTS / "calibration").glob("*"))
    ordinal_files = sorted((RESULTS / "ordinal_evaluation").glob("*"))
    table_files = sorted((RESULTS / "tables").glob("*"))
    figure_source_files = sorted(
        (RESULTS / "figures" / "source_data").glob("*")
    )
    figure_assets = sorted(
        path
        for extension in ("png", "pdf", "svg", "tiff")
        for path in (RESULTS / "figures").glob(f"*.{extension}")
    )
    document_files = sorted(OUTPUTS.glob("*"))
    primary_deliverables = sorted(
        path
        for path in (
            ROOT / "1104Manuscript.docx",
            OUTPUTS / "1104Manuscript.pdf",
        )
        if path.is_file()
    )
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "software": {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "torch": torch.__version__,
            "xgboost": xgboost.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "seeds": [1, 7, 42, 123, 2024],
        "outer_folds": 5,
        "bootstrap_replicates": 2000,
        "primary_landmark_hours": 24,
        "secondary_landmarks_hours": [6, 72],
        "sensitivity_scenarios": 16,
        "commands": ANALYSIS_COMMANDS,
        "inputs": hashed_files([ROOT / path for path in KEY_INPUTS]),
        "results": {
            "prediction_assets": hashed_files(prediction_files),
            "metric_assets": hashed_files(metric_files),
            "audit_assets": hashed_files(audit_files),
            "calibration_assets": hashed_files(calibration_files),
            "ordinal_evaluation_assets": hashed_files(ordinal_files),
            "table_assets": hashed_files(table_files),
            "figure_assets": hashed_files(figure_assets),
            "figure_source_assets": hashed_files(figure_source_files),
            "document_assets": hashed_files(document_files),
            "primary_deliverable_assets": hashed_files(
                primary_deliverables
            ),
        },
        "notes": [
            "Raw PhysioNet databases are read-only and are not included.",
            "Formal and smoke or archived results are kept in separate directories.",
            "Patient-grouped outer folds and a patient-level bootstrap are used in MIMIC.",
            "eICU site-level analyses use hospital_key as the resampling unit.",
        ],
    }
    path = RESULTS / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Reproducibility record",
        "",
        f"Generated: `{payload['generated_utc']}`",
        "",
        f"Python: `{payload['python'].splitlines()[0]}`",
        "",
        "## Fixed analysis settings",
        "",
        "- Primary landmark: ICU hour 24.",
        "- Outer validation: five patient-grouped stratified folds.",
        "- Model seeds: 1, 7, 42, 123, and 2024.",
        "- Uncertainty: 2,000 bootstrap replicates.",
        "- Primary metric: five-class one-vs-rest macro-AUC.",
        "",
        "## Software",
        "",
        "| Package | Version |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name} | `{version}` |"
        for name, version in payload["software"].items()
    )
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```text",
            *payload["commands"],
            "```",
            "",
            "## Input hashes",
            "",
            "| File | SHA-256 |",
            "|---|---|",
        ]
    )
    lines.extend(
        f"| `{name}` | `{entry['sha256']}` |"
        for name, entry in payload["inputs"].items()
    )
    lines.extend(
        [
            "",
            "Raw MIMIC-IV, MIMIC-III, and eICU-CRD records remain under their "
            "credentialed PhysioNet access terms and are not redistributed.",
        ]
    )
    (OUTPUTS / "reproducibility_record.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote {path}")
    print(f"Wrote {OUTPUTS / 'reproducibility_record.md'}")


if __name__ == "__main__":
    main()

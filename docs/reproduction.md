# Reproduction Guide

## Environment

The analysis used Python 3.10.11 on CPU. Install the locked dependencies from
`requirements.txt`.

## Local Data Layout

Credentialed source files must be obtained separately from PhysioNet. Processed
cohorts are written under `data/processed/`, which is excluded from Git.

Expected processed datasets include:

- `data/processed/mimic4/`
- `data/processed/mimic4_6h/`
- `data/processed/mimic4_72h/`
- `data/processed/mimic3/`
- `data/processed/eicu/`

## Core Analysis

Run commands from the repository root:

```powershell
py -3.10 -m src.ami_dynamic.build_cohorts --datasets mimic4 mimic3 eicu
py -3.10 work/derive_diagnosis_flags.py
py -3.10 work/audit_processed_cohorts.py
py -3.10 -m src.ami_dynamic.train_evaluate --dataset mimic4 --folds 5 --seeds 1 7 42 123 2024 --epochs 30 --models xgb_static xgb_static_ts mlp_static mlp_static_ts gru grud transformer stgcn_lite dggru_static dggru dggru_tr dggru_ctr --lambda-curvature 0.01 --transport --leakage-audit --no-resume
py -3.10 work/run_severity_benchmark.py --seeds 1 7 42 123 2024 --folds 5
py -3.10 -m src.ami_dynamic.train_evaluate --dataset eicu --folds 5 --seeds 1 7 42 123 2024 --epochs 30 --models xgb_static xgb_static_ts dggru_tr --no-resume
```

## Transport, Calibration, and Ordinal Metrics

```powershell
py -3.10 work/analyze_external.py
py -3.10 work/run_transport_sensitivity.py
py -3.10 work/run_subgroup_analysis.py
py -3.10 work/run_ordinal_metrics_external_dca.py --bootstrap-replicates 2000
py -3.10 work/run_external_calibration_common3.py --replicates 2000
```

## Temporal Controls

```powershell
py -3.10 work/run_temporal_controls.py --seed 42
foreach ($seed in 1,7,42,123,2024) { py -3.10 work/run_temporal_controls.py --seed $seed --skip-controls --lambdas 0.01 1.5 }
py -3.10 work/build_multiseed_temporal_summary.py
py -3.10 work/run_cumulative_temporal_path.py --seed 42 --folds 5 --epochs 30
```

## Sensitivity Analyses

```powershell
py -3.10 work/run_sensitivity_models.py --analysis six_level --model xgb_static_ts
py -3.10 work/run_sensitivity_models.py --analysis primary_position --model xgb_static_ts
py -3.10 work/run_sensitivity_models.py --analysis exclude_type2 --model xgb_static_ts
py -3.10 work/run_sensitivity_models.py --analysis exclude_against_advice --model xgb_static_ts
py -3.10 work/run_sensitivity_models.py --analysis exclude_troponin --model xgb_static_ts
py -3.10 work/build_landmark_cohorts.py --landmarks 6 72
py -3.10 work/run_additional_sensitivities.py --analyses pre24h_revascularization pre24h_vasopressor_mcs label_merge_rehab_facility --bootstrap 2000 --overwrite
py -3.10 work/run_additional_sensitivities.py --analyses multiple_imputation --imputations 3 --mi-max-iter 1 --bootstrap 2000 --overwrite
```

## Audits, Tables, and Figures

```powershell
py -3.10 work/audit_count_feature_ablation.py
py -3.10 work/audit_eicu_hospital_cluster_cv.py
py -3.10 work/build_site_heterogeneity_report.py
py -3.10 work/build_inclusion_comparison.py
py -3.10 work/make_tables.py
py -3.10 work/make_figures.py
py -3.10 work/audit_sign_conventions.py
py -3.10 work/verify_prediction_cache.py --temporal-controls
py -3.10 work/build_run_manifest.py
py -3.10 -m pytest -q
```

The repository omits row-level predictions and site-identifiable outputs.
Recreating them requires the credentialed source datasets.

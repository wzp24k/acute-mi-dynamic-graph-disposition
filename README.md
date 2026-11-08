# Invariant Continuous-Time Organ Graph Learning

This project evaluates continuous-time organ graph learning and complementary
temporal ensembling for a diagnosis-coded MACCE phenotype after a 24-hour ICU
landmark in coronary atherosclerosis, with development in MIMIC-IV and
transport to MIMIC-III and eICU-CRD.

The public repository contains analysis code, tests, configuration files,
non-sensitive aggregate results, and R figure outputs. It does not contain the
manuscript, credentialed patient-level records, processed patient tables, or
prediction caches.

The complementary temporal ensemble is explicitly exploratory because it was
formed after observing the organ-graph result. The original equal-weight
XGB+TS plus Transformer model achieved a development AUROC of 0.764 and a
2020-2022 AUROC of 0.745. A second-stage model added organ adverse-slope,
recovery-direction, range, ratio, and observation-fraction features. Its
feature set and weight were selected on 2008-2016 versus 2017-2019 and then
frozen. The locked 75% enhanced-XGBoost plus 25% Transformer model achieved
2020-2022 AUROC 0.7462 versus 0.7337 for original XGB+TS, a gain of 0.0125
(95% CI 0.0073-0.0172). Strict nested XGBoost tuning of the earlier ensemble
gave a gain of 0.0098, narrowly below the 0.01 material threshold. The
seven-organ token graph reached 0.750 and the original full CT-OGT reached
0.756; neither supported a material organ-graph increment.
Joint nested XGBoost and Transformer selection gave a positive gain of 0.0073
(95% CI 0.0043-0.0106), still below the material threshold.

## Critical endpoint boundary

The main endpoint is a diagnosis-coded 7-day MACCE **proxy**. MIMIC diagnosis
tables do not contain event-onset times, and MIMIC-III procedure records do not
contain procedure dates. Death is timestamped, MIMIC-IV revascularization is
dated to the procedure day, and eICU diagnoses carry offsets. The timestamped
sensitivity endpoint is stored separately as `macce7_timestamped`. Absolute
performance from the coded endpoint should not be described as fully
adjudicated 7-day MACCE performance.

A separate interval-censored sensitivity endpoint is stored as
`interval_endpoint.csv` in each processed cohort. MIMIC-IV contains 45 strongly
confirmed events, too few for reliable model inference, and 1,265 events with a
specific anchor interval overlapping the target window. Retrained XGB+TS
achieved an AUROC of 0.795 for the anchor-overlap endpoint. This is a
sensitivity phenotype, not event-onset truth.

The 38-record structured chart-review pilot was AI-assisted and is not
independent clinician adjudication. It is stored as
`results/manual_endpoint_review_pilot.csv`; the 100-record MIMIC review pack
has a two-reader schema in `results/manual_endpoint_review_structured.csv`.
Human PPV, timing error, and Cohen kappa remain pending and are not imputed.

## Data sources

- MIMIC-IV v3.1: `D:\TT paper\0811Temporal Pathway\mimic-iv-3.1`
- MIMIC-III v1.4: `D:\TT paper\0811Temporal Pathway\mimic-iii-clinical-database-1.4_1`
- eICU-CRD subset: `D:\TT paper\0811Temporal Pathway\eicu`

The primary eICU transport cohort allows a CAD diagnosis in any diagnosis
position (`n=5,913`, 642 events, 166 sites). The original principal-diagnosis
eICU cohort (`n=903`, 57 events) is retained as a sensitivity analysis.
A hospital-held-out local model in expanded eICU achieved AUROC 0.683 across
160 sites. This is a cross-hospital local-learning result, not transport of the
MIMIC-IV model.
The enhanced XGBoost improved MIMIC-III transport AUROC to 0.670, and its fixed
75/25 fusion with Transformer reached 0.671.

The eICU extract available in this workspace does not contain treatment,
medication, or intake/output tables. Urine output is therefore retained as a
missing channel in eICU and is not part of the minimum-channel requirement.

## Reproduction

Run from this directory in PowerShell:

```powershell
$env:PYTHONPATH = "src"
py -m ctogt.audit_endpoint --database all
py -m ctogt.build_cohorts --database all
py -m ctogt.train_evaluate --mode cv --models primary --folds 5 --seeds 7
py -m ctogt.train_evaluate --mode cv --models controls --folds 5 --seeds 7
py -m ctogt.train_evaluate --mode external --models xgb_static_ts,transformer,ctogt --seeds 7 --external mimic3,eicu
py -m ctogt.train_evaluate --mode external --models xgb_static_ts,transformer --seeds 7 --external eicu_anycad
py -m ctogt.temporal_holdout
py -m ctogt.nested_selection
py -m ctogt.nested_transformer_selection
py -m ctogt.strong_temporal_holdout
py -m ctogt.strong_temporal_holdout_fusion
py -m ctogt.strong_tabular_candidates
py -m ctogt.strong_fusion_triple
py -m ctogt.organ_dynamics_holdout
py -m ctogt.interval_threshold_sensitivity
py -m ctogt.clinical_utility_analysis
py -m ctogt.sensitivity
py -m ctogt.build_manual_endpoint_review
py work\build_structured_endpoint_review.py
py work\score_manual_endpoint_review.py
py -m ctogt.make_tables_figures
py -m ctogt.make_manuscript
```

Tabular development baselines can be repeated over multiple seeds:

```powershell
py -m ctogt.train_evaluate --mode cv --models logit_static,xgb_static,xgb_static_ts --folds 5 --seeds 7,42,2026
```

## Outputs

- Cohort flow and endpoints: `results/endpoint_feasibility.json`
- Processed trajectories: `data/processed/{mimic4,mimic3,eicu}`
- Fold-level predictions: `results/predictions`
- Development and transport metrics: `results/metrics_development.json`, `results/metrics_external.json`
- Locked temporal holdout: `results/temporal_holdout_validation.json`
- Internal-era locked strong model: `results/strong_temporal_holdout_fusion.json`
- Enhanced external transport: `results/strong_external_transport.json`
- Expanded eICU hospital-held-out model: `results/eicu_site_holdout.json`
- Nested selection and utility intervals: `results/nested_selection_and_utility.json`
- Endpoint chart-review artifacts: `results/manual_endpoint_review*.csv`
- Interval-rule sensitivity: `results/interval_threshold_sensitivity.json`
- Calibration and threshold utility: `results/clinical_utility_analysis.json`
- Main tables and figures: `results/tables`, `results/figures`
- Main manuscript tables: 1-8; figures: 1-7
- Manuscript: `outputs/CTOGT_MACCE_manuscript.docx`
- Supplement: `outputs/CTOGT_MACCE_supplement.docx`

## Tests

```powershell
$env:PYTHONPATH = "src"
py -m pytest tests -q
```

The tests cover the static prior graph, continuous-time features, and output
shape contracts of the full CT-OGT model.

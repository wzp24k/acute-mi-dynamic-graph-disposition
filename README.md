# Acute MI Ordinal Discharge-Disposition Prediction

This repository contains the analysis code and non-sensitive aggregate results
for a retrospective study of ordinal discharge disposition after acute
myocardial infarction in intensive care. The primary landmark is ICU hour 24.

The repository is private during peer review. It excludes the manuscript,
supplementary manuscript files, credentialed patient-level records, prediction
caches, and site-identifiable eICU outputs.

## Study Scope

- Development: MIMIC-IV v3.1, 6,487 patients.
- Secondary same-institution validation: MIMIC-III v1.4, 4,750 patients.
- Multicentre transport: eICU-CRD v2.0, 6,696 ICU units.
- Primary outcome: five-level discharge-disposition scale in MIMIC.
- Transport outcome: harmonized four-class scale in eICU.
- Primary comparison: DG-GRU+TR versus XGB-static+TS.
- Regularization comparison: DG-GRU+TR versus DG-GRU.

The study defines disposition as a care-placement endpoint. It is not a
validated functional-recovery scale.

## Key Aggregate Findings

- XGB-static+TS achieved a development macro-AUC of 0.778.
- DG-GRU+TR achieved a development macro-AUC of 0.729.
- The paired difference was -0.049 (95% CI -0.057 to -0.042).
- Adding temporal summaries increased XGB macro-AUC by 0.028.
- Temporal regularization changed DG-GRU performance by about 0.001.
- eICU patient-grouped macro-AUC was 0.807.
- eICU patient-safe hospital-cluster macro-AUC was 0.784.
- Grouped out-of-fold recalibration restored slopes near one.

Signed contrasts always follow the named first model minus the named comparison
model. Null and unfavorable estimates are retained without sign transformation.

## Repository Map

- `src/ami_dynamic/`: cohort construction, features, models, training and
  evaluation.
- `work/`: analysis, sensitivity, audit, table and figure scripts.
- `tests/`: focused unit and integration tests.
- `configs/study.json`: locked analysis configuration.
- `results/tables/`: aggregate manuscript tables and source data.
- `results/audits/`: cohort, leakage, calibration and sensitivity audits.
- `results/calibration/`: aggregate external-calibration outputs.
- `results/ordinal_evaluation/`: ordinal metrics, threshold and DCA outputs.
- `results/sensitivity/`: aggregate sensitivity results.
- `results/figures/`: publication figures and non-identifiable source data.
- `docs/reproduction.md`: analysis command sequence.

## Data Access

MIMIC-IV, MIMIC-III, and eICU-CRD are available from PhysioNet under
credentialed access. No credentialed records are redistributed here.

Users must obtain independent data-use approval before reproducing the
analysis. Processed cohorts are generated locally and remain outside version
control.

## Quick Start

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m pytest -q
```

After placing credentialed source data in the expected local paths, follow
`docs/reproduction.md`.

## Verification

The tests cover outcome mapping rules, ordinal metrics, temporal masking,
calibration utilities, and cohort integrity checks. Aggregate outputs include
seed-level estimates and bootstrap intervals where applicable.

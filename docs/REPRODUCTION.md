# Reproduction guide

## Data access

Request credentialed access to MIMIC-IV v3.1, MIMIC-III v1.4, and eICU-CRD
through PhysioNet. Place the source extracts at the paths defined in
`src/ctogt/common.py`, or update those path constants for the local system.

## Python environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

## Cohort construction

```powershell
py -m ctogt.audit_endpoint --database all
py -m ctogt.build_cohorts --database all
py -m ctogt.build_interval_endpoint --database all
```

## Development and transport

```powershell
py -m ctogt.train_evaluate --mode cv --models primary --folds 5 --seeds 7
py -m ctogt.train_evaluate --mode cv --models controls --folds 5 --seeds 7
py -m ctogt.train_evaluate --mode external --models xgb_static_ts,transformer,ctogt --seeds 7 --external mimic3,eicu
py -m ctogt.train_evaluate --mode external --models xgb_static_ts,transformer --seeds 7 --external eicu_anycad
```

## Strengthening and sensitivity analyses

```powershell
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
py -m ctogt.eicu_site_holdout
```

## Figures

Figure source data are exported with the Python script, and all figures are
drawn in R.

```powershell
py work\export_r_figure_data.py
& "C:\Program Files\R\R-4.6.2\bin\Rscript.exe" work\replot_figures.R
```

## Verification

```powershell
py work\verify_reference_registry.py
py work\finalize_reference_registry.py
py work\audit_1108_requirements.py
py work\peer_review_1108.py
py -m pytest tests -q
```

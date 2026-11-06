# AMI Project Status

Date: 2026-11-05

Project: Temporal Regularization of a Dynamic Graph Network for Ordinal
Prediction of Discharge Disposition After Acute Myocardial Infarction in
Intensive Care

## Locked cohorts

| Database | Role | Analytic n | Outcome scale |
|---|---|---:|---|
| MIMIC-IV v3.1 | Development, ICU hour 24 | 6,487 | Five-level ordinal |
| MIMIC-IV 6-hour landmark | Independent sensitivity cohort | 7,514 | Five-level ordinal |
| MIMIC-IV 72-hour landmark | Independent sensitivity cohort | 2,962 | Five-level ordinal |
| MIMIC-III v1.4 | Secondary same-institution validation | 4,750 | Five-level ordinal |
| eICU-CRD v2.0 | Multicentre transport | 6,696 | Harmonized four-class |

The 6-hour and 72-hour cohorts were reconstructed from source records with
landmark-specific eligibility, features, trajectories and outcomes. They were
not created by truncating the primary 24-hour sequences.

## Completed analyses

- All prespecified model comparisons at the primary 24-hour landmark.
- Strict factorial controls: SeqOrdinal GRU versus DG-GRU with the same raw
  sequence-plus-availability tensor, and DG-GRU versus DG-GRU-static with the
  same graph-recurrent implementation but the hourly values and masks zeroed.
- External calibration in the main manuscript: raw and out-of-fold
  recalibrated slopes, intercepts and Brier scores for MIMIC-III and eICU
  (Table 7 and Figure 7), with eICU resampling at the patient-safe
  hospital-cluster level.
- A pragmatic common three-class outcome evaluated in MIMIC-IV, MIMIC-III and
  eICU (Table 8). Class 1 is explicitly reported as non-construct-identical
  harmonization.
- Secondary cumulative-curvature temporal regularization at
  `lambda_c=0.01`, with a six-point curvature path and matched controls.
- Five patient-grouped outer folds crossed with seeds 1, 7, 42, 123 and 2024.
- Paired patient-level bootstrap intervals, calibration, DeLong tests,
  NRI/IDI and decision-curve summaries.
- Eighteen-point temporal-regularization path and matched controls.
- Independent 6-hour and 72-hour landmark analyses.
- Sixteen prespecified sensitivity scenarios, including multiple imputation,
  pre-24-hour revascularization, vasoactive/MCS support and merged
  disposition labels.
- XGB-static+TS feature gain and held-out permutation importance using all
  154 locked predictors.
- Included-versus-excluded comparison with standardized differences.
- MIMIC-III secondary same-institution validation, eICU transport and eICU internal
  cross-validation.
- eICU hospital-level heterogeneity, patient-safe leave-one-cluster-out
  stability and patient-safe cluster bootstrap.
- Prespecified subgroup descriptions.

## Main findings

- XGB-static+TS achieved the highest development five-class macro-AUC
  (0.7764), followed by XGB-static (0.7491); DG-GRU+TR achieved 0.7261.
- DG-GRU+TR minus XGB-static+TS was -0.0503 (95% CI -0.0583 to -0.0422).
  The dynamic graph model did not close the discrimination gap to the
  summary-feature benchmark.
- DG-GRU+TR minus DG-GRU was +0.0000094 (95% CI -0.000625 to +0.000647),
  consistent with a null temporal-regularization effect.
- DG-GRU minus DG-GRU-static was -0.0007117, and SeqOrdinal GRU minus DG-GRU
  was +0.0009165. These strict factorial controls do not show an advantage
  for the dynamic graph structure.
- DG-GRU+CTR minus DG-GRU was -0.0008911. The redesigned cumulative-curvature
  term was secondary and did not improve discrimination.
- XGB-static+TS minus XGB-static was +0.0272 (95% CI +0.0223 to +0.0326),
  supporting predictive value for compact temporal summaries.
- Direct transport was not uniformly favorable. eICU four-class macro-AUC was
  0.7114 for XGB-static+TS, 0.7229 for XGB-static and 0.7287 for DG-GRU+TR.
- The DG-GRU+TR eICU hospital-level macro-AUC median was 0.742 (IQR
  0.704-0.767) across the 35 sites where the four-class macro-AUC was
  estimable. Patient-safe leave-one-cluster-out pooled estimates ranged
  from 0.7263 to 0.7309.
- The 12 largest included-versus-excluded differences were dominated by
  lower pre-landmark laboratory availability among excluded patients. Among
  patients with lactate measured, the excluded group had higher first lactate
  (median 2.20 vs 1.90 mmol/L; SMD -0.482 for included minus excluded).

## Scientific boundaries

- The endpoint is hospital care placement, not functional recovery, patient
  preference or treatment benefit.
- The primary estimand conditions on survival and continued intensive care at
  ICU hour 24.
- MIMIC-III is secondary same-institution validation, not an independent
  temporal or external cohort; shifted dates do not establish window
  exclusivity.
- eICU rehabilitation and death are not construct-identical to the MIMIC
  home-health/rehabilitation composite and hospice category.
- The common three-class scale is a pragmatic harmonization for transport
  evaluation. Its middle class combines non-home post-acute and other care
  and is not an identical clinical construct across databases.
- Direct transport, calibration and hospital-level summaries are affected by
  case mix, measurement and care-resource differences.
- Raw external probabilities were not uniformly calibrated. Out-of-fold
  recalibration improved the probability scale but requires outcome data from
  the target database.
- Treatment variables are pre-landmark proxies, not causal exposures.
- No prospective clinical deployment, workflow evaluation or fairness
  analysis was performed.
- Negative, null and uncertain estimates are retained. No result was
  sign-flipped, absolute-transformed or replaced after inspection.

## Main deliverables

- `1104Manuscript.docx`
- `outputs/1104Manuscript.pdf`
- `outputs/AMI_Dynamic_Graph_Manuscript.md`
- `outputs/AMI_Dynamic_Graph_Manuscript.docx`
- `outputs/AMI_Dynamic_Graph_Manuscript.pdf`
- `outputs/AMI_Dynamic_Graph_Supplementary_Material.docx`
- `outputs/AMI_Dynamic_Graph_Supplementary_Material.pdf`
- `outputs/TRIPOD_AI_checklist.docx`
- `outputs/TRIPOD_AI_checklist.pdf`
- `outputs/AMI_Code_and_Reproduction.zip`
- `results/run_manifest.json`

## Key audits

- `results/audits/cohort_audit.json`
- `results/audits/leakage_audit.json`
- `results/audits/sign_convention_audit.json`
- `results/audits/p1_reviewer_items_audit.json`
- `results/audits/prediction_cache_audit.json`
- `results/audits/docx_delivery_audit.json`
- `results/ordinal_evaluation/ordinal_evaluation_report.json`
- `results/audits/eicu_site_heterogeneity_report.json`
- `results/audits/inclusion_comparison.json`
- `results/audits/tableS11_feature_importance_audit.json`
- `work/outline_compliance_matrix.md`

## Reproduction

Python 3.10.11, CPU training. Exact commands, software versions, input hashes
and result hashes are recorded in `results/run_manifest.json` and
`outputs/reproducibility_record.md`. Raw PhysioNet patient records are not
redistributed.

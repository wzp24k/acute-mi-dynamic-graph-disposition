# Table S11. Feature importance for XGB-static+TS

Feature importance was estimated in the locked MIMIC-IV analytic cohort using the same 154-variable XGB-static+TS feature contract and five-fold patient-grouped, outcome-stratified outer cross-validation used by the primary model comparison. Normalized gain was calculated within each training fold and averaged across the five fitted models.

Permutation importance was evaluated only on the held-out outer test rows. Each predictor and each prespecified variable group was permuted five times per fold; the reported estimand is the decrease in five-class one-vs-rest macro-AUC. Variable-group permutation preserved within-group feature correlations by permuting the group rows jointly. Intervals use the five fold-level permutation means. Negative values reflect permutation noise and were retained rather than truncated.

## Audit

| Audit item                            | Result                                |
|:--------------------------------------|:--------------------------------------|
| MIMIC-IV analytic cohort              | 6,487 admissions                      |
| Patients                              | 6,487                                 |
| Model features                        | 154                                   |
| Variable groups                       | 5                                     |
| Temporal horizon                      | Hours 0-23; no post-landmark features |
| Outer validation                      | 5-fold patient-grouped stratified CV  |
| Cross-validation seed                 | 42                                    |
| Permutation repeats                   | 5                                     |
| Permutation base seed                 | 1104                                  |
| XGBoost threads                       | 4                                     |
| Locked feature contract               | PASS                                  |
| Forbidden outcome/post-landmark names | PASS (none found)                     |
| Duplicate predictors                  | PASS (none found)                     |
| Patient-disjoint outer folds          | PASS                                  |
| Mean outer-fold baseline macro-AUC    | 0.7725                                |

## Variable-group importance

| Variable group       |   Features |   Normalized gain (%) | Permutation drop (95% CI)    |
|:---------------------|-----------:|----------------------:|:-----------------------------|
| Demographics         |         14 |                  9.34 | +0.0386 (+0.0023 to +0.0750) |
| Troponin / CK-MB     |         11 |                  5.28 | +0.0372 (-0.0009 to +0.0754) |
| Other laboratories   |         80 |                 45.76 | +0.0366 (-0.0020 to +0.0751) |
| Vital-sign summaries |         42 |                 22.16 | +0.0438 (+0.0024 to +0.0853) |
| GCS / neuro          |          7 |                 17.46 | +0.0376 (-0.0006 to +0.0759) |

## Top 20 features by outer-fold permutation importance

| Feature                      | Variable group       |   Macro-AUC drop |   95% CI low |   95% CI high |   Gain (%) |
|:-----------------------------|:---------------------|-----------------:|-------------:|--------------:|-----------:|
| age                          | Demographics         |           0.0363 |         0.03 |   0.0382476   |   1.54824  |
| gcs_last                     | GCS / neuro          |           0.0139 |         0.01 |   0.0158322   |   3.01003  |
| gcs_mean                     | GCS / neuro          |           0.0114 |         0.01 |   0.0157707   |   2.95286  |
| care_unit_cvicu              | Demographics         |           0.0074 |         0.01 |   0.00934837  |   2.28718  |
| gcs_count                    | GCS / neuro          |           0.0059 |         0.00 |   0.00949357  |   0.936317 |
| care_unit_micu               | Demographics         |           0.0050 |         0.00 |   0.0076704   |   0.808263 |
| gcs_max                      | GCS / neuro          |           0.0026 |         0.00 |   0.00416376  |   1.66947  |
| gcs_change                   | GCS / neuro          |           0.0024 |         0.00 |   0.00416415  |   4.34972  |
| lab_blood_urea_nitrogen_mean | Other laboratories   |           0.0014 |         0.00 |   0.00234731  |   1.45453  |
| lab_blood_urea_nitrogen_max  | Other laboratories   |           0.0011 |         0.00 |   0.00147383  |   1.67095  |
| lab_hemoglobin_max           | Other laboratories   |           0.0011 |         0.00 |   0.00152223  |   0.802679 |
| lab_hemoglobin_min           | Other laboratories   |           0.0009 |        -0.00 |   0.00245628  |   1.3065   |
| rr_mean                      | Vital-sign summaries |           0.0009 |         0.00 |   0.00148132  |   0.784502 |
| lab_albumin_min              | Other laboratories   |           0.0007 |         0.00 |   0.00111888  |   0.664956 |
| lab_albumin_mean             | Other laboratories   |           0.0005 |        -0.00 |   0.00127435  |   0.72737  |
| lab_lactate_mean             | Other laboratories   |           0.0005 |        -0.00 |   0.00199133  |   1.18436  |
| hr_last                      | Vital-sign summaries |           0.0005 |        -0.00 |   0.00124345  |   0.69789  |
| dbp_mean                     | Vital-sign summaries |           0.0005 |        -0.00 |   0.0016165   |   0.855946 |
| sbp_first                    | Vital-sign summaries |           0.0005 |         0.00 |   0.000903222 |   0.422713 |
| lab_sodium_mean              | Other laboratories   |           0.0005 |        -0.00 |   0.000939094 |   0.565536 |

## Top 15 features by normalized XGBoost gain

| Feature                       | Variable group     |   Gain (%) |   Macro-AUC drop |
|:------------------------------|:-------------------|-----------:|-----------------:|
| gcs_change                    | GCS / neuro        |    4.34972 |      0.00243455  |
| gcs_last                      | GCS / neuro        |    3.01003 |      0.0138935   |
| gcs_mean                      | GCS / neuro        |    2.95286 |      0.0114128   |
| gcs_min                       | GCS / neuro        |    2.78936 |      0.000451926 |
| care_unit_cvicu               | Demographics       |    2.28718 |      0.00737603  |
| gcs_first                     | GCS / neuro        |    1.75172 |      0.000111766 |
| lab_blood_urea_nitrogen_max   | Other laboratories |    1.67095 |      0.00109433  |
| gcs_max                       | GCS / neuro        |    1.66947 |      0.00258457  |
| age                           | Demographics       |    1.54824 |      0.0362653   |
| lab_blood_urea_nitrogen_mean  | Other laboratories |    1.45453 |      0.00135204  |
| lab_bicarbonate_min           | Other laboratories |    1.31301 |     -7.57606e-07 |
| lab_hemoglobin_min            | Other laboratories |    1.3065  |      0.000913302 |
| lab_blood_urea_nitrogen_first | Other laboratories |    1.28164 |      0.000160914 |
| lab_total_bilirubin_max       | Other laboratories |    1.1948  |      0.000257634 |
| lab_lactate_mean              | Other laboratories |    1.18436 |      0.000518578 |

Gain measures how a feature was used by the fitted trees and is not a causal effect. Permutation importance measures predictive dependence on the held-out fold and can be attenuated when correlated variables have interchangeable information.

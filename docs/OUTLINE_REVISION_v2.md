# Outline revision v2

## Reason for revision

The original outline assigned a 7-day MACCE endpoint to all three databases.
The data audit shows that event timing is not equally observable:

- MIMIC death time is exact.
- MIMIC-IV revascularization is available at procedure-day resolution.
- MIMIC-III revascularization codes have no procedure date.
- MIMIC diagnosis codes have no MI/stroke onset timestamps.
- eICU has diagnosis offsets, but these are documentation-entry offsets rather
  than validated clinical onset times.
- The available eICU extract has no notes, physical examination, past history,
  treatment, or medication tables.

## Revised endpoint hierarchy

### Co-primary endpoint A: 7-day all-cause death

This endpoint is the cleanest common time-resolved outcome. It is clinically
unambiguous and can be compared across all three databases.

### Co-primary endpoint B: diagnosis-coded MACCE phenotype

The phenotype combines acute MI, stroke, death, and revascularization recorded
during the index admission. It must always be described as "coded" and is not
equivalent to adjudicated incident MACCE after the landmark.

### Sensitivity endpoint: timed diagnosis-offset MACCE

For eICU, use `diagnosisoffset > 1440` and `<= 10080` minutes for MI and
stroke. For MIMIC-IV, use death and dated revascularization. For MIMIC-III,
only death is reliably timed. This endpoint is expected to have low external
event counts and is not the primary basis for clinical deployment claims.

## Revised research questions

RQ1: Does 24-hour temporal information improve prediction of the coded MACCE
phenotype over static severity?

RQ2: Does a complementary temporal ensemble improve discrimination over
XGBoost with temporal summaries?

RQ3: Does a dynamic organ graph add discrimination or transport value beyond
the ensemble?

RQ4: Are gains preserved in external databases after local recalibration?

RQ5: Which results remain when the endpoint is restricted to time-resolved
death or diagnosis-offset events?

## Revised hypothesis labels

- H1: Temporal summaries improve over static severity.
- H2: A complementary temporal ensemble improves over XGB+TS.
- H3: Organ-graph structure adds value beyond complementary temporal models.
- H4: Environment-invariant regularization improves external calibration or
  transport.
- H5: Gains are not explained by random graphs, reversed time, permuted time,
  or omission of continuous-time encoding.

## Revised model strategy

The temporal ensemble should combine XGBoost using temporal summaries and a
temporal Transformer with equal weights. It was locked after the organ-graph
hypothesis failed and before the 2020-2022 MIMIC-IV temporal holdout labels
were used. It avoids external-label tuning and is reported with paired
bootstrap against XGB+TS, strict nested XGBoost selection, and clinical-utility
confidence intervals.

CT-OGT remains the mechanistic model for testing organ graph structure and
environment invariance. Its negative or positive result is reported separately
from the ensemble.

## Revised sample-size and transport rules

- Development requires at least 150 coded MACCE events.
- eICU requires at least 75 coded MACCE events for stable external inference.
  The expanded any-position cohort has 642 events; the original principal-
  diagnosis cohort has 57 and remains descriptive.
- Sites with fewer than 30 patients or zero events are not used for
  site-specific performance claims.
- Recalibration is reported separately from discrimination.

## Revised title options

1. Invariant Continuous-Time Organ Graph Learning with Complementary Temporal
   Ensembling for Diagnosis-Coded MACCE in Coronary Atherosclerosis: A
   Three-Database Transport Study
2. Complementary Temporal Ensembling for Diagnosis-Coded MACCE in Coronary
   Atherosclerosis: A Three-Database Transport Study
3. Continuous-Time Organ Graph Learning and Complementary Temporal Ensembling
   for MACCE in Coronary Atherosclerosis: A Three-Database Benchmark

Option 1 preserves the original scientific focus while making the ensemble and
the coded endpoint visible.

## Final implemented decisions

- The main model is the equal-weight XGB+TS + three-seed Transformer temporal
  ensemble.
- CT-OGT is retained as the mechanistic organ-graph model and ablation target.
- The main endpoint is explicitly diagnosis-coded.
- Seven-day all-cause death is retained as the cleanest time-resolved outcome.
- An interval-censored endpoint is included as a sensitivity analysis.
- The primary temporal result is positive but marginal: ensemble AUROC 0.764
  versus 0.754 for XGB+TS, paired difference 0.0102 (95% CI 0.0077-0.0127).
- The locked 2020-2022 temporal holdout reproduced the gain: 0.7449 versus
  0.7337, difference 0.0112 (95% CI 0.0052-0.0166).
- A second-stage exploratory model added organ adverse slopes, recovery
  directions, ranges, ratios, and observation fractions. Features and a
  75/25 enhanced-XGBoost/Transformer weight were selected on 2017-2019 and
  frozen. The 2020-2022 AUROC was 0.7462 versus 0.7337 for XGB+TS,
  difference 0.0125 (95% CI 0.0073-0.0172).
- Strict nested XGBoost tuning attenuated the gain to 0.0098 (95% CI
  0.0071-0.0124), narrowly below the 0.01 material threshold.
- Joint nested XGBoost and Transformer selection gave a gain of 0.0073
  (95% CI 0.0043-0.0106), also positive but below 0.01.
- The organ-graph result is modest and below the 0.01 material threshold:
  full CT-OGT AUROC 0.756 versus 0.754 for XGB+TS.
- The redesigned seven-organ token graph achieved AUROC 0.750 and did not
  rescue the organ-graph hypothesis.
- Interval endpoint results:
  - MIMIC-IV: 45 fully confirmed events, insufficient for inference.
  - MIMIC-IV overlap layer: 1,265 events, AUROC 0.795.
  - MIMIC-III overlap layer: 76 events, AUROC 0.755.
  - Expanded eICU overlap layer: 119 events, AUROC 0.668.
  - Original eICU overlap layer: 9 events, insufficient for inference.
- The 38-record endpoint chart-review pilot supports a target-window event in
  10 records but remains AI-assisted rather than independent clinical
  adjudication.
- A two-reader schema and scoring script for PPV, timing error, and Cohen kappa
  are complete; 100 human reviews remain pending.
- Troponin threshold 0.5-2.0 and eICU lag 0-12 hour sensitivity analyses do
  not change the main interval-endpoint conclusion.
- The enhanced 75/25 fusion improved MIMIC-III AUROC from 0.659 to 0.671
  (paired difference 0.0120, 95% CI 0.0058-0.0179). Expanded eICU direct
  transport was neutral (difference -0.0003), while hospital-held-out local
  learning reached AUROC 0.683.

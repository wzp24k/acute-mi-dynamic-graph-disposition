# Study Protocol and Analysis Contract

## 1. Working title

Invariant Continuous-Time Organ Graph Learning with Complementary Temporal
Ensembling for Diagnosis-Coded MACCE in Coronary Atherosclerosis: A
Three-Database Transport Study

## 2. Design

Retrospective, multi-database prediction study with a fixed 24-hour ICU
landmark. MIMIC-IV is the development database, MIMIC-III is the
same-institution temporal transport database, and eICU-CRD is the
multi-hospital external transport database. Reporting follows TRIPOD+AI.

## 3. Hypotheses

- H1: A 24-hour temporal representation improves discrimination over static
  severity.
- H2: The continuous-time dynamic organ graph adds at least 0.01 AUROC beyond
  XGB+TS.
- H3: Environment-invariant regularization improves external discrimination
  or calibration relative to the dynamic graph without invariance.
- H4: An exploratory ensemble of XGBoost with temporal summaries and a
  three-seed temporal Transformer improves AUROC by at least 0.01 over XGB+TS.
- H5: Observed temporal gains are not explained by hospital signatures, random graphs,
  reversed time, permuted time, or omission of continuous-time encoding.

## 4. Eligibility

- Age at least 18 years.
- Diagnosis of coronary atherosclerosis using ICD-10 I25.x or ICD-9 414.x.
- First eligible ICU stay per patient.
- Alive at ICU hour 24.
- At least six observed organ channels during hours 0-23.
- No outcome used before the prediction landmark.

Development and MIMIC-III require a principal diagnosis position. The primary
eICU transport analysis allows any diagnosis position to increase events and
site coverage; the original principal-diagnosis eICU population remains a
sensitivity analysis. ACS, chronic CAD, and prior-PCI/CABG groups are also
sensitivity populations. A subgroup defined by acute MI codes cannot be used
for outcome inference because those codes overlap the endpoint.

## 5. Endpoint

The composite includes all-cause death, acute myocardial infarction, ischemic
or hemorrhagic stroke, and urgent or unplanned revascularization. The intended
window is ICU hour 24 through day 7.

### 5.1 Actual ascertainment boundary

- MIMIC death time is exact.
- MIMIC-IV revascularization date is available at procedure-day resolution.
- MIMIC-III procedure tables contain codes but no procedure date.
- MIMIC diagnosis tables do not provide event-onset time.
- eICU diagnosis and treatment offsets provide time when the corresponding
  table is available.
- The available eICU extract lacks notes, treatment, and medication tables.

The main endpoint is therefore described as a **diagnosis-coded MACCE
phenotype**. A 7-day all-cause death endpoint is retained as a time-resolved
comparator. The `macce7_timestamped` sensitivity endpoint is retained for
death, MIMIC-IV revascularization, and eICU diagnosis-offset events. If a fully
timestamped common endpoint is unavailable, the main analysis is not allowed to
claim fully adjudicated 7-day MACCE ascertainment.

The eICU timing audit is documented in `docs/EICU_EVENT_TIMING_AUDIT.md`. The
available extract contains `apacheadmissiondx` and `diagnosisoffset` but no
notes, physical examination, past history, treatment, or medication tables.
`diagnosisoffset` is treated as a documentation-time proxy, not clinical onset.

The interval sensitivity endpoint counts a diagnosis-coded event when a
specific clinical, procedure, radiology, troponin, or eICU diagnosis-entry
anchor has a plausible interval overlapping ICU hours 24-168. A stricter
full-interval-inside-window stratum is retained for audit but is not used for
model inference when it has fewer than 150 events. Events with no specific
anchor are excluded from the anchor-overlap endpoint.

The interval endpoint remains a sensitivity analysis. Troponin assay thresholds
are varied at 0.5, 1.0, and 2.0 times the implemented cut-point, and eICU
diagnosis documentation lag is varied at 0, 3, 6, and 12 hours. Frozen
predictions are applied to every variant without model retuning.

The human adjudication pack uses two independent readers and records serial
troponin change, ECG evidence, angiography or PCI, acute stroke imaging, focal
neurologic deficit, clinical-record support, onset window, confidence, and a
consensus label. PPV, timing error, and Cohen kappa are calculated only after
both readers complete the form.

## 6. Predictors

Only measurements at or before ICU hour 24 are predictors. Seven organ systems
are represented by up to 20 channels:

- Neurologic: GCS, pupil reactivity
- Circulatory: heart rate, SBP, DBP, MBP, lactate
- Respiratory: respiratory rate, SpO2, FiO2
- Renal: creatinine, urine output
- Coagulation: platelets, INR, aPTT
- Inflammation: WBC, temperature
- Metabolic: glucose, pH, bicarbonate

Static predictors include age, sex, race, emergency admission, prior MI, prior
stroke, heart failure, atrial fibrillation, diabetes, hypertension, CKD, COPD,
PAD, smoking, prior PCI, prior CABG, and comorbidity count. Hospital identifiers
are never direct predictors.

## 7. Model

CT-OGT contains:

1. Observation-aware node features containing value, mask, elapsed time,
   absolute time, and their interaction.
2. A prior-constrained dynamic organ graph with same-organ and clinically
   specified cross-organ links.
3. Masked organ attention and a continuous-time temporal Transformer.
4. A static-context encoder and risk-trajectory head.
5. Temporal smoothness and drift penalties.
6. Graph-sparsity regularization.
7. Environment-balanced risk and gradient-reversed environment classification
   penalties.

An exploratory seven-organ token graph aggregates channels into neurologic,
circulatory, respiratory, renal, coagulation, inflammation, and metabolic
tokens. It tests whether channel-level graph dilution explains a negative
result and is reported as a negative or exploratory redesign, not a new
primary hypothesis after performance inspection.

An additional exploratory tabular representation adds organ adverse-slope
counts, recovery directions, measurement ranges, clinical ratios, and observed
organ fractions. Its candidate features and hyperparameters are selected only
on 2008-2016 development versus 2017-2019 internal temporal validation. The
locked model is then evaluated once in 2020-2022.

## 8. Comparators

- Logistic regression using static variables.
- Static XGBoost.
- XGBoost using static variables and 24-hour temporal summaries (XGB+TS).
- GRU.
- Transformer.
- TFT-lite.
- Equal-weight temporal ensemble of XGB+TS and Transformer.
- Continuous-time fixed graph.
- Continuous-time dynamic graph without environment invariance.
- Full CT-OGT.
- Seven-organ token dynamic graph.

Negative controls:

- Random graph.
- Reversed time.
- Permuted hour order.
- No continuous-time encoding.

## 9. Cross-validation and transport

- Development: stratified five-fold cross-validation.
- Nested selection: three-fold inner XGBoost tuning inside each outer fold.
- Nested Transformer sensitivity: two candidate configurations selected by an
  inner validation split inside each outer fold.
- Locked temporal holdout: 2020-2022 MIMIC-IV anchor-year periods, frozen
  after 2008-2019 development predictions are fixed.
- Second-stage era lock: fit candidate models on 2008-2016, select features,
  hyperparameters, and fusion weight on 2017-2019, refit on 2008-2019, and test
  in 2020-2022.
- Tabular baselines: repeat across three seeds.
- Deep models and the temporal ensemble components: five folds across three
  seeds.
- External transport: fit in MIMIC-IV without external labels and apply
  directly to MIMIC-III and eICU.
- External calibration: cross-fitted Platt scaling within the external
  database. eICU calibration folds are grouped by hospital where possible.
- eICU sites are summarized only when they contain at least 30 patients.

## 10. Metrics

Primary metric: AUROC.

Secondary metrics:

- AUPRC
- Brier score
- Log loss
- Calibration slope and intercept
- Expected calibration error
- Top-decile sensitivity and specificity
- Decision-curve net benefit

Model contrasts use patient-level paired bootstrap confidence intervals. The
material AUROC threshold is 0.01 and the confidence interval must exclude zero.
Decision-curve net benefit is reported with bootstrap confidence intervals from
threshold probabilities 0.05 through 0.30.
Brier score and sensitivity, specificity, PPV, and NPV at fixed 0.10 and 0.15
reporting thresholds are also reported with bootstrap intervals before
and after local recalibration.

The primary model candidate is the equal-weight XGB+TS + Transformer ensemble,
locked before temporal-holdout evaluation. CT-OGT and its ablations are
mechanism analyses rather than the primary deployment model.

After the exploratory era-locked analysis selected enhanced organ-dynamics
features, the fixed 75/25 enhanced-XGBoost/Transformer model was also applied
unchanged to MIMIC-III and eICU. A separate hospital-held-out model was fit
only inside expanded eICU and is labeled local cross-hospital learning, not
cross-database transport.

## 11. Go/No-Go rules

- Development should have at least 150 events.
- eICU should have at least 75 events for stable external discrimination. The
  expanded any-position cohort has 642 events; the original principal-
  diagnosis cohort has 57 and remains descriptive.
- If event timing is not recoverable for a component, that component is not
  promoted to a fully timestamped endpoint.
- If the full model does not exceed XGB+TS, the study is reported as a strict
  negative benchmark rather than redefining success post hoc.
- If calibration or net benefit is not improved, increased AUROC alone is not
  interpreted as clinical utility.

## 12. Interpretation rules

- H1 supported: temporal model exceeds static model by the material threshold.
- H2 supported: CT-OGT exceeds XGB+TS with a paired confidence interval
  excluding zero and at least 0.01 AUROC difference.
- H3 supported only if environment-invariant CT-OGT improves transport or
  calibration relative to the dynamic graph without invariance.
- H4 is supported in the original five-fold analysis and temporal holdout when
  the point estimate reaches 0.01 with a CI excluding zero. It is explicitly
  reported as marginal when strict nested tuning is below 0.01.
- H5 requires negative controls to be materially worse; a control matching the
  full model invalidates a mechanistic claim.
- Attention maps are exploratory and cannot establish causal organ coupling.

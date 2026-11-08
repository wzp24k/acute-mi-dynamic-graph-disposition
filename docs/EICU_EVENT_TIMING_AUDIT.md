# eICU event-time audit

Source audit: `work/eicu_timing_audit.py`

Result artifact: `results/eicu_timing_audit.json`

## Available eICU tables

The available eICU extract contains:

- `patient.csv.gz`
- `diagnosis.csv.gz`
- `apacheApsVar.csv.gz`
- `apachePatientResult.csv.gz`
- `customLab.csv.gz`
- `lab.csv.gz`
- `nurseCharting.csv.gz`
- `vitalAperiodic.csv.gz`
- `vitalPeriodic.csv.gz`
- `hospital.csv.gz`

It does not contain `note`, `physicalExam`, `pastHistory`, `socialHistory`,
`treatment`, `medication`, `admissionDrug`, `infusionDrug`, `carePlan`,
`respiratoryCharting`, `respiratoryCare`, or `intakeOutput`.

## What is available

### Admission reason

`patient.apacheadmissiondx` is a categorical ICU admission-reason field. It has
no event timestamp.

Examples across the full eICU extract:

| Admission reason | Rows |
|---|---:|
| Infarction, acute myocardial | 7,228 |
| CVA / cerebrovascular accident / stroke | 6,647 |
| CABG alone | 4,543 |
| Unstable angina | 2,658 |
| Chest pain, unknown origin | 735 |
| Stable angina | 645 |

This field can identify why a patient entered the ICU. It cannot determine
whether an MI or stroke occurred after the 24-hour landmark.

### Diagnosis offset

`diagnosis.diagnosisoffset` is available in minutes from ICU admission. It
provides a structured relative time for diagnosis rows. However, it represents
when the diagnosis was documented in the eICU record, not necessarily the
clinical onset of the event.

Using the first MI or stroke diagnosis per stay:

| Event | Timing bin | Stays |
|---|---|---:|
| Acute MI | Before ICU | 123 |
| Acute MI | 0-24 h | 8,951 |
| Acute MI | 24 h-day 7 | 666 |
| Acute MI | After day 7 | 58 |
| Stroke | Before ICU | 245 |
| Stroke | 0-24 h | 10,210 |
| Stroke | 24 h-day 7 | 494 |
| Stroke | After day 7 | 114 |

Diagnosis offsets are useful for sensitivity analysis. They should not be
described as adjudicated event-onset times without notes or source-record
validation.

The expanded any-position CAD cohort contained 5,913 patients and 642 coded
MACCE events. Only 119 events had a diagnosis-offset anchor interval
overlapping ICU hours 24-168 after the interval filter. No event met the
stricter full-interval-inside-window definition because diagnosis offset is a
documentation timestamp rather than a bounded event-onset interval. The
original principal-diagnosis cohort contained 57 coded events and nine
anchor-overlap events.

Documentation-lag sensitivity at 0, 3, 6, and 12 hours left both the expanded
and principal-diagnosis overlap counts unchanged. Expanded-cohort
temporal-ensemble AUROC was 0.653 at all four values. This indicates that the
overlap classification was stable to the lag assumption in the available
records, not that diagnosis offset is a validated onset time.

### Nursing and physiology

`nurseCharting`, `vitalAperiodic`, and `vitalPeriodic` contain numeric or coded
measurements. They do not contain narrative chief complaints or clinical notes.

## Direct answer

eICU provides a usable **diagnosis-entry time proxy**, not a fully reliable
event-onset time. The current extract has no free-text notes or history tables
that could be used to recover a more exact onset.

## Consequence for the outline

The original statement that the primary endpoint is "7-day MACCE" with
verified event times is not supportable across all three databases.

The revised endpoint hierarchy should be:

1. **Time-resolved endpoint**: 7-day all-cause death. This is the only component
   with broadly comparable exact or near-exact timing.
2. **Diagnosis-coded MACCE phenotype**: acute MI, stroke, death, and
   revascularization recorded during the index admission. This is the main
   cross-database phenotype but should retain "coded" in its name.
3. **7-day diagnosis-offset sensitivity endpoint**: use eICU
   `diagnosisoffset > 1440` and `<= 10080` minutes for MI/stroke, together with
   death, as a timed sensitivity analysis. Report this separately from the main
   coded phenotype.

The manuscript should not claim that all MACCE components were adjudicated to
the exact event time. It should state that the primary endpoint includes coded
events whose onset may precede the 24-hour landmark.

## Recommended outline changes

- Rename the primary endpoint from "7-day MACCE" to
  "diagnosis-coded 7-day MACCE phenotype".
- Make 7-day all-cause death a co-primary or key secondary endpoint.
- Move exact-time MI/stroke event analysis to exploratory/sensitivity status.
- Add a limitation that eICU diagnosis offsets are documentation timestamps.
- Report the expanded eICU cohort separately from the original
  principal-diagnosis sensitivity cohort.
- Add MIMIC note-based extraction only as a separate sensitivity analysis if
  full notes and validation resources are available.

## MIMIC comparison

Source audit: `work/mimic_event_timing_audit.py`

Result artifact: `results/mimic_event_timing_audit.json`

### MIMIC-IV

- `diagnoses_icd` columns are `subject_id`, `hadm_id`, `seq_num`, `icd_code`,
  and `icd_version`. There is no chart time, offset, or diagnosis-entry field.
- `procedures_icd.chartdate` is populated for 61,014 rows across 11,901
  admissions in the eligible audit cohort. It is available at day resolution.
- Discharge summaries are available for 8,905 of 13,131 eligible admissions
  (67.8%).
- The current MIMIC-IV note extract contains discharge summaries and radiology
  reports, but not a general progress-note table.
- Radiology reports containing acute infarct or intracranial hemorrhage terms
  are present for 1,214 admissions with hemorrhagic or cerebrovascular terms.

### MIMIC-III

- `DIAGNOSES_ICD` has no event or diagnosis-entry time.
- `PROCEDURES_ICD` has no date column.
- `NOTEEVENTS` has `CHARTDATE` and `CHARTTIME`, but note time is not identical
  to clinical event onset.
- Timed note categories in the eligible cohort include:
  - Nursing/other: 21,700 notes across 2,318 admissions.
  - Radiology: 15,361 notes across 2,923 admissions.
  - Nursing: 6,352 notes across 454 admissions.
  - Physician: 2,443 notes across 446 admissions.
- ECG notes are present for 3,454 admissions, but `CHARTTIME` is not populated
  in this extract.
- Discharge summaries are present for 3,491 admissions, but `CHARTTIME` is not
  populated in this extract.

### Can MIMIC diagnosis-entry time be obtained?

No. The MIMIC diagnosis tables do not provide an equivalent of eICU
`diagnosisoffset`.

MIMIC can support approximate event timing through three separate routes:

1. Dated procedures for revascularization or selected operations.
2. Timed notes or radiology reports with explicit event language.
3. Laboratory and order trajectories, such as troponin rise/fall or start of
   ACS treatment.

None of these is diagnosis-entry time. All require a separate validated
phenotyping study, and the available coverage selects a non-random subset of
patients.

### What approximation is defensible now?

An event-time interval can be assigned:

- Lower bound: ICU admission or the last known event-free time.
- Upper bound: the first supporting procedure, note, imaging report, or order.
- Classification: confirmed timed event, possible event, or no timed evidence.

This is an interval-censored endpoint. It supports a sensitivity analysis but
not a claim that true onset was observed for all events. Restricting to confirmed
timed events changes the cohort and event prevalence and can introduce
verification bias.

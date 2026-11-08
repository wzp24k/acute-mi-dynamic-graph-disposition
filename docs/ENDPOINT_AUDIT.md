# Endpoint feasibility audit

Source artifact: `results/endpoint_feasibility.json`

## Cohort and event counts

| Database | Eligible at 24 h | Death 24 h-7 d | Revascularization 24 h-7 d | Coded AMI | Coded stroke | Coded MACCE proxy |
|---|---:|---:|---:|---:|---:|---:|
| MIMIC-IV | 13,131 | 524 | 409 | 3,214 | 1,079 | 4,475 |
| MIMIC-III | 3,523 | 13 | unavailable | 224 | 201 | 416 |
| eICU | 929 | 4 | unavailable | 51 timed | 5 timed | 58 timed |
| eICU expanded CAD | 6,226 | 93 | unavailable | 479 timed | 128 timed | 674 timed |

The final analytic cohorts after the six-channel threshold are 13,131,
3,470, 5,913, and 903 patients, respectively. The corresponding event counts
are 4,475, 409, 642, and 57. The expanded eICU cohort is the primary
multi-hospital transport cohort; the principal-diagnosis cohort is a
sensitivity analysis.

## Go/No-Go decisions

- MIMIC-IV passed the prespecified development threshold of at least 150
  events.
- Expanding eICU to any admission diagnosis position produced 642 analytic
  events and passed the 75-event transport threshold.
- The original principal-diagnosis eICU cohort had 57 events. Estimates from
  that cohort remain descriptive.
- Expanded eICU contains 166 sites; 43 analytic sites have at least 30
  patients. The source audit identified 44 sites with at least 30 patients and
  at least one event before the six-channel filter.

## Event-time quality

| Component | Time quality |
|---|---|
| Death | Exact timestamp in MIMIC and discharge offset in eICU |
| MIMIC-IV revascularization | Procedure date, day-level resolution |
| MIMIC-III revascularization | Code exists, procedure date unavailable |
| MIMIC acute MI | Coded diagnosis, onset time unavailable |
| MIMIC stroke | Coded diagnosis, onset time unavailable |
| eICU acute MI | Diagnosis offset available |
| eICU stroke | Diagnosis offset available |
| eICU revascularization | Treatment table absent from the available extract |

## Consequence for the study

The main endpoint is designated a **diagnosis-coded 7-day MACCE proxy**. It is
not a fully adjudicated or fully timestamped composite endpoint.

The timestamped sensitivity endpoint includes:

- death from the 24-hour landmark through day 7;
- MIMIC-IV revascularization from day-level procedure dates;
- eICU acute MI or stroke when the diagnosis offset is after 1,440 minutes
  and no later than 10,080 minutes.

The timestamped endpoint has 921 events in MIMIC-IV. XGB+TS achieves AUROC
0.870 on this endpoint, compared with 0.753 on the coded proxy endpoint. The
difference is interpreted as endpoint-ascertainment sensitivity, not as
evidence that either endpoint is a gold standard.

## Required manuscript language

- Do not write that every MACCE component was ascertained to the exact hour.
- State that coded myocardial infarction and stroke may include
  present-on-admission events.
- Report event-component counts separately.
- Report both the expanded eICU event count and the original cohort shortfall.
- Treat direct external calibration as invalid before local recalibration.
- Treat attention maps as exploratory and non-causal.

## Interval-censored sensitivity endpoint

See `docs/EICU_EVENT_TIMING_AUDIT.md` for source-data timing coverage.

| Database | Coded events | Fully confirmed interval events | Anchor-overlap events |
|---|---:|---:|---:|
| MIMIC-IV | 4,475 | 45 | 1,265 |
| MIMIC-III | 409 | 12 | 76 |
| eICU principal diagnosis | 57 | 0 | 9 |
| eICU expanded CAD | 642 | 0 | 119 |

The interval endpoint uses exact death times, procedure-day revascularization,
timed radiology/notes, troponin timing, and eICU diagnosis offsets. Events with
only a diagnosis code are excluded from the anchor-overlap endpoint rather than
assigned a broad admission-to-discharge interval. The 45 MIMIC-IV fully
confirmed events and the 12 fully confirmed MIMIC-III events are below the
150-event inference threshold; no AUROC is reported for those strata.

Retrained XGB+TS achieved AUROC 0.795 for MIMIC-IV anchor overlap, 0.755 for
MIMIC-III anchor overlap, and 0.668 for expanded eICU anchor overlap. The
original eICU cohort had only nine overlap events, so no model estimate is
reported.

Rule sensitivity:

- MIMIC-IV troponin threshold multipliers 0.5, 1.0, and 2.0 produced 1,266,
  1,265, and 1,254 anchor-overlap events; temporal-ensemble AUROC was 0.769
  at all three values.
- MIMIC-III remained at 76 overlap events with temporal-ensemble AUROC 0.729.
- eICU documentation lags 0, 3, 6, and 12 hours produced 119 expanded-cohort
  overlap events and temporal-ensemble AUROC 0.653 at every value.

## Structured chart-review pilot

The review pack contains 120 records. One hundred MIMIC records are available
for independent clinical review, and 20 eICU records lack source notes in the
available extract.

A 38-record AI-assisted structured chart-review pilot produced:

- 4 confirmed target-window events;
- 6 probable target-window events;
- 7 pre-landmark records;
- 5 historical or non-event records;
- 1 post-horizon record;
- 15 uncertain or insufficient records.

Thus 10 of 38 pilot records supported a target-window event. This is a
structured extraction pilot, not independent clinician adjudication, and cannot
be reported as a human gold-standard positive predictive value.

The full review pack now has a two-reader schema for serial troponin change,
ECG evidence, angiography/PCI, acute stroke imaging, focal deficit,
clinical-record support, onset window, confidence, consensus label, and notes.
The scoring script will calculate PPV, timing misclassification, and Cohen
kappa once both readers finish. At the current snapshot, all 100 reviewable
MIMIC records remain pending; no human PPV or kappa is imputed.

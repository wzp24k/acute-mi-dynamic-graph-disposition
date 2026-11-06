# Table S13. Included versus excluded patients at the MIMIC-IV AMI 24-hour landmark

The source tables were rebuilt read-only. The primary comparison is between the 6,487 included patients and the 1,330 patients who died on or before ICU hour 24, left the ICU alive before hour 24, had no ICU discharge time, or had an unmappable discharge disposition after surviving the landmark.

## Cohort reconciliation

| Disposition | n |
|---|---:|
| AMI hospital admissions in the source table | 16,509 |
| AMI admissions without a first ICU stay | 8,242 |
| First ICU stay | 8,267 |
| First ICU stay in an adult AMI admission | 8,267 |
| Died on or before ICU hour 24 | 232 |
| Left ICU alive before hour 24 | 1,089 |
| Missing ICU discharge time | 2 |
| Survived landmark but outcome was unmappable | 7 |
| Included in the analytic cohort | 6,487 |
| Eligible landmark but later hospitalization for the same patient (not in Table S13 comparison column) | 450 |

The last row is a patient-level deduplication step, not a clinical landmark exclusion. Adding it to the included and primary excluded groups returns the first-ICU adult cohort exactly.

## Included versus excluded

Values are median (IQR) for continuous variables and n (%) for categorical variables. Continuous-variable availability is shown as n (%) in the preceding row. Standardized differences use included minus excluded; no P values are reported.

| Characteristic | Included (n=6,487) | Excluded (n=1,330) | SMD |
|---|---:|---:|---:|
| **Demographics** |  |  |  |
| Age, years: available | 6,487 (100.0%) | 1,330 (100.0%) | +0.000 |
| Age, years: median (IQR) | 72 (62-80) | 72 (62-82) | -0.053 |
| Sex: Female | 2,336 (36.0%) | 515 (38.7%) | -0.056 |
| Sex: Male | 4,151 (64.0%) | 815 (61.3%) | +0.056 |
| Race: White | 4,113 (63.4%) | 844 (63.5%) | -0.001 |
| Race: Black | 538 (8.3%) | 137 (10.3%) | -0.069 |
| Race: Hispanic or Latino | 192 (3.0%) | 36 (2.7%) | +0.015 |
| Race: Asian | 143 (2.2%) | 40 (3.0%) | -0.050 |
| Race: Other or unknown | 1,501 (23.1%) | 273 (20.5%) | +0.063 |
| **Admission and AMI characteristics** |  |  |  |
| Admission type: Urgent | 2,412 (37.2%) | 386 (29.0%) | +0.174 |
| Admission type: Elective or surgical same-day | 166 (2.6%) | 11 (0.8%) | +0.135 |
| Admission type: Observation | 996 (15.4%) | 237 (17.8%) | -0.066 |
| Admission type: Other or unknown | 2,913 (44.9%) | 696 (52.3%) | -0.149 |
| AMI subtype: STEMI | 1,326 (20.4%) | 383 (28.8%) | -0.195 |
| AMI subtype: NSTEMI | 3,774 (58.2%) | 614 (46.2%) | +0.242 |
| AMI subtype: Type 2 MI | 1,194 (18.4%) | 285 (21.4%) | -0.076 |
| AMI subtype: Unspecified | 156 (2.4%) | 41 (3.1%) | -0.042 |
| AMI subtype: Other | 37 (0.6%) | 7 (0.5%) | +0.006 |
| AMI primary position: Primary-position AMI | 3,206 (49.4%) | 662 (49.8%) | -0.007 |
| AMI primary position: Not primary | 3,281 (50.6%) | 668 (50.2%) | +0.007 |
| **Pre-landmark vital signs** |  |  |  |
| Heart rate, first: available | 6,480 (99.9%) | 1,324 (99.5%) | +0.065 |
| Heart rate, first: median (IQR) | 84.0 (74.0-98.0) | 84.0 (73.0-98.0) | +0.014 |
| Respiratory rate, first: available | 6,477 (99.8%) | 1,320 (99.2%) | +0.089 |
| Respiratory rate, first: median (IQR) | 18.0 (15.0-23.0) | 19.0 (16.0-23.0) | -0.086 |
| Peripheral oxygen saturation, first: available | 6,480 (99.9%) | 1,304 (98.0%) | +0.184 |
| Peripheral oxygen saturation, first: median (IQR) | 98.0 (95.0-100.0) | 97.0 (95.0-99.0) | +0.137 |
| Systolic blood pressure, first: available | 6,456 (99.5%) | 1,312 (98.6%) | +0.092 |
| Systolic blood pressure, first: median (IQR) | 119.0 (104.0-135.0) | 121.0 (106.0-138.0) | -0.086 |
| Diastolic blood pressure, first: available | 6,456 (99.5%) | 1,312 (98.6%) | +0.092 |
| Diastolic blood pressure, first: median (IQR) | 65.0 (55.0-77.0) | 69.0 (59.0-81.0) | -0.177 |
| Mean arterial pressure, first: available | 6,457 (99.5%) | 1,312 (98.6%) | +0.094 |
| Mean arterial pressure, first: median (IQR) | 81.0 (71.0-93.0) | 84.0 (73.0-96.0) | -0.108 |
| Temperature, first: available | 6,220 (95.9%) | 1,270 (95.5%) | +0.019 |
| Temperature, first: median (IQR) | 36.6 (36.4-36.9) | 36.7 (36.4-36.9) | +0.031 |
| **Pre-landmark laboratory values** |  |  |  |
| Troponin I or T, first: available | 4,190 (64.6%) | 880 (66.2%) | -0.033 |
| Troponin I or T, first: median (IQR) | 0.520 (0.160-1.708) | 0.535 (0.147-2.103) | +0.001 |
| CK-MB, first: available | 3,983 (61.4%) | 825 (62.0%) | -0.013 |
| CK-MB, first: median (IQR) | 16.00 (6.00-57.00) | 15.00 (5.00-59.00) | +0.034 |
| Creatinine, first: available | 6,458 (99.6%) | 1,196 (89.9%) | +0.442 |
| Creatinine, first: median (IQR) | 1.10 (0.80-1.90) | 1.20 (0.80-1.90) | -0.024 |
| Blood urea nitrogen, first: available | 6,456 (99.5%) | 1,196 (89.9%) | +0.440 |
| Blood urea nitrogen, first: median (IQR) | 23.0 (16.0-40.0) | 23.0 (16.0-40.0) | -0.011 |
| Lactate, first: available | 4,601 (70.9%) | 569 (42.8%) | +0.593 |
| Lactate, first: median (IQR) | 1.90 (1.30-2.80) | 2.20 (1.50-4.30) | -0.482 |
| White blood cell count, first: available | 6,431 (99.1%) | 1,179 (88.6%) | +0.449 |
| White blood cell count, first: median (IQR) | 11.8 (8.7-15.8) | 10.2 (7.7-13.9) | +0.179 |
| Hemoglobin, first: available | 6,432 (99.2%) | 1,179 (88.6%) | +0.450 |
| Hemoglobin, first: median (IQR) | 10.4 (8.7-12.2) | 11.1 (9.2-12.8) | -0.187 |
| Platelets, first: available | 6,439 (99.3%) | 1,191 (89.5%) | +0.432 |
| Platelets, first: median (IQR) | 192.0 (140.0-251.0) | 203.0 (156.0-254.0) | -0.092 |
| Sodium, first: available | 6,458 (99.6%) | 1,196 (89.9%) | +0.442 |
| Sodium, first: median (IQR) | 138 (135-141) | 138 (136-141) | -0.029 |
| Potassium, first: available | 6,455 (99.5%) | 1,200 (90.2%) | +0.430 |
| Potassium, first: median (IQR) | 4.30 (3.90-4.70) | 4.20 (3.80-4.60) | +0.052 |
| Glucose, first: available | 6,411 (98.8%) | 1,189 (89.4%) | +0.409 |
| Glucose, first: median (IQR) | 134 (110-181) | 131 (108-180) | -0.015 |
| INR, first: available | 5,985 (92.3%) | 1,020 (76.7%) | +0.440 |
| INR, first: median (IQR) | 1.30 (1.10-1.50) | 1.20 (1.10-1.50) | -0.055 |
| Total bilirubin, first: available | 3,446 (53.1%) | 624 (46.9%) | +0.124 |
| Total bilirubin, first: median (IQR) | 0.60 (0.40-1.00) | 0.60 (0.40-1.00) | -0.051 |
| ALT, first: available | 3,467 (53.4%) | 632 (47.5%) | +0.119 |
| ALT, first: median (IQR) | 33 (19-79) | 36 (20-80) | -0.042 |
| AST, first: available | 3,492 (53.8%) | 636 (47.8%) | +0.120 |
| AST, first: median (IQR) | 62 (32-157) | 60 (29-194) | -0.053 |
| Albumin, first: available | 2,067 (31.9%) | 360 (27.1%) | +0.105 |
| Albumin, first: median (IQR) | 3.2 (2.8-3.6) | 3.3 (2.7-3.8) | -0.045 |
| Bicarbonate, first: available | 6,454 (99.5%) | 1,190 (89.5%) | +0.450 |
| Bicarbonate, first: median (IQR) | 22 (20-25) | 23 (20-25) | +0.014 |
| Chloride, first: available | 6,460 (99.6%) | 1,197 (90.0%) | +0.442 |
| Chloride, first: median (IQR) | 104 (100-108) | 103 (99-106) | +0.184 |

## Key differences

### Measurement availability

Excluded patients had fewer routine laboratory results because their observation window ended before hour 24. Availability therefore contributes to selection and is reported as an effect size, not as a significance test.

- Lactate, first: included 4,601 (70.9%); excluded 569 (42.8%); SMD +0.593.
- Hemoglobin, first: included 6,432 (99.2%); excluded 1,179 (88.6%); SMD +0.450.
- Bicarbonate, first: included 6,454 (99.5%); excluded 1,190 (89.5%); SMD +0.450.
- White blood cell count, first: included 6,431 (99.1%); excluded 1,179 (88.6%); SMD +0.449.
- Chloride, first: included 6,460 (99.6%); excluded 1,197 (90.0%); SMD +0.442.
- Creatinine, first: included 6,458 (99.6%); excluded 1,196 (89.9%); SMD +0.442.
- Sodium, first: included 6,458 (99.6%); excluded 1,196 (89.9%); SMD +0.442.
- INR, first: included 5,985 (92.3%); excluded 1,020 (76.7%); SMD +0.440.

### Clinical and demographic composition

- Lactate, first, median: included 1.90 (1.30-2.80); excluded 2.20 (1.50-4.30); SMD -0.482.
- AMI subtype: NSTEMI: included 3,774 (58.2%); excluded 614 (46.2%); SMD +0.242.
- AMI subtype: STEMI: included 1,326 (20.4%); excluded 383 (28.8%); SMD -0.195.
- Hemoglobin, first, median: included 10.4 (8.7-12.2); excluded 11.1 (9.2-12.8); SMD -0.187.
- Chloride, first, median: included 104 (100-108); excluded 103 (99-106); SMD +0.184.
- White blood cell count, first, median: included 11.8 (8.7-15.8); excluded 10.2 (7.7-13.9); SMD +0.179.
- Diastolic blood pressure, first, median: included 65.0 (55.0-77.0); excluded 69.0 (59.0-81.0); SMD -0.177.
- Admission type: Urgent: included 2,412 (37.2%); excluded 386 (29.0%); SMD +0.174.
- Admission type: Other or unknown: included 2,913 (44.9%); excluded 696 (52.3%); SMD -0.149.
- Peripheral oxygen saturation, first, median: included 98.0 (95.0-100.0); excluded 97.0 (95.0-99.0); SMD +0.137.
- Admission type: Elective or surgical same-day: included 166 (2.6%); excluded 11 (0.8%); SMD +0.135.
- Mean arterial pressure, first, median: included 81.0 (71.0-93.0); excluded 84.0 (73.0-96.0); SMD -0.108.

These differences describe selection at the landmark and must not be interpreted as causal effects.

## Notes

- The observation window for excluded patients was censored at the earlier of ICU hour 24, ICU discharge, or death. Values first recorded in that window were compared.
- Temperature was converted from Fahrenheit to Celsius when needed. Troponin I and T were combined; their units are assay dependent.
- Standardized differences for observed continuous values use the difference in means divided by the average pooled standard deviation. The table reports medians and IQRs for distributional transparency.
- Missing values were not imputed. Availability is reported separately for every vital sign and laboratory variable.

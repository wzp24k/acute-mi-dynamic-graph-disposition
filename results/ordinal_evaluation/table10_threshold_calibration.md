# Table 10. Cumulative-threshold performance, calibration, and reliability

Cumulative probabilities were P(y>=2)=sum(p2..p4), P(y>=3)=sum(p3..p4), and P(y>=4)=p4 on the five-level MIMIC-IV outcome scale. Calibration models were logit(y)=intercept+slope*logit(p). Intervals use 2,000 patient-bootstrap replicates.

## Threshold performance

| Model | Threshold | Events/N | AUROC (95% CI) | Calibration slope | Intercept | Brier (95% CI) |
|---|---|---:|---:|---:|---:|---:|
| XGB-static+TS | P(y>=2) | 3,540/6,487 | 0.8288 (0.8187 to 0.8386) | 1.216 (1.157 to 1.278) | -0.030 (-0.092 to 0.030) | 0.1681 (0.1637 to 0.1725) |
| XGB-static+TS | P(y>=3) | 3,068/6,487 | 0.8069 (0.7962 to 0.8173) | 1.187 (1.128 to 1.250) | 0.003 (-0.056 to 0.059) | 0.1795 (0.1754 to 0.1838) |
| XGB-static+TS | P(y>=4) | 1,171/6,487 | 0.8641 (0.8537 to 0.8741) | 1.234 (1.165 to 1.312) | 0.251 (0.145 to 0.364) | 0.1056 (0.1010 to 0.1102) |
| DG-GRU+TR | P(y>=2) | 3,540/6,487 | 0.8075 (0.7973 to 0.8186) | 0.724 (0.682 to 0.773) | -0.372 (-0.438 to -0.309) | 0.1873 (0.1813 to 0.1931) |
| DG-GRU+TR | P(y>=3) | 3,068/6,487 | 0.7868 (0.7760 to 0.7980) | 0.638 (0.599 to 0.682) | 0.006 (-0.052 to 0.062) | 0.1923 (0.1866 to 0.1979) |
| DG-GRU+TR | P(y>=4) | 1,171/6,487 | 0.8438 (0.8321 to 0.8558) | 0.760 (0.712 to 0.813) | -1.034 (-1.112 to -0.959) | 0.1310 (0.1258 to 0.1359) |
| DG-GRU+CTR | P(y>=2) | 3,540/6,487 | 0.8067 (0.7966 to 0.8173) | 0.720 (0.679 to 0.767) | -0.372 (-0.439 to -0.310) | 0.1879 (0.1819 to 0.1937) |
| DG-GRU+CTR | P(y>=3) | 3,068/6,487 | 0.7867 (0.7759 to 0.7977) | 0.638 (0.599 to 0.680) | 0.001 (-0.058 to 0.058) | 0.1924 (0.1867 to 0.1979) |
| DG-GRU+CTR | P(y>=4) | 1,171/6,487 | 0.8428 (0.8312 to 0.8549) | 0.755 (0.709 to 0.808) | -1.044 (-1.121 to -0.968) | 0.1319 (0.1267 to 0.1369) |

## Risk-stratified reliability

Fixed strata were low risk [0.00,0.20), intermediate [0.20,0.50), and high [0.50,1.00]. Calibration gap is observed rate minus mean predicted risk.

| Model | Threshold | Risk stratum | N | Events | Mean predicted | Observed | Gap |
|---|---|---|---:|---:|---:|---:|---:|
| XGB-static+TS | P(y>=2) | Low [0.00,0.20) | 843 | 57 | 0.1222 | 0.0676 | -0.0546 |
| XGB-static+TS | P(y>=2) | Intermediate [0.20,0.50) | 1,846 | 637 | 0.3598 | 0.3451 | -0.0148 |
| XGB-static+TS | P(y>=2) | High [0.50,1.00) | 3,798 | 2,846 | 0.7285 | 0.7493 | +0.0209 |
| XGB-static+TS | P(y>=3) | Low [0.00,0.20) | 1,114 | 83 | 0.1138 | 0.0745 | -0.0393 |
| XGB-static+TS | P(y>=3) | Intermediate [0.20,0.50) | 2,277 | 792 | 0.3601 | 0.3478 | -0.0122 |
| XGB-static+TS | P(y>=3) | High [0.50,1.00) | 3,096 | 2,193 | 0.6864 | 0.7083 | +0.0220 |
| XGB-static+TS | P(y>=4) | Low [0.00,0.20) | 4,279 | 237 | 0.0699 | 0.0554 | -0.0146 |
| XGB-static+TS | P(y>=4) | Intermediate [0.20,0.50) | 1,695 | 587 | 0.3201 | 0.3463 | +0.0262 |
| XGB-static+TS | P(y>=4) | High [0.50,1.00) | 513 | 347 | 0.6317 | 0.6764 | +0.0447 |
| DG-GRU+TR | P(y>=2) | Low [0.00,0.20) | 742 | 65 | 0.0666 | 0.0876 | +0.0210 |
| DG-GRU+TR | P(y>=2) | Intermediate [0.20,0.50) | 1,180 | 316 | 0.3814 | 0.2678 | -0.1136 |
| DG-GRU+TR | P(y>=2) | High [0.50,1.00) | 4,565 | 3,159 | 0.7911 | 0.6920 | -0.0991 |
| DG-GRU+TR | P(y>=3) | Low [0.00,0.20) | 1,530 | 205 | 0.0881 | 0.1340 | +0.0459 |
| DG-GRU+TR | P(y>=3) | Intermediate [0.20,0.50) | 2,197 | 895 | 0.3416 | 0.4074 | +0.0658 |
| DG-GRU+TR | P(y>=3) | High [0.50,1.00) | 2,760 | 1,968 | 0.7694 | 0.7130 | -0.0564 |
| DG-GRU+TR | P(y>=4) | Low [0.00,0.20) | 3,456 | 128 | 0.0801 | 0.0370 | -0.0431 |
| DG-GRU+TR | P(y>=4) | Intermediate [0.20,0.50) | 1,488 | 289 | 0.3300 | 0.1942 | -0.1358 |
| DG-GRU+TR | P(y>=4) | High [0.50,1.00) | 1,543 | 754 | 0.7381 | 0.4887 | -0.2495 |
| DG-GRU+CTR | P(y>=2) | Low [0.00,0.20) | 741 | 70 | 0.0669 | 0.0945 | +0.0276 |
| DG-GRU+CTR | P(y>=2) | Intermediate [0.20,0.50) | 1,167 | 304 | 0.3806 | 0.2605 | -0.1201 |
| DG-GRU+CTR | P(y>=2) | High [0.50,1.00) | 4,579 | 3,166 | 0.7906 | 0.6914 | -0.0992 |
| DG-GRU+CTR | P(y>=3) | Low [0.00,0.20) | 1,495 | 201 | 0.0856 | 0.1344 | +0.0489 |
| DG-GRU+CTR | P(y>=3) | Intermediate [0.20,0.50) | 2,238 | 894 | 0.3410 | 0.3995 | +0.0584 |
| DG-GRU+CTR | P(y>=3) | High [0.50,1.00) | 2,754 | 1,973 | 0.7712 | 0.7164 | -0.0548 |
| DG-GRU+CTR | P(y>=4) | Low [0.00,0.20) | 3,455 | 135 | 0.0805 | 0.0391 | -0.0415 |
| DG-GRU+CTR | P(y>=4) | Intermediate [0.20,0.50) | 1,490 | 299 | 0.3323 | 0.2007 | -0.1317 |
| DG-GRU+CTR | P(y>=4) | High [0.50,1.00) | 1,542 | 737 | 0.7408 | 0.4780 | -0.2628 |

## External decision-curve methods

External DCA used raw transported probabilities, not recalibrated probabilities. eICU net-benefit intervals resampled patient-safe hospital clusters, defined as connected components of the patient-hospital graph so that hospitals sharing any patient were resampled together. MIMIC-III intervals resampled patients. Exact cluster counts and methods are stored in ordinal_evaluation_report.json and the bootstrap metadata columns of external_dca_curves.csv.

Negative values, including negative gaps, slopes, net benefits, QWK, and C-index contrasts, were retained without absolute transformation or truncation.

| Cohort            | Interval endpoint                        |   Events | Retrained XGB+TS AUROC   | Existing temporal ensemble AUROC   |
|:------------------|:-----------------------------------------|---------:|:-------------------------|:-----------------------------------|
| MIMIC-IV          | Fully confirmed interval in target       |       45 | NA                       | NA                                 |
| MIMIC-IV          | Specific anchor interval overlaps target |     1265 | 0.795                    | 0.769                              |
| MIMIC-III         | Fully confirmed interval in target       |       12 | NA                       | NA                                 |
| MIMIC-III         | Specific anchor interval overlaps target |       76 | 0.755                    | 0.729                              |
| eICU              | Fully confirmed interval in target       |        0 | NA                       | NA                                 |
| eICU              | Specific anchor interval overlaps target |        9 | NA                       | NA                                 |
| eICU expanded CAD | Fully confirmed interval in target       |        0 | NA                       | NA                                 |
| eICU expanded CAD | Specific anchor interval overlaps target |      119 | 0.668                    | 0.653                              |
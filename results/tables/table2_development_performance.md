| Model                                    | AUROC (95% CI)      |   AUPRC |   Brier |   Calibration slope |   ECE |
|:-----------------------------------------|:--------------------|--------:|--------:|--------------------:|------:|
| Logistic regression, static              | 0.632 (0.623-0.643) |   0.459 |   0.214 |               0.966 | 0.007 |
| XGBoost, static                          | 0.641 (0.632-0.651) |   0.469 |   0.212 |               0.973 | 0.004 |
| XGBoost, static + temporal summaries     | 0.754 (0.745-0.763) |   0.636 |   0.181 |               1.116 | 0.013 |
| GRU                                      | 0.749 (0.740-0.758) |   0.63  |   0.183 |               0.874 | 0.02  |
| Transformer                              | 0.759 (0.751-0.768) |   0.647 |   0.179 |               0.977 | 0.01  |
| TFT-lite                                 | 0.749 (0.740-0.758) |   0.631 |   0.183 |               0.851 | 0.023 |
| Continuous-time fixed organ graph        | 0.751 (0.743-0.760) |   0.632 |   0.182 |               0.888 | 0.016 |
| Continuous-time dynamic organ graph      | 0.756 (0.748-0.765) |   0.639 |   0.181 |               0.988 | 0.01  |
| Seven-organ token dynamic graph          | 0.750 (0.742-0.759) |   0.631 |   0.183 |               0.896 | 0.016 |
| Temporal ensemble (XGB+TS + Transformer) | 0.764 (0.755-0.773) |   0.654 |   0.178 |               1.132 | 0.017 |
| CT-OGT full model                        | 0.756 (0.748-0.765) |   0.64  |   0.181 |               0.965 | 0.011 |
| CT-OGT, random graph control             | 0.750 (0.742-0.759) |   0.631 |   0.183 |               0.923 | 0.012 |
| CT-OGT, reversed time control            | 0.750 (0.741-0.758) |   0.627 |   0.183 |               0.91  | 0.016 |
| CT-OGT, permuted time control            | 0.749 (0.740-0.758) |   0.627 |   0.183 |               0.897 | 0.017 |
| CT-OGT, no continuous-time encoding      | 0.747 (0.738-0.756) |   0.622 |   0.184 |               0.929 | 0.012 |
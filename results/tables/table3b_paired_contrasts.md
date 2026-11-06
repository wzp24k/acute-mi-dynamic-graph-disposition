| contrast                                                                                          |     estimate |       ci_low |     ci_high | metric    |
|:--------------------------------------------------------------------------------------------------|-------------:|-------------:|------------:|:----------|
| DG-GRU+TR minus XGB-static+TS                                                                     | -0.0493836   | -0.0570815   | -0.0416449  | macro-AUC |
| DG-GRU+TR minus DG-GRU                                                                            |  0.000612371 | -4.57583e-05 |  0.00129373 | macro-AUC |
| XGB-static+TS minus XGB-static                                                                    |  0.027551    |  0.0226163   |  0.0326428  | macro-AUC |
| MLP-static+TS minus MLP-static                                                                    |  0.0163433   |  0.00977499  |  0.0230419  | macro-AUC |
| MLP-static minus XGB-static                                                                       | -0.0339229   | -0.0418515   | -0.0259385  | macro-AUC |
| XGB-static+TS minus MLP-static+TS                                                                 |  0.0451306   |  0.0370147   |  0.0528847  | macro-AUC |
| MLP-static+TS minus DG-GRU (representation plus structure)                                        |  0.00486538  |  0.000977416 |  0.00892381 | macro-AUC |
| SeqOrdinal GRU minus DG-GRU                                                                       |  0.00143607  | -0.00241426  |  0.0052813  | macro-AUC |
| DG-GRU minus DG-GRU-static (raw sequence-plus-availability input toggle; fixed network structure) | -0.00227526  | -0.00555614  |  0.00105262 | macro-AUC |
| DG-GRU+CTR minus DG-GRU                                                                           | -0.000260022 | -0.00353842  |  0.00273509 | macro-AUC |
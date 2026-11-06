# Table 8. Common three-class outcome across three databases

Values are estimate (95% grouped bootstrap CI). Macro AUC is one-vs-rest and unweighted. Ordinal MAE and quadratic weighted kappa use the probability-maximizing class. MIMIC-IV values average out-of-fold probabilities across five seeds. eICU intervals resample patient-safe hospital clusters; MIMIC-III and MIMIC-IV intervals resample patients.

Common outcome: 0 = home or against advice; 1 = non-home post-acute or other care; 2 = hospice or in-hospital death. The middle level is a pragmatic harmonization and does not imply equivalent care intensity across databases.

| Dataset | Model | n (class 0/1/2) | Macro AUC | Ordinal MAE | Quadratic weighted kappa |
|---|---|---:|---|---|---|
| MIMIC-IV | XGB-static | 6487 (1011/4305/1171) | 0.815 (0.806 to 0.825) | 0.287 (0.276 to 0.298) | 0.412 (0.389 to 0.436) |
| MIMIC-IV | XGB-static+TS | 6487 (1011/4305/1171) | 0.839 (0.831 to 0.848) | 0.267 (0.257 to 0.278) | 0.475 (0.454 to 0.498) |
| MIMIC-IV | DG-GRU+TR | 6487 (1011/4305/1171) | 0.810 (0.801 to 0.820) | 0.318 (0.306 to 0.331) | 0.535 (0.513 to 0.554) |
| MIMIC-III | XGB-static | 4750 (1048/3043/659) | 0.815 (0.804 to 0.825) | 0.283 (0.270 to 0.296) | 0.476 (0.451 to 0.498) |
| MIMIC-III | XGB-static+TS | 4750 (1048/3043/659) | 0.831 (0.821 to 0.841) | 0.262 (0.249 to 0.275) | 0.527 (0.503 to 0.551) |
| MIMIC-III | DG-GRU+TR | 4750 (1048/3043/659) | 0.831 (0.821 to 0.841) | 0.291 (0.277 to 0.304) | 0.589 (0.568 to 0.611) |
| eICU direct transport | XGB-static | 6696 (4420/1647/629) | 0.761 (0.747 to 0.774) | 0.533 (0.510 to 0.556) | 0.323 (0.301 to 0.346) |
| eICU direct transport | XGB-static+TS | 6696 (4420/1647/629) | 0.751 (0.728 to 0.771) | 0.581 (0.541 to 0.624) | 0.312 (0.275 to 0.346) |
| eICU direct transport | DG-GRU+TR | 6696 (4420/1647/629) | 0.747 (0.733 to 0.759) | 0.447 (0.422 to 0.477) | 0.436 (0.411 to 0.456) |

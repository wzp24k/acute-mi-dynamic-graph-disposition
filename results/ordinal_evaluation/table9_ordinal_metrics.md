# Table 9. Ordinal performance from five-seed OOF predictions

Probabilities were averaged across seeds 1, 7, 42, 123, and 2024 before metric calculation. Ordinal C-index used the predicted expected class as the continuous score and counted tied scores as 0.5. QWK and ordinal MAE used the maximum-probability class. Intervals are patient-bootstrap percentile 95% CIs.

| Model | Ordinal C-index (95% CI) | QWK (95% CI) | Ordinal MAE (95% CI) | Multi-class Brier (95% CI) |
|---|---:|---:|---:|---:|
| XGB-static | 0.7552 (0.7466 to 0.7641) | 0.4990 (0.4777 to 0.5199) | 0.9067 (0.8813 to 0.9326) | 0.6343 (0.6262 to 0.6421) |
| XGB-static+TS | 0.7844 (0.7767 to 0.7924) | 0.5568 (0.5377 to 0.5764) | 0.8389 (0.8136 to 0.8637) | 0.6127 (0.6039 to 0.6207) |
| MLP-static | 0.7530 (0.7439 to 0.7624) | 0.5082 (0.4886 to 0.5285) | 1.0313 (1.0048 to 1.0564) | 0.7339 (0.7242 to 0.7437) |
| MLP-static+TS | 0.7739 (0.7658 to 0.7825) | 0.5473 (0.5290 to 0.5673) | 0.9809 (0.9544 to 1.0057) | 0.7297 (0.7191 to 0.7406) |
| SeqOrdinal GRU | 0.7729 (0.7648 to 0.7817) | 0.5382 (0.5204 to 0.5574) | 1.0113 (0.9855 to 1.0359) | 0.7307 (0.7200 to 0.7412) |
| GRU-D | 0.7735 (0.7654 to 0.7819) | 0.5428 (0.5242 to 0.5614) | 1.0023 (0.9764 to 1.0267) | 0.7288 (0.7183 to 0.7392) |
| Transformer | 0.7763 (0.7679 to 0.7848) | 0.5455 (0.5266 to 0.5640) | 1.0032 (0.9780 to 1.0288) | 0.7239 (0.7134 to 0.7340) |
| STGCN-lite | 0.7744 (0.7665 to 0.7832) | 0.5358 (0.5178 to 0.5550) | 1.0197 (0.9937 to 1.0446) | 0.7295 (0.7189 to 0.7399) |
| DG-GRU-static | 0.7737 (0.7655 to 0.7823) | 0.5430 (0.5253 to 0.5626) | 1.0039 (0.9784 to 1.0281) | 0.7274 (0.7171 to 0.7375) |
| DG-GRU | 0.7732 (0.7650 to 0.7820) | 0.5443 (0.5258 to 0.5642) | 1.0039 (0.9773 to 1.0284) | 0.7314 (0.7207 to 0.7418) |
| DG-GRU+TR | 0.7734 (0.7653 to 0.7823) | 0.5416 (0.5233 to 0.5611) | 1.0071 (0.9807 to 1.0318) | 0.7310 (0.7203 to 0.7413) |
| DG-GRU+CTR | 0.7731 (0.7653 to 0.7816) | 0.5403 (0.5220 to 0.5594) | 1.0153 (0.9886 to 1.0404) | 0.7331 (0.7224 to 0.7434) |

Primary cohort: MIMIC-IV, n=6,487; 6,487 patients; 5 outcome classes.

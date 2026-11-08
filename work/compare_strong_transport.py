"""Add paired bootstrap contrasts to saved enhanced-transport predictions."""
from __future__ import annotations

import json

import numpy as np

from ctogt.common import PREDICTIONS, RESULTS
from ctogt.train_evaluate import bootstrap_paired_auc


def main() -> None:
    path = RESULTS / "strong_external_transport.json"
    output = json.loads(path.read_text(encoding="utf-8"))
    for cohort in ("mimic3", "eicu", "eicu_anycad"):
        with np.load(
            PREDICTIONS / f"{cohort}_external_strong_xgb.npz",
            allow_pickle=True,
        ) as strong:
            labels = strong["labels"].astype(int)
            strong_xgb = strong["strong_xgb"].astype(float)
            strong_fusion = strong["locked_75_25_fusion"].astype(float)
        with np.load(
            PREDICTIONS / f"{cohort}_external_xgb_static_ts.npz",
            allow_pickle=True,
        ) as baseline:
            baseline_xgb = baseline["xgb_static_ts"].astype(float)
        with np.load(
            PREDICTIONS / f"{cohort}_external_temporal_ensemble.npz",
            allow_pickle=True,
        ) as baseline:
            baseline_fusion = baseline["temporal_ensemble"].astype(float)
        output["models"][cohort]["contrasts"] = {
            "strong_xgb_minus_baseline_xgb": bootstrap_paired_auc(
                labels,
                strong_xgb,
                baseline_xgb,
                2000,
                2050,
            ),
            "locked_fusion_minus_original_fusion": bootstrap_paired_auc(
                labels,
                strong_fusion,
                baseline_fusion,
                2000,
                2051,
            ),
        }
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

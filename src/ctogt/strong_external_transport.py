"""Transport the enhanced organ-dynamics XGBoost without external tuning."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from .common import PREDICTIONS, RESULTS, ensure_dirs, load_config
from .strong_temporal_holdout import enhanced_features
from .train_evaluate import (
    crossfit_platt,
    fit_xgboost,
    load_dataset,
    metric_block,
)


def main() -> None:
    ensure_dirs()
    config = load_config()
    development = load_dataset("mimic4")
    development_frame, _ = enhanced_features(development)
    model_config = copy.deepcopy(config)
    model_config["xgboost"] = {
        **config["xgboost"],
        "n_estimators": 620,
        "max_depth": 4,
        "learning_rate": 0.02,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "reg_lambda": 4.0,
    }
    output = {"models": {}}
    for cohort in ("mimic3", "eicu", "eicu_anycad"):
        external = load_dataset(cohort)
        external_frame, _ = enhanced_features(external)
        strong_xgb = fit_xgboost(
            development_frame,
            external_frame,
            development.labels,
            model_config,
            7,
        )
        groups = (
            external.cohort.get(
                "hospitalid",
                pd.Series(0, index=external.cohort.index),
            )
            .fillna(0)
            .astype(str)
            .to_numpy()
        )
        recalibrated = crossfit_platt(
            external.labels,
            strong_xgb,
            groups,
            int(config["cv_folds"]),
            7,
        )
        with np.load(
            PREDICTIONS / f"{cohort}_external_transformer.npz",
            allow_pickle=True,
        ) as data:
            transformer = data["transformer"].astype(float)
        locked_fusion = 0.75 * strong_xgb + 0.25 * transformer
        output["models"][cohort] = {
            "n": int(len(external.labels)),
            "events": int(external.labels.sum()),
            "strong_xgb": {
                "unrecalibrated": metric_block(external.labels, strong_xgb),
                "recalibrated": metric_block(external.labels, recalibrated),
            },
            "locked_75_25_fusion": metric_block(external.labels, locked_fusion),
        }
        np.savez_compressed(
            PREDICTIONS / f"{cohort}_external_strong_xgb.npz",
            labels=external.labels,
            stay_id=external.cohort["stay_id"].to_numpy(dtype=np.int64),
            site=groups,
            strong_xgb=strong_xgb.astype(np.float32),
            strong_xgb__recalibrated=recalibrated.astype(np.float32),
            locked_75_25_fusion=locked_fusion.astype(np.float32),
        )
        print(
            f"[strong transport] {cohort}: "
            f"XGB {metric_block(external.labels, strong_xgb)['auroc']:.4f}, "
            f"fusion "
            f"{metric_block(external.labels, locked_fusion)['auroc']:.4f}",
            flush=True,
        )
    (RESULTS / "strong_external_transport.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

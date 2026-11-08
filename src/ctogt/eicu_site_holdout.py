"""Hospital-held-out validation of the enhanced tabular model in expanded eICU."""
from __future__ import annotations

import copy
import json

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .common import PREDICTIONS, RESULTS, ensure_dirs, load_config
from .strong_temporal_holdout import enhanced_features
from .train_evaluate import fit_xgboost, load_dataset, metric_block


def main() -> None:
    ensure_dirs()
    config = load_config()
    dataset = load_dataset("eicu_anycad")
    frame, _ = enhanced_features(dataset)
    sites = (
        dataset.cohort["hospitalid"]
        .fillna("unknown")
        .astype(str)
        .to_numpy()
    )
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=7,
    )
    probability = np.full(len(dataset.labels), np.nan)
    folds = []
    for fold, (train, test) in enumerate(
        splitter.split(
            frame,
            dataset.labels,
            sites,
        )
    ):
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
        probability[test] = fit_xgboost(
            frame.iloc[train],
            frame.iloc[test],
            dataset.labels[train],
            model_config,
            7 + fold,
        )
        folds.append(
            {
                "fold": fold + 1,
                "train_n": int(len(train)),
                "test_n": int(len(test)),
                "train_sites": int(len(np.unique(sites[train]))),
                "test_sites": int(len(np.unique(sites[test]))),
                "test_events": int(dataset.labels[test].sum()),
            }
        )
        print(
            f"[eICU site holdout] fold={fold + 1} "
            f"test_sites={len(np.unique(sites[test]))} "
            f"events={dataset.labels[test].sum()}",
            flush=True,
        )
    metrics = metric_block(dataset.labels, probability)
    output = {
        "design": "hospital-held-out 5-fold validation within expanded eICU",
        "n": int(len(dataset.labels)),
        "events": int(dataset.labels.sum()),
        "sites": int(len(np.unique(sites))),
        "folds": folds,
        "metrics": metrics,
        "note": (
            "This is a local cross-hospital model, not transport of the "
            "MIMIC-IV-trained model."
        ),
    }
    (RESULTS / "eicu_site_holdout.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        PREDICTIONS / "eicu_anycad_site_holdout_enhanced_xgb.npz",
        labels=dataset.labels,
        stay_id=dataset.cohort["stay_id"].to_numpy(dtype=np.int64),
        site=sites,
        probability=probability.astype(np.float32),
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

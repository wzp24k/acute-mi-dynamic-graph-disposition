"""Nested hyperparameter selection and paired bootstrap clinical utility."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from .common import PREDICTIONS, RESULTS, ensure_dirs, load_config
from .temporal_holdout import bootstrap_auc_ci, bootstrap_net_benefit
from .train_evaluate import (
    ALL_FEATURES,
    fit_xgboost,
    load_dataset,
    metric_block,
    prepare_tabular,
)


def candidate_grid(config):
    base = copy.deepcopy(config["xgboost"])
    return [
        base,
        {
            **base,
            "n_estimators": 260,
            "max_depth": 2,
            "learning_rate": 0.04,
            "min_child_weight": 6,
            "reg_lambda": 4.0,
        },
        {
            **base,
            "n_estimators": 420,
            "max_depth": 3,
            "learning_rate": 0.025,
            "min_child_weight": 3,
            "colsample_bytree": 0.7,
        },
    ]


def nested_xgb(dataset, config, outer_folds=5, inner_folds=3, seed=7):
    labels = dataset.labels
    outer = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=seed
    )
    prediction = np.full(len(labels), np.nan)
    selection = []
    candidates = candidate_grid(config)
    for fold_index, (outer_train, outer_test) in enumerate(
        outer.split(np.zeros((len(labels), 1)), labels)
    ):
        inner = StratifiedKFold(
            n_splits=inner_folds, shuffle=True, random_state=seed + fold_index
        )
        scores = np.zeros((len(candidates), inner_folds), dtype=float)
        for inner_fold, (inner_train, inner_valid) in enumerate(
            inner.split(np.zeros((len(outer_train), 1)), labels[outer_train])
        ):
            train_rows = outer_train[inner_train]
            valid_rows = outer_train[inner_valid]
            train_x, valid_x = prepare_tabular(
                dataset.frame(ALL_FEATURES).iloc[train_rows],
                dataset.frame(ALL_FEATURES).iloc[valid_rows],
            )
            for candidate_index, parameters in enumerate(candidates):
                model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=seed + fold_index,
                    n_jobs=8,
                    **parameters,
                )
                model.fit(train_x, labels[train_rows], verbose=False)
                probability = model.predict_proba(valid_x)[:, 1]
                scores[candidate_index, inner_fold] = metric_block(
                    labels[valid_rows], probability
                )["log_loss"]
        best_index = int(np.argmin(scores.mean(axis=1)))
        chosen = candidates[best_index]
        train_x, test_x = prepare_tabular(
            dataset.frame(ALL_FEATURES).iloc[outer_train],
            dataset.frame(ALL_FEATURES).iloc[outer_test],
        )
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed + fold_index,
            n_jobs=8,
            **chosen,
        )
        model.fit(train_x, labels[outer_train], verbose=False)
        prediction[outer_test] = model.predict_proba(test_x)[:, 1]
        selection.append(
            {
                "fold": fold_index + 1,
                "candidate": best_index,
                "parameters": chosen,
                "inner_log_loss": scores.mean(axis=1).tolist(),
            }
        )
    return prediction, selection


def paired_difference(labels, first, second, seed=7, replicates=2000):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        index = rng.integers(0, len(labels), len(labels))
        sampled = labels[index]
        if len(np.unique(sampled)) < 2:
            continue
        values.append(
            roc_auc_score(sampled, first[index])
            - roc_auc_score(sampled, second[index])
        )
    return {
        "estimate": float(
            roc_auc_score(labels, first) - roc_auc_score(labels, second)
        ),
        "ci": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
    }


def utility_for_prediction(labels, probability):
    return {
        "metrics": {
            **metric_block(labels, probability),
            "auroc_ci": bootstrap_auc_ci(labels, probability),
        },
        "net_benefit_ci": bootstrap_net_benefit(labels, probability),
    }


def main() -> None:
    ensure_dirs()
    config = load_config()
    dataset = load_dataset("mimic4")
    with np.load(PREDICTIONS / "mimic4_cv_transformer.npz", allow_pickle=True) as data:
        transformer = data["probability"].astype(float)
    with np.load(PREDICTIONS / "mimic4_cv_xgb_static_ts.npz", allow_pickle=True) as data:
        standard_xgb = data["probability"].astype(float)
    nested_xgb_probability, selection = nested_xgb(dataset, config)
    nested_ensemble = (nested_xgb_probability + transformer) / 2
    output = {
        "selection": selection,
        "development": {
            "standard_xgb_ts": utility_for_prediction(dataset.labels, standard_xgb),
            "nested_xgb_ts": utility_for_prediction(
                dataset.labels, nested_xgb_probability
            ),
            "nested_temporal_ensemble": utility_for_prediction(
                dataset.labels, nested_ensemble
            ),
        },
        "contrasts": {
            "nested_ensemble_minus_standard_xgb": paired_difference(
                dataset.labels, nested_ensemble, standard_xgb
            ),
            "nested_xgb_minus_standard_xgb": paired_difference(
                dataset.labels, nested_xgb_probability, standard_xgb
            ),
        },
        "external_clinical_utility": {},
    }
    for cohort in ("mimic3", "eicu", "eicu_anycad"):
        path = PREDICTIONS / f"{cohort}_external_temporal_ensemble.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=True) as data:
            labels = data["labels"].astype(int)
            probability = data["temporal_ensemble"].astype(float)
        output["external_clinical_utility"][cohort] = utility_for_prediction(
            labels, probability
        )
    (RESULTS / "nested_selection_and_utility.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        PREDICTIONS / "mimic4_cv_nested_xgb_ts.npz",
        labels=dataset.labels,
        stay_id=dataset.cohort["stay_id"].to_numpy(dtype=np.int64),
        probability=nested_xgb_probability.astype(np.float32),
    )
    np.savez_compressed(
        PREDICTIONS / "mimic4_cv_nested_temporal_ensemble.npz",
        labels=dataset.labels,
        stay_id=dataset.cohort["stay_id"].to_numpy(dtype=np.int64),
        probability=nested_ensemble.astype(np.float32),
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

"""Compare additional strong tabular learners on the internal era split."""
from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer

from .common import PREDICTIONS, RESULTS, ensure_dirs, load_config
from .strong_temporal_holdout import enhanced_features
from .temporal_holdout import load_years
from .train_evaluate import (
    bootstrap_paired_auc,
    load_dataset,
    metric_block,
)


def fit_predict(model, train_x, train_y, validation_x):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_values = imputer.fit_transform(train_x)
    validation_values = imputer.transform(validation_x)
    model.fit(train_values, train_y)
    return model.predict_proba(validation_values)[:, 1]


def main() -> None:
    ensure_dirs()
    config = load_config()
    dataset = load_dataset("mimic4")
    groups = load_years(dataset)
    frame, _ = enhanced_features(dataset)
    development_rows = np.flatnonzero(
        np.isin(groups, ["2008 - 2010", "2011 - 2013", "2014 - 2016"])
    )
    validation_rows = np.flatnonzero(groups == "2017 - 2019")
    final_rows = np.flatnonzero(groups == "2020 - 2022")

    models = {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=500,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=7,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=700,
            max_features=0.5,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=14,
            random_state=7,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=700,
            max_features=0.4,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=14,
            random_state=7,
        ),
    }
    candidates = []
    validation_predictions = {}
    for name, model in models.items():
        probability = fit_predict(
            model,
            frame.iloc[development_rows],
            dataset.labels[development_rows],
            frame.iloc[validation_rows],
        )
        metrics = metric_block(dataset.labels[validation_rows], probability)
        candidates.append({"name": name, **metrics})
        validation_predictions[name] = probability
        print(
            f"[strong tabular validation] {name} AUROC={metrics['auroc']:.4f}",
            flush=True,
        )
    selected = max(candidates, key=lambda record: record["auroc"])
    selected_model = models[selected["name"]]
    final_probability = fit_predict(
        selected_model,
        frame.iloc[np.flatnonzero(groups != "2020 - 2022")],
        dataset.labels[groups != "2020 - 2022"],
        frame.iloc[final_rows],
    )
    with np.load(
        PREDICTIONS / "mimic4_temporal_holdout.npz",
        allow_pickle=True,
    ) as data:
        labels = data["labels"].astype(int)
        baseline_xgb = data["xgb_static_ts"].astype(float)
        transformer = data["transformer"].astype(float)
    output = {
        "validation_n": int(len(validation_rows)),
        "validation_events": int(dataset.labels[validation_rows].sum()),
        "candidates": candidates,
        "locked_selected_model": selected["name"],
        "final_holdout": {
            "n": int(len(final_rows)),
            "events": int(labels.sum()),
            "selected_model": metric_block(labels, final_probability),
            "selected_minus_baseline_xgb": bootstrap_paired_auc(
                labels,
                final_probability,
                baseline_xgb,
                2000,
                2030,
            ),
            "selected_minus_transformer": bootstrap_paired_auc(
                labels,
                final_probability,
                transformer,
                2000,
                2031,
            ),
        },
    }
    (RESULTS / "strong_tabular_candidates.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        PREDICTIONS / "mimic4_internal_era_validation_strong_tabular.npz",
        labels=dataset.labels[validation_rows],
        stay_id=dataset.cohort["stay_id"].iloc[validation_rows].to_numpy(
            dtype=np.int64
        ),
        **validation_predictions,
    )
    np.savez_compressed(
        PREDICTIONS / "mimic4_temporal_holdout_strong_tabular.npz",
        labels=labels,
        stay_id=np.load(
            PREDICTIONS / "mimic4_temporal_holdout.npz",
            allow_pickle=True,
        )["stay_id"].astype(np.int64),
        baseline_xgb=baseline_xgb.astype(np.float32),
        selected=final_probability.astype(np.float32),
        transformer=transformer.astype(np.float32),
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

"""Locked temporal holdout using real MIMIC-IV anchor-year groups."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .common import CLINICAL_PRIOR_EDGES, MIMIC4, ORGAN_CHANNELS, PREDICTIONS, RESULTS, ensure_dirs, load_config
from .models import (
    DeepConfig,
    SequenceStandardizer,
    build_prior_adjacency,
    predict_deep_model,
    train_deep_model,
)
from .train_evaluate import (
    ALL_FEATURES,
    fit_xgboost,
    load_dataset,
    metric_block,
)


def load_years(dataset):
    patients = pd.read_csv(
        MIMIC4 / "hosp" / "patients.csv.gz",
        usecols=["subject_id", "anchor_year_group"],
    ).rename(columns={"subject_id": "patient_id"})
    return dataset.cohort[["patient_id"]].merge(
        patients, on="patient_id", how="left", validate="many_to_one"
    )["anchor_year_group"].fillna("unknown").astype(str).to_numpy()


def fit_transformer(
    dataset,
    train_rows,
    test_rows,
    config,
    seeds=(7, 42, 2026),
):
    probabilities = []
    for seed in seeds:
        inner_train, inner_valid = train_test_split(
            train_rows,
            test_size=0.12,
            random_state=seed,
            stratify=dataset.labels[train_rows],
        )
        standardizer = SequenceStandardizer.fit(
            dataset.values[inner_train],
            dataset.masks[inner_train],
            dataset.frame(ALL_FEATURES).to_numpy(dtype=np.float32)[inner_train],
        )
        train_values, train_masks = standardizer.transform_sequence(
            dataset.values[inner_train], dataset.masks[inner_train]
        )
        valid_values, valid_masks = standardizer.transform_sequence(
            dataset.values[inner_valid], dataset.masks[inner_valid]
        )
        train_static = standardizer.transform_static(
            dataset.frame(ALL_FEATURES).to_numpy(dtype=np.float32)[inner_train]
        )
        valid_static = standardizer.transform_static(
            dataset.frame(ALL_FEATURES).to_numpy(dtype=np.float32)[inner_valid]
        )
        model = train_deep_model(
            train_values,
            train_masks,
            train_static,
            dataset.labels[inner_train],
            dataset.environments[inner_train],
            valid_values,
            valid_masks,
            valid_static,
            dataset.labels[inner_valid],
            dataset.environments[inner_valid],
            config=DeepConfig(architecture="transformer"),
            prior=build_prior_adjacency(ORGAN_CHANNELS, CLINICAL_PRIOR_EDGES),
            hidden=int(config["hidden_size"]),
            dropout=float(config["dropout"]),
            epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
            patience=int(config["patience"]),
            lambda_smooth=float(config["lambda_smooth"]),
            lambda_drift=float(config["lambda_drift"]),
            lambda_environment=float(config["lambda_environment"]),
            lambda_graph_sparsity=float(config["lambda_graph_sparsity"]),
            seed=seed,
        )
        test_values, test_masks = standardizer.transform_sequence(
            dataset.values[test_rows], dataset.masks[test_rows]
        )
        test_static = standardizer.transform_static(
            dataset.frame(ALL_FEATURES).to_numpy(dtype=np.float32)[test_rows]
        )
        output = predict_deep_model(
            model,
            test_values,
            test_masks,
            test_static,
            config=DeepConfig(architecture="transformer"),
            seed=seed,
        )
        probabilities.append(output["probability"])
    return np.mean(probabilities, axis=0)


def bootstrap_auc_ci(labels, probability, seed=7, replicates=2000):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        index = rng.integers(0, len(labels), len(labels))
        sampled = labels[index]
        if len(np.unique(sampled)) < 2:
            continue
        values.append(roc_auc_score(sampled, probability[index]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap_net_benefit(
    labels,
    probability,
    thresholds=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    replicates=2000,
    seed=7,
):
    rng = np.random.default_rng(seed)
    output = {}
    n = len(labels)
    for threshold in thresholds:
        predicted = probability >= threshold
        point_tp = int(np.sum(predicted & (labels == 1)))
        point_fp = int(np.sum(predicted & (labels == 0)))
        point = point_tp / n - point_fp / n * threshold / (1 - threshold)
        samples = []
        for _ in range(replicates):
            index = rng.integers(0, n, n)
            sampled_labels = labels[index]
            sampled_prediction = predicted[index]
            tp = int(np.sum(sampled_prediction & (sampled_labels == 1)))
            fp = int(np.sum(sampled_prediction & (sampled_labels == 0)))
            samples.append(tp / n - fp / n * threshold / (1 - threshold))
        output[str(threshold)] = {
            "estimate": float(point),
            "ci": [
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)),
            ],
        }
    return output


def main() -> None:
    ensure_dirs()
    config = load_config()
    dataset = load_dataset("mimic4")
    groups = load_years(dataset)
    test_mask = groups == "2020 - 2022"
    if test_mask.sum() == 0:
        raise RuntimeError("2020-2022 temporal holdout group is missing")
    train_rows = np.flatnonzero(~test_mask)
    test_rows = np.flatnonzero(test_mask)
    labels = dataset.labels[test_rows]
    xgb_probability = fit_xgboost(
        dataset.frame(ALL_FEATURES).iloc[train_rows],
        dataset.frame(ALL_FEATURES).iloc[test_rows],
        dataset.labels[train_rows],
        config,
        7,
    )
    transformer_probability = fit_transformer(
        dataset,
        train_rows,
        test_rows,
        config,
    )
    ensemble_probability = (xgb_probability + transformer_probability) / 2
    results = {}
    for name, probability in (
        ("xgb_static_ts", xgb_probability),
        ("transformer", transformer_probability),
        ("temporal_ensemble", ensemble_probability),
    ):
        metrics = metric_block(labels, probability)
        metrics["auroc_ci"] = bootstrap_auc_ci(labels, probability)
        results[name] = {
            "metrics": metrics,
            "net_benefit_ci": bootstrap_net_benefit(labels, probability),
        }
    rng = np.random.default_rng(2026)
    differences = []
    for _ in range(2000):
        index = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[index])) < 2:
            continue
        differences.append(
            roc_auc_score(labels[index], ensemble_probability[index])
            - roc_auc_score(labels[index], xgb_probability[index])
        )
    output = {
        "split": {
            "development_groups": sorted(set(groups[train_rows])),
            "holdout_groups": sorted(set(groups[test_rows])),
            "train_n": int(len(train_rows)),
            "train_events": int(dataset.labels[train_rows].sum()),
            "test_n": int(len(test_rows)),
            "test_events": int(labels.sum()),
        },
        "models": results,
        "ensemble_minus_xgb_static_ts": {
            "estimate": float(
                roc_auc_score(labels, ensemble_probability)
                - roc_auc_score(labels, xgb_probability)
            ),
            "ci": [
                float(np.percentile(differences, 2.5)),
                float(np.percentile(differences, 97.5)),
            ],
        },
    }
    (RESULTS / "temporal_holdout_validation.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        PREDICTIONS / "mimic4_temporal_holdout.npz",
        labels=labels,
        stay_id=dataset.cohort["stay_id"].iloc[test_rows].to_numpy(dtype=np.int64),
        xgb_static_ts=xgb_probability.astype(np.float32),
        transformer=transformer_probability.astype(np.float32),
        temporal_ensemble=ensemble_probability.astype(np.float32),
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

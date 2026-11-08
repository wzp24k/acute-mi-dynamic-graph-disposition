"""Gate an organ-rate graph on 2017-2019 before 2020-2022 evaluation."""
from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import train_test_split

from .common import CLINICAL_PRIOR_EDGES, ORGAN_CHANNELS, PREDICTIONS, RESULTS, ensure_dirs, load_config
from .models import (
    DeepConfig,
    SequenceStandardizer,
    build_prior_adjacency,
    predict_deep_model,
    train_deep_model,
)
from .temporal_holdout import load_years
from .train_evaluate import ALL_FEATURES, load_dataset, metric_block


CONFIG = DeepConfig(
    architecture="organ_graph",
    graph_mode="dynamic",
    include_continuous_time=True,
    use_environment_head=False,
    use_environment_penalty=False,
    use_organ_rates=True,
)


def fit_predict(dataset, train_rows, test_rows, config, seed=7):
    fit_rows, stop_rows = train_test_split(
        train_rows,
        test_size=0.12,
        random_state=seed,
        stratify=dataset.labels[train_rows],
    )
    static = dataset.frame(ALL_FEATURES).to_numpy(dtype=np.float32)
    standardizer = SequenceStandardizer.fit(
        dataset.values[fit_rows],
        dataset.masks[fit_rows],
        static[fit_rows],
    )
    fit_values, fit_masks = standardizer.transform_sequence(
        dataset.values[fit_rows], dataset.masks[fit_rows]
    )
    stop_values, stop_masks = standardizer.transform_sequence(
        dataset.values[stop_rows], dataset.masks[stop_rows]
    )
    test_values, test_masks = standardizer.transform_sequence(
        dataset.values[test_rows], dataset.masks[test_rows]
    )
    fit_static = standardizer.transform_static(static[fit_rows])
    stop_static = standardizer.transform_static(static[stop_rows])
    test_static = standardizer.transform_static(static[test_rows])
    model = train_deep_model(
        fit_values,
        fit_masks,
        fit_static,
        dataset.labels[fit_rows],
        dataset.environments[fit_rows],
        stop_values,
        stop_masks,
        stop_static,
        dataset.labels[stop_rows],
        dataset.environments[stop_rows],
        config=CONFIG,
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
    return predict_deep_model(
        model,
        test_values,
        test_masks,
        test_static,
        config=CONFIG,
        seed=seed + 9_000_000,
    )["probability"]


def main() -> None:
    ensure_dirs()
    config = load_config()
    dataset = load_dataset("mimic4")
    groups = load_years(dataset)
    development_rows = np.flatnonzero(
        np.isin(groups, ["2008 - 2010", "2011 - 2013", "2014 - 2016"])
    )
    validation_rows = np.flatnonzero(groups == "2017 - 2019")
    final_rows = np.flatnonzero(groups == "2020 - 2022")
    validation_probability = fit_predict(
        dataset,
        development_rows,
        validation_rows,
        config,
    )
    validation_metrics = metric_block(
        dataset.labels[validation_rows],
        validation_probability,
    )
    gate = 0.7380239015689462
    output = {
        "architecture": "seven-organ token graph with organ rates and elapsed time",
        "validation_n": int(len(validation_rows)),
        "validation_events": int(dataset.labels[validation_rows].sum()),
        "validation": validation_metrics,
        "gate_auroc": gate,
        "gate_passed": bool(validation_metrics["auroc"] > gate),
        "final_holdout": None,
    }
    if output["gate_passed"]:
        final_probability = fit_predict(
            dataset,
            np.flatnonzero(groups != "2020 - 2022"),
            final_rows,
            config,
            seed=7,
        )
        output["final_holdout"] = {
            "n": int(len(final_rows)),
            "events": int(dataset.labels[final_rows].sum()),
            **metric_block(dataset.labels[final_rows], final_probability),
        }
        np.savez_compressed(
            PREDICTIONS / "mimic4_temporal_holdout_organ_dynamics.npz",
            labels=dataset.labels[final_rows],
            stay_id=dataset.cohort["stay_id"].iloc[final_rows].to_numpy(
                dtype=np.int64
            ),
            probability=final_probability.astype(np.float32),
        )
    (RESULTS / "organ_dynamics_holdout.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        PREDICTIONS / "mimic4_internal_era_validation_organ_dynamics.npz",
        labels=dataset.labels[validation_rows],
        stay_id=dataset.cohort["stay_id"].iloc[validation_rows].to_numpy(
            dtype=np.int64
        ),
        probability=validation_probability.astype(np.float32),
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

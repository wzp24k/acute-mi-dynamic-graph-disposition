"""Nested Transformer tuning and joint nested temporal-ensemble evaluation."""
from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .common import CLINICAL_PRIOR_EDGES, ORGAN_CHANNELS, PREDICTIONS, RESULTS, ensure_dirs, load_config
from .models import (
    SequenceStandardizer,
    build_prior_adjacency,
    predict_deep_model,
    train_deep_model,
)
from .nested_selection import (
    paired_difference,
    utility_for_prediction,
)
from .train_evaluate import (
    ALL_FEATURES,
    DEEP_CONFIGS,
    load_dataset,
    metric_block,
    split_validation,
)


def transformer_candidates(config: dict) -> list[dict[str, float | int]]:
    return [
        {
            "hidden": int(config["hidden_size"]),
            "dropout": float(config["dropout"]),
            "learning_rate": float(config["learning_rate"]),
        },
        {
            "hidden": 32,
            "dropout": 0.15,
            "learning_rate": 0.0012,
        },
    ]


def fit_candidate(
    dataset,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    candidate: dict[str, float | int],
    config: dict,
    seed: int,
) -> np.ndarray:
    fit_rows, early_stop_rows = split_validation(
        train_rows,
        dataset.labels,
        seed,
    )
    static = dataset.frame(ALL_FEATURES).to_numpy(dtype=np.float32)
    standardizer = SequenceStandardizer.fit(
        dataset.values[fit_rows],
        dataset.masks[fit_rows],
        static[fit_rows],
    )
    fit_values, fit_masks = standardizer.transform_sequence(
        dataset.values[fit_rows],
        dataset.masks[fit_rows],
    )
    fit_static = standardizer.transform_static(static[fit_rows])
    stop_values, stop_masks = standardizer.transform_sequence(
        dataset.values[early_stop_rows],
        dataset.masks[early_stop_rows],
    )
    stop_static = standardizer.transform_static(static[early_stop_rows])
    validation_values, validation_masks = standardizer.transform_sequence(
        dataset.values[validation_rows],
        dataset.masks[validation_rows],
    )
    validation_static = standardizer.transform_static(static[validation_rows])
    model = train_deep_model(
        fit_values,
        fit_masks,
        fit_static,
        dataset.labels[fit_rows],
        dataset.environments[fit_rows],
        stop_values,
        stop_masks,
        stop_static,
        dataset.labels[early_stop_rows],
        dataset.environments[early_stop_rows],
        config=DEEP_CONFIGS["transformer"],
        prior=build_prior_adjacency(ORGAN_CHANNELS, CLINICAL_PRIOR_EDGES),
        hidden=int(candidate["hidden"]),
        dropout=float(candidate["dropout"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(candidate["learning_rate"]),
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
        validation_values,
        validation_masks,
        validation_static,
        config=DEEP_CONFIGS["transformer"],
        seed=seed + 9_000_000,
    )["probability"]


def nested_transformer(
    dataset,
    config: dict,
    outer_folds: int = 5,
    inner_folds: int = 1,
    seed: int = 7,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    labels = dataset.labels
    outer = StratifiedKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=seed,
    )
    prediction = np.full(len(labels), np.nan)
    selection = []
    candidates = transformer_candidates(config)
    for fold_index, (outer_train, outer_test) in enumerate(
        outer.split(np.zeros((len(labels), 1)), labels)
    ):
        inner_train, inner_valid = train_test_split(
            outer_train,
            test_size=0.2,
            random_state=seed + fold_index,
            stratify=labels[outer_train],
        )
        scores = np.zeros(len(candidates), dtype=float)
        aurocs = np.zeros(len(candidates), dtype=float)
        for candidate_index, candidate in enumerate(candidates):
            probability = fit_candidate(
                dataset,
                inner_train,
                inner_valid,
                candidate,
                config,
                seed + 100 * fold_index + candidate_index,
            )
            metrics = metric_block(labels[inner_valid], probability)
            scores[candidate_index] = metrics["log_loss"]
            aurocs[candidate_index] = metrics["auroc"]
            print(
                f"  nested-transformer fold={fold_index + 1} "
                f"candidate={candidate_index} "
                f"logloss={metrics['log_loss']:.4f}",
                flush=True,
            )
        best_index = int(np.argmin(scores))
        chosen = candidates[best_index]
        prediction[outer_test] = fit_candidate(
            dataset,
            outer_train,
            outer_test,
            chosen,
            config,
            seed + 1000 + fold_index,
        )
        selection.append(
            {
                "fold": fold_index + 1,
                "candidate": best_index,
                "parameters": chosen,
                "inner_folds": int(inner_folds),
                "inner_log_loss": scores.tolist(),
                "inner_auroc": aurocs.tolist(),
            }
        )
        print(
            f"nested-transformer fold={fold_index + 1} "
            f"selected={chosen}",
            flush=True,
        )
    if np.isnan(prediction).any():
        raise RuntimeError("Incomplete nested Transformer OOF predictions")
    return prediction, selection


def main() -> None:
    ensure_dirs()
    config = load_config()
    dataset = load_dataset("mimic4")
    with np.load(
        PREDICTIONS / "mimic4_cv_xgb_static_ts.npz",
        allow_pickle=True,
    ) as data:
        standard_xgb = data["probability"].astype(float)
    with np.load(
        PREDICTIONS / "mimic4_cv_nested_xgb_ts.npz",
        allow_pickle=True,
    ) as data:
        nested_xgb = data["probability"].astype(float)

    nested_transformer_probability, selection = nested_transformer(
        dataset,
        config,
    )
    joint_nested_ensemble = (nested_xgb + nested_transformer_probability) / 2
    output = {
        "transformer_selection": selection,
        "development": {
            "nested_transformer": utility_for_prediction(
                dataset.labels,
                nested_transformer_probability,
            ),
            "joint_nested_temporal_ensemble": utility_for_prediction(
                dataset.labels,
                joint_nested_ensemble,
            ),
        },
        "contrasts": {
            "joint_nested_ensemble_minus_standard_xgb": paired_difference(
                dataset.labels,
                joint_nested_ensemble,
                standard_xgb,
            ),
            "joint_nested_ensemble_minus_nested_xgb": paired_difference(
                dataset.labels,
                joint_nested_ensemble,
                nested_xgb,
            ),
        },
    }
    (RESULTS / "nested_transformer_selection.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        PREDICTIONS / "mimic4_cv_nested_transformer.npz",
        labels=dataset.labels,
        stay_id=dataset.cohort["stay_id"].to_numpy(dtype=np.int64),
        probability=nested_transformer_probability.astype(np.float32),
    )
    np.savez_compressed(
        PREDICTIONS / "mimic4_cv_joint_nested_temporal_ensemble.npz",
        labels=dataset.labels,
        stay_id=dataset.cohort["stay_id"].to_numpy(dtype=np.int64),
        probability=joint_nested_ensemble.astype(np.float32),
    )

    combined_path = RESULTS / "nested_selection_and_utility.json"
    combined = (
        json.loads(combined_path.read_text(encoding="utf-8"))
        if combined_path.is_file()
        else {"development": {}, "contrasts": {}}
    )
    combined.setdefault("development", {}).update(output["development"])
    combined.setdefault("contrasts", {}).update(output["contrasts"])
    combined["transformer_selection"] = selection
    combined_path.write_text(
        json.dumps(combined, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

"""Train, cross-validate, transport, and audit AMI prediction models."""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold

from .common import (
    DEMOGRAPHIC_FEATURES,
    LAB_NAMES,
    PROCESSED,
    RESULTS,
    ensure_dirs,
    load_config,
    static_feature_columns,
    transform_lab_for_model,
)
from .evaluation import (
    bootstrap_binary_auc_ci,
    bootstrap_metric_ci,
    calibration_slope_intercept,
    cumulative_nri_idi,
    decision_curve,
    delong_per_class,
    paired_bootstrap_delta,
    point_metrics,
    reliability_bins,
)
from .models import (
    SequencePreprocessor,
    TabularPreprocessor,
    predict_tabular_model,
    predict_temporal_model,
    train_tabular_model,
    train_temporal_model,
)


warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_num_threads(
    int(os.environ.get("AMI_TORCH_THREADS", str(max(1, min(6, os.cpu_count() or 1)))))
)

TEMPORAL_MODELS = {
    "gru": {"architecture": "gru", "input_kind": "full", "regularized": False},
    "grud": {"architecture": "grud", "input_kind": "full", "regularized": False},
    "transformer": {
        "architecture": "transformer",
        "input_kind": "full",
        "regularized": False,
    },
    "stgcn_lite": {
        "architecture": "stgcn_lite",
        "input_kind": "full",
        "regularized": False,
    },
    "dggru_static": {
        "architecture": "dggru",
        "input_kind": "full",
        "regularized": False,
        "use_sequence": False,
    },
    "dggru": {"architecture": "dggru", "input_kind": "full", "regularized": False},
    "dggru_tr": {
        "architecture": "dggru",
        "input_kind": "full",
        "regularized": True,
        "regularization_kind": "scalar_trajectory",
    },
    "dggru_ctr": {
        "architecture": "dggru",
        "input_kind": "full",
        "regularized": True,
        "regularization_kind": "cumulative_curvature",
    },
}
TABULAR_MODELS = {
    "xgb_static": {"input_kind": "static"},
    "xgb_static_ts": {"input_kind": "full"},
    "mlp_static": {"input_kind": "static"},
    "mlp_static_ts": {"input_kind": "full"},
}
MODEL_ORDER = (
    "xgb_static",
    "xgb_static_ts",
    "mlp_static",
    "mlp_static_ts",
    "gru",
    "grud",
    "transformer",
    "stgcn_lite",
    "dggru_static",
    "dggru",
    "dggru_tr",
    "dggru_ctr",
)
XGB_DEFAULTS = {
    "n_estimators": 260,
    "max_depth": 3,
    "learning_rate": 0.035,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
}
TRAIN_DEFAULTS = {
    "epochs": 20,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "hidden": 64,
    "edge_dim": 16,
    "dropout": 0.2,
}
DEFAULT_LAMBDA_CURVATURE = 0.01


def _collapse_five_to_four_probabilities(
    probability_5: np.ndarray,
) -> np.ndarray:
    probability = np.asarray(probability_5, dtype=float)
    if probability.ndim != 2 or probability.shape[1] != 5:
        raise ValueError("Expected five-class probability matrix")
    return np.stack(
        [
            probability[:, 0],
            probability[:, 1:3].sum(axis=1),
            probability[:, 3],
            probability[:, 4],
        ],
        axis=1,
    )


def _json_default(value: object):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _static_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    matrix = np.empty((len(frame), len(columns)), dtype=np.float64)
    for column_index, column in enumerate(columns):
        if column not in frame:
            matrix[:, column_index] = np.nan
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if column.startswith("lab_") and not column.endswith("_missing"):
            parts = column.split("_")
            if len(parts) >= 3 and parts[-1] in {"first", "min", "max", "mean"}:
                lab_name = "_".join(parts[1:-1])
                values = np.asarray(
                    [transform_lab_for_model(lab_name, value) for value in values],
                    dtype=float,
                )
        matrix[:, column_index] = values
    return matrix


def load_dataset(name: str) -> dict:
    directory = PROCESSED / name
    cohort = pd.read_csv(directory / "cohort.csv")
    sequence_data = np.load(directory / "sequences.npz")
    values = sequence_data["values"].astype(np.float32)
    masks = sequence_data["masks"].astype(np.float32)
    sequence_ids = sequence_data["stay_id"].astype(np.int64)
    if not np.array_equal(
        sequence_ids, cohort["stay_id"].to_numpy(dtype=np.int64)
    ):
        raise ValueError(f"{name}: sequence and cohort IDs differ")
    return {
        "name": name,
        "cohort": cohort,
        "values": values,
        "masks": masks,
    }


def _feature_columns(cohort: pd.DataFrame) -> tuple[list[str], list[str]]:
    lab_columns = [
        column
        for column in cohort.columns
        if column.startswith("lab_") and column != "lab_rows"
    ]
    static_columns = [
        column
        for column in DEMOGRAPHIC_FEATURES + tuple(lab_columns) + ("troponin_assay_t",)
        if column in cohort
    ]
    summary_columns = static_feature_columns(cohort)
    summary_only = [
        column
        for column in summary_columns
        if column not in static_columns and column != "troponin_assay_t"
    ]
    return static_columns, static_columns + summary_only


def _assert_feature_contract(columns: list[str]) -> None:
    forbidden = {
        "outcome",
        "outcome4",
        "outcome_six",
        "disposition_raw",
        "death_at_hospital",
        "icu_out_hour",
        "death_hour_from_icu",
        "hospital_expire_flag",
    }
    leaked = forbidden.intersection(columns)
    if leaked:
        raise ValueError(f"Post-landmark or outcome fields entered predictors: {leaked}")


def _load_cached_prediction(
    path: Path,
    *,
    labels: np.ndarray,
    groups: np.ndarray,
    fold_id: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=True) as cached:
            probability = cached["probabilities"].astype(np.float32)
            cached_labels = cached["labels"].astype(int)
            cached_groups = cached["subject_key"].astype(str)
            cached_fold = cached["fold"].astype(int)
            trajectory = cached["trajectories"].astype(np.float32)
    except (KeyError, OSError, ValueError):
        return None
    expected_shape = (len(labels), n_classes)
    if probability.shape != expected_shape:
        return None
    if not np.array_equal(cached_labels, labels):
        return None
    if not np.array_equal(cached_groups, groups):
        return None
    if not np.array_equal(cached_fold, fold_id):
        return None
    if not np.isfinite(probability).all():
        return None
    cached_trajectory = (
        trajectory if trajectory.ndim == 2 and trajectory.size else None
    )
    return probability, cached_trajectory


def _fit_xgb(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    n_classes: int,
    seed: int,
    n_jobs: int,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=n_jobs,
        **XGB_DEFAULTS,
    )
    model.fit(train_x, train_y, verbose=False)
    return model


def _temporal_predictions(
    train_values: np.ndarray,
    train_masks: np.ndarray,
    train_static: np.ndarray,
    train_labels: np.ndarray,
    test_values: np.ndarray,
    test_masks: np.ndarray,
    test_static: np.ndarray,
    *,
    model_name: str,
    n_classes: int,
    seed: int,
    epochs: int,
    train_defaults: dict,
    lambda_smooth: float,
    lambda_drift: float,
    lambda_curvature: float = 0.01,
    regularization_mode: str = "smooth_drift",
    use_sequence: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    preprocessor = SequencePreprocessor.fit(
        train_values, train_masks, train_static
    )
    sequence_train = preprocessor.transform_sequence(train_values, train_masks)
    sequence_test = preprocessor.transform_sequence(test_values, test_masks)
    if not use_sequence:
        sequence_train = np.zeros_like(sequence_train)
        sequence_test = np.zeros_like(sequence_test)
    static_train = preprocessor.transform_static(train_static)
    static_test = preprocessor.transform_static(test_static)
    specification = TEMPORAL_MODELS[model_name]
    model = train_temporal_model(
        sequence_train,
        static_train,
        train_labels,
        architecture=specification["architecture"],
        use_regularization=bool(specification["regularized"]),
        regularization_kind=str(
            specification.get("regularization_kind", "scalar_trajectory")
        ),
        lambda_smooth=lambda_smooth,
        lambda_drift=lambda_drift,
        lambda_curvature=lambda_curvature,
        regularization_mode=regularization_mode,
        n_classes=n_classes,
        epochs=epochs,
        batch_size=int(train_defaults["batch_size"]),
        learning_rate=float(train_defaults["learning_rate"]),
        weight_decay=float(train_defaults["weight_decay"]),
        hidden=int(train_defaults["hidden"]),
        edge_dim=int(train_defaults["edge_dim"]),
        dropout=float(train_defaults["dropout"]),
        seed=seed,
    )
    return predict_temporal_model(model, sequence_test, static_test)


def _tabular_predictions(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    test_matrix: np.ndarray,
    *,
    n_classes: int,
    seed: int,
    epochs: int,
    train_defaults: dict,
) -> np.ndarray:
    preprocessor = TabularPreprocessor.fit(train_matrix)
    model = train_tabular_model(
        preprocessor.transform(train_matrix),
        train_labels,
        n_classes=n_classes,
        epochs=epochs,
        batch_size=int(train_defaults["batch_size"]),
        learning_rate=float(train_defaults["learning_rate"]),
        weight_decay=float(train_defaults["weight_decay"]),
        hidden=128,
        dropout=float(train_defaults["dropout"]),
        seed=seed,
    )
    return predict_tabular_model(model, preprocessor.transform(test_matrix))


def run_cv(
    dataset_name: str,
    *,
    seeds: list[int],
    folds: int,
    epochs: int,
    models: list[str],
    max_rows: int | None,
    lambda_smooth: float,
    lambda_drift: float,
    lambda_curvature: float,
    resume: bool,
    force: bool,
) -> dict:
    data = load_dataset(dataset_name)
    cohort = data["cohort"]
    values = data["values"]
    masks = data["masks"]
    if max_rows is not None and len(cohort) > max_rows:
        rng = np.random.default_rng(20261104)
        selected = np.sort(rng.choice(len(cohort), size=max_rows, replace=False))
        cohort = cohort.iloc[selected].reset_index(drop=True)
        values = values[selected]
        masks = masks[selected]
    labels = cohort["outcome"].to_numpy(dtype=int)
    groups = cohort["subject_key"].astype(str).to_numpy()
    static_columns, full_columns = _feature_columns(cohort)
    _assert_feature_contract(static_columns)
    _assert_feature_contract(full_columns)
    static_matrix = _static_matrix(cohort, static_columns)
    full_matrix = _static_matrix(cohort, full_columns)
    n_classes = int(labels.max()) + 1
    predictions: dict[str, dict[int, np.ndarray]] = {
        model: {} for model in models
    }
    trajectories: dict[str, dict[int, np.ndarray]] = {
        model: {} for model in TEMPORAL_MODELS if model in models
    }
    for seed in seeds:
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=seed
        )
        fold_id = np.full(len(cohort), -1, dtype=int)
        for fold, (_, test_index) in enumerate(
            splitter.split(full_matrix, labels, groups)
        ):
            fold_id[test_index] = fold
        if (fold_id < 0).any():
            raise ValueError("Cross-validation did not assign every row to a test fold")
        for model_name in models:
            oof_probability = np.full(
                (len(cohort), n_classes), np.nan, dtype=np.float32
            )
            oof_trajectory = (
                np.full((len(cohort), values.shape[1]), np.nan, dtype=np.float32)
                if model_name in trajectories
                else None
            )
            prediction_path = (
                RESULTS
                / "predictions"
                / f"{dataset_name}_{model_name}_seed{seed}.npz"
            )
            cached = (
                None
                if force or not resume
                else _load_cached_prediction(
                    prediction_path,
                    labels=labels,
                    groups=groups,
                    fold_id=fold_id,
                    n_classes=n_classes,
                )
            )
            if cached is not None:
                oof_probability, oof_trajectory = cached
                status = "cached"
            else:
                for fold in range(folds):
                    train = np.flatnonzero(fold_id != fold)
                    test = np.flatnonzero(fold_id == fold)
                    if model_name in TABULAR_MODELS:
                        input_kind = TABULAR_MODELS[model_name]["input_kind"]
                        matrix = (
                            static_matrix if input_kind == "static" else full_matrix
                        )
                        if model_name.startswith("mlp"):
                            probability = _tabular_predictions(
                                matrix[train],
                                labels[train],
                                matrix[test],
                                n_classes=n_classes,
                                seed=seed + fold,
                                epochs=epochs,
                                train_defaults=TRAIN_DEFAULTS,
                            )
                            if probability.shape[1] != n_classes:
                                raise ValueError(
                                    f"{model_name}: predicted "
                                    f"{probability.shape[1]} classes, "
                                    f"expected {n_classes}"
                                )
                            oof_probability[test] = probability
                            continue
                        model = _fit_xgb(
                            matrix[train],
                            labels[train],
                            n_classes=n_classes,
                            seed=seed + fold,
                            n_jobs=int(os.environ.get("AMI_XGB_THREADS", "6")),
                        )
                        probability = model.predict_proba(matrix[test])
                    else:
                        temporal_input_kind = TEMPORAL_MODELS[model_name][
                            "input_kind"
                        ]
                        temporal_static = (
                            static_matrix
                            if temporal_input_kind == "static"
                            else full_matrix
                        )
                        probability, trajectory = _temporal_predictions(
                            values[train],
                            masks[train],
                            temporal_static[train],
                            labels[train],
                            values[test],
                            masks[test],
                            temporal_static[test],
                            model_name=model_name,
                            n_classes=n_classes,
                            seed=seed + fold,
                            epochs=epochs,
                            train_defaults=TRAIN_DEFAULTS,
                            lambda_smooth=lambda_smooth,
                            lambda_drift=lambda_drift,
                            lambda_curvature=lambda_curvature,
                            use_sequence=bool(
                                TEMPORAL_MODELS[model_name].get(
                                    "use_sequence", True
                                )
                            ),
                        )
                        if oof_trajectory is not None:
                            oof_trajectory[test] = trajectory
                    if probability.shape[1] != n_classes:
                        raise ValueError(
                            f"{model_name}: predicted {probability.shape[1]} classes, "
                            f"expected {n_classes}"
                        )
                    oof_probability[test] = probability
                status = "trained"
            predictions[model_name][seed] = oof_probability
            if oof_trajectory is not None:
                trajectories[model_name][seed] = oof_trajectory
            seed_metrics = point_metrics(labels, oof_probability)
            _write_json(
                RESULTS / "predictions" / f"{dataset_name}_{model_name}_seed{seed}.json",
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": seed,
                    "fold_seed": seed,
                    "status": status,
                    "metrics": {
                        key: value
                        for key, value in seed_metrics.items()
                        if key != "cumulative_probabilities"
                    },
                },
            )
            np.savez_compressed(
                prediction_path,
                labels=labels,
                probabilities=oof_probability,
                trajectories=(
                    oof_trajectory
                    if oof_trajectory is not None
                    else np.empty((0, 0), dtype=np.float32)
                ),
                fold=fold_id,
                fold_seed=np.asarray(seed),
                stay_id=cohort["stay_id"].to_numpy(dtype=np.int64),
                subject_key=groups,
            )
            print(
                f"[{dataset_name}] model={model_name} seed={seed} "
                f"status={status} macro_auc={seed_metrics['macro_auc']:.4f}",
                flush=True,
            )
    averaged = {
        model: np.mean(
            np.stack([predictions[model][seed] for seed in seeds]), axis=0
        )
        for model in models
    }
    metrics = {}
    for model_name, probability in averaged.items():
        point = point_metrics(labels, probability)
        point.pop("cumulative_probabilities", None)
        metrics[model_name] = point
        metrics[model_name]["macro_auc_ci"] = bootstrap_metric_ci(
            labels,
            probability,
            metric="macro_auc",
            groups=groups,
            replicates=int(os.environ.get("AMI_BOOTSTRAP_REPLICATES", "2000")),
        )
    output = {
        "dataset": dataset_name,
        "n": int(len(cohort)),
        "folds": folds,
        "seeds": seeds,
        "outcome_counts": {
            str(key): int(value) for key, value in pd.Series(labels).value_counts().sort_index().items()
        },
        "static_features": static_columns,
        "full_static_features": full_columns,
        "metrics": metrics,
    }
    if {"xgb_static_ts", "dggru", "dggru_tr"}.issubset(averaged):
        output["primary_contrast"] = paired_bootstrap_delta(
            labels,
            averaged["dggru_tr"],
            averaged["xgb_static_ts"],
            metric="macro_auc",
            groups=groups,
            replicates=int(os.environ.get("AMI_BOOTSTRAP_REPLICATES", "2000")),
        )
        output["regularization_contrast"] = paired_bootstrap_delta(
            labels,
            averaged["dggru_tr"],
            averaged["dggru"],
            metric="macro_auc",
            groups=groups,
            replicates=int(os.environ.get("AMI_BOOTSTRAP_REPLICATES", "2000")),
        )
        for comparison, old_name, new_name in (
            ("time_series_increment", "xgb_static", "xgb_static_ts"),
            (
                "time_series_increment_mlp",
                "mlp_static",
                "mlp_static_ts",
            ),
            ("regularization_increment", "dggru", "dggru_tr"),
            ("deep_vs_static_ts", "xgb_static_ts", "dggru_tr"),
            ("static_input_dggru_increment", "dggru_static", "dggru"),
            ("cumulative_curvature_increment", "dggru", "dggru_ctr"),
            (
                "same_static_input_mlp_vs_xgb",
                "xgb_static",
                "mlp_static",
            ),
            ("same_full_input_xgb_vs_mlp", "mlp_static_ts", "xgb_static_ts"),
            ("same_full_input_mlp_vs_dggru", "dggru", "mlp_static_ts"),
            ("same_full_input_gru_vs_dggru", "dggru", "gru"),
        ):
            if old_name in averaged and new_name in averaged:
                output[comparison] = {
                    "nri_idi": cumulative_nri_idi(
                        labels, averaged[old_name], averaged[new_name]
                    ).to_dict(orient="records"),
                    "delta_macro_auc": paired_bootstrap_delta(
                        labels,
                        averaged[new_name],
                        averaged[old_name],
                        groups=groups,
                        replicates=int(
                            os.environ.get("AMI_BOOTSTRAP_REPLICATES", "2000")
                        ),
                    ),
                }
    if {"xgb_static_ts", "dggru_tr"}.issubset(averaged):
        output["per_class_delong"] = delong_per_class(
            labels, averaged["dggru_tr"], averaged["xgb_static_ts"]
        ).to_dict(orient="records")
    calibration = {}
    for model_name, probability in averaged.items():
        high_need = (labels >= 3).astype(int)
        mortality = (labels == n_classes - 1).astype(int)
        calibration[model_name] = {
            "p_y_ge_3": calibration_slope_intercept(
                high_need, probability[:, 3:].sum(axis=1)
            ),
            "p_y_4": calibration_slope_intercept(
                mortality, probability[:, -1]
            ),
            "reliability_y_ge_3": reliability_bins(
                high_need, probability[:, 3:].sum(axis=1)
            ).to_dict(orient="records"),
        }
    output["calibration"] = calibration
    for model_name, probability in averaged.items():
        decision_curve(
            (labels >= 3).astype(int),
            probability[:, 3:].sum(axis=1),
        ).assign(model=model_name).to_csv(
            RESULTS
            / "tables"
            / f"{dataset_name}_{model_name}_dca_y_ge_3.csv",
            index=False,
        )
        decision_curve(
            (labels == n_classes - 1).astype(int),
            probability[:, -1],
        ).assign(model=model_name).to_csv(
            RESULTS
            / "tables"
            / f"{dataset_name}_{model_name}_dca_y_{n_classes - 1}.csv",
            index=False,
        )
    _write_json(RESULTS / f"metrics_{dataset_name}.json", output)
    return output


def transport_validation(
    *,
    seeds: list[int],
    xgb_only: bool,
    epochs: int,
) -> dict:
    dev = load_dataset("mimic4")
    dev_cohort = dev["cohort"]
    labels_5 = dev_cohort["outcome"].to_numpy(dtype=int)
    static_columns, full_columns = _feature_columns(dev_cohort)
    _assert_feature_contract(full_columns)
    matrix_cache = {
        "xgb_static": _static_matrix(dev_cohort, static_columns),
        "xgb_static_ts": _static_matrix(dev_cohort, full_columns),
    }
    external = {
        name: load_dataset(name) for name in ("mimic3", "eicu")
    }
    external_matrix_cache = {
        model_name: {
            name: _static_matrix(
                data["cohort"],
                static_columns if model_name == "xgb_static" else full_columns,
            )
            for name, data in external.items()
        }
        for model_name in ("xgb_static", "xgb_static_ts")
    }
    transport_results = {
        name: {
            "n": int(len(data["cohort"])),
            "status": (
                "secondary same-institution validation"
                if name == "mimic3"
                else "external transport"
            ),
            "outcome_counts": {
                str(int(key)): int(value)
                for key, value in data["cohort"]["outcome4"]
                .value_counts()
                .sort_index()
                .items()
            },
            "models": {},
        }
        for name, data in external.items()
    }

    fitted_xgb: dict[str, list[xgb.XGBClassifier]] = {}
    for model_name in ("xgb_static", "xgb_static_ts"):
        fitted_xgb[model_name] = [
            _fit_xgb(
                matrix_cache[model_name],
                labels_5,
                n_classes=5,
                seed=seed,
                n_jobs=int(os.environ.get("AMI_XGB_THREADS", "6")),
            )
            for seed in seeds
        ]
    for external_name in ("mimic3", "eicu"):
        cohort = external[external_name]["cohort"]
        external_labels = cohort["outcome4"].to_numpy(dtype=int)
        strict_death_labels = (
            pd.to_numeric(
                cohort["hospital_expire_flag"], errors="coerce"
            )
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        groups = cohort["subject_key"].astype(str).to_numpy()
        for model_name in ("xgb_static", "xgb_static_ts"):
            probability_5 = np.mean(
                np.stack(
                    [
                        model.predict_proba(
                            external_matrix_cache[model_name][external_name]
                        )
                        for model in fitted_xgb[model_name]
                    ]
                ),
                axis=0,
            )
            probability = _collapse_five_to_four_probabilities(probability_5)
            metrics = point_metrics(external_labels, probability)
            metrics.pop("cumulative_probabilities", None)
            highest_class_auc = bootstrap_metric_ci(
                external_labels,
                probability,
                metric="highest_class_auc",
                groups=groups,
                replicates=int(
                    os.environ.get("AMI_BOOTSTRAP_REPLICATES", "2000")
                ),
            )
            result = {
                "n_seed_models": len(seeds),
                "training_label_contract": (
                    "five-class development outcome collapsed to four classes"
                ),
                "metrics": metrics,
                "macro_auc_ci": bootstrap_metric_ci(
                    external_labels,
                    probability,
                    groups=groups,
                    replicates=int(
                        os.environ.get("AMI_BOOTSTRAP_REPLICATES", "2000")
                    ),
                ),
                "highest_class_auc_ci": highest_class_auc,
                "strict_mortality_auc_ci": bootstrap_binary_auc_ci(
                    strict_death_labels,
                    probability[:, 3],
                    groups=groups,
                    replicates=int(
                        os.environ.get("AMI_BOOTSTRAP_REPLICATES", "2000")
                    ),
                ),
                "calibration": {
                    "highest_class_y_3": calibration_slope_intercept(
                        (external_labels == 3).astype(int), probability[:, 3]
                    ),
                    "high_need_y_ge_2": calibration_slope_intercept(
                        (external_labels >= 2).astype(int),
                        probability[:, 2:].sum(axis=1),
                    ),
                },
            }
            transport_results[external_name]["models"][model_name] = result
            np.savez_compressed(
                RESULTS
                / "predictions"
                / f"{external_name}_{model_name}_transport.npz",
                training_label_contract=np.asarray(
                    "five-class development outcome collapsed to four classes"
                ),
                labels=external_labels,
                strict_death_labels=strict_death_labels,
                probabilities=probability,
                stay_id=cohort["stay_id"].to_numpy(dtype=np.int64),
                subject_key=groups,
            )
    if not xgb_only:
        static_dev_5 = _static_matrix(dev_cohort, full_columns)
        preprocessor = SequencePreprocessor.fit(
            dev["values"], dev["masks"], static_dev_5
        )
        sequence_dev = preprocessor.transform_sequence(
            dev["values"], dev["masks"]
        )
        normalized_dev = preprocessor.transform_static(static_dev_5)
        probability_by_seed = {
            name: [] for name in ("mimic3", "eicu")
        }
        for seed in seeds:
            model = train_temporal_model(
                sequence_dev,
                normalized_dev,
                labels_5,
                architecture="dggru",
                use_regularization=True,
                lambda_smooth=0.2,
                lambda_drift=0.05,
                epochs=epochs,
                seed=seed,
                batch_size=int(TRAIN_DEFAULTS["batch_size"]),
                learning_rate=float(TRAIN_DEFAULTS["learning_rate"]),
                weight_decay=float(TRAIN_DEFAULTS["weight_decay"]),
                hidden=int(TRAIN_DEFAULTS["hidden"]),
                edge_dim=int(TRAIN_DEFAULTS["edge_dim"]),
                dropout=float(TRAIN_DEFAULTS["dropout"]),
                n_classes=5,
            )
            for external_name in ("mimic3", "eicu"):
                external_data = external[external_name]
                static_external = _static_matrix(
                    external_data["cohort"], full_columns
                )
                sequence_external = preprocessor.transform_sequence(
                    external_data["values"], external_data["masks"]
                )
                normalized_external = preprocessor.transform_static(
                    static_external
                )
                probability_5, _ = predict_temporal_model(
                    model, sequence_external, normalized_external
                )
                probability = _collapse_five_to_four_probabilities(
                    probability_5
                )
                probability_by_seed[external_name].append(probability)
        for external_name in ("mimic3", "eicu"):
            cohort = external[external_name]["cohort"]
            labels_external = cohort["outcome4"].to_numpy(dtype=int)
            strict_death_labels = (
                pd.to_numeric(
                    cohort["hospital_expire_flag"], errors="coerce"
                )
                .fillna(0)
                .astype(int)
                .to_numpy()
            )
            groups = cohort["subject_key"].astype(str).to_numpy()
            probability = np.mean(
                np.stack(probability_by_seed[external_name]), axis=0
            )
            metrics = point_metrics(labels_external, probability)
            metrics.pop("cumulative_probabilities", None)
            highest_class_auc = bootstrap_metric_ci(
                labels_external,
                probability,
                metric="highest_class_auc",
                groups=groups,
                replicates=int(
                    os.environ.get("AMI_BOOTSTRAP_REPLICATES", "2000")
                ),
            )
            transport_results[external_name]["models"]["dggru_tr"] = {
                "n_seed_models": len(seeds),
                "training_label_contract": (
                    "five-class development outcome collapsed to four classes"
                ),
                "metrics": metrics,
                "macro_auc_ci": bootstrap_metric_ci(
                    labels_external,
                    probability,
                    groups=groups,
                    replicates=int(
                        os.environ.get("AMI_BOOTSTRAP_REPLICATES", "2000")
                    ),
                ),
                "highest_class_auc_ci": highest_class_auc,
                "strict_mortality_auc_ci": bootstrap_binary_auc_ci(
                    strict_death_labels,
                    probability[:, 3],
                    groups=groups,
                    replicates=int(
                        os.environ.get("AMI_BOOTSTRAP_REPLICATES", "2000")
                    ),
                ),
                "calibration": {
                    "highest_class_y_3": calibration_slope_intercept(
                        (labels_external == 3).astype(int), probability[:, 3]
                    ),
                    "high_need_y_ge_2": calibration_slope_intercept(
                        (labels_external >= 2).astype(int),
                        probability[:, 2:].sum(axis=1),
                    ),
                },
            }
            np.savez_compressed(
                RESULTS
                / "predictions"
                / f"{external_name}_dggru_tr_transport.npz",
                training_label_contract=np.asarray(
                    "five-class development outcome collapsed to four classes"
                ),
                labels=labels_external,
                strict_death_labels=strict_death_labels,
                probabilities=probability,
                stay_id=cohort["stay_id"].to_numpy(dtype=np.int64),
                subject_key=groups,
            )
    _write_json(RESULTS / "metrics_transport.json", transport_results)
    return transport_results


def leakage_audit(*, seed: int) -> dict:
    dev = load_dataset("mimic4")
    cohort = dev["cohort"]
    labels = cohort["outcome"].to_numpy(dtype=int)
    groups = cohort["subject_key"].astype(str).to_numpy()
    static_columns, _ = _feature_columns(cohort)
    matrix = _static_matrix(cohort, static_columns)
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=20261104
    )
    random = np.random.default_rng(seed)
    shuffled = random.permutation(labels)
    negative = np.full((len(cohort), 5), np.nan)
    positive = np.full((len(cohort), 5), np.nan)
    disposition = pd.get_dummies(
        cohort["disposition_raw"].fillna("UNKNOWN"), dtype=float
    ).to_numpy()
    positive_matrix = np.concatenate([matrix, disposition], axis=1)
    for fold, (train, test) in enumerate(
        splitter.split(matrix, labels, groups)
    ):
        model = _fit_xgb(
            matrix[train],
            shuffled[train],
            n_classes=5,
            seed=seed + fold,
            n_jobs=int(os.environ.get("AMI_XGB_THREADS", "6")),
        )
        negative[test] = model.predict_proba(matrix[test])
        leak = _fit_xgb(
            positive_matrix[train],
            labels[train],
            n_classes=5,
            seed=seed + fold,
            n_jobs=int(os.environ.get("AMI_XGB_THREADS", "6")),
        )
        positive[test] = leak.predict_proba(positive_matrix[test])
    result = {
        "negative_control_randomized_label_macro_auc": point_metrics(
            shuffled, negative
        )["macro_auc"],
        "positive_control_disposition_macro_auc": point_metrics(
            labels, positive
        )["macro_auc"],
        "interpretation": (
            "Randomized labels should be near 0.5; direct disposition leakage "
            "should approach 1.0."
        ),
    }
    _write_json(RESULTS / "audits" / "leakage_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mimic4")
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=TRAIN_DEFAULTS["epochs"])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--lambda-smooth", type=float, default=0.2)
    parser.add_argument("--lambda-drift", type=float, default=0.05)
    parser.add_argument(
        "--lambda-curvature",
        type=float,
        default=DEFAULT_LAMBDA_CURVATURE,
    )
    parser.add_argument("--transport", action="store_true")
    parser.add_argument("--xgb-only-transport", action="store_true")
    parser.add_argument("--leakage-audit", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    unknown = set(args.models).difference(MODEL_ORDER)
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    result = run_cv(
        args.dataset,
        seeds=args.seeds,
        folds=args.folds,
        epochs=args.epochs,
        models=args.models,
        max_rows=args.max_rows,
        lambda_smooth=args.lambda_smooth,
        lambda_drift=args.lambda_drift,
        lambda_curvature=args.lambda_curvature,
        resume=not args.no_resume,
        force=args.force,
    )
    if args.transport:
        transport_validation(
            seeds=args.seeds,
            xgb_only=args.xgb_only_transport,
            epochs=args.epochs,
        )
    if args.leakage_audit:
        leakage_audit(seed=args.seeds[0])
    print(
        json.dumps(
            {
                "n": result["n"],
                "models": list(result["metrics"]),
                "best_macro_auc": max(
                    (value["macro_auc"], key)
                    for key, value in result["metrics"].items()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

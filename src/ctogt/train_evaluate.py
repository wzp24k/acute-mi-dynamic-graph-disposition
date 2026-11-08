"""Cross-validated development and three-database transport evaluation."""
from __future__ import annotations

import argparse
import copy
import json
import os
import warnings
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import (
    CLINICAL_PRIOR_EDGES,
    MIMIC4,
    ORGAN_CHANNELS,
    PREDICTIONS,
    PROCESSED,
    RESULTS,
    ensure_dirs,
    load_config,
)
from .models import (
    DeepConfig,
    SequenceStandardizer,
    build_prior_adjacency,
    predict_deep_model,
    train_deep_model,
)


warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_num_threads(max(1, min(10, int(os.environ.get("CTOGT_TORCH_THREADS", "8")))))

STATIC_FEATURES = (
    "age",
    "sex_male",
    "race_white",
    "race_black",
    "emergency_admission",
    "prior_mi",
    "prior_stroke",
    "heart_failure",
    "atrial_fib",
    "diabetes",
    "hypertension",
    "ckd",
    "copd",
    "pad",
    "smoking",
    "prior_pci",
    "prior_cabg",
    "comorbidity_burden",
)

TS_FEATURES = tuple(
    f"{channel}_{statistic}"
    for channel in ORGAN_CHANNELS
    for statistic in ("n", "mean", "min", "max", "sd", "slope", "missing")
)
ALL_FEATURES = (*STATIC_FEATURES, *TS_FEATURES)

MODEL_LABELS = {
    "logit_static": "Logistic regression, static",
    "xgb_static": "XGBoost, static",
    "xgb_static_ts": "XGBoost, static + temporal summaries",
    "gru": "GRU",
    "transformer": "Transformer",
    "tft": "TFT-lite",
    "ct_fixed": "Continuous-time fixed organ graph",
    "organ_graph": "Seven-organ token dynamic graph",
    "ct_graph": "Continuous-time dynamic organ graph",
    "ctogt": "CT-OGT full model",
    "temporal_ensemble": "Temporal ensemble (XGB+TS + Transformer)",
    "ct_random": "CT-OGT, random graph control",
    "ct_reverse": "CT-OGT, reversed time control",
    "ct_permute": "CT-OGT, permuted time control",
    "ct_no_time": "CT-OGT, no continuous-time encoding",
}

DEEP_CONFIGS = {
    "gru": DeepConfig(architecture="gru"),
    "transformer": DeepConfig(architecture="transformer"),
    "tft": DeepConfig(architecture="tft"),
    "ct_fixed": DeepConfig(
        architecture="ctogt",
        graph_mode="fixed",
        include_continuous_time=True,
        use_environment_head=False,
        use_environment_penalty=False,
    ),
    "organ_graph": DeepConfig(
        architecture="organ_graph",
        graph_mode="dynamic",
        include_continuous_time=True,
        use_environment_head=False,
        use_environment_penalty=False,
    ),
    "ct_graph": DeepConfig(
        architecture="ctogt",
        graph_mode="dynamic",
        include_continuous_time=True,
        use_environment_head=False,
        use_environment_penalty=False,
    ),
    "ctogt": DeepConfig(
        architecture="ctogt",
        graph_mode="dynamic",
        include_continuous_time=True,
        use_environment_head=True,
        use_environment_penalty=True,
    ),
    "ct_random": DeepConfig(
        architecture="ctogt",
        graph_mode="dynamic",
        include_continuous_time=True,
        use_environment_head=False,
        use_environment_penalty=False,
        random_graph=True,
    ),
    "ct_reverse": DeepConfig(
        architecture="ctogt",
        graph_mode="dynamic",
        include_continuous_time=True,
        use_environment_head=True,
        use_environment_penalty=True,
        time_mode="reverse",
    ),
    "ct_permute": DeepConfig(
        architecture="ctogt",
        graph_mode="dynamic",
        include_continuous_time=True,
        use_environment_head=True,
        use_environment_penalty=True,
        time_mode="permute",
    ),
    "ct_no_time": DeepConfig(
        architecture="ctogt",
        graph_mode="dynamic",
        include_continuous_time=False,
        use_environment_head=True,
        use_environment_penalty=True,
    ),
}

PRIMARY_MODELS = (
    "logit_static",
    "xgb_static",
    "xgb_static_ts",
    "gru",
    "transformer",
    "tft",
    "ct_fixed",
    "organ_graph",
    "ct_graph",
    "ctogt",
)
CONTROL_MODELS = ("ct_random", "ct_reverse", "ct_permute", "ct_no_time")
ALL_MODELS = PRIMARY_MODELS + CONTROL_MODELS


class Dataset:
    def __init__(
        self,
        name: str,
        cohort: pd.DataFrame,
        values: np.ndarray,
        masks: np.ndarray,
        observed: np.ndarray,
    ):
        self.name = name
        self.cohort = cohort.reset_index(drop=True)
        self.values = values.astype(np.float32)
        self.masks = masks.astype(np.float32)
        self.observed = observed.astype(np.float32)
        self.labels = (
            pd.to_numeric(self.cohort["macce7"], errors="raise")
            .astype(int)
            .to_numpy()
        )
        environments = (
            self.cohort.get(
                "environment", pd.Series("environment", index=self.cohort.index)
            )
            .astype("string")
            .fillna("unknown")
            .astype(str)
        )
        self.environment_names = environments.to_numpy()
        if self.name == "mimic4":
            patients = pd.read_csv(
                MIMIC4 / "hosp" / "patients.csv.gz",
                usecols=["subject_id", "anchor_year_group"],
            ).rename(columns={"subject_id": "patient_id"})
            environment_frame = self.cohort[["patient_id", "first_careunit"]].merge(
                patients,
                on="patient_id",
                how="left",
                validate="many_to_one",
            )
            self.environment_names = (
                environment_frame["first_careunit"]
                .fillna("unknown")
                .astype(str)
                + "_"
                + environment_frame["anchor_year_group"]
                .fillna("unknown")
                .astype(str)
            ).to_numpy()
        self.environments = pd.factorize(self.environment_names)[0].astype(np.int64)

    def frame(self, columns: tuple[str, ...]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                column: (
                    pd.to_numeric(self.cohort[column], errors="coerce")
                    if column in self.cohort
                    else pd.Series(np.nan, index=self.cohort.index)
                )
                for column in columns
            }
        )


def load_dataset(name: str) -> Dataset:
    root = PROCESSED / name
    cohort = pd.read_csv(root / "cohort.csv")
    with np.load(root / "sequences_24h.npz", allow_pickle=True) as data:
        values = data["values"].astype(np.float32)
        masks = data["masks"].astype(np.float32)
        observed = data["observed"].astype(np.float32)
        ids = data["stay_id"].astype(np.int64)
        channels = tuple(data["channels"].tolist())
    if channels != ORGAN_CHANNELS:
        raise ValueError(f"{name}: sequence channels differ from the contract")
    if not np.array_equal(ids, cohort["stay_id"].astype(np.int64).to_numpy()):
        raise ValueError(f"{name}: identifiers are misaligned")
    if values.shape != (len(cohort), 24, len(ORGAN_CHANNELS)):
        raise ValueError(f"{name}: unexpected sequence shape {values.shape}")
    return Dataset(name, cohort, values, masks, observed)


def prepare_tabular(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_values = imputer.fit_transform(train)
    test_values = imputer.transform(test)
    return (
        np.asarray(train_values, dtype=np.float32),
        np.asarray(test_values, dtype=np.float32),
    )


def fit_logistic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(
            C=0.3,
            solver="lbfgs",
            max_iter=3000,
            random_state=seed,
        ),
    )
    model.fit(train, labels)
    return model.predict_proba(test)[:, 1]


def fit_xgboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: np.ndarray,
    config: dict,
    seed: int,
) -> np.ndarray:
    train_x, test_x = prepare_tabular(train, test)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=max(1, int(os.environ.get("CTOGT_XGB_THREADS", "8"))),
        **copy.deepcopy(config["xgboost"]),
    )
    model.fit(train_x, labels, verbose=False)
    return model.predict_proba(test_x)[:, 1]


def split_validation(
    rows: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train, valid = train_test_split(
        rows,
        test_size=0.12,
        random_state=seed,
        stratify=labels[rows],
    )
    return np.asarray(train), np.asarray(valid)


def prepare_sequence(
    dataset: Dataset,
    train_rows: np.ndarray,
    test_dataset: Dataset,
    test_rows: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    standardizer = SequenceStandardizer.fit(
        dataset.values[train_rows],
        dataset.masks[train_rows],
        dataset.frame(ALL_FEATURES).to_numpy(dtype=np.float32)[train_rows],
    )
    train_values, train_masks = standardizer.transform_sequence(
        dataset.values[train_rows], dataset.masks[train_rows]
    )
    train_static = standardizer.transform_static(
        dataset.frame(ALL_FEATURES).to_numpy(dtype=np.float32)[train_rows]
    )
    test_values, test_masks = standardizer.transform_sequence(
        test_dataset.values[test_rows], test_dataset.masks[test_rows]
    )
    test_static = standardizer.transform_static(
        test_dataset.frame(ALL_FEATURES).to_numpy(dtype=np.float32)[test_rows]
    )
    return (
        train_values,
        train_masks,
        train_static,
        test_values,
        test_masks,
        test_static,
    )


def train_deep_predict(
    dataset: Dataset,
    train_rows: np.ndarray,
    test_dataset: Dataset,
    test_rows: np.ndarray,
    model_name: str,
    config: dict,
    seed: int,
) -> dict[str, np.ndarray]:
    inner_train, inner_valid = split_validation(
        train_rows, dataset.labels, seed
    )
    (
        train_values,
        train_masks,
        train_static,
        valid_values,
        valid_masks,
        valid_static,
    ) = prepare_sequence(dataset, inner_train, dataset, inner_valid)
    # The inner test arrays are the validation rows; replace final test below.
    deep_config = DEEP_CONFIGS[model_name]
    prior = build_prior_adjacency(ORGAN_CHANNELS, CLINICAL_PRIOR_EDGES)
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
        config=deep_config,
        prior=prior,
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
    standardizer = SequenceStandardizer.fit(
        dataset.values[inner_train],
        dataset.masks[inner_train],
        dataset.frame(ALL_FEATURES).to_numpy(dtype=np.float32)[inner_train],
    )
    test_values, test_masks = standardizer.transform_sequence(
        test_dataset.values[test_rows], test_dataset.masks[test_rows]
    )
    test_static = standardizer.transform_static(
        test_dataset.frame(ALL_FEATURES).to_numpy(dtype=np.float32)[test_rows]
    )
    return predict_deep_model(
        model,
        test_values,
        test_masks,
        test_static,
        config=deep_config,
        seed=seed + 9_000_000,
    )


def fit_fold_predictions(
    dataset: Dataset,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    model_name: str,
    config: dict,
    seed: int,
) -> dict[str, np.ndarray]:
    if model_name == "logit_static":
        probability = fit_logistic(
            dataset.frame(STATIC_FEATURES).iloc[train_rows],
            dataset.frame(STATIC_FEATURES).iloc[test_rows],
            dataset.labels[train_rows],
            seed,
        )
        return {"probability": probability}
    if model_name in {"xgb_static", "xgb_static_ts"}:
        features = (
            STATIC_FEATURES if model_name == "xgb_static" else ALL_FEATURES
        )
        probability = fit_xgboost(
            dataset.frame(features).iloc[train_rows],
            dataset.frame(features).iloc[test_rows],
            dataset.labels[train_rows],
            config,
            seed,
        )
        return {"probability": probability}
    return train_deep_predict(
        dataset,
        train_rows,
        dataset,
        test_rows,
        model_name,
        config,
        seed,
    )


def metric_block(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    logits = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    calibration = LogisticRegression(
        C=1e6, solver="lbfgs", max_iter=1000
    ).fit(logits, labels)
    calibration_slope = float(calibration.coef_[0, 0])
    calibration_intercept = float(calibration.intercept_[0])
    bins = np.linspace(0.0, 1.0, 11)
    bin_index = np.clip(np.digitize(probability, bins) - 1, 0, 9)
    ece = 0.0
    for index in range(10):
        selected = bin_index == index
        if selected.any():
            ece += selected.mean() * abs(
                probability[selected].mean() - labels[selected].mean()
            )
    threshold = float(np.quantile(probability, 0.9))
    prediction = probability >= threshold
    sensitivity = float(prediction[labels == 1].mean()) if (labels == 1).any() else np.nan
    specificity = float((~prediction[labels == 0]).mean()) if (labels == 0).any() else np.nan
    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "log_loss": float(log_loss(labels, probability)),
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "ece": float(ece),
        "sensitivity_top10": sensitivity,
        "specificity_top10": specificity,
        "threshold_top10": threshold,
        "prevalence": float(labels.mean()),
    }


def net_benefit(
    labels: np.ndarray,
    probability: np.ndarray,
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
) -> pd.DataFrame:
    rows = []
    n = len(labels)
    prevalence = float(labels.mean())
    for threshold in thresholds:
        predicted = probability >= threshold
        tp = int((predicted & (labels == 1)).sum())
        fp = int((predicted & (labels == 0)).sum())
        net = tp / n - fp / n * threshold / (1.0 - threshold)
        treat_all = prevalence - (1.0 - prevalence) * threshold / (
            1.0 - threshold
        )
        rows.append(
            {
                "threshold": threshold,
                "net_benefit": float(net),
                "treat_all": float(treat_all),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_paired_auc(
    labels: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(labels)
    for _ in range(int(replicates)):
        index = rng.integers(0, n, n)
        sampled = labels[index]
        if len(np.unique(sampled)) < 2:
            continue
        values.append(
            roc_auc_score(sampled, probability_a[index])
            - roc_auc_score(sampled, probability_b[index])
        )
    point = roc_auc_score(labels, probability_a) - roc_auc_score(
        labels, probability_b
    )
    return {
        "difference": float(point),
        "ci": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
    }


def bootstrap_metric(
    labels: np.ndarray,
    probability: np.ndarray,
    metric: str,
    replicates: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(labels)
    for _ in range(int(replicates)):
        index = rng.integers(0, n, n)
        sampled = labels[index]
        if len(np.unique(sampled)) < 2:
            continue
        if metric == "auroc":
            values.append(roc_auc_score(sampled, probability[index]))
        elif metric == "auprc":
            values.append(average_precision_score(sampled, probability[index]))
        elif metric == "brier":
            values.append(brier_score_loss(sampled, probability[index]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def select_models(value: str) -> tuple[str, ...]:
    if value == "all":
        return ALL_MODELS
    if value == "primary":
        return PRIMARY_MODELS
    if value == "controls":
        return CONTROL_MODELS
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(requested).difference(ALL_MODELS)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    return requested


def run_development(
    config: dict,
    models: tuple[str, ...],
    folds: int,
    seeds: tuple[int, ...],
) -> None:
    dataset = load_dataset("mimic4")
    splitter = StratifiedKFold(
        n_splits=int(folds), shuffle=True, random_state=int(seeds[0])
    )
    splits = list(
        splitter.split(np.zeros((len(dataset.labels), 1)), dataset.labels)
    )
    for model_name in models:
        print(f"[development] {model_name}", flush=True)
        per_seed = []
        attention_payload: dict[str, np.ndarray] = {}
        for seed in seeds:
            oof = np.full(len(dataset.labels), np.nan)
            organ_attention = np.zeros(
                (len(dataset.labels), len(ORGAN_CHANNELS)), dtype=np.float32
            )
            temporal_attention = np.zeros((len(dataset.labels), 24), dtype=np.float32)
            for fold_index, (train_rows, test_rows) in enumerate(splits):
                print(
                    f"  seed={seed} fold={fold_index + 1}/{folds}",
                    flush=True,
                )
                output = fit_fold_predictions(
                    dataset,
                    np.asarray(train_rows),
                    np.asarray(test_rows),
                    model_name,
                    config,
                    seed + fold_index,
                )
                oof[test_rows] = output["probability"]
                if "organ_attention" in output:
                    organ_attention[test_rows] = output["organ_attention"]
                if "temporal_attention" in output:
                    temporal_attention[test_rows] = output["temporal_attention"]
            if np.isnan(oof).any():
                raise RuntimeError(f"{model_name}: incomplete OOF predictions")
            per_seed.append(oof)
            attention_payload[f"seed_{seed}_organ"] = organ_attention
            attention_payload[f"seed_{seed}_temporal"] = temporal_attention
        seed_predictions = np.stack(per_seed)
        prediction = np.mean(per_seed, axis=0)
        metrics = metric_block(dataset.labels, prediction)
        bootstrap = {
            "auroc_ci": bootstrap_metric(
                dataset.labels,
                prediction,
                "auroc",
                int(config["bootstrap_replicates"]),
                int(seeds[0]),
            ),
            "auprc_ci": bootstrap_metric(
                dataset.labels,
                prediction,
                "auprc",
                int(config["bootstrap_replicates"]),
                int(seeds[0]) + 1,
            ),
        }
        model_result = {
            "dataset": "mimic4",
            "n": int(len(dataset.labels)),
            "events": int(dataset.labels.sum()),
            "prevalence": float(dataset.labels.mean()),
            "seeds": list(seeds),
            "folds": int(folds),
            "model": model_name,
            "metrics": metrics,
            "bootstrap": bootstrap,
        }
        (RESULTS / f"metrics_development_{model_name}.json").write_text(
            json.dumps(model_result, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            PREDICTIONS / f"mimic4_cv_{model_name}.npz",
            labels=dataset.labels,
            stay_id=dataset.cohort["stay_id"].to_numpy(dtype=np.int64),
            probability=prediction.astype(np.float32),
            seed_probability=seed_predictions.astype(np.float32),
        )
        np.savez_compressed(
            PREDICTIONS / f"mimic4_attention_{model_name}.npz",
            **attention_payload,
        )
        print(json.dumps(metrics, indent=2), flush=True)
    aggregate_development(config, seeds)


def aggregate_development(config: dict, seeds: tuple[int, ...]) -> None:
    metric_files = sorted(RESULTS.glob("metrics_development_*.json"))
    if not metric_files:
        return
    loaded = [
        json.loads(path.read_text(encoding="utf-8")) for path in metric_files
    ]
    loaded = [
        entry for entry in loaded if entry.get("folds") == int(config["cv_folds"])
    ]
    if not loaded:
        return
    model_names = [str(entry["model"]) for entry in loaded]
    labels = None
    all_predictions = {}
    for model_name in model_names:
        path = PREDICTIONS / f"mimic4_cv_{model_name}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=True) as data:
            current_labels = data["labels"].astype(int)
            if labels is None:
                labels = current_labels
            elif not np.array_equal(labels, current_labels):
                continue
            all_predictions[model_name] = data["probability"].astype(float)
    if labels is None:
        return
    metrics = {
        entry["model"]: entry["metrics"]
        for entry in loaded
        if entry["model"] in all_predictions
    }
    bootstrap = {
        entry["model"]: entry["bootstrap"]
        for entry in loaded
        if entry["model"] in all_predictions
    }
    pairs = {}
    if "ctogt" in all_predictions and "xgb_static_ts" in all_predictions:
        pairs["ctogt_minus_xgb_static_ts"] = bootstrap_paired_auc(
            labels,
            all_predictions["ctogt"],
            all_predictions["xgb_static_ts"],
            int(config["bootstrap_replicates"]),
            int(seeds[0]) + 2,
        )
    if "ct_graph" in all_predictions and "ct_fixed" in all_predictions:
        pairs["ct_graph_minus_ct_fixed"] = bootstrap_paired_auc(
            labels,
            all_predictions["ct_graph"],
            all_predictions["ct_fixed"],
            int(config["bootstrap_replicates"]),
            int(seeds[0]) + 3,
        )
    if "ctogt" in all_predictions and "ct_graph" in all_predictions:
        pairs["ctogt_minus_ct_graph"] = bootstrap_paired_auc(
            labels,
            all_predictions["ctogt"],
            all_predictions["ct_graph"],
            int(config["bootstrap_replicates"]),
            int(seeds[0]) + 4,
        )
    if "ct_reverse" in all_predictions and "ctogt" in all_predictions:
        pairs["ct_reverse_minus_ctogt"] = bootstrap_paired_auc(
            labels,
            all_predictions["ct_reverse"],
            all_predictions["ctogt"],
            int(config["bootstrap_replicates"]),
            int(seeds[0]) + 5,
        )
    if "ct_permute" in all_predictions and "ctogt" in all_predictions:
        pairs["ct_permute_minus_ctogt"] = bootstrap_paired_auc(
            labels,
            all_predictions["ct_permute"],
            all_predictions["ctogt"],
            int(config["bootstrap_replicates"]),
            int(seeds[0]) + 6,
        )
    result = {
        "dataset": "mimic4",
        "n": int(len(labels)),
        "events": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "seeds": list(seeds),
        "folds": int(config["cv_folds"]),
        "metrics": metrics,
        "bootstrap": bootstrap,
        "paired_contrasts": pairs,
    }
    (RESULTS / "metrics_development.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    payload = {"labels": labels}
    for model, prediction in all_predictions.items():
        payload[model] = prediction.astype(np.float32)
        with np.load(
            PREDICTIONS / f"mimic4_cv_{model}.npz", allow_pickle=True
        ) as data:
            payload[f"{model}__seed"] = data["seed_probability"].astype(
                np.float32
            )
            if "stay_id" in payload:
                if not np.array_equal(payload["stay_id"], data["stay_id"]):
                    payload["stay_id"] = data["stay_id"].astype(np.int64)
            else:
                payload["stay_id"] = data["stay_id"].astype(np.int64)
    np.savez_compressed(PREDICTIONS / "mimic4_cv.npz", **payload)
    print(json.dumps(metrics, indent=2), flush=True)


def train_external_model(
    development: Dataset,
    external: Dataset,
    model_name: str,
    config: dict,
    seeds: tuple[int, ...],
) -> dict[str, np.ndarray]:
    predictions = []
    attention_organ = []
    attention_temporal = []
    for seed in seeds:
        print(f"[external] {external.name} {model_name} seed={seed}", flush=True)
        if model_name == "logit_static":
            probability = fit_logistic(
                development.frame(STATIC_FEATURES),
                external.frame(STATIC_FEATURES),
                development.labels,
                seed,
            )
            predictions.append(probability)
            continue
        if model_name in {"xgb_static", "xgb_static_ts"}:
            features = (
                STATIC_FEATURES if model_name == "xgb_static" else ALL_FEATURES
            )
            probability = fit_xgboost(
                development.frame(features),
                external.frame(features),
                development.labels,
                config,
                seed,
            )
            predictions.append(probability)
            continue
        all_rows = np.arange(len(development.labels))
        inner_train, inner_valid = split_validation(
            all_rows, development.labels, seed
        )
        (
            train_values,
            train_masks,
            train_static,
            valid_values,
            valid_masks,
            valid_static,
        ) = prepare_sequence(
            development, inner_train, development, inner_valid
        )
        deep_config = DEEP_CONFIGS[model_name]
        model = train_deep_model(
            train_values,
            train_masks,
            train_static,
            development.labels[inner_train],
            development.environments[inner_train],
            valid_values,
            valid_masks,
            valid_static,
            development.labels[inner_valid],
            development.environments[inner_valid],
            config=deep_config,
            prior=build_prior_adjacency(
                ORGAN_CHANNELS, CLINICAL_PRIOR_EDGES
            ),
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
        standardizer = SequenceStandardizer.fit(
            development.values[inner_train],
            development.masks[inner_train],
            development.frame(ALL_FEATURES).to_numpy(dtype=np.float32)[
                inner_train
            ],
        )
        external_values, external_masks = standardizer.transform_sequence(
            external.values, external.masks
        )
        external_static = standardizer.transform_static(
            external.frame(ALL_FEATURES).to_numpy(dtype=np.float32)
        )
        output = predict_deep_model(
            model,
            external_values,
            external_masks,
            external_static,
            config=deep_config,
            seed=seed,
        )
        predictions.append(output["probability"])
        if "organ_attention" in output:
            attention_organ.append(output["organ_attention"])
        if "temporal_attention" in output:
            attention_temporal.append(output["temporal_attention"])
    result = {"probability": np.mean(predictions, axis=0)}
    if attention_organ:
        result["organ_attention"] = np.mean(attention_organ, axis=0)
    if attention_temporal:
        result["temporal_attention"] = np.mean(attention_temporal, axis=0)
    return result


def crossfit_platt(
    labels: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> np.ndarray:
    if len(np.unique(labels)) < 2:
        return probability.copy()
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        if len(np.unique(groups)) < int(folds):
            raise ValueError("Insufficient groups for grouped calibration")
        splitter = StratifiedGroupKFold(
            n_splits=int(folds), shuffle=True, random_state=seed
        )
        splits = list(
            splitter.split(probability.reshape(-1, 1), labels, groups)
        )
    except Exception:
        splitter = StratifiedKFold(
            n_splits=int(folds), shuffle=True, random_state=seed
        )
        splits = list(
            splitter.split(probability.reshape(-1, 1), labels)
        )
    calibrated = np.full(len(labels), np.nan)
    for train_index, test_index in splits:
        logits_train = np.log(
            np.clip(probability[train_index], 1e-7, 1 - 1e-7)
            / np.clip(1.0 - probability[train_index], 1e-7, 1 - 1e-7)
        ).reshape(-1, 1)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(logits_train, labels[train_index])
        logits_test = np.log(
            np.clip(probability[test_index], 1e-7, 1 - 1e-7)
            / np.clip(1.0 - probability[test_index], 1e-7, 1 - 1e-7)
        ).reshape(-1, 1)
        calibrated[test_index] = model.predict_proba(logits_test)[:, 1]
    return calibrated


def run_external(
    config: dict,
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    external_names: tuple[str, ...],
) -> None:
    development = load_dataset("mimic4")
    for external_name in external_names:
        external = load_dataset(external_name)
        site_groups = (
            external.cohort.get("hospitalid", pd.Series(0, index=external.cohort.index))
            .fillna(0)
            .astype(str)
            .to_numpy()
        )
        payload: dict[str, np.ndarray] = {
            "labels": external.labels,
            "stay_id": external.cohort["stay_id"].to_numpy(dtype=np.int64),
            "site": site_groups,
        }
        for model_name in models:
            output = train_external_model(
                development, external, model_name, config, seeds
            )
            probability = output["probability"]
            calibrated = crossfit_platt(
                external.labels,
                probability,
                site_groups,
                int(config["cv_folds"]),
                int(seeds[0]),
            )
            model_result = {
                "unrecalibrated": metric_block(external.labels, probability),
                "recalibrated": metric_block(external.labels, calibrated),
                "net_benefit": net_benefit(
                    external.labels, probability
                ).to_dict(orient="records"),
            }
            if external_name == "eicu":
                site_frame = pd.DataFrame(
                    {
                        "site": site_groups,
                        "label": external.labels,
                        "probability": probability,
                    }
                )
                summary = (
                    site_frame.groupby("site", as_index=False)
                    .agg(
                        n=("label", "size"),
                        events=("label", "sum"),
                        mean_prediction=("probability", "mean"),
                        observed=("label", "mean"),
                    )
                    .query("n >= @config['eicu_minimum_site_n']")
                )
                model_result["site_summary"] = summary.to_dict(orient="records")
            payload[model_name] = probability.astype(np.float32)
            payload[f"{model_name}__recalibrated"] = calibrated.astype(np.float32)
            if "organ_attention" in output:
                payload[
                    f"{model_name}__organ_attention"
                ] = output["organ_attention"].astype(np.float32)
            if "temporal_attention" in output:
                payload[
                    f"{model_name}__temporal_attention"
                ] = output["temporal_attention"].astype(np.float32)
            model_record = {
                "external": external_name,
                "n": int(len(external.labels)),
                "events": int(external.labels.sum()),
                "prevalence": float(external.labels.mean()),
                "model": model_name,
                "metrics": model_result,
            }
            (
                RESULTS
                / f"metrics_external_{external_name}_{model_name}.json"
            ).write_text(
                json.dumps(model_record, indent=2), encoding="utf-8"
            )
            np.savez_compressed(
                PREDICTIONS / f"{external_name}_external_{model_name}.npz",
                **payload,
            )
            print(
                json.dumps(
                    {
                        "external": external_name,
                        "model": model_name,
                        "auroc": model_result["unrecalibrated"]["auroc"],
                        "recalibrated_auroc": model_result["recalibrated"][
                            "auroc"
                        ],
                    }
                ),
                flush=True,
            )
    aggregate_external()


def aggregate_external() -> None:
    metric_files = sorted(RESULTS.glob("metrics_external_*_*.json"))
    if not metric_files:
        return
    aggregate: dict[str, object] = {
        "development": "mimic4",
        "external": {},
    }
    for path in metric_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        external_name = str(record["external"])
        external_result = aggregate["external"].setdefault(
            external_name,
            {
                "n": int(record["n"]),
                "events": int(record["events"]),
                "prevalence": float(record["prevalence"]),
                "models": {},
            },
        )
        external_result["models"][str(record["model"])] = record["metrics"]
    (RESULTS / "metrics_external.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cv", "external", "all"], default="cv")
    parser.add_argument("--models", default="primary")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--external", default="mimic3,eicu")
    args = parser.parse_args()
    ensure_dirs()
    config = load_config()
    models = select_models(args.models)
    seeds = (
        tuple(int(value) for value in config["model_seeds"])
        if args.seeds == "all"
        else tuple(int(value) for value in args.seeds.split(","))
    )
    if args.mode in {"cv", "all"}:
        run_development(config, models, args.folds, seeds)
    if args.mode in {"external", "all"}:
        external_names = tuple(
            value.strip() for value in args.external.split(",") if value.strip()
        )
        run_external(config, models, seeds, external_names)


if __name__ == "__main__":
    main()

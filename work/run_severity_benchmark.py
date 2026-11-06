"""Fit the prespecified admission-severity benchmark on shared CV folds."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.miscmodels.ordinal_model import OrderedModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import RESULTS  # noqa: E402
from src.ami_dynamic.evaluation import (  # noqa: E402
    bootstrap_metric_ci,
    calibration_slope_intercept,
    point_metrics,
)
from src.ami_dynamic.train_evaluate import load_dataset  # noqa: E402


SEVERITY_COLUMNS = (
    "age",
    "hr_first",
    "sbp_first",
    "gcs_first",
    "lab_creatinine_first",
    "lab_lactate_first",
    "lab_troponin_first",
)


def _matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if column in frame
            else np.full(len(frame), np.nan)
            for column in SEVERITY_COLUMNS
        ]
    )


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, str, float]:
    median = np.nanmedian(train_x, axis=0)
    median[~np.isfinite(median)] = 0.0
    train_filled = np.where(np.isfinite(train_x), train_x, median)
    test_filled = np.where(np.isfinite(test_x), test_x, median)
    mean = train_filled.mean(axis=0)
    std = train_filled.std(axis=0)
    std[std < 1e-8] = 1.0
    train_scaled = (train_filled - mean) / std
    test_scaled = (test_filled - mean) / std

    ordered = OrderedModel(
        train_y,
        train_scaled,
        distr="logit",
    ).fit(method="bfgs", maxiter=300, disp=False)
    multinomial = MNLogit(train_y, train_scaled).fit(
        method="bfgs", maxiter=300, disp=False
    )
    statistic = max(0.0, 2.0 * (multinomial.llf - ordered.llf))
    degrees = max(1, len(multinomial.params) - len(ordered.params))
    proportional_odds_p = float(stats.chi2.sf(statistic, degrees))
    if proportional_odds_p < 0.05:
        probability = np.asarray(multinomial.predict(test_scaled), dtype=float)
        selected = "multinomial"
    else:
        probability = np.asarray(ordered.predict(test_scaled), dtype=float)
        selected = "proportional_odds"
    if probability.ndim == 1:
        probability = probability.reshape(-1, n_classes)
    probability = np.clip(probability, 1e-7, 1.0)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability, selected, proportional_odds_p


def run(seed: int, folds: int) -> dict:
    prediction_dir = RESULTS / "predictions"
    cached_json = (
        prediction_dir / f"mimic4_severity_benchmark_seed{seed}.json"
    )
    cached_npz = (
        prediction_dir / f"mimic4_severity_benchmark_seed{seed}.npz"
    )
    data = load_dataset("mimic4")
    cohort = data["cohort"]
    labels = cohort["outcome"].to_numpy(dtype=int)
    groups = cohort["subject_key"].astype(str).to_numpy()
    if cached_json.is_file() and cached_npz.is_file():
        with np.load(cached_npz, allow_pickle=True) as cached:
            cache_matches = (
                np.array_equal(cached["labels"].astype(int), labels)
                and np.array_equal(
                    cached["subject_key"].astype(str), groups
                )
            )
        if cache_matches:
            return json.loads(cached_json.read_text(encoding="utf-8"))
    matrix = _matrix(cohort)
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    n_classes = int(labels.max()) + 1
    probability = np.full((len(cohort), n_classes), np.nan, dtype=float)
    fold_id = np.full(len(cohort), -1, dtype=int)
    selected_models = []
    po_p_values = []
    for fold, (train, test) in enumerate(
        splitter.split(matrix, labels, groups)
    ):
        fold_id[test] = fold
        fold_probability, selected, po_p = _fit_predict(
            matrix[train],
            labels[train],
            matrix[test],
            n_classes,
        )
        probability[test] = fold_probability
        selected_models.append(selected)
        po_p_values.append(po_p)

    metrics = point_metrics(labels, probability)
    metrics.pop("cumulative_probabilities", None)
    metrics["macro_auc_ci"] = bootstrap_metric_ci(
        labels,
        probability,
        groups=groups,
        replicates=int(os.environ.get("AMI_BOOTSTRAP_REPLICATES", "2000")),
    )
    result = {
        "dataset": "mimic4",
        "model": "admission_severity_benchmark",
        "seed": seed,
        "folds": folds,
        "n": int(len(cohort)),
        "variables": list(SEVERITY_COLUMNS),
        "selected_model_by_fold": selected_models,
        "proportional_odds_test_p_by_fold": po_p_values,
        "metrics": metrics,
        "calibration": {
            "p_y_ge_3": calibration_slope_intercept(
                (labels >= 3).astype(int), probability[:, 3:].sum(axis=1)
            ),
            "p_y_4": calibration_slope_intercept(
                (labels == 4).astype(int), probability[:, 4]
            ),
        },
    }
    prediction_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_dir / f"mimic4_severity_benchmark_seed{seed}.npz",
        labels=labels,
        probabilities=probability.astype(np.float32),
        fold=fold_id,
        stay_id=cohort["stay_id"].to_numpy(dtype=np.int64),
        subject_key=groups,
    )
    (prediction_dir / f"mimic4_severity_benchmark_seed{seed}.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 42, 123, 2024])
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    seed_results = [run(seed, args.folds) for seed in args.seeds]
    labels = np.load(
        RESULTS / "predictions" / f"mimic4_severity_benchmark_seed{args.seeds[0]}.npz",
        allow_pickle=True,
    )["labels"].astype(int)
    groups = np.load(
        RESULTS / "predictions" / f"mimic4_severity_benchmark_seed{args.seeds[0]}.npz",
        allow_pickle=True,
    )["subject_key"].astype(str)
    probabilities = np.stack(
        [
            np.load(
                RESULTS
                / "predictions"
                / f"mimic4_severity_benchmark_seed{seed}.npz",
                allow_pickle=True,
            )["probabilities"].astype(float)
            for seed in args.seeds
        ]
    )
    averaged = probabilities.mean(axis=0)
    metrics = point_metrics(labels, averaged)
    metrics.pop("cumulative_probabilities", None)
    metrics["macro_auc_ci"] = bootstrap_metric_ci(
        labels,
        averaged,
        groups=groups,
        replicates=int(os.environ.get("AMI_BOOTSTRAP_REPLICATES", "2000")),
    )
    output = {
        "dataset": "mimic4",
        "model": "admission_severity_benchmark",
        "seeds": args.seeds,
        "folds": args.folds,
        "n": int(len(labels)),
        "variables": list(SEVERITY_COLUMNS),
        "proportional_odds_rejected_fraction": float(
            np.mean(
                [
                    np.asarray(result["proportional_odds_test_p_by_fold"]) < 0.05
                    for result in seed_results
                ]
            )
        ),
        "metrics": metrics,
        "calibration": {
            "p_y_ge_3": calibration_slope_intercept(
                (labels >= 3).astype(int), averaged[:, 3:].sum(axis=1)
            ),
            "p_y_4": calibration_slope_intercept(
                (labels == 4).astype(int), averaged[:, 4]
            ),
        },
    }
    path = RESULTS / "metrics_mimic4_severity_benchmark.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

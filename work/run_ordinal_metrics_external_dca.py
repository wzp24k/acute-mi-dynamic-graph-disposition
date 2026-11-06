"""Ordinal metrics, threshold calibration, and external decision-curve analysis.

This evaluation-only script reads frozen five-seed OOF predictions and frozen
external transport predictions. It does not fit or retrain any prediction
model. All outputs are written to results/ordinal_evaluation/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Keep parallel workers from oversubscribing BLAS threads.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import special
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import (  # noqa: E402
    PROCESSED,
    RESULTS,
    sha256_file,
)
from src.ami_dynamic.evaluation import (  # noqa: E402
    calibration_slope_intercept,
    decision_curve,
)
from work.run_external_calibration_common3 import (  # noqa: E402
    patient_safe_hospital_clusters,
)


OUTPUT_DIR = RESULTS / "ordinal_evaluation"
PREDICTION_DIR = RESULTS / "predictions"
SEEDS = (1, 7, 42, 123, 2024)
BOOTSTRAP_SEED = 20261104
DEFAULT_REPLICATES = 2000

MODEL_SPECS = (
    ("xgb_static", "XGB-static"),
    ("xgb_static_ts", "XGB-static+TS"),
    ("mlp_static", "MLP-static"),
    ("mlp_static_ts", "MLP-static+TS"),
    ("gru", "SeqOrdinal GRU"),
    ("grud", "GRU-D"),
    ("transformer", "Transformer"),
    ("stgcn_lite", "STGCN-lite"),
    ("dggru_static", "DG-GRU-static"),
    ("dggru", "DG-GRU"),
    ("dggru_tr", "DG-GRU+TR"),
    ("dggru_ctr", "DG-GRU+CTR"),
)
MODEL_NAMES = tuple(name for name, _ in MODEL_SPECS)
MODEL_LABELS = dict(MODEL_SPECS)
THRESHOLD_MODELS = ("xgb_static_ts", "dggru_tr", "dggru_ctr")
EXTERNAL_MODELS = ("xgb_static", "xgb_static_ts", "dggru_tr")
ORDINAL_CONTRASTS = (
    (
        "xgb_static_ts",
        "dggru_tr",
        "XGB-static+TS minus DG-GRU+TR",
    ),
    (
        "xgb_static_ts",
        "xgb_static",
        "XGB-static+TS minus XGB-static",
    ),
    (
        "dggru_tr",
        "dggru",
        "DG-GRU+TR minus DG-GRU",
    ),
    (
        "dggru_ctr",
        "dggru",
        "DG-GRU+CTR minus DG-GRU",
    ),
)
THRESHOLDS = tuple(np.round(np.arange(0.05, 0.6001, 0.01), 2))

OUTCOME_LABELS = {
    0: "Home or against advice",
    1: "Home health care",
    2: "Inpatient rehabilitation or assisted living",
    3: "Skilled nursing or other facility",
    4: "Hospice or in-hospital death",
}
CUMULATIVE_TARGETS = (
    (2, "P(y>=2)", "Inpatient rehabilitation, facility, hospice, or death"),
    (3, "P(y>=3)", "Skilled nursing or other facility, hospice, or death"),
    (4, "P(y>=4)", "Hospice or in-hospital death"),
)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def _format_ci(
    estimate: float,
    low: float,
    high: float,
    digits: int = 4,
) -> str:
    if not np.isfinite(estimate):
        return "NA"
    if not np.isfinite(low) or not np.isfinite(high):
        return f"{estimate:.{digits}f}"
    return f"{estimate:.{digits}f} ({low:.{digits}f} to {high:.{digits}f})"


def _group_contract(
    groups: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    rows: dict[int, np.ndarray] = {}
    for index in range(len(unique)):
        rows[index] = np.flatnonzero(inverse == index)
    return unique, rows


def _bootstrap_row_indices(
    group_rows: dict[int, np.ndarray],
    *,
    seed: int,
    replicates: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_groups = len(group_rows)
    sampled_groups = rng.integers(0, n_groups, size=(replicates, n_groups))
    indices = np.empty((replicates, sum(len(x) for x in group_rows.values())), dtype=np.int32)
    for row in range(replicates):
        indices[row] = np.concatenate(
            [group_rows[int(group)] for group in sampled_groups[row]]
        ).astype(np.int32, copy=False)
    return indices


def ordinal_c_index(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    n_classes: int,
) -> float:
    """Harrell-type C-index for ordered labels, with score ties counting 0.5.

    A comparable pair has different true ordered labels. The pair is
    concordant when the patient with the higher label also has a higher
    predicted expected class.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")

    _, inverse = np.unique(scores, return_inverse=True)
    n_score_groups = int(inverse.max()) + 1 if len(inverse) else 0
    counts = np.zeros((n_score_groups, n_classes), dtype=np.int64)
    np.add.at(counts, (inverse, labels), 1)

    before = np.cumsum(counts, axis=0) - counts
    before_lower = np.cumsum(before, axis=1)
    lower_before = np.zeros_like(before_lower)
    lower_before[:, 1:] = before_lower[:, :-1]
    concordant_between = int(np.sum(counts * lower_before))

    group_totals = counts.sum(axis=1)
    tied_pairs = int(
        np.sum(group_totals * (group_totals - 1) // 2)
    )
    tied_same_label = int(
        np.sum(counts * (counts - 1) // 2)
    )
    tied_different_label = tied_pairs - tied_same_label

    label_counts = np.bincount(labels, minlength=n_classes).astype(np.int64)
    lower_label_counts = np.cumsum(label_counts) - label_counts
    comparable_pairs = int(np.sum(label_counts * lower_label_counts))
    if comparable_pairs == 0:
        return float("nan")

    numerator = concordant_between + 0.5 * tied_different_label
    return float(numerator / comparable_pairs)


def quadratic_weighted_kappa(
    labels: np.ndarray,
    predicted: np.ndarray,
    *,
    n_classes: int,
) -> float:
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    observed = np.bincount(
        labels * n_classes + predicted,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes).astype(float)
    row = observed.sum(axis=1)
    column = observed.sum(axis=0)
    expected = np.outer(row, column) / max(len(labels), 1)
    index = np.arange(n_classes, dtype=float)
    weights = (index[:, None] - index[None, :]) ** 2
    numerator = float(np.sum(weights * observed))
    denominator = float(np.sum(weights * expected))
    if denominator <= 0:
        return float("nan")
    return float(1.0 - numerator / denominator)


def multiclass_brier(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    n_classes = probabilities.shape[1]
    one_hot = np.eye(n_classes, dtype=float)[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def core_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    n_classes = probabilities.shape[1]
    classes = np.arange(n_classes, dtype=float)
    expected_class = probabilities @ classes
    predicted_class = probabilities.argmax(axis=1)
    return {
        "ordinal_c_index": ordinal_c_index(
            labels,
            expected_class,
            n_classes=n_classes,
        ),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(
            labels,
            predicted_class,
            n_classes=n_classes,
        ),
        "ordinal_mae": float(np.mean(np.abs(labels - predicted_class))),
        "multiclass_brier": multiclass_brier(labels, probabilities),
        "mean_expected_class": float(expected_class.mean()),
    }


def binary_auc(target: np.ndarray, probability: np.ndarray) -> float:
    target = np.asarray(target, dtype=int)
    if len(np.unique(target)) < 2:
        return float("nan")
    return float(roc_auc_score(target, np.asarray(probability, dtype=float)))


def binary_brier(target: np.ndarray, probability: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    probability = np.asarray(probability, dtype=float)
    return float(np.mean((target - probability) ** 2))


def _calibration_fit(
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float]:
    """Near-unpenalized calibration slope and intercept via one-predictor IRLS."""
    target = np.asarray(target, dtype=float)
    if len(np.unique(target)) < 2:
        return float("nan"), float("nan")
    probability = np.clip(
        np.asarray(probability, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )
    logit = np.log(probability / (1.0 - probability))
    design = np.column_stack([np.ones(len(logit)), logit])
    beta = np.array([0.0, 1.0], dtype=float)
    penalty = 1e-6
    identity = np.eye(2, dtype=float)
    for _ in range(25):
        linear = design @ beta
        mean = special.expit(linear)
        weight = np.clip(mean * (1.0 - mean), 1e-9, None)
        working = linear + (target - mean) / weight
        information = design.T @ (weight[:, None] * design) + penalty * identity
        right = design.T @ (weight * working) - penalty * beta
        try:
            updated = np.linalg.solve(information, right)
        except np.linalg.LinAlgError:
            updated = np.linalg.lstsq(information, right, rcond=None)[0]
        step = updated - beta
        beta = updated
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(beta[0]), float(beta[1])


def threshold_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    include_calibration: bool = True,
) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    output = {
        "auroc": binary_auc(target, probability),
        "brier": binary_brier(target, probability),
    }
    if include_calibration:
        intercept, slope = _calibration_fit(target, probability)
        output["calibration_intercept"] = intercept
        output["calibration_slope"] = slope
    return output


def _ci_summary(
    estimates: list[float] | np.ndarray,
    point: float,
) -> dict[str, float]:
    values = np.asarray(estimates, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "estimate": float(point),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "valid_replicates": 0,
        }
    return {
        "estimate": float(point),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "valid_replicates": int(len(values)),
    }


def _bootstrap_chunk(
    start: int,
    stop: int,
    *,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold_probabilities: dict[str, np.ndarray],
    group_rows: dict[int, np.ndarray],
    seed: int,
) -> dict:
    indices = _bootstrap_row_indices(
        group_rows,
        seed=seed + 7919 * start,
        replicates=stop - start,
    )
    metric_names = (
        "ordinal_c_index",
        "quadratic_weighted_kappa",
        "ordinal_mae",
        "multiclass_brier",
    )
    model_values = {
        model: {metric: [] for metric in metric_names}
        for model in MODEL_NAMES
    }
    threshold_values = {
        model: {
            level: {metric: [] for metric in (
                "auroc",
                "brier",
                "calibration_intercept",
                "calibration_slope",
            )}
            for level, _, _ in CUMULATIVE_TARGETS
        }
        for model in THRESHOLD_MODELS
    }

    for sampled_rows in indices:
        sampled_labels = labels[sampled_rows]
        for model_index, model in enumerate(MODEL_NAMES):
            sampled_probability = probabilities[model_index, sampled_rows]
            metrics = core_metrics(sampled_labels, sampled_probability)
            for metric in metric_names:
                model_values[model][metric].append(float(metrics[metric]))
        for model in THRESHOLD_MODELS:
            for target_index, (level, _, _) in enumerate(CUMULATIVE_TARGETS):
                target = (sampled_labels >= level).astype(int)
                probability = threshold_probabilities[model][target_index][
                    sampled_rows
                ]
                metrics = threshold_metrics(target, probability)
                for metric, value in metrics.items():
                    threshold_values[model][level][metric].append(float(value))
    return {
        "model_values": model_values,
        "threshold_values": threshold_values,
    }


def _merge_bootstrap_chunks(
    chunks: list[dict],
) -> tuple[dict, dict]:
    model_values = {
        model: {
            metric: []
            for metric in (
                "ordinal_c_index",
                "quadratic_weighted_kappa",
                "ordinal_mae",
                "multiclass_brier",
            )
        }
        for model in MODEL_NAMES
    }
    threshold_values = {
        model: {
            level: {
                metric: []
                for metric in (
                    "auroc",
                    "brier",
                    "calibration_intercept",
                    "calibration_slope",
                )
            }
            for level, _, _ in CUMULATIVE_TARGETS
        }
        for model in THRESHOLD_MODELS
    }
    for chunk in chunks:
        for model, metrics in chunk["model_values"].items():
            for metric, values in metrics.items():
                model_values[model][metric].extend(values)
        for model, levels in chunk["threshold_values"].items():
            for level, metrics in levels.items():
                for metric, values in metrics.items():
                    threshold_values[model][level][metric].extend(values)
    return model_values, threshold_values


def load_primary_predictions() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    list[dict],
]:
    cohort = pd.read_csv(
        PROCESSED / "mimic4" / "cohort.csv",
        usecols=["stay_id", "subject_key", "outcome"],
    )
    labels = cohort["outcome"].to_numpy(dtype=int)
    groups = cohort["subject_key"].astype(str).to_numpy()
    stay_id = cohort["stay_id"].to_numpy(dtype=np.int64)
    probabilities: dict[str, np.ndarray] = {}
    inputs = []

    for model in MODEL_NAMES:
        seed_probabilities = []
        for seed in SEEDS:
            path = PREDICTION_DIR / f"mimic4_{model}_seed{seed}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=True) as data:
                cache_labels = data["labels"].astype(int)
                cache_probability = data["probabilities"].astype(float)
                cache_stay = data["stay_id"].astype(np.int64)
                cache_groups = data["subject_key"].astype(str)
            if not np.array_equal(cache_labels, labels):
                raise ValueError(f"{path.name}: labels differ from locked cohort")
            if not np.array_equal(cache_stay, stay_id):
                raise ValueError(f"{path.name}: stay_id order differs from cohort")
            if not np.array_equal(cache_groups, groups):
                raise ValueError(f"{path.name}: patient order differs from cohort")
            if cache_probability.shape != (len(labels), 5):
                raise ValueError(f"{path.name}: expected (n, 5) probabilities")
            if not np.isfinite(cache_probability).all():
                raise ValueError(f"{path.name}: non-finite probabilities")
            if not np.allclose(
                cache_probability.sum(axis=1),
                1.0,
                atol=2e-5,
            ):
                raise ValueError(f"{path.name}: probabilities do not sum to one")
            seed_probabilities.append(cache_probability)
            inputs.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "model": model,
                    "seed": seed,
                }
            )
        probabilities[model] = np.mean(np.stack(seed_probabilities), axis=0)
    return labels, groups, stay_id, probabilities, inputs


def build_table9(
    labels: np.ndarray,
    groups: np.ndarray,
    probabilities: dict[str, np.ndarray],
    bootstrap_values: dict,
) -> tuple[pd.DataFrame, dict]:
    probability_stack = np.stack(
        [probabilities[model] for model in MODEL_NAMES],
        axis=0,
    )
    rows = []
    detailed = {}
    for model_index, model in enumerate(MODEL_NAMES):
        point = core_metrics(labels, probability_stack[model_index])
        summary = {
            metric: _ci_summary(
                bootstrap_values[model][metric],
                point[metric],
            )
            for metric in (
                "ordinal_c_index",
                "quadratic_weighted_kappa",
                "ordinal_mae",
                "multiclass_brier",
            )
        }
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n": int(len(labels)),
                "n_patients": int(len(np.unique(groups))),
                "n_seed_models": len(SEEDS),
                "n_classes": int(probability_stack.shape[2]),
                "ordinal_c_index": point["ordinal_c_index"],
                "ordinal_c_index_ci_low": summary["ordinal_c_index"]["ci_low"],
                "ordinal_c_index_ci_high": summary["ordinal_c_index"]["ci_high"],
                "quadratic_weighted_kappa": point["quadratic_weighted_kappa"],
                "qwk_ci_low": summary["quadratic_weighted_kappa"]["ci_low"],
                "qwk_ci_high": summary["quadratic_weighted_kappa"]["ci_high"],
                "ordinal_mae": point["ordinal_mae"],
                "ordinal_mae_ci_low": summary["ordinal_mae"]["ci_low"],
                "ordinal_mae_ci_high": summary["ordinal_mae"]["ci_high"],
                "multiclass_brier": point["multiclass_brier"],
                "multiclass_brier_ci_low": summary["multiclass_brier"]["ci_low"],
                "multiclass_brier_ci_high": summary["multiclass_brier"]["ci_high"],
                "mean_expected_class": point["mean_expected_class"],
            }
        )
        detailed[model] = {
            "model_label": MODEL_LABELS[model],
            "point": point,
            "bootstrap": summary,
        }
    return pd.DataFrame(rows), detailed


def build_table9b(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    bootstrap_values: dict,
) -> tuple[pd.DataFrame, dict]:
    point_metrics = {
        model: core_metrics(labels, probabilities[model])
        for model in MODEL_NAMES
    }
    rows = []
    detailed = {}
    for model_a, model_b, label in ORDINAL_CONTRASTS:
        row = {
            "model_a": model_a,
            "model_b": model_b,
            "contrast": label,
        }
        detail = {"contrast": label}
        for metric, point_label in (
            ("ordinal_c_index", "ordinal_c_index"),
            ("quadratic_weighted_kappa", "qwk"),
            ("ordinal_mae", "ordinal_mae"),
        ):
            point = float(
                point_metrics[model_a][metric]
                - point_metrics[model_b][metric]
            )
            values_a = np.asarray(
                bootstrap_values[model_a][metric], dtype=float
            )
            values_b = np.asarray(
                bootstrap_values[model_b][metric], dtype=float
            )
            paired = values_a - values_b
            summary = _ci_summary(paired, point)
            row[f"{point_label}_delta"] = point
            row[f"{point_label}_ci_low"] = summary["ci_low"]
            row[f"{point_label}_ci_high"] = summary["ci_high"]
            row[f"{point_label}_valid_replicates"] = summary[
                "valid_replicates"
            ]
            detail[metric] = summary
        rows.append(row)
        detailed[label] = detail
    return pd.DataFrame(rows), detailed


def _risk_strata(
    target: np.ndarray,
    probability: np.ndarray,
) -> list[dict]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    definitions = (
        ("Low", 0.0, 0.2),
        ("Intermediate", 0.2, 0.5),
        ("High", 0.5, 1.0000001),
    )
    rows = []
    for name, lower, upper in definitions:
        rows_mask = (probability >= lower) & (probability < upper)
        rows.append(
            {
                "risk_stratum": name,
                "lower_bound": lower,
                "upper_bound": min(upper, 1.0),
                "n": int(rows_mask.sum()),
                "events": int(target[rows_mask].sum()),
                "mean_predicted": (
                    float(probability[rows_mask].mean())
                    if rows_mask.any()
                    else float("nan")
                ),
                "observed_rate": (
                    float(target[rows_mask].mean())
                    if rows_mask.any()
                    else float("nan")
                ),
            }
        )
    return rows


def _reliability_deciles(
    target: np.ndarray,
    probability: np.ndarray,
) -> list[dict]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    order = np.argsort(probability, kind="mergesort")
    rows = []
    for index, selected in enumerate(np.array_split(order, 10), start=1):
        if len(selected) == 0:
            continue
        rows.append(
            {
                "decile": index,
                "n": int(len(selected)),
                "mean_predicted": float(probability[selected].mean()),
                "observed_rate": float(target[selected].mean()),
                "calibration_gap": float(
                    target[selected].mean() - probability[selected].mean()
                ),
            }
        )
    return rows


def build_table10(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    bootstrap_values: dict,
) -> tuple[pd.DataFrame, dict]:
    threshold_probabilities = {
        model: [
            probabilities[model][:, level:].sum(axis=1)
            for level, _, _ in CUMULATIVE_TARGETS
        ]
        for model in THRESHOLD_MODELS
    }
    rows = []
    detailed = {}
    for model in THRESHOLD_MODELS:
        detailed[model] = {}
        for target_index, (level, label, definition) in enumerate(
            CUMULATIVE_TARGETS
        ):
            target = (labels >= level).astype(int)
            probability = threshold_probabilities[model][target_index]
            point = threshold_metrics(target, probability)
            regression = calibration_slope_intercept(target, probability)
            bootstrap = {
                metric: _ci_summary(
                    bootstrap_values[model][level][metric],
                    (
                        regression["slope"]
                        if metric == "calibration_slope"
                        else (
                            regression["intercept"]
                            if metric == "calibration_intercept"
                            else point[metric]
                        )
                    ),
                )
                for metric in (
                    "auroc",
                    "brier",
                    "calibration_intercept",
                    "calibration_slope",
                )
            }
            row = {
                "record_type": "threshold_performance",
                "model": model,
                "model_label": MODEL_LABELS[model],
                "threshold_label": label,
                "target_definition": definition,
                "risk_stratum": "",
                "n": int(len(target)),
                "events": int(target.sum()),
                "non_events": int((1 - target).sum()),
                "event_fraction": float(target.mean()),
                "mean_predicted": float(probability.mean()),
                "observed_rate": float(target.mean()),
                "calibration_gap": float(target.mean() - probability.mean()),
                "auroc": point["auroc"],
                "auroc_ci_low": bootstrap["auroc"]["ci_low"],
                "auroc_ci_high": bootstrap["auroc"]["ci_high"],
                "calibration_slope": regression["slope"],
                "calibration_slope_ci_low": bootstrap[
                    "calibration_slope"
                ]["ci_low"],
                "calibration_slope_ci_high": bootstrap[
                    "calibration_slope"
                ]["ci_high"],
                "calibration_intercept": regression["intercept"],
                "calibration_intercept_ci_low": bootstrap[
                    "calibration_intercept"
                ]["ci_low"],
                "calibration_intercept_ci_high": bootstrap[
                    "calibration_intercept"
                ]["ci_high"],
                "brier": point["brier"],
                "brier_ci_low": bootstrap["brier"]["ci_low"],
                "brier_ci_high": bootstrap["brier"]["ci_high"],
            }
            rows.append(row)
            strata = _risk_strata(target, probability)
            for stratum in strata:
                rows.append(
                    {
                        "record_type": "risk_stratum",
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "threshold_label": label,
                        "target_definition": definition,
                        "risk_stratum": (
                            f"{stratum['risk_stratum']} "
                            f"[{stratum['lower_bound']:.2f},"
                            f"{stratum['upper_bound']:.2f})"
                        ),
                        "n": stratum["n"],
                        "events": stratum["events"],
                        "non_events": stratum["n"] - stratum["events"],
                        "event_fraction": (
                            stratum["observed_rate"]
                            if np.isfinite(stratum["observed_rate"])
                            else float("nan")
                        ),
                        "mean_predicted": stratum["mean_predicted"],
                        "observed_rate": stratum["observed_rate"],
                        "calibration_gap": (
                            stratum["observed_rate"] - stratum["mean_predicted"]
                            if np.isfinite(stratum["observed_rate"])
                            and np.isfinite(stratum["mean_predicted"])
                            else float("nan")
                        ),
                        "auroc": float("nan"),
                        "auroc_ci_low": float("nan"),
                        "auroc_ci_high": float("nan"),
                        "calibration_slope": float("nan"),
                        "calibration_slope_ci_low": float("nan"),
                        "calibration_slope_ci_high": float("nan"),
                        "calibration_intercept": float("nan"),
                        "calibration_intercept_ci_low": float("nan"),
                        "calibration_intercept_ci_high": float("nan"),
                        "brier": float("nan"),
                        "brier_ci_low": float("nan"),
                        "brier_ci_high": float("nan"),
                    }
                )
            detailed[model][label] = {
                "level": level,
                "target_definition": definition,
                "n": int(len(target)),
                "events": int(target.sum()),
                "non_events": int((1 - target).sum()),
                "event_fraction": float(target.mean()),
                "point": {
                    **point,
                    **regression,
                },
                "bootstrap": bootstrap,
                "risk_strata": strata,
                "reliability_deciles": _reliability_deciles(
                    target,
                    probability,
                ),
            }
    return pd.DataFrame(rows), detailed


def load_external_dataset(
    dataset: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, str],
    list[dict],
]:
    required_columns = ["stay_id", "subject_key", "outcome4"]
    if dataset == "eicu":
        required_columns.append("hospital_key")
    cohort = pd.read_csv(
        PROCESSED / dataset / "cohort.csv",
        usecols=required_columns,
    )
    labels = cohort["outcome4"].to_numpy(dtype=int)
    stay_id = cohort["stay_id"].to_numpy(dtype=np.int64)
    subject_key = cohort["subject_key"].astype(str).to_numpy()
    probabilities: dict[str, np.ndarray] = {}
    inputs = []
    source_by_model = {}

    for model in EXTERNAL_MODELS:
        path = PREDICTION_DIR / f"{dataset}_{model}_transport.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as data:
            cache_labels = data["labels"].astype(int)
            cache_probability = data["probabilities"].astype(float)
            cache_stay = data["stay_id"].astype(np.int64)
            cache_subject = data["subject_key"].astype(str)
        if not np.array_equal(cache_labels, labels):
            raise ValueError(f"{path.name}: labels differ from external cohort")
        if not np.array_equal(cache_stay, stay_id):
            raise ValueError(f"{path.name}: stay_id order differs from cohort")
        if not np.array_equal(cache_subject, subject_key):
            raise ValueError(f"{path.name}: patient order differs from cohort")
        if cache_probability.shape != (len(labels), 4):
            raise ValueError(f"{path.name}: expected (n, 4) probabilities")
        if not np.isfinite(cache_probability).all():
            raise ValueError(f"{path.name}: non-finite probabilities")
        if not np.allclose(
            cache_probability.sum(axis=1),
            1.0,
            atol=2e-5,
        ):
            raise ValueError(f"{path.name}: probabilities do not sum to one")
        probabilities[model] = cache_probability
        source_by_model[model] = str(path)
        inputs.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "dataset": dataset,
                "model": model,
            }
        )

    patient_safe_clustering = None
    if dataset == "eicu":
        bootstrap_groups, patient_safe_clustering = (
            patient_safe_hospital_clusters(
                subject_key,
                cohort["hospital_key"].astype(str).to_numpy(),
            )
        )
        bootstrap_unit = "patient-safe hospital cluster"
    elif dataset == "mimic3":
        bootstrap_groups = subject_key
        bootstrap_unit = "patient"
    else:
        raise ValueError(dataset)
    contract = {
        "bootstrap_unit": bootstrap_unit,
        "n_bootstrap_groups": int(len(np.unique(bootstrap_groups))),
        "patient_safe_hospital_clustering": patient_safe_clustering,
        "probability_source": "raw transported probabilities",
        "source_by_model": json.dumps(source_by_model, sort_keys=True),
    }
    return labels, bootstrap_groups, probabilities, contract, inputs


def _bootstrap_counts(
    *,
    n_groups: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = np.zeros((replicates, n_groups), dtype=np.int32)
    for row in range(replicates):
        sampled = rng.integers(0, n_groups, size=n_groups)
        counts[row] = np.bincount(sampled, minlength=n_groups)
    return counts


def _dca_bootstrap(
    *,
    group_tp: np.ndarray,
    group_fp: np.ndarray,
    group_sizes: np.ndarray,
    prevalence: float,
    thresholds: np.ndarray,
    replicates: int,
    seed: int,
    n_jobs: int,
) -> np.ndarray:
    odds = thresholds / (1.0 - thresholds)
    group_contribution = group_tp - group_fp * odds[None, :]
    n_chunks = max(1, min(replicates, n_jobs * 4))
    bounds = np.linspace(0, replicates, n_chunks + 1, dtype=int)
    tasks = [
        (int(bounds[index]), int(bounds[index + 1]), index)
        for index in range(n_chunks)
        if bounds[index] < bounds[index + 1]
    ]

    def run_chunk(start: int, stop: int, index: int) -> np.ndarray:
        counts = _bootstrap_counts(
            n_groups=len(group_sizes),
            replicates=stop - start,
            seed=seed + index * 104729,
        )
        sampled_size = counts @ group_sizes
        sampled_contribution = counts @ group_contribution
        return sampled_contribution / sampled_size[:, None]

    chunks = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
        delayed(run_chunk)(start, stop, index)
        for start, stop, index in tasks
    )
    return np.vstack(chunks)


def build_external_dca(
    *,
    replicates: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    targets = (
        ("high_need_y_ge_2", "high-need disposition (4-class y>=2)", 2),
        ("death_y_3", "in-hospital death (4-class y=3)", 3),
    )
    rows = []
    detailed = {}
    inputs = []
    tasks = []
    for dataset in ("eicu", "mimic3"):
        labels, groups, probabilities, contract, dataset_inputs = (
            load_external_dataset(dataset)
        )
        inputs.extend(dataset_inputs)
        unique_groups, inverse = np.unique(groups, return_inverse=True)
        group_sizes = np.bincount(inverse, minlength=len(unique_groups)).astype(float)
        detailed[dataset] = {
            "n": int(len(labels)),
            "bootstrap_unit": contract["bootstrap_unit"],
            "n_bootstrap_groups": int(len(unique_groups)),
            "probability_source": contract["probability_source"],
            "patient_safe_hospital_clustering": contract[
                "patient_safe_hospital_clustering"
            ],
            "outcome_counts": {
                str(level): int(np.sum(labels == level))
                for level in range(4)
            },
            "models": {},
        }
        for target_name, target_label, level in targets:
            target = (labels >= level).astype(int) if level == 2 else (labels == level).astype(int)
            prevalence = float(target.mean())
            for model in EXTERNAL_MODELS:
                probability = (
                    probabilities[model][:, 2:].sum(axis=1)
                    if level == 2
                    else probabilities[model][:, 3]
                )
                point_frame = decision_curve(
                    target,
                    probability,
                    thresholds=np.asarray(THRESHOLDS),
                )
                tasks.append(
                    (
                        dataset,
                        target_name,
                        target_label,
                        model,
                        labels,
                        target,
                        probability,
                        groups,
                        inverse,
                        len(unique_groups),
                        group_sizes,
                        prevalence,
                        point_frame,
                        contract,
                    )
                )

    def evaluate_task(task: tuple) -> tuple:
        (
            dataset,
            target_name,
            target_label,
            model,
            labels,
            target,
            probability,
            groups,
            inverse,
            n_groups,
            group_sizes,
            prevalence,
            point_frame,
            contract,
        ) = task
        thresholds = np.asarray(THRESHOLDS)
        group_tp = np.zeros((n_groups, len(thresholds)), dtype=float)
        group_fp = np.zeros((n_groups, len(thresholds)), dtype=float)
        for index, threshold in enumerate(thresholds):
            selected = probability >= threshold
            group_tp[:, index] = np.bincount(
                inverse,
                weights=(selected & (target == 1)).astype(float),
                minlength=n_groups,
            )
            group_fp[:, index] = np.bincount(
                inverse,
                weights=(selected & (target == 0)).astype(float),
                minlength=n_groups,
            )
        bootstrap = _dca_bootstrap(
            group_tp=group_tp,
            group_fp=group_fp,
            group_sizes=group_sizes,
            prevalence=prevalence,
            thresholds=thresholds,
            replicates=replicates,
            seed=BOOTSTRAP_SEED + (1000 if dataset == "eicu" else 2000)
            + (100 if target_name == "death_y_3" else 0)
            + 10 * EXTERNAL_MODELS.index(model),
            n_jobs=1,
        )
        curve_rows = []
        for index, threshold in enumerate(thresholds):
            values = bootstrap[:, index]
            values = values[np.isfinite(values)]
            curve_rows.append(
                {
                    "dataset": dataset,
                    "validation_type": (
                        "external multicentre transport"
                        if dataset == "eicu"
                        else "secondary same-institution validation"
                    ),
                    "bootstrap_unit": contract["bootstrap_unit"],
                    "n_bootstrap_groups": n_groups,
                    "probability_source": contract["probability_source"],
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "target": target_name,
                    "target_label": target_label,
                    "threshold": float(threshold),
                    "n": int(len(target)),
                    "events": int(target.sum()),
                    "prevalence": prevalence,
                    "net_benefit": float(point_frame.iloc[index]["net_benefit"]),
                    "net_benefit_ci_low": (
                        float(np.percentile(values, 2.5))
                        if len(values)
                        else float("nan")
                    ),
                    "net_benefit_ci_high": (
                        float(np.percentile(values, 97.5))
                        if len(values)
                        else float("nan")
                    ),
                    "treat_all_net_benefit": float(
                        point_frame.iloc[index]["treat_all"]
                    ),
                    "treat_none_net_benefit": 0.0,
                    "bootstrap_replicates": int(replicates),
                    "valid_bootstrap_replicates": int(len(values)),
                }
            )
        summary = {
            "target_label": target_label,
            "n": int(len(target)),
            "events": int(target.sum()),
            "prevalence": prevalence,
            "n_thresholds": int(len(thresholds)),
            "bootstrap_unit": contract["bootstrap_unit"],
            "n_bootstrap_groups": n_groups,
            "probability_source": contract["probability_source"],
            "bootstrap_replicates": int(replicates),
            "points_at_standard_thresholds": {
                f"{threshold:.2f}": {
                    "net_benefit": float(
                        point_frame.loc[
                            np.isclose(
                                point_frame["threshold"],
                                threshold,
                            ),
                            "net_benefit",
                        ].iloc[0]
                    ),
                    "treat_all_net_benefit": float(
                        point_frame.loc[
                            np.isclose(
                                point_frame["threshold"],
                                threshold,
                            ),
                            "treat_all",
                        ].iloc[0]
                    ),
                }
                for threshold in (0.10, 0.20, 0.30, 0.40, 0.50)
            },
        }
        return (
            dataset,
            target_name,
            model,
            pd.DataFrame(curve_rows),
            summary,
        )

    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=5)(
        delayed(evaluate_task)(task) for task in tasks
    )
    for dataset, target_name, model, frame, summary in results:
        rows.append(frame)
        detailed[dataset]["models"].setdefault(model, {})[
            target_name
        ] = summary
    curve_frame = pd.concat(rows, ignore_index=True)
    return curve_frame, detailed, inputs


def write_table9_markdown(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Table 9. Ordinal performance from five-seed OOF predictions",
        "",
        (
            "Probabilities were averaged across seeds 1, 7, 42, 123, and "
            "2024 before metric calculation. Ordinal C-index used the "
            "predicted expected class as the continuous score and counted "
            "tied scores as 0.5. QWK and ordinal MAE used the maximum-"
            "probability class. Intervals are patient-bootstrap percentile "
            "95% CIs."
        ),
        "",
        "| Model | Ordinal C-index (95% CI) | QWK (95% CI) | Ordinal MAE (95% CI) | Multi-class Brier (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    row.model_label,
                    _format_ci(
                        row.ordinal_c_index,
                        row.ordinal_c_index_ci_low,
                        row.ordinal_c_index_ci_high,
                    ),
                    _format_ci(
                        row.quadratic_weighted_kappa,
                        row.qwk_ci_low,
                        row.qwk_ci_high,
                    ),
                    _format_ci(
                        row.ordinal_mae,
                        row.ordinal_mae_ci_low,
                        row.ordinal_mae_ci_high,
                    ),
                    _format_ci(
                        row.multiclass_brier,
                        row.multiclass_brier_ci_low,
                        row.multiclass_brier_ci_high,
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Primary cohort: MIMIC-IV, n={int(frame.iloc[0]['n']):,}; "
            f"{int(frame.iloc[0]['n_patients']):,} patients; "
            f"{int(frame.iloc[0]['n_classes'])} outcome classes.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_table9b_markdown(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Table 9b. Paired ordinal-metric contrasts",
        "",
        (
            "Values are the first named model minus the second named model. "
            "Intervals use the same patient-bootstrap replicates as Table 9. "
            "Negative values are retained without sign flipping or absolute "
            "transformation."
        ),
        "",
        (
            "| Contrast | Delta ordinal C-index (95% CI) | "
            "Delta QWK (95% CI) | Delta ordinal MAE (95% CI) |"
        ),
        "|---|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    row.contrast,
                    _format_ci(
                        row.ordinal_c_index_delta,
                        row.ordinal_c_index_ci_low,
                        row.ordinal_c_index_ci_high,
                    ),
                    _format_ci(
                        row.qwk_delta,
                        row.qwk_ci_low,
                        row.qwk_ci_high,
                    ),
                    _format_ci(
                        row.ordinal_mae_delta,
                        row.ordinal_mae_ci_low,
                        row.ordinal_mae_ci_high,
                    ),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table10_markdown(frame: pd.DataFrame, path: Path) -> None:
    performance = frame[frame["record_type"].eq("threshold_performance")]
    strata = frame[frame["record_type"].eq("risk_stratum")]
    lines = [
        "# Table 10. Cumulative-threshold performance, calibration, and reliability",
        "",
        (
            "Cumulative probabilities were P(y>=2)=sum(p2..p4), "
            "P(y>=3)=sum(p3..p4), and P(y>=4)=p4 on the five-level "
            "MIMIC-IV outcome scale. Calibration models were "
            "logit(y)=intercept+slope*logit(p). Intervals use 2,000 "
            "patient-bootstrap replicates."
        ),
        "",
        "## Threshold performance",
        "",
        "| Model | Threshold | Events/N | AUROC (95% CI) | Calibration slope | Intercept | Brier (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in performance.itertuples(index=False):
        slope_text = (
            _format_ci(
                row.calibration_slope,
                row.calibration_slope_ci_low,
                row.calibration_slope_ci_high,
                3,
            )
            if np.isfinite(row.calibration_slope)
            else "NA"
        )
        intercept_text = (
            _format_ci(
                row.calibration_intercept,
                row.calibration_intercept_ci_low,
                row.calibration_intercept_ci_high,
                3,
            )
            if np.isfinite(row.calibration_intercept)
            else "NA"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row.model_label,
                    row.threshold_label,
                    f"{int(row.events):,}/{int(row.n):,}",
                    _format_ci(
                        row.auroc,
                        row.auroc_ci_low,
                        row.auroc_ci_high,
                    ),
                    slope_text,
                    intercept_text,
                    _format_ci(
                        row.brier,
                        row.brier_ci_low,
                        row.brier_ci_high,
                    ),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Risk-stratified reliability",
            "",
            (
                "Fixed strata were low risk [0.00,0.20), intermediate "
                "[0.20,0.50), and high [0.50,1.00]. Calibration gap is "
                "observed rate minus mean predicted risk."
            ),
            "",
            "| Model | Threshold | Risk stratum | N | Events | Mean predicted | Observed | Gap |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in strata.itertuples(index=False):
        if int(row.n) == 0:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row.model_label,
                    row.threshold_label,
                    row.risk_stratum,
                    f"{int(row.n):,}",
                    f"{int(row.events):,}",
                    f"{row.mean_predicted:.4f}",
                    f"{row.observed_rate:.4f}",
                    f"{row.calibration_gap:+.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## External decision-curve methods",
            "",
            (
                "External DCA used raw transported probabilities, not "
                "recalibrated probabilities. eICU net-benefit intervals "
                "resampled patient-safe hospital clusters, defined as "
                "connected components of the patient-hospital graph so that "
                "hospitals sharing any patient were resampled together. "
                "MIMIC-III intervals resampled patients. Exact cluster "
                "counts and methods are stored in "
                "ordinal_evaluation_report.json and the bootstrap metadata "
                "columns of external_dca_curves.csv."
            ),
            "",
            (
                "Negative values, including negative gaps, slopes, net "
                "benefits, QWK, and C-index contrasts, were retained without "
                "absolute transformation or truncation."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def self_checks(
    *,
    table9: pd.DataFrame,
    table9b: pd.DataFrame,
    table10: pd.DataFrame,
    dca: pd.DataFrame,
    outputs: list[Path],
) -> dict:
    toy_labels = np.array([0, 1, 2, 3, 4, 2, 1], dtype=int)
    toy_scores = toy_labels.astype(float)
    c_index_identity = ordinal_c_index(
        toy_labels,
        toy_scores,
        n_classes=5,
    )
    c_index_ties = ordinal_c_index(
        toy_labels,
        np.ones(len(toy_labels)),
        n_classes=5,
    )
    if not np.isclose(c_index_identity, 1.0):
        raise AssertionError("Ordinal C-index identity test failed")
    if not np.isclose(c_index_ties, 0.5):
        raise AssertionError("Ordinal C-index tie test failed")
    if len(table9) != len(MODEL_NAMES):
        raise AssertionError("Table 9 does not contain every prespecified model")
    if len(table9b) != len(ORDINAL_CONTRASTS):
        raise AssertionError("Table 9b does not contain every paired contrast")
    if not np.all(
        (table9["ordinal_c_index"] >= 0)
        & (table9["ordinal_c_index"] <= 1)
    ):
        raise AssertionError("Ordinal C-index outside [0, 1]")
    performance = table10[table10["record_type"].eq("threshold_performance")]
    if len(performance) != len(THRESHOLD_MODELS) * len(CUMULATIVE_TARGETS):
        raise AssertionError("Table 10 threshold rows are incomplete")
    expected_dca_rows = (
        2
        * len(EXTERNAL_MODELS)
        * 2
        * len(THRESHOLDS)
    )
    if len(dca) != expected_dca_rows:
        raise AssertionError("External DCA row count is incomplete")
    if dca["threshold"].min() != 0.05 or dca["threshold"].max() != 0.60:
        raise AssertionError("External DCA threshold range is incorrect")
    if not outputs or not all(path.is_file() and path.stat().st_size > 0 for path in outputs):
        raise AssertionError("One or more output files are missing or empty")
    return {
        "status": "passed",
        "ordinal_c_index_identity": float(c_index_identity),
        "ordinal_c_index_all_ties": float(c_index_ties),
        "table9_rows": int(len(table9)),
        "table9b_rows": int(len(table9b)),
        "table10_rows": int(len(table10)),
        "table10_threshold_rows": int(len(performance)),
        "external_dca_rows": int(len(dca)),
        "negative_values_retained": True,
        "absolute_value_or_sign_flip_applied": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_REPLICATES,
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(16, os.cpu_count() or 1)),
    )
    args = parser.parse_args()
    if args.bootstrap_replicates < 2:
        raise ValueError("At least two bootstrap replicates are required")

    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Loading frozen predictions; bootstrap={args.bootstrap_replicates}, "
        f"jobs={args.jobs}",
        flush=True,
    )
    labels, groups, stay_id, probabilities, primary_inputs = (
        load_primary_predictions()
    )
    probability_stack = np.stack(
        [probabilities[model] for model in MODEL_NAMES],
        axis=0,
    )
    _, group_rows = _group_contract(groups)
    threshold_probabilities = {
        model: np.stack(
            [
                probabilities[model][:, level:].sum(axis=1)
                for level, _, _ in CUMULATIVE_TARGETS
            ],
            axis=0,
        )
        for model in THRESHOLD_MODELS
    }

    print("Running patient bootstrap for ordinal and calibration metrics", flush=True)
    n_chunks = max(1, min(args.bootstrap_replicates, args.jobs * 4))
    bounds = np.linspace(
        0,
        args.bootstrap_replicates,
        n_chunks + 1,
        dtype=int,
    )
    chunk_results = Parallel(
        n_jobs=args.jobs,
        prefer="processes",
        verbose=5,
    )(
        delayed(_bootstrap_chunk)(
            int(bounds[index]),
            int(bounds[index + 1]),
            labels=labels,
            probabilities=probability_stack,
            threshold_probabilities=threshold_probabilities,
            group_rows=group_rows,
            seed=BOOTSTRAP_SEED,
        )
        for index in range(n_chunks)
        if bounds[index] < bounds[index + 1]
    )
    model_bootstrap, threshold_bootstrap = _merge_bootstrap_chunks(
        chunk_results
    )

    table9, detailed9 = build_table9(
        labels,
        groups,
        probabilities,
        model_bootstrap,
    )
    table9b, detailed9b = build_table9b(
        labels,
        probabilities,
        model_bootstrap,
    )
    table10, detailed10 = build_table10(
        labels,
        probabilities,
        threshold_bootstrap,
    )

    print(
        "Running external DCA with patient-safe hospital-cluster bootstrap "
        "for eICU and patient bootstrap for MIMIC-III",
        flush=True,
    )
    dca, detailed_dca, external_inputs = build_external_dca(
        replicates=args.bootstrap_replicates,
        n_jobs=args.jobs,
    )

    table9_path = OUTPUT_DIR / "table9_ordinal_metrics.csv"
    table9_md_path = OUTPUT_DIR / "table9_ordinal_metrics.md"
    table9b_path = OUTPUT_DIR / "table9b_ordinal_contrasts.csv"
    table9b_md_path = OUTPUT_DIR / "table9b_ordinal_contrasts.md"
    table10_path = OUTPUT_DIR / "table10_threshold_calibration.csv"
    table10_md_path = OUTPUT_DIR / "table10_threshold_calibration.md"
    dca_path = OUTPUT_DIR / "external_dca_curves.csv"
    report_path = OUTPUT_DIR / "ordinal_evaluation_report.json"

    table9.to_csv(table9_path, index=False)
    table9b.to_csv(table9b_path, index=False)
    table10.to_csv(table10_path, index=False)
    dca.to_csv(dca_path, index=False)
    write_table9_markdown(table9, table9_md_path)
    write_table9b_markdown(table9b, table9b_md_path)
    write_table10_markdown(table10, table10_md_path)

    checks = self_checks(
        table9=table9,
        table9b=table9b,
        table10=table10,
        dca=dca,
        outputs=[
            table9_path,
            table9_md_path,
            table9b_path,
            table9b_md_path,
            table10_path,
            table10_md_path,
            dca_path,
        ],
    )
    report = {
        "status": "passed",
        "analysis": "ordinal metrics, cumulative-threshold calibration, and external DCA",
        "runtime_seconds": float(time.time() - started),
        "protocol": {
            "primary_dataset": "mimic4",
            "primary_n": int(len(labels)),
            "primary_n_patients": int(len(np.unique(groups))),
            "primary_n_stays": int(len(np.unique(stay_id))),
            "seeds": list(SEEDS),
            "seed_aggregation": "mean predicted probability at each patient",
            "n_classes": 5,
            "models": {
                model: MODEL_LABELS[model] for model in MODEL_NAMES
            },
            "outcome_labels": OUTCOME_LABELS,
            "ordinal_c_index": (
                "Comparable pairs have different true ordered labels; "
                "predicted expected class is the score; score ties count 0.5."
            ),
            "quadratic_weighted_kappa": (
                "Maximum-probability class versus observed class; "
                "squared-distance weights."
            ),
            "ordinal_mae": (
                "Mean absolute difference between maximum-probability class "
                "and observed class."
            ),
            "multiclass_brier": (
                "Mean patient-level sum over classes of squared probability "
                "error against the one-hot observed class."
            ),
            "confidence_intervals": {
                "method": "percentile bootstrap",
                "replicates_requested": int(args.bootstrap_replicates),
                "primary_resampling_unit": "patient (subject_key)",
                "eicu_dca_resampling_unit": "patient-safe hospital cluster",
                "mimic3_dca_resampling_unit": "patient (subject_key)",
                "seed": BOOTSTRAP_SEED,
            },
            "external_dca_methods": {
                "probability_source": "raw transported probabilities",
                "eicu_bootstrap_rule": (
                    "connected components of the patient-hospital graph; "
                    "hospitals sharing any patient are resampled together "
                    "as one patient-safe hospital cluster"
                ),
                "mimic3_bootstrap_rule": "patient (subject_key)",
            },
            "threshold_calibration": (
                "logit(y)=intercept+slope*logit(p); near-unpenalized "
                "one-predictor logistic regression."
            ),
            "risk_strata": {
                "low": [0.0, 0.2],
                "intermediate": [0.2, 0.5],
                "high": [0.5, 1.0],
            },
            "external_dca_thresholds": {
                "minimum": 0.05,
                "maximum": 0.60,
                "increment": 0.01,
                "n_thresholds": len(THRESHOLDS),
            },
            "external_targets": {
                "high_need_y_ge_2": "4-class y>=2",
                "death_y_3": "4-class y=3",
            },
        },
        "outcome_counts": {
            str(level): int(np.sum(labels == level))
            for level in range(5)
        },
        "inputs": {
            "primary_predictions": primary_inputs,
            "external_predictions": external_inputs,
        },
        "table9_ordinal_metrics": detailed9,
        "table9b_ordinal_contrasts": detailed9b,
        "table10_threshold_calibration": detailed10,
        "external_dca": detailed_dca,
        "outputs": {
            "table9_ordinal_metrics_csv": str(table9_path),
            "table9_ordinal_metrics_md": str(table9_md_path),
            "table9b_ordinal_contrasts_csv": str(table9b_path),
            "table9b_ordinal_contrasts_md": str(table9b_md_path),
            "table10_threshold_calibration_csv": str(table10_path),
            "table10_threshold_calibration_md": str(table10_md_path),
            "external_dca_curves_csv": str(dca_path),
            "ordinal_evaluation_report_json": str(report_path),
        },
        "self_checks": checks,
    }
    _write_json(report_path, report)

    # Re-read outputs to catch serialization failures before reporting success.
    if len(pd.read_csv(table9_path)) != len(table9):
        raise AssertionError("Table 9 CSV round-trip failed")
    if len(pd.read_csv(table9b_path)) != len(table9b):
        raise AssertionError("Table 9b CSV round-trip failed")
    if len(pd.read_csv(table10_path)) != len(table10):
        raise AssertionError("Table 10 CSV round-trip failed")
    if len(pd.read_csv(dca_path)) != len(dca):
        raise AssertionError("External DCA CSV round-trip failed")
    json.loads(report_path.read_text(encoding="utf-8"))

    print(
        json.dumps(
            {
                "status": "passed",
                "outputs": report["outputs"],
                "runtime_seconds": report["runtime_seconds"],
                "self_checks": checks,
                "headline": {
                    row["model_label"]: {
                        "ordinal_c_index": row["ordinal_c_index"],
                        "qwk": row["quadratic_weighted_kappa"],
                        "ordinal_mae": row["ordinal_mae"],
                        "multiclass_brier": row["multiclass_brier"],
                    }
                    for row in table9.to_dict(orient="records")
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

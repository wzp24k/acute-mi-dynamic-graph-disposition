"""Discrimination, ordinal, calibration, and clinical-utility metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    log_loss,
    roc_auc_score,
)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def point_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float), 1e-7, 1.0
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    classes = np.arange(probabilities.shape[1])
    pred = probabilities.argmax(axis=1)
    per_class = {
        int(k): safe_auc((labels == k).astype(int), probabilities[:, k])
        for k in classes
    }
    try:
        macro_auc = float(
            roc_auc_score(
                labels,
                probabilities,
                multi_class="ovr",
                average="macro",
                labels=classes,
            )
        )
    except ValueError:
        macro_auc = float("nan")
    cumulative = np.cumsum(probabilities, axis=1)[:, :-1]
    return {
        "macro_auc": macro_auc,
        "accuracy": float(accuracy_score(labels, pred)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(labels, pred, weights="quadratic")
        ),
        "ordinal_mae": float(np.mean(np.abs(labels - pred))),
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "multiclass_brier": float(
            np.mean(np.sum((probabilities - np.eye(len(classes))[labels]) ** 2, axis=1))
        ),
        "per_class_auc": per_class,
        "mean_predicted_class": float(np.mean(np.sum(probabilities * classes, axis=1))),
        "cumulative_probabilities": cumulative,
    }


def _metric_scalar(
    labels: np.ndarray,
    probabilities: np.ndarray,
    name: str,
) -> float:
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float), 1e-7, 1.0
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if name == "macro_auc":
        classes = np.arange(probabilities.shape[1])
        try:
            return float(
                roc_auc_score(
                    labels,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                    labels=classes,
                )
            )
        except ValueError:
            return float("nan")
    if name == "highest_class_auc":
        # Fix the binary target to the model's highest probability column.
        # A bootstrap resample can omit that class; deriving the target from
        # labels.max() would then silently evaluate a different outcome.
        highest_class = (labels == probabilities.shape[1] - 1).astype(int)
        return safe_auc(highest_class, probabilities[:, -1])
    metrics = point_metrics(labels, probabilities)
    if name in metrics:
        return float(metrics[name])
    raise ValueError(f"Unknown metric {name}")


def bootstrap_metric_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    metric: str = "macro_auc",
    groups: np.ndarray | None = None,
    replicates: int = 2000,
    seed: int = 20261104,
) -> dict:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    groups = (
        np.arange(len(labels))
        if groups is None
        else np.asarray(groups)
    )
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_rows = [
        np.flatnonzero(inverse == group_index)
        for group_index in range(len(unique_groups))
    ]
    sampled = []
    for _ in range(replicates):
        chosen = rng.choice(len(unique_groups), size=len(unique_groups), replace=True)
        rows = np.concatenate([group_rows[group_index] for group_index in chosen])
        value = _metric_scalar(labels[rows], probabilities[rows], metric)
        if np.isfinite(value):
            sampled.append(value)
    if not sampled:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    estimate = _metric_scalar(labels, probabilities, metric)
    return {
        "estimate": float(estimate),
        "ci_low": float(np.percentile(sampled, 2.5)),
        "ci_high": float(np.percentile(sampled, 97.5)),
        "bootstrap_replicates_valid": int(len(sampled)),
    }


def bootstrap_binary_auc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    replicates: int = 2000,
    seed: int = 20261104,
) -> dict:
    """Bootstrap a binary AUROC with optional cluster resampling."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    groups = (
        np.arange(len(labels))
        if groups is None
        else np.asarray(groups)
    )
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_rows = [
        np.flatnonzero(inverse == group_index)
        for group_index in range(len(unique_groups))
    ]
    sampled = []
    for _ in range(replicates):
        chosen = rng.choice(
            len(unique_groups), size=len(unique_groups), replace=True
        )
        rows = np.concatenate(
            [group_rows[group_index] for group_index in chosen]
        )
        value = safe_auc(labels[rows], scores[rows])
        if np.isfinite(value):
            sampled.append(value)
    if not sampled:
        return {
            "estimate": safe_auc(labels, scores),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_replicates_valid": 0,
        }
    return {
        "estimate": safe_auc(labels, scores),
        "ci_low": float(np.percentile(sampled, 2.5)),
        "ci_high": float(np.percentile(sampled, 97.5)),
        "bootstrap_replicates_valid": int(len(sampled)),
    }


def paired_bootstrap_delta(
    labels: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    *,
    metric: str = "macro_auc",
    groups: np.ndarray | None = None,
    replicates: int = 2000,
    seed: int = 20261104,
) -> dict:
    """Delta defined as model A minus model B."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    probabilities_a = np.asarray(probabilities_a, dtype=float)
    probabilities_b = np.asarray(probabilities_b, dtype=float)
    groups = (
        np.arange(len(labels)) if groups is None else np.asarray(groups)
    )
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_rows = [
        np.flatnonzero(inverse == group_index)
        for group_index in range(len(unique_groups))
    ]
    deltas = []
    for _ in range(replicates):
        chosen = rng.choice(len(unique_groups), size=len(unique_groups), replace=True)
        rows = np.concatenate([group_rows[group_index] for group_index in chosen])
        value_a = _metric_scalar(labels[rows], probabilities_a[rows], metric)
        value_b = _metric_scalar(labels[rows], probabilities_b[rows], metric)
        if np.isfinite(value_a) and np.isfinite(value_b):
            deltas.append(value_a - value_b)
    estimate = _metric_scalar(labels, probabilities_a, metric) - _metric_scalar(
        labels, probabilities_b, metric
    )
    if not deltas:
        return {
            "delta": float(estimate),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    return {
        "delta": float(estimate),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "bootstrap_replicates_valid": int(len(deltas)),
    }


def calibration_slope_intercept(
    target: np.ndarray,
    probability: np.ndarray,
) -> dict:
    target = np.asarray(target, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(target)) < 2:
        return {"slope": float("nan"), "intercept": float("nan")}
    logit = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(logit, target)
    return {
        "slope": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def reliability_bins(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    order = np.argsort(probability)
    groups = np.array_split(order, n_bins)
    rows = []
    for index, rows_index in enumerate(groups):
        if len(rows_index) == 0:
            continue
        rows.append(
            {
                "bin": index + 1,
                "n": int(len(rows_index)),
                "mean_predicted": float(probability[rows_index].mean()),
                "observed": float(target[rows_index].mean()),
            }
        )
    return pd.DataFrame(rows)


def continuous_nri(
    target: np.ndarray,
    old_probability: np.ndarray,
    new_probability: np.ndarray,
) -> float:
    target = np.asarray(target, dtype=int)
    delta = np.asarray(new_probability) - np.asarray(old_probability)
    events = target == 1
    non_events = ~events
    if events.sum() == 0 or non_events.sum() == 0:
        return float("nan")
    event_component = (
        np.mean(delta[events] > 0) - np.mean(delta[events] < 0)
    )
    non_event_component = (
        np.mean(delta[non_events] < 0) - np.mean(delta[non_events] > 0)
    )
    return float(event_component + non_event_component)


def integrated_discrimination_improvement(
    target: np.ndarray,
    old_probability: np.ndarray,
    new_probability: np.ndarray,
) -> float:
    target = np.asarray(target, dtype=int)
    events = target == 1
    non_events = ~events
    if events.sum() == 0 or non_events.sum() == 0:
        return float("nan")
    old_discrimination = old_probability[events].mean() - old_probability[
        non_events
    ].mean()
    new_discrimination = new_probability[events].mean() - new_probability[
        non_events
    ].mean()
    return float(new_discrimination - old_discrimination)


def cumulative_nri_idi(
    labels: np.ndarray,
    old_probabilities: np.ndarray,
    new_probabilities: np.ndarray,
) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=int)
    old_cumulative = np.cumsum(old_probabilities, axis=1)[:, :-1]
    new_cumulative = np.cumsum(new_probabilities, axis=1)[:, :-1]
    rows = []
    for threshold_index in range(old_cumulative.shape[1]):
        target = (labels >= threshold_index + 1).astype(int)
        rows.append(
            {
                "threshold": f"y>={threshold_index + 1}",
                "nri": continuous_nri(
                    target,
                    old_cumulative[:, threshold_index],
                    new_cumulative[:, threshold_index],
                ),
                "idi": integrated_discrimination_improvement(
                    target,
                    old_cumulative[:, threshold_index],
                    new_cumulative[:, threshold_index],
                ),
            }
        )
    return pd.DataFrame(rows)


def decision_curve(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)
    prevalence = target.mean()
    rows = []
    for threshold in thresholds:
        predicted = probability >= threshold
        tp = int(np.sum(predicted & (target == 1)))
        fp = int(np.sum(predicted & (target == 0)))
        net_benefit = tp / len(target) - fp / len(target) * (
            threshold / (1.0 - threshold)
        )
        rows.append(
            {
                "threshold": float(threshold),
                "net_benefit": float(net_benefit),
                "treat_all": float(
                    prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)
                ),
            }
        )
    return pd.DataFrame(rows)


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    midranks = np.zeros(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    result = np.empty_like(midranks)
    result[order] = midranks
    return result


def delong_auc_test(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict:
    labels = np.asarray(labels, dtype=int)
    positive = labels == 1
    negative = ~positive
    if positive.sum() == 0 or negative.sum() == 0:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "p_value": float("nan")}

    def components(scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        pos = scores[positive]
        neg = scores[negative]
        pos_ranks = _compute_midrank(pos)
        neg_ranks = _compute_midrank(neg)
        tz = _compute_midrank(np.concatenate([pos, neg]))
        auc = (
            tz[: len(pos)].sum()
            - len(pos) * (len(pos) + 1) / 2.0
        ) / (len(pos) * len(neg))
        v01 = (tz[: len(pos)] - pos_ranks) / len(neg)
        v10 = 1.0 - (tz[len(pos) :] - neg_ranks) / len(pos)
        return float(auc), v01, v10

    auc_a, v01_a, v10_a = components(np.asarray(scores_a, dtype=float))
    auc_b, v01_b, v10_b = components(np.asarray(scores_b, dtype=float))
    covariance = (
        np.cov(np.stack([v01_a, v01_b])) / negative.sum()
        + np.cov(np.stack([v10_a, v10_b])) / positive.sum()
    )
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    if variance <= 0:
        p_value = 1.0
    else:
        z_value = (auc_a - auc_b) / np.sqrt(variance)
        p_value = float(2 * stats.norm.sf(abs(z_value)))
    return {"auc_a": auc_a, "auc_b": auc_b, "p_value": p_value}


def delong_per_class(
    labels: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for class_index in range(probabilities_a.shape[1]):
        result = delong_auc_test(
            (np.asarray(labels) == class_index).astype(int),
            probabilities_a[:, class_index],
            probabilities_b[:, class_index],
        )
        rows.append({"class": class_index, **result})
    out = pd.DataFrame(rows)
    ordered = out.sort_values("p_value")
    rank = np.arange(1, len(out) + 1)
    ordered["holm_p"] = np.minimum(
        1.0, np.maximum.accumulate((ordered["p_value"] * (len(out) - rank + 1)).clip(0, 1))
    )
    out.loc[ordered.index, "holm_p"] = ordered["holm_p"]
    out["bonferroni_p"] = np.minimum(1.0, out["p_value"] * len(out))
    return out

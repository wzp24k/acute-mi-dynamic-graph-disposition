"""Evaluate complementary information among already-saved model predictions."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
PRED = ROOT / "results" / "predictions"
RESULTS = ROOT / "results"

CANDIDATES = (
    "xgb_static_ts",
    "transformer",
    "gru",
    "tft",
    "ct_fixed",
    "ct_graph",
    "ctogt",
)


def load_development():
    labels = None
    predictions = {}
    for model in CANDIDATES:
        path = PRED / f"mimic4_cv_{model}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=True) as data:
            current = data["labels"].astype(int)
            if labels is None:
                labels = current
            if np.array_equal(labels, current):
                predictions[model] = data["probability"].astype(float)
    return labels, predictions


def rank_probability(values: np.ndarray) -> np.ndarray:
    return (rankdata(values) - 0.5) / len(values)


def stack_oof(labels: np.ndarray, predictions: dict[str, np.ndarray], models):
    matrix = np.column_stack([predictions[model] for model in models])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    output = np.full(len(labels), np.nan)
    for train, test in splitter.split(matrix, labels):
        model = LogisticRegression(C=0.5, solver="lbfgs", max_iter=2000)
        model.fit(matrix[train], labels[train])
        output[test] = model.predict_proba(matrix[test])[:, 1]
    return output


def metric(labels: np.ndarray, probability: np.ndarray, name: str) -> dict:
    return {
        "name": name,
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
    }


def external_predictions(cohort: str, models):
    labels = None
    predictions = {}
    for model in models:
        path = PRED / f"{cohort}_external_{model}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=True) as data:
            current = data["labels"].astype(int)
            if labels is None:
                labels = current
            if np.array_equal(labels, current):
                predictions[model] = data[model].astype(float)
    return labels, predictions


def main() -> None:
    labels, predictions = load_development()
    rows = [metric(labels, values, model) for model, values in predictions.items()]
    for size in (2, 3, 4):
        for models in itertools.combinations(predictions, size):
            mean_probability = np.mean([predictions[model] for model in models], axis=0)
            rank = np.mean(
                [rank_probability(predictions[model]) for model in models], axis=0
            )
            rows.append(metric(labels, mean_probability, "mean:" + "+".join(models)))
            rows.append(metric(labels, rank, "rank:" + "+".join(models)))
            if size <= 3:
                rows.append(
                    metric(
                        labels,
                        stack_oof(labels, predictions, models),
                        "stack:" + "+".join(models),
                    )
                )
    rows = sorted(rows, key=lambda item: item["auroc"], reverse=True)
    external = {}
    for cohort in ("mimic3", "eicu"):
        cohort_labels, cohort_predictions = external_predictions(cohort, CANDIDATES)
        cohort_rows = [
            metric(cohort_labels, values, model)
            for model, values in cohort_predictions.items()
        ]
        for size in (2, 3, 4):
            for models in itertools.combinations(cohort_predictions, size):
                mean_probability = np.mean(
                    [cohort_predictions[model] for model in models], axis=0
                )
                rank = np.mean(
                    [rank_probability(cohort_predictions[model]) for model in models],
                    axis=0,
                )
                cohort_rows.append(
                    metric(
                        cohort_labels,
                        mean_probability,
                        "mean:" + "+".join(models),
                    )
                )
                cohort_rows.append(
                    metric(
                        cohort_labels,
                        rank,
                        "rank:" + "+".join(models),
                    )
                )
        external[cohort] = sorted(
            cohort_rows, key=lambda item: item["auroc"], reverse=True
        )
    output = {
        "development_top": rows[:30],
        "external_top": {name: values[:20] for name, values in external.items()},
    }
    (RESULTS / "ensemble_diagnostics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

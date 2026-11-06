"""Audit whether observed-measurement counts drive XGB-static+TS performance."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.evaluation import (  # noqa: E402
    bootstrap_metric_ci,
    paired_bootstrap_delta,
    point_metrics,
)
from src.ami_dynamic.train_evaluate import (  # noqa: E402
    _assert_feature_contract,
    _fit_xgb,
    _json_default,
    _static_matrix,
    load_dataset,
)


SEEDS = (1, 7, 42, 123, 2024)
FOLDS = 5
THREADS = min(16, max(1, os.cpu_count() or 1))
OUTPUT = PROJECT_ROOT / "results" / "audits" / "count_feature_ablation.json"
PREDICTION_OUTPUT = (
    PROJECT_ROOT / "results" / "audits" / "count_feature_ablation_predictions"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def main() -> None:
    data = load_dataset("mimic4")
    cohort = data["cohort"].reset_index(drop=True)
    labels = cohort["outcome"].to_numpy(dtype=int)
    groups = cohort["subject_key"].astype(str).to_numpy()
    metrics_path = PROJECT_ROOT / "results" / "metrics_mimic4.json"
    locked = json.loads(metrics_path.read_text(encoding="utf-8"))
    full_columns = list(locked["full_static_features"])
    _assert_feature_contract(full_columns)
    count_columns = [
        column for column in full_columns if column.endswith("_count")
    ]
    reduced_columns = [
        column for column in full_columns if column not in count_columns
    ]
    expected_count_columns = {
        f"{channel}_count"
        for channel in (
            "hr",
            "rr",
            "spo2",
            "sbp",
            "dbp",
            "mbp",
            "temp",
            "gcs",
        )
    }
    if set(count_columns) != expected_count_columns:
        raise ValueError(
            f"Unexpected count columns: {sorted(count_columns)}"
        )
    matrix = _static_matrix(cohort, reduced_columns)
    n_classes = int(labels.max()) + 1
    baseline_probabilities = {}
    reduced_probabilities = {}
    seed_rows = []
    PREDICTION_OUTPUT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        baseline_path = (
            PROJECT_ROOT
            / "results"
            / "predictions"
            / f"mimic4_xgb_static_ts_seed{seed}.npz"
        )
        with np.load(baseline_path, allow_pickle=True) as cached:
            baseline_probability = cached["probabilities"].astype(float)
            cached_fold = cached["fold"].astype(int)
            cached_labels = cached["labels"].astype(int)
            cached_groups = cached["subject_key"].astype(str)
        if not np.array_equal(cached_labels, labels):
            raise ValueError(f"{baseline_path}: labels differ from cohort")
        if not np.array_equal(cached_groups, groups):
            raise ValueError(f"{baseline_path}: subject keys differ from cohort")
        splitter = StratifiedGroupKFold(
            n_splits=FOLDS, shuffle=True, random_state=seed
        )
        fold_id = np.full(len(labels), -1, dtype=int)
        for fold, (_, test) in enumerate(
            splitter.split(matrix, labels, groups)
        ):
            fold_id[test] = fold
        if not np.array_equal(fold_id, cached_fold):
            raise ValueError(f"seed {seed}: fold assignment differs from lock")

        reduced_probability = np.full(
            (len(labels), n_classes), np.nan, dtype=np.float32
        )
        for fold in range(FOLDS):
            train = np.flatnonzero(fold_id != fold)
            test = np.flatnonzero(fold_id == fold)
            model = _fit_xgb(
                matrix[train],
                labels[train],
                n_classes=n_classes,
                seed=seed + fold,
                n_jobs=THREADS,
            )
            reduced_probability[test] = model.predict_proba(matrix[test])
            print(
                f"[count ablation] seed={seed} fold={fold + 1}/{FOLDS}",
                flush=True,
            )
        if not np.isfinite(reduced_probability).all():
            raise ValueError(f"seed {seed}: non-finite reduced-model predictions")
        baseline_metrics = point_metrics(labels, baseline_probability)
        reduced_metrics = point_metrics(labels, reduced_probability)
        baseline_metrics.pop("cumulative_probabilities", None)
        reduced_metrics.pop("cumulative_probabilities", None)
        delta = reduced_metrics["macro_auc"] - baseline_metrics["macro_auc"]
        seed_rows.append(
            {
                "seed": seed,
                "with_counts_macro_auc": baseline_metrics["macro_auc"],
                "without_counts_macro_auc": reduced_metrics["macro_auc"],
                "without_minus_with": delta,
            }
        )
        baseline_probabilities[seed] = baseline_probability
        reduced_probabilities[seed] = reduced_probability
        np.savez_compressed(
            PREDICTION_OUTPUT / f"xgb_static_ts_no_counts_seed{seed}.npz",
            labels=labels,
            probabilities=reduced_probability,
            fold=fold_id,
            stay_id=cohort["stay_id"].to_numpy(dtype=np.int64),
            subject_key=groups,
        )
        print(
            f"[count ablation] seed={seed} "
            f"with={baseline_metrics['macro_auc']:.6f} "
            f"without={reduced_metrics['macro_auc']:.6f} "
            f"delta={delta:+.6f}",
            flush=True,
        )

    baseline_average = np.mean(
        np.stack([baseline_probabilities[seed] for seed in SEEDS]), axis=0
    )
    reduced_average = np.mean(
        np.stack([reduced_probabilities[seed] for seed in SEEDS]), axis=0
    )
    baseline_metrics = point_metrics(labels, baseline_average)
    reduced_metrics = point_metrics(labels, reduced_average)
    baseline_metrics.pop("cumulative_probabilities", None)
    reduced_metrics.pop("cumulative_probabilities", None)
    baseline_metrics["macro_auc_ci"] = bootstrap_metric_ci(
        labels,
        baseline_average,
        groups=groups,
        replicates=2000,
    )
    reduced_metrics["macro_auc_ci"] = bootstrap_metric_ci(
        labels,
        reduced_average,
        groups=groups,
        replicates=2000,
    )
    paired_delta = paired_bootstrap_delta(
        labels,
        reduced_average,
        baseline_average,
        groups=groups,
        replicates=2000,
    )
    output = {
        "analysis": "same folds, same rows, XGB-static+TS with versus without measurement counts",
        "n": int(len(labels)),
        "seeds": list(SEEDS),
        "folds": FOLDS,
        "count_columns_removed": count_columns,
        "with_counts_feature_count": len(full_columns),
        "without_counts_feature_count": len(reduced_columns),
        "with_counts": baseline_metrics,
        "without_counts": reduced_metrics,
        "delta_definition": "without_counts minus with_counts",
        "seed_averaged_delta": paired_delta,
        "seed_specific": seed_rows,
    }
    _write_json(OUTPUT, output)
    print(json.dumps(output, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()

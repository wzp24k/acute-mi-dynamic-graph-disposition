"""Compare eICU patient-grouped CV with patient-safe hospital-cluster CV."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
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
from work.run_external_calibration_common3 import (  # noqa: E402
    patient_safe_hospital_clusters,
)


SEEDS = (1, 7, 42, 123, 2024)
FOLDS = 5
THREADS = min(16, max(1, os.cpu_count() or 1))
OUTPUT = (
    PROJECT_ROOT / "results" / "audits" / "eicu_hospital_cluster_cv_audit.json"
)
PREDICTION_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "audits"
    / "eicu_hospital_cluster_cv_predictions"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _without_cumulative(metrics: dict) -> dict:
    output = dict(metrics)
    output.pop("cumulative_probabilities", None)
    return output


def main() -> None:
    data = load_dataset("eicu")
    cohort = data["cohort"].reset_index(drop=True)
    labels = cohort["outcome"].to_numpy(dtype=int)
    patients = cohort["subject_key"].astype(str).to_numpy()
    hospitals = cohort["hospital_key"].astype(str).to_numpy()
    clusters, clustering = patient_safe_hospital_clusters(patients, hospitals)
    metrics_path = PROJECT_ROOT / "results" / "metrics_eicu.json"
    locked = json.loads(metrics_path.read_text(encoding="utf-8"))
    full_columns = list(locked["full_static_features"])
    _assert_feature_contract(full_columns)
    matrix = _static_matrix(cohort, full_columns)
    n_classes = int(labels.max()) + 1
    PREDICTION_OUTPUT.mkdir(parents=True, exist_ok=True)

    patient_grouped_probabilities = {}
    hospital_cluster_probabilities = {}
    seed_rows = []
    for seed in SEEDS:
        patient_path = (
            PROJECT_ROOT
            / "results"
            / "predictions"
            / f"eicu_xgb_static_ts_seed{seed}.npz"
        )
        with np.load(patient_path, allow_pickle=True) as cached:
            patient_probability = cached["probabilities"].astype(float)
            cached_labels = cached["labels"].astype(int)
            cached_patients = cached["subject_key"].astype(str)
        if not np.array_equal(cached_labels, labels):
            raise ValueError(f"{patient_path}: labels differ from cohort")
        if not np.array_equal(cached_patients, patients):
            raise ValueError(f"{patient_path}: patients differ from cohort")

        splitter = StratifiedGroupKFold(
            n_splits=FOLDS, shuffle=True, random_state=seed
        )
        fold_id = np.full(len(labels), -1, dtype=int)
        for fold, (_, test) in enumerate(
            splitter.split(matrix, labels, clusters)
        ):
            fold_id[test] = fold
        if (fold_id < 0).any():
            raise ValueError(f"seed {seed}: incomplete hospital-cluster folds")
        cluster_probability = np.full(
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
            cluster_probability[test] = model.predict_proba(matrix[test])
            print(
                f"[hospital CV] seed={seed} fold={fold + 1}/{FOLDS}",
                flush=True,
            )
        if not np.isfinite(cluster_probability).all():
            raise ValueError(f"seed {seed}: non-finite predictions")

        patient_metrics = _without_cumulative(
            point_metrics(labels, patient_probability)
        )
        cluster_metrics = _without_cumulative(
            point_metrics(labels, cluster_probability)
        )
        delta = (
            cluster_metrics["macro_auc"] - patient_metrics["macro_auc"]
        )
        seed_rows.append(
            {
                "seed": seed,
                "patient_grouped_macro_auc": patient_metrics["macro_auc"],
                "hospital_cluster_macro_auc": cluster_metrics["macro_auc"],
                "hospital_cluster_minus_patient_grouped": delta,
            }
        )
        patient_grouped_probabilities[seed] = patient_probability
        hospital_cluster_probabilities[seed] = cluster_probability
        np.savez_compressed(
            PREDICTION_OUTPUT
            / f"eicu_xgb_static_ts_hospital_cluster_seed{seed}.npz",
            labels=labels,
            probabilities=cluster_probability,
            fold=fold_id,
            clusters=clusters,
            hospital_key=hospitals,
            subject_key=patients,
            stay_id=cohort["stay_id"].to_numpy(dtype=np.int64),
        )
        print(
            f"[hospital CV] seed={seed} "
            f"patient={patient_metrics['macro_auc']:.6f} "
            f"hospital={cluster_metrics['macro_auc']:.6f} "
            f"delta={delta:+.6f}",
            flush=True,
        )

    patient_average = np.mean(
        np.stack([patient_grouped_probabilities[seed] for seed in SEEDS]),
        axis=0,
    )
    cluster_average = np.mean(
        np.stack([hospital_cluster_probabilities[seed] for seed in SEEDS]),
        axis=0,
    )
    patient_metrics = _without_cumulative(
        point_metrics(labels, patient_average)
    )
    cluster_metrics = _without_cumulative(
        point_metrics(labels, cluster_average)
    )
    patient_metrics["macro_auc_ci"] = bootstrap_metric_ci(
        labels,
        patient_average,
        groups=clusters,
        replicates=2000,
    )
    cluster_metrics["macro_auc_ci"] = bootstrap_metric_ci(
        labels,
        cluster_average,
        groups=clusters,
        replicates=2000,
    )
    paired_delta = paired_bootstrap_delta(
        labels,
        cluster_average,
        patient_average,
        groups=clusters,
        replicates=2000,
    )
    output = {
        "analysis": (
            "eICU internal CV with patient grouping versus patient-safe "
            "hospital-cluster grouping"
        ),
        "n": int(len(labels)),
        "n_patients": int(len(np.unique(patients))),
        "n_hospitals": int(len(np.unique(hospitals))),
        "n_hospital_clusters": int(len(np.unique(clusters))),
        "clustering": clustering,
        "seeds": list(SEEDS),
        "folds": FOLDS,
        "feature_count": len(full_columns),
        "patient_grouped": patient_metrics,
        "hospital_cluster_grouped": cluster_metrics,
        "delta_definition": (
            "hospital_cluster_grouped minus patient_grouped"
        ),
        "seed_averaged_delta": paired_delta,
        "seed_specific": seed_rows,
    }
    _write_json(OUTPUT, output)
    print(json.dumps(output, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()

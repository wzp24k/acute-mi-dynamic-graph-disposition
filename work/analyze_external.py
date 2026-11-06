"""Hospital-level transport analysis for the eICU AMI validation cohort."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import PROCESSED, RESULTS  # noqa: E402
from src.ami_dynamic.evaluation import (  # noqa: E402
    calibration_slope_intercept,
    point_metrics,
    safe_auc,
)
from work.run_external_calibration_common3 import (  # noqa: E402
    patient_safe_hospital_clusters,
)


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1.0 - probability)).reshape(-1, 1)


def _pooled_metric(
    labels: np.ndarray,
    probability: np.ndarray,
    rows: np.ndarray,
    metric: str,
) -> float:
    metrics = point_metrics(labels[rows], probability[rows])
    if metric == "macro_auc":
        return float(metrics["macro_auc"])
    if metric == "highest_class_auc":
        highest_class = probability.shape[1] - 1
        return safe_auc(
            (labels[rows] == highest_class).astype(int),
            probability[rows, highest_class],
        )
    raise ValueError(metric)


def patient_safe_cluster_bootstrap(
    labels: np.ndarray,
    probability: np.ndarray,
    hospital_clusters: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict:
    unique, inverse = np.unique(hospital_clusters, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    values = {"macro_auc": [], "highest_class_auc": []}
    for _ in range(replicates):
        selected = rng.choice(len(groups), size=len(groups), replace=True)
        rows = np.concatenate([groups[index] for index in selected])
        for metric in values:
            value = _pooled_metric(labels, probability, rows, metric)
            if np.isfinite(value):
                values[metric].append(value)
    return {
        metric: {
            "ci_low": float(np.percentile(samples, 2.5)),
            "ci_high": float(np.percentile(samples, 97.5)),
            "valid_replicates": int(len(samples)),
        }
        for metric, samples in values.items()
        if samples
    }


def leave_one_cluster_out(
    labels: np.ndarray,
    probability: np.ndarray,
    hospital_clusters: np.ndarray,
) -> dict:
    full = _pooled_metric(
        labels,
        probability,
        np.arange(len(labels)),
        "macro_auc",
    )
    estimates = []
    for cluster in np.unique(hospital_clusters):
        rows = np.flatnonzero(hospital_clusters != cluster)
        value = _pooled_metric(labels, probability, rows, "macro_auc")
        estimates.append(
            {
                "omitted_hospital_cluster": str(cluster),
                "macro_auc_without_site": float(value),
                "delta_from_full": float(value - full),
            }
        )
    frame = pd.DataFrame(estimates)
    return {
        "full_macro_auc": float(full),
        "minimum": float(frame["macro_auc_without_site"].min()),
        "maximum": float(frame["macro_auc_without_site"].max()),
        "largest_absolute_delta": float(
            frame["delta_from_full"].abs().max()
        ),
        "sites": frame.to_dict(orient="records"),
    }


def cross_fitted_cluster_recalibration(
    labels: np.ndarray,
    probability: np.ndarray,
    hospital_clusters: np.ndarray,
) -> dict:
    target = (labels >= 2).astype(int)
    score = probability[:, 2:].sum(axis=1)
    recalibrated = np.full(len(labels), np.nan, dtype=float)
    for cluster in np.unique(hospital_clusters):
        test = hospital_clusters == cluster
        train = ~test
        if len(np.unique(target[train])) < 2:
            continue
        model = LogisticRegression(C=1e6, solver="lbfgs")
        model.fit(_logit(score[train]), target[train])
        recalibrated[test] = model.predict_proba(_logit(score[test]))[:, 1]
    valid = np.isfinite(recalibrated)
    return {
        "n_cross_fitted": int(valid.sum()),
        "design": "patient-safe leave-one-hospital-cluster-out",
        "before": calibration_slope_intercept(target[valid], score[valid]),
        "after_cross_fitted": calibration_slope_intercept(
            target[valid], recalibrated[valid]
        ),
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    cohort = pd.read_csv(PROCESSED / "eicu" / "cohort.csv")
    sites = cohort["hospital_key"].astype(str).to_numpy()
    subjects = cohort["subject_key"].astype(str).to_numpy()
    hospital_clusters, cluster_metadata = patient_safe_hospital_clusters(
        subjects,
        sites,
    )
    site_counts = pd.Series(sites).value_counts()
    sites_at_least_30 = set(site_counts[site_counts >= 30].index)
    model_files = sorted((RESULTS / "predictions").glob("eicu_*_transport.npz"))
    if not model_files:
        raise FileNotFoundError("Run transport validation first")

    table_rows = []
    models = {}
    for path in model_files:
        model_name = path.stem.removeprefix("eicu_").removesuffix("_transport")
        with np.load(path, allow_pickle=True) as data:
            labels = data["labels"].astype(int)
            probability = data["probabilities"].astype(float)
            stay_id = data["stay_id"].astype(np.int64)
        if not np.array_equal(
            stay_id, cohort["stay_id"].to_numpy(dtype=np.int64)
        ):
            raise ValueError(f"{path.name}: cohort order does not match")
        models[model_name] = (labels, probability)
        for site in sorted(sites_at_least_30):
            rows = sites == site
            table_rows.append(
                {
                    "model": model_name,
                    "hospital_key": site,
                    "n": int(rows.sum()),
                    "n_high_need": int((labels[rows] >= 2).sum()),
                    "macro_auc": point_metrics(
                        labels[rows], probability[rows]
                    )["macro_auc"],
                    "highest_class_auc": safe_auc(
                        (labels[rows] == 3).astype(int),
                        probability[rows, 3],
                    ),
                }
            )
    site_table = pd.DataFrame(table_rows)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    site_table.to_csv(
        RESULTS / "tables" / "eicu_site_metrics.csv", index=False
    )

    summary = {
        "n_hospitals": int(len(site_counts)),
        "n_hospitals_at_least_30": int(len(sites_at_least_30)),
        "patient_safe_hospital_clusters": cluster_metadata,
        "n_admissions_at_least_30": int(
            site_counts[site_counts.index.isin(sites_at_least_30)].sum()
        ),
        "models": {},
    }
    for model_name, (labels, probability) in models.items():
        summary["models"][model_name] = {
            "patient_safe_leave_one_cluster_out": leave_one_cluster_out(
                labels, probability, hospital_clusters
            ),
            "patient_safe_cluster_bootstrap": patient_safe_cluster_bootstrap(
                labels,
                probability,
                hospital_clusters,
                replicates=2000,
                seed=20261104,
            ),
            "patient_safe_cluster_recalibration_high_need": (
                cross_fitted_cluster_recalibration(
                    labels,
                    probability,
                    hospital_clusters,
                )
            ),
        }
    path = RESULTS / "audits" / "eicu_site_analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""External calibration and common three-class validation for the AMI models.

This script is intentionally self-contained.  It reads the locked processed
cohorts and existing prediction caches, performs grouped recalibration and
bootstrap analyses, and writes only to results/calibration/.

Definitions
-----------
Calibration targets
    high_need = P(y >= 2)
    death = highest-class outcome P(y = 3); for MIMIC-III this is
    hospice-or-in-hospital-death rather than strict mortality

Affine recalibration
    logit(q) = intercept + slope * logit(p)

MIMIC-III
    Patient-grouped five-fold out-of-fold affine recalibration.

eICU direct transport
    Patient-safe leave-one-hospital-cluster-out affine recalibration. Hospitals
    that share a patient are merged before any fold is constructed.

Common three-class outcome
    MIMIC-IV five-class labels: [0] -> 0, [1, 2, 3] -> 1, [4] -> 2
    MIMIC-III/eICU four-class labels: [0] -> 0, [1, 2] -> 1, [3] -> 2

All estimates retain the original model's positive-score direction.  No
absolute-value transformation or sign correction is applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
PREDICTIONS = PROJECT_ROOT / "results" / "predictions"
OUTPUT_DIR = PROJECT_ROOT / "results" / "calibration"

SEEDS = (1, 7, 42, 123, 2024)
BOOTSTRAP_SEED = 20261105
DEFAULT_REPLICATES = 2000
LOGIT_EPS = 1e-6
PROB_EPS = 1e-12

MODELS = {
    "xgb_static": "XGB-static",
    "xgb_static_ts": "XGB-static+TS",
    "dggru_tr": "DG-GRU+TR",
}

TARGETS = {
    "high_need": {
        "label": "High-need disposition, P(y>=2)",
        "short_label": "P(y>=2)",
        "columns": (2, 3),
    },
    "death": {
        "label": "Highest-class outcome, P(y=3)",
        "short_label": "P(y=3)",
        "columns": (3,),
    },
}

CALIBRATION_CSV = OUTPUT_DIR / "table7_external_calibration.csv"
CALIBRATION_MD = OUTPUT_DIR / "table7_external_calibration.md"
COMMON3_CSV = OUTPUT_DIR / "table8_common_three_class.csv"
COMMON3_MD = OUTPUT_DIR / "table8_common_three_class.md"
REPORT_JSON = OUTPUT_DIR / "calibration_and_common3_report.json"
CALIBRATION_PREDICTIONS_CSV = OUTPUT_DIR / "calibration_oof_predictions.csv"
COMMON3_PREDICTIONS_CSV = OUTPUT_DIR / "common3_predictions.csv"
EXTERNAL_CALIBRATION_CURVES_CSV = OUTPUT_DIR / "external_calibration_curves.csv"


@dataclass(frozen=True)
class TransportData:
    dataset: str
    cohort: pd.DataFrame
    ids: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    groups: np.ndarray
    group_label: str
    validation_design: str
    n_folds: int


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(
        np.asarray(probability, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS
    )
    return np.log(probability / (1.0 - probability))


def _safe_probability(probability: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(probability, dtype=float), PROB_EPS, 1.0 - PROB_EPS
    )


def fit_logistic_affine(
    target: np.ndarray,
    probability: np.ndarray,
) -> dict:
    """Fit logit(y=1) = intercept + slope * logit(p) by Newton-Raphson.

    The model is unpenalized to mirror a logistic calibration map with a very
    weak numerical regularizer.  Centering the score improves conditioning.
    """
    y = np.asarray(target, dtype=float)
    if len(y) == 0 or np.unique(y).size < 2:
        return {
            "intercept": float("nan"),
            "slope": float("nan"),
            "converged": False,
            "n_iter": 0,
        }

    x = _safe_logit(probability)
    x_center = float(np.mean(x))
    xc = x - x_center
    prevalence = float(np.clip(np.mean(y), LOGIT_EPS, 1.0 - LOGIT_EPS))
    offset = math.log(prevalence / (1.0 - prevalence))
    slope = 0.0
    converged = False
    n_iter = 0

    def negative_log_likelihood(a: float, b: float) -> float:
        eta = np.clip(a + b * xc, -35.0, 35.0)
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    current_loss = negative_log_likelihood(offset, slope)
    for n_iter in range(1, 101):
        eta = np.clip(offset + slope * xc, -35.0, 35.0)
        fitted = expit(eta)
        weight = np.maximum(fitted * (1.0 - fitted), 1e-12)
        score = np.array(
            [
                np.sum(y - fitted),
                np.sum((y - fitted) * xc),
            ],
            dtype=float,
        )
        information = np.array(
            [
                [np.sum(weight), np.sum(weight * xc)],
                [np.sum(weight * xc), np.sum(weight * xc * xc)],
            ],
            dtype=float,
        )
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(step).all():
            break

        step_scale = 1.0
        accepted = False
        while step_scale >= 1.0 / 1024.0:
            candidate_offset = offset + step_scale * step[0]
            candidate_slope = slope + step_scale * step[1]
            candidate_loss = negative_log_likelihood(
                candidate_offset, candidate_slope
            )
            if candidate_loss <= current_loss + 1e-10:
                offset = candidate_offset
                slope = candidate_slope
                current_loss = candidate_loss
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
        if float(np.max(np.abs(step_scale * step))) < 1e-10:
            converged = True
            break

    if not np.isfinite(offset) or not np.isfinite(slope):
        return {
            "intercept": float("nan"),
            "slope": float("nan"),
            "converged": False,
            "n_iter": n_iter,
        }
    intercept = float(offset - slope * x_center)
    return {
        "intercept": intercept,
        "slope": float(slope),
        "converged": bool(converged),
        "n_iter": int(n_iter),
    }


def apply_logistic_affine(
    probability: np.ndarray,
    intercept: float,
    slope: float,
) -> np.ndarray:
    return expit(intercept + slope * _safe_logit(probability))


def _target_probability(
    probabilities: np.ndarray,
    target_key: str,
) -> np.ndarray:
    columns = TARGETS[target_key]["columns"]
    return probabilities[:, list(columns)].sum(axis=1)


def binary_calibration_metrics(
    target: np.ndarray,
    probability: np.ndarray,
) -> dict:
    y = np.asarray(target, dtype=int)
    p = _safe_probability(probability)
    affine = fit_logistic_affine(y, p)
    brier = float(np.mean((p - y) ** 2))
    log_loss_value = float(
        -np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p))
    )
    return {
        "slope": affine["slope"],
        "intercept": affine["intercept"],
        "brier": brier,
        "log_loss": log_loss_value,
        "n": int(len(y)),
        "events": int(y.sum()),
        "prevalence": float(y.mean()),
        "affine_converged": affine["converged"],
        "affine_iterations": affine["n_iter"],
    }


def _group_rows(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    rows = [
        np.flatnonzero(inverse == group_index)
        for group_index in range(len(unique))
    ]
    return unique, rows


def _sample_group_rows(
    rng: np.random.Generator,
    rows: list[np.ndarray],
) -> np.ndarray:
    selected = rng.integers(0, len(rows), size=len(rows))
    return np.concatenate([rows[index] for index in selected])


def grouped_calibration_bootstrap(
    target: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict:
    """Cluster bootstrap intervals for calibration metrics.

    The fixed probability vector is resampled by patient/site.  For the
    cross-fitted probabilities, the out-of-fold recalibration maps are held
    fixed; this captures sampling uncertainty conditional on the fitted maps.
    """
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    unique, rows = _group_rows(groups)
    rng = np.random.default_rng(seed)
    samples = {"slope": [], "intercept": [], "brier": [], "log_loss": []}
    for _ in range(replicates):
        chosen = _sample_group_rows(rng, rows)
        metrics = binary_calibration_metrics(y[chosen], p[chosen])
        for key in samples:
            value = metrics[key]
            if np.isfinite(value):
                samples[key].append(value)
    point = binary_calibration_metrics(y, p)
    output = {
        "estimate": point,
        "n_groups": int(len(unique)),
        "bootstrap_replicates_requested": int(replicates),
    }
    for key, values in samples.items():
        if values:
            output[key] = {
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "valid_replicates": int(len(values)),
            }
        else:
            output[key] = {
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "valid_replicates": 0,
            }
    return output


def _wilson_interval(
    successes: int,
    n: int,
    *,
    z_value: float = 1.959963984540054,
) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    proportion = successes / n
    denominator = 1.0 + z_value**2 / n
    center = (proportion + z_value**2 / (2.0 * n)) / denominator
    half_width = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / n
            + z_value**2 / (4.0 * n**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def reliability_deciles(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Equal-frequency reliability bins with Wilson observed-rate intervals."""
    y = np.asarray(target, dtype=int)
    p = _safe_probability(probability)
    order = np.argsort(p, kind="mergesort")
    grouped_rows = np.array_split(order, n_bins)
    rows = []
    for bin_id, bin_rows in enumerate(grouped_rows, start=1):
        if len(bin_rows) == 0:
            raise ValueError("Reliability bin is empty")
        successes = int(y[bin_rows].sum())
        n = int(len(bin_rows))
        observed = successes / n
        ci_low, ci_high = _wilson_interval(successes, n)
        rows.append(
            {
                "bin_id": int(bin_id),
                "n": n,
                "predicted_mean": float(p[bin_rows].mean()),
                "observed_rate": float(observed),
                "observed_ci_low": float(ci_low),
                "observed_ci_high": float(ci_high),
                "predicted_min": float(p[bin_rows].min()),
                "predicted_max": float(p[bin_rows].max()),
            }
        )
    return pd.DataFrame(rows)


def build_external_calibration_curves(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create raw and grouped-CV recalibrated reliability curves."""
    score_columns = {
        "raw": "raw_probability",
        "grouped_cv_recalibrated": "oof_recal_probability",
    }
    rows: list[dict] = []
    grouped = predictions.groupby(
        ["dataset", "model", "target"], sort=False, observed=True
    )
    for (dataset, model, target_key), group in grouped:
        target = group["y"].to_numpy(dtype=int)
        for score_type, score_column in score_columns.items():
            curve = reliability_deciles(
                target, group[score_column].to_numpy(dtype=float)
            )
            for row in curve.to_dict(orient="records"):
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "target": target_key,
                        "score_type": score_type,
                        **row,
                    }
                )
    frame = pd.DataFrame(rows)
    expected = predictions[
        ["dataset", "model", "target"]
    ].drop_duplicates().shape[0] * len(score_columns) * 10
    if len(frame) != expected:
        raise ValueError(
            f"External calibration curve row check failed: {len(frame)} != {expected}"
        )
    return frame


def _fit_cross_validated_recalibration(
    target: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    split_iterator: Iterable[tuple[np.ndarray, np.ndarray, str]],
    *,
    design: str,
) -> tuple[np.ndarray, list[dict]]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    recalibrated = np.full(len(y), np.nan, dtype=float)
    fold_maps: list[dict] = []
    for train_rows, test_rows, fold_label in split_iterator:
        affine = fit_logistic_affine(y[train_rows], p[train_rows])
        if not np.isfinite(affine["intercept"]) or not np.isfinite(
            affine["slope"]
        ):
            raise ValueError(
                f"Cannot fit affine recalibration for {design} fold "
                f"{fold_label}: training set has one outcome class"
            )
        recalibrated[test_rows] = apply_logistic_affine(
            p[test_rows], affine["intercept"], affine["slope"]
        )
        fold_maps.append(
            {
                "fold": str(fold_label),
                "train_n": int(len(train_rows)),
                "test_n": int(len(test_rows)),
                "train_events": int(y[train_rows].sum()),
                "test_events": int(y[test_rows].sum()),
                "intercept": float(affine["intercept"]),
                "slope": float(affine["slope"]),
                "converged": bool(affine["converged"]),
                "iterations": int(affine["n_iter"]),
            }
        )
    if not np.isfinite(recalibrated).all():
        raise ValueError(f"{design} did not produce complete out-of-fold predictions")
    return recalibrated, fold_maps


def mimic3_patient_grouped_recalibration(
    target: np.ndarray,
    probability: np.ndarray,
    subjects: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    splitter = GroupKFold(n_splits=5)
    splits = [
        (train_rows, test_rows, f"patient_group_fold_{index + 1}")
        for index, (train_rows, test_rows) in enumerate(
            splitter.split(probability, target, groups=subjects)
        )
    ]
    return _fit_cross_validated_recalibration(
        target,
        probability,
        subjects,
        splits,
        design="patient-grouped five-fold",
    )


def patient_safe_hospital_clusters(
    patients: np.ndarray,
    hospitals: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Merge hospitals that share a patient into disjoint leave-out clusters.

    A patient observed in more than one eICU hospital connects those hospitals.
    Treating each hospital as an independent fold would allow the same patient
    to contribute to both training and testing.  Connected components of the
    patient-hospital graph are the smallest patient-safe hospital clusters.
    """
    patient_values = np.asarray(patients).astype(str)
    hospital_values = np.asarray(hospitals).astype(str)
    if len(patient_values) != len(hospital_values):
        raise ValueError("patients and hospitals must have equal length")
    if len(patient_values) == 0:
        raise ValueError("patient and hospital arrays are empty")

    unique_hospitals, inverse = np.unique(
        hospital_values, return_inverse=True
    )
    parent = np.arange(len(unique_hospitals), dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    patient_hospitals: dict[str, set[int]] = {}
    for patient, hospital_index in zip(patient_values, inverse):
        patient_hospitals.setdefault(patient, set()).add(int(hospital_index))
    cross_hospital_patients = 0
    for hospital_indices in patient_hospitals.values():
        if len(hospital_indices) > 1:
            cross_hospital_patients += 1
            first = min(hospital_indices)
            for hospital_index in hospital_indices:
                union(first, hospital_index)

    roots = np.array(
        [find(index) for index in range(len(unique_hospitals))],
        dtype=int,
    )
    unique_roots = np.unique(roots)
    cluster_number = {
        int(root): index + 1 for index, root in enumerate(unique_roots)
    }
    hospital_to_cluster = np.array(
        [cluster_number[int(root)] for root in roots],
        dtype=int,
    )
    cluster_labels = np.array(
        [
            f"hospital_cluster_{hospital_to_cluster[int(hospital_index)]:03d}"
            for hospital_index in inverse
        ],
        dtype=object,
    )
    cluster_sizes = pd.Series(cluster_labels).value_counts()
    return cluster_labels, {
        "n_hospitals": int(len(unique_hospitals)),
        "n_patients": int(len(patient_hospitals)),
        "n_cross_hospital_patients": int(cross_hospital_patients),
        "n_hospital_clusters": int(len(unique_roots)),
        "n_clusters_with_multiple_hospitals": int(
            sum(size > 1 for size in pd.Series(hospital_to_cluster).value_counts())
        ),
        "n_cluster_sizes": int(len(cluster_sizes)),
        "cluster_size_min": int(cluster_sizes.min()),
        "cluster_size_median": float(cluster_sizes.median()),
        "cluster_size_max": int(cluster_sizes.max()),
        "split_rule": (
            "connected components of the patient-hospital graph; every "
            "held-out hospital cluster is disjoint from training patients"
        ),
    }


def patient_safe_leave_one_hospital_cluster_out_recalibration(
    target: np.ndarray,
    probability: np.ndarray,
    hospital_clusters: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    unique_clusters = np.unique(hospital_clusters)
    splits = [
        (
            np.flatnonzero(hospital_clusters != cluster),
            np.flatnonzero(hospital_clusters == cluster),
            f"omit_{cluster}",
        )
        for cluster in unique_clusters
    ]
    return _fit_cross_validated_recalibration(
        target,
        probability,
        hospital_clusters,
        splits,
        design="patient-safe leave-one-hospital-cluster-out",
    )


def _load_transport_dataset(
    dataset: str,
    *,
    group_column: str,
    group_label: str,
    validation_design: str,
    n_folds: int,
    cohort_label_map: dict[int, int],
    patient_safe_hospital_clusters_enabled: bool = False,
) -> tuple[TransportData, dict]:
    cohort = pd.read_csv(PROCESSED / dataset / "cohort.csv")
    required = {"stay_id", "outcome", group_column}
    if patient_safe_hospital_clusters_enabled:
        required.add("subject_key")
    missing = required.difference(cohort.columns)
    if missing:
        raise ValueError(f"{dataset}: missing cohort columns {sorted(missing)}")
    if cohort["stay_id"].duplicated().any():
        raise ValueError(f"{dataset}: stay_id is not unique in cohort.csv")
    cohort = cohort.set_index("stay_id", drop=False)
    all_models: list[np.ndarray] = []
    ids_reference: np.ndarray | None = None
    labels_reference: np.ndarray | None = None
    alignment_records = []

    for model_key in MODELS:
        path = PREDICTIONS / f"{dataset}_{model_key}_transport.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as data:
            if not {"stay_id", "labels", "probabilities"}.issubset(data.files):
                raise ValueError(f"{path.name}: required arrays are missing")
            ids = data["stay_id"].astype(np.int64)
            labels = data["labels"].astype(int)
            probabilities = data["probabilities"].astype(float)
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"{path.name}: duplicate stay_id values")
        if not np.isin(ids, cohort.index.to_numpy()).all():
            raise ValueError(f"{path.name}: stay_id values are absent from cohort")
        aligned = cohort.loc[ids]
        cohort_outcome = aligned["outcome"].to_numpy(dtype=int)
        unmapped = sorted(set(cohort_outcome).difference(cohort_label_map))
        if unmapped:
            raise ValueError(
                f"{dataset}: cohort outcome levels lack a transport mapping: {unmapped}"
            )
        expected_labels = np.array(
            [cohort_label_map[int(value)] for value in cohort_outcome],
            dtype=int,
        )
        if probabilities.shape != (len(ids), 4):
            raise ValueError(f"{path.name}: expected four-class probabilities")
        if not np.array_equal(labels, expected_labels):
            raise ValueError(f"{path.name}: labels do not match aligned cohort")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-5):
            raise ValueError(f"{path.name}: probabilities do not sum to one")
        if ids_reference is None:
            ids_reference = ids
            labels_reference = labels
        elif not np.array_equal(ids, ids_reference):
            raise ValueError(f"{path.name}: model caches use different cohort order")
        if not np.array_equal(labels, labels_reference):
            raise ValueError(f"{path.name}: labels differ across model caches")
        all_models.append(probabilities)
        alignment_records.append(
            {
                "model": MODELS[model_key],
                "source": _relative(path),
                "n": int(len(ids)),
                "stay_id_order_matches_reference": True,
                "labels_match_cohort": True,
            }
        )

    assert ids_reference is not None and labels_reference is not None
    aligned = cohort.loc[ids_reference]
    if pd.isna(aligned[group_column]).any():
        raise ValueError(f"{dataset}: missing {group_column} values")
    grouping_metadata = None
    if patient_safe_hospital_clusters_enabled:
        groups, grouping_metadata = patient_safe_hospital_clusters(
            aligned["subject_key"].to_numpy(dtype=str),
            aligned[group_column].to_numpy(dtype=str),
        )
        n_folds = int(grouping_metadata["n_hospital_clusters"])
    else:
        groups = aligned[group_column].astype(str).to_numpy()
    return (
        TransportData(
            dataset=dataset,
            cohort=aligned.reset_index(drop=True),
            ids=ids_reference,
            labels=labels_reference,
            probabilities=np.stack(all_models),
            groups=groups,
            group_label=group_label,
            validation_design=validation_design,
            n_folds=n_folds,
        ),
        {
            "n": int(len(ids_reference)),
            "n_unique_stay_id": int(len(np.unique(ids_reference))),
            "n_groups": int(len(np.unique(groups))),
            "group_column": group_column,
            "patient_safe_hospital_clustering": grouping_metadata,
            "all_rows_aligned": True,
            "models": alignment_records,
        },
    )


def _load_mimic4_seed_average() -> tuple[
    pd.DataFrame,
    np.ndarray,
    dict[str, np.ndarray],
    list[dict],
]:
    cohort = pd.read_csv(PROCESSED / "mimic4" / "cohort.csv")
    required = {"stay_id", "subject_key", "outcome"}
    missing = required.difference(cohort.columns)
    if missing:
        raise ValueError(f"MIMIC-IV: missing cohort columns {sorted(missing)}")
    if cohort["stay_id"].duplicated().any():
        raise ValueError("MIMIC-IV: stay_id is not unique")
    cohort = cohort.set_index("stay_id", drop=False)
    ids_reference: np.ndarray | None = None
    labels_reference: np.ndarray | None = None
    probability_by_model: dict[str, np.ndarray] = {}
    alignment_records: list[dict] = []

    for model_key in MODELS:
        seed_probabilities = []
        for seed in SEEDS:
            path = PREDICTIONS / f"mimic4_{model_key}_seed{seed}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=True) as data:
                if not {"stay_id", "labels", "probabilities"}.issubset(data.files):
                    raise ValueError(f"{path.name}: required arrays are missing")
                ids = data["stay_id"].astype(np.int64)
                labels = data["labels"].astype(int)
                probabilities = data["probabilities"].astype(float)
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"{path.name}: duplicate stay_id values")
            aligned = cohort.loc[ids]
            expected_labels = aligned["outcome"].to_numpy(dtype=int)
            if probabilities.shape != (len(ids), 5):
                raise ValueError(f"{path.name}: expected five-class probabilities")
            if not np.array_equal(labels, expected_labels):
                raise ValueError(f"{path.name}: labels do not match aligned cohort")
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-5):
                raise ValueError(f"{path.name}: probabilities do not sum to one")
            seed_probabilities.append(probabilities)
            if ids_reference is None:
                ids_reference = ids
                labels_reference = labels
            if not np.array_equal(ids, ids_reference):
                raise ValueError(f"{path.name}: seed caches use different cohort order")
            if not np.array_equal(labels, labels_reference):
                raise ValueError(f"{path.name}: seed labels differ")
        averaged = np.mean(np.stack(seed_probabilities), axis=0)
        averaged /= averaged.sum(axis=1, keepdims=True)
        probability_by_model[model_key] = averaged
        alignment_records.append(
            {
                "model": MODELS[model_key],
                "n": int(len(ids_reference)),
                "seeds": list(SEEDS),
                "all_seed_orders_aligned": True,
                "labels_match_cohort": True,
            }
        )

    assert ids_reference is not None and labels_reference is not None
    aligned = cohort.loc[ids_reference].reset_index(drop=True)
    return aligned, labels_reference, probability_by_model, alignment_records


def _map_mimic4_common3(labels: np.ndarray) -> np.ndarray:
    values = np.full(len(labels), -1, dtype=int)
    values[labels == 0] = 0
    values[np.isin(labels, [1, 2, 3])] = 1
    values[labels == 4] = 2
    if (values < 0).any():
        raise ValueError("MIMIC-IV labels must be integers from 0 through 4")
    return values


def _map_four_to_common3(labels: np.ndarray) -> np.ndarray:
    values = np.full(len(labels), -1, dtype=int)
    values[labels == 0] = 0
    values[np.isin(labels, [1, 2])] = 1
    values[labels == 3] = 2
    if (values < 0).any():
        raise ValueError("Four-class labels must be integers from 0 through 3")
    return values


def _common3_probabilities(
    probabilities: np.ndarray,
    *,
    source_classes: int,
) -> np.ndarray:
    if source_classes == 4:
        output = np.column_stack(
            [
                probabilities[:, 0],
                probabilities[:, 1] + probabilities[:, 2],
                probabilities[:, 3],
            ]
        )
    elif source_classes == 5:
        output = np.column_stack(
            [
                probabilities[:, 0],
                probabilities[:, 1:4].sum(axis=1),
                probabilities[:, 4],
            ]
        )
    else:
        raise ValueError(f"Unsupported source class count: {source_classes}")
    return output / output.sum(axis=1, keepdims=True)


def common3_point_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("Common three-class evaluation requires all three classes")
    predicted = p.argmax(axis=1)
    macro_auc = float(
        roc_auc_score(
            y,
            p,
            multi_class="ovr",
            average="macro",
            labels=[0, 1, 2],
        )
    )
    ordinal_mae = float(np.mean(np.abs(y - predicted)))
    quadratic_weighted_kappa = float(
        cohen_kappa_score(y, predicted, weights="quadratic", labels=[0, 1, 2])
    )
    return {
        "macro_auc": macro_auc,
        "ordinal_mae": ordinal_mae,
        "quadratic_weighted_kappa": quadratic_weighted_kappa,
        "predicted_class": predicted,
    }


def grouped_common3_bootstrap(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    unique, rows = _group_rows(groups)
    rng = np.random.default_rng(seed)
    samples = {
        "macro_auc": [],
        "ordinal_mae": [],
        "quadratic_weighted_kappa": [],
    }
    for _ in range(replicates):
        chosen = _sample_group_rows(rng, rows)
        if np.unique(y[chosen]).size < 3:
            continue
        metrics = common3_point_metrics(y[chosen], p[chosen])
        for key in samples:
            value = metrics[key]
            if np.isfinite(value):
                samples[key].append(value)
    point = common3_point_metrics(y, p)
    output = {
        "estimate": {
            key: value
            for key, value in point.items()
            if key != "predicted_class"
        },
        "n_groups": int(len(unique)),
        "bootstrap_replicates_requested": int(replicates),
    }
    for key, values in samples.items():
        if values:
            output[key] = {
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "valid_replicates": int(len(values)),
            }
        else:
            output[key] = {
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "valid_replicates": 0,
            }
    return output


def _format_ci(
    estimate: float,
    ci_low: float,
    ci_high: float,
    digits: int = 3,
) -> str:
    return (
        f"{estimate:.{digits}f} "
        f"({ci_low:.{digits}f} to {ci_high:.{digits}f})"
    )


def _serialize_calibration_rows(
    dataset: str,
    transport: TransportData | None,
    model_key: str,
    target_key: str,
    target: np.ndarray,
    raw_probability: np.ndarray,
    oof_probability: np.ndarray,
    in_sample_affine: dict,
    raw_bootstrap: dict,
    oof_bootstrap: dict,
    fold_maps: list[dict],
    *,
    bootstrap_replicates: int,
) -> dict:
    raw_estimate = raw_bootstrap["estimate"]
    oof_estimate = oof_bootstrap["estimate"]
    in_sample_probability = apply_logistic_affine(
        raw_probability,
        in_sample_affine["intercept"],
        in_sample_affine["slope"],
    )
    in_sample_metrics = binary_calibration_metrics(target, in_sample_probability)
    if transport is None:
        validation_design = "five-fold out-of-fold predictions; five-seed average"
        n_folds = 5
    else:
        validation_design = transport.validation_design
        n_folds = transport.n_folds
    target_label = TARGETS[target_key]["label"]
    if target_key == "death" and dataset == "MIMIC-III":
        target_label = "Hospice-or-death, P(y=3)"
    elif target_key == "death" and dataset == "eICU direct transport":
        target_label = "In-hospital death, P(y=3)"
    return {
        "dataset": dataset,
        "validation_design": validation_design,
        "model_key": model_key,
        "model": MODELS[model_key],
        "target_key": target_key,
        "target": target_label,
        "n": int(len(target)),
        "events": int(target.sum()),
        "prevalence": float(target.mean()),
        "n_groups": int(raw_bootstrap["n_groups"]),
        "n_folds": int(n_folds),
        "bootstrap_replicates_requested": int(bootstrap_replicates),
        "raw_slope": raw_estimate["slope"],
        "raw_slope_ci_low": raw_bootstrap["slope"]["ci_low"],
        "raw_slope_ci_high": raw_bootstrap["slope"]["ci_high"],
        "raw_slope_valid_replicates": raw_bootstrap["slope"]["valid_replicates"],
        "raw_intercept": raw_estimate["intercept"],
        "raw_intercept_ci_low": raw_bootstrap["intercept"]["ci_low"],
        "raw_intercept_ci_high": raw_bootstrap["intercept"]["ci_high"],
        "raw_intercept_valid_replicates": raw_bootstrap["intercept"][
            "valid_replicates"
        ],
        "raw_brier": raw_estimate["brier"],
        "raw_brier_ci_low": raw_bootstrap["brier"]["ci_low"],
        "raw_brier_ci_high": raw_bootstrap["brier"]["ci_high"],
        "raw_brier_valid_replicates": raw_bootstrap["brier"]["valid_replicates"],
        "raw_log_loss": raw_estimate["log_loss"],
        "raw_log_loss_ci_low": raw_bootstrap["log_loss"]["ci_low"],
        "raw_log_loss_ci_high": raw_bootstrap["log_loss"]["ci_high"],
        "raw_log_loss_valid_replicates": raw_bootstrap["log_loss"][
            "valid_replicates"
        ],
        "in_sample_recal_intercept": in_sample_affine["intercept"],
        "in_sample_recal_slope": in_sample_affine["slope"],
        "in_sample_recal_brier": in_sample_metrics["brier"],
        "in_sample_recal_log_loss": in_sample_metrics["log_loss"],
        "in_sample_recal_converged": in_sample_affine["converged"],
        "oof_recal_slope": oof_estimate["slope"],
        "oof_recal_slope_ci_low": oof_bootstrap["slope"]["ci_low"],
        "oof_recal_slope_ci_high": oof_bootstrap["slope"]["ci_high"],
        "oof_recal_slope_valid_replicates": oof_bootstrap["slope"][
            "valid_replicates"
        ],
        "oof_recal_intercept": oof_estimate["intercept"],
        "oof_recal_intercept_ci_low": oof_bootstrap["intercept"]["ci_low"],
        "oof_recal_intercept_ci_high": oof_bootstrap["intercept"]["ci_high"],
        "oof_recal_intercept_valid_replicates": oof_bootstrap["intercept"][
            "valid_replicates"
        ],
        "oof_recal_brier": oof_estimate["brier"],
        "oof_recal_brier_ci_low": oof_bootstrap["brier"]["ci_low"],
        "oof_recal_brier_ci_high": oof_bootstrap["brier"]["ci_high"],
        "oof_recal_brier_valid_replicates": oof_bootstrap["brier"][
            "valid_replicates"
        ],
        "oof_recal_log_loss": oof_estimate["log_loss"],
        "oof_recal_log_loss_ci_low": oof_bootstrap["log_loss"]["ci_low"],
        "oof_recal_log_loss_ci_high": oof_bootstrap["log_loss"]["ci_high"],
        "oof_recal_log_loss_valid_replicates": oof_bootstrap["log_loss"][
            "valid_replicates"
        ],
        "fold_maps": fold_maps,
    }


def _serialize_common3_row(
    dataset: str,
    validation_design: str,
    group_label: str,
    model_key: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    bootstrap: dict,
) -> dict:
    estimate = bootstrap["estimate"]
    return {
        "dataset": dataset,
        "validation_design": validation_design,
        "grouping_unit": group_label,
        "model_key": model_key,
        "model": MODELS[model_key],
        "n": int(len(labels)),
        "class_0_n": int((labels == 0).sum()),
        "class_1_n": int((labels == 1).sum()),
        "class_2_n": int((labels == 2).sum()),
        "n_groups": int(bootstrap["n_groups"]),
        "bootstrap_replicates_requested": int(
            bootstrap["bootstrap_replicates_requested"]
        ),
        "macro_auc": estimate["macro_auc"],
        "macro_auc_ci_low": bootstrap["macro_auc"]["ci_low"],
        "macro_auc_ci_high": bootstrap["macro_auc"]["ci_high"],
        "macro_auc_valid_replicates": bootstrap["macro_auc"][
            "valid_replicates"
        ],
        "ordinal_mae": estimate["ordinal_mae"],
        "ordinal_mae_ci_low": bootstrap["ordinal_mae"]["ci_low"],
        "ordinal_mae_ci_high": bootstrap["ordinal_mae"]["ci_high"],
        "ordinal_mae_valid_replicates": bootstrap["ordinal_mae"][
            "valid_replicates"
        ],
        "quadratic_weighted_kappa": estimate["quadratic_weighted_kappa"],
        "quadratic_weighted_kappa_ci_low": bootstrap[
            "quadratic_weighted_kappa"
        ]["ci_low"],
        "quadratic_weighted_kappa_ci_high": bootstrap[
            "quadratic_weighted_kappa"
        ]["ci_high"],
        "quadratic_weighted_kappa_valid_replicates": bootstrap[
            "quadratic_weighted_kappa"
        ]["valid_replicates"],
    }


def _build_markdown_calibration(frame: pd.DataFrame) -> str:
    lines = [
        "# Table 7. External and cross-validated calibration",
        "",
        (
            "Values are estimate (95% grouped bootstrap CI). Recalibration used "
            "an affine logistic map on logit(p). eICU intervals resample "
            "patient-safe hospital clusters; MIMIC-III intervals resample "
            "patients. Cross-validated intervals condition on the fitted "
            "out-of-fold recalibration maps. Raw calibration slope and "
            "intercept are calibration-model coefficients for the original "
            "probabilities; OOF-recalibrated calibration slope and intercept "
            "are calibration-model coefficients for the cross-fitted "
            "recalibrated probabilities, not the coefficients of the affine "
            "recalibration maps."
        ),
        "",
        (
            "| Dataset | Target | Model | n/events | Raw calibration slope | "
            "Raw calibration intercept | Raw Brier | Raw log loss | In-sample "
            "recalibration intercept | In-sample recalibration slope | "
            "OOF-recalibrated calibration slope | OOF-recalibrated calibration "
            "intercept | OOF Brier | OOF log loss |"
        ),
        "|---|---|---|---:|---|---|---|---:|---:|---:|---|---|---|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            "| {dataset} | {target} | {model} | {n}/{events} | {raw_slope} | "
            "{raw_intercept} | {raw_brier} | {raw_log_loss} | {is_i:.3f} | "
            "{is_s:.3f} | {oof_slope} | {oof_intercept} | {oof_brier} | "
            "{oof_log_loss} |".format(
                dataset=row.dataset,
                target=row.target,
                model=row.model,
                n=row.n,
                events=row.events,
                raw_slope=_format_ci(
                    row.raw_slope, row.raw_slope_ci_low, row.raw_slope_ci_high
                ),
                raw_intercept=_format_ci(
                    row.raw_intercept,
                    row.raw_intercept_ci_low,
                    row.raw_intercept_ci_high,
                ),
                raw_brier=_format_ci(
                    row.raw_brier, row.raw_brier_ci_low, row.raw_brier_ci_high
                ),
                raw_log_loss=_format_ci(
                    row.raw_log_loss,
                    row.raw_log_loss_ci_low,
                    row.raw_log_loss_ci_high,
                ),
                is_i=row.in_sample_recal_intercept,
                is_s=row.in_sample_recal_slope,
                oof_slope=_format_ci(
                    row.oof_recal_slope,
                    row.oof_recal_slope_ci_low,
                    row.oof_recal_slope_ci_high,
                ),
                oof_intercept=_format_ci(
                    row.oof_recal_intercept,
                    row.oof_recal_intercept_ci_low,
                    row.oof_recal_intercept_ci_high,
                ),
                oof_brier=_format_ci(
                    row.oof_recal_brier,
                    row.oof_recal_brier_ci_low,
                    row.oof_recal_brier_ci_high,
                ),
                oof_log_loss=_format_ci(
                    row.oof_recal_log_loss,
                    row.oof_recal_log_loss_ci_low,
                    row.oof_recal_log_loss_ci_high,
                ),
            )
        )
    lines.extend(
        [
            "",
            (
                "Abbreviations: OOF, out-of-fold; TS, temporal-summary features; "
                "TR, temporal regularization."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _build_markdown_common3(frame: pd.DataFrame) -> str:
    lines = [
        "# Table 8. Common three-class outcome across three databases",
        "",
        (
            "Values are estimate (95% grouped bootstrap CI). Macro AUC is "
            "one-vs-rest and unweighted. Ordinal MAE and quadratic weighted kappa "
            "use the probability-maximizing class. MIMIC-IV values average "
            "out-of-fold probabilities across five seeds. eICU intervals resample "
            "patient-safe hospital clusters; MIMIC-III and MIMIC-IV intervals "
            "resample patients."
        ),
        "",
        (
            "Common outcome: 0 = home or against advice; 1 = non-home "
            "post-acute or other care; 2 = hospice or in-hospital death. "
            "The middle level is a pragmatic harmonization and does not imply "
            "equivalent care intensity across databases."
        ),
        "",
        (
            "| Dataset | Model | n (class 0/1/2) | Macro AUC | Ordinal MAE | "
            "Quadratic weighted kappa |"
        ),
        "|---|---|---:|---|---|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            "| {dataset} | {model} | {n} ({c0}/{c1}/{c2}) | {auc} | {mae} | "
            "{kappa} |".format(
                dataset=row.dataset,
                model=row.model,
                n=row.n,
                c0=row.class_0_n,
                c1=row.class_1_n,
                c2=row.class_2_n,
                auc=_format_ci(
                    row.macro_auc, row.macro_auc_ci_low, row.macro_auc_ci_high
                ),
                mae=_format_ci(
                    row.ordinal_mae,
                    row.ordinal_mae_ci_low,
                    row.ordinal_mae_ci_high,
                ),
                kappa=_format_ci(
                    row.quadratic_weighted_kappa,
                    row.quadratic_weighted_kappa_ci_low,
                    row.quadratic_weighted_kappa_ci_high,
                ),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _validate_ci_columns(
    frame: pd.DataFrame,
    prefixes: Iterable[str],
) -> dict:
    checks = {}
    for prefix in prefixes:
        low_column = f"{prefix}_ci_low"
        high_column = f"{prefix}_ci_high"
        if low_column not in frame or high_column not in frame:
            raise ValueError(f"Missing bootstrap CI columns for {prefix}")
        low = frame[low_column].to_numpy(dtype=float)
        high = frame[high_column].to_numpy(dtype=float)
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError(f"Non-finite bootstrap CI found for {prefix}")
        if not np.all(low <= high):
            raise ValueError(f"Reversed bootstrap CI found for {prefix}")
        checks[prefix] = {
            "n": int(len(frame)),
            "all_finite": True,
            "all_lower_not_greater_than_upper": True,
        }
    return checks


def run(replicates: int) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mimic3, mimic3_alignment = _load_transport_dataset(
        "mimic3",
        group_column="subject_key",
        group_label="patient (subject_key)",
        validation_design="patient-grouped five-fold out-of-fold recalibration",
        n_folds=5,
        cohort_label_map={0: 0, 1: 1, 2: 1, 3: 2, 4: 3},
    )
    eicu, eicu_alignment = _load_transport_dataset(
        "eicu",
        group_column="hospital_key",
        group_label="patient-safe hospital cluster",
        validation_design=(
            "patient-safe leave-one-hospital-cluster-out out-of-fold "
            "recalibration"
        ),
        n_folds=0,
        cohort_label_map={0: 0, 1: 1, 2: 2, 3: 3},
        patient_safe_hospital_clusters_enabled=True,
    )
    mimic4, mimic4_labels, mimic4_probabilities, mimic4_alignment = (
        _load_mimic4_seed_average()
    )

    if len(mimic3.ids) != 4750:
        raise ValueError(f"Unexpected MIMIC-III cohort size: {len(mimic3.ids)}")
    if len(eicu.ids) != 6696:
        raise ValueError(f"Unexpected eICU cohort size: {len(eicu.ids)}")
    if len(mimic4) != 6487:
        raise ValueError(f"Unexpected MIMIC-IV cohort size: {len(mimic4)}")

    calibration_rows: list[dict] = []
    calibration_prediction_frames: list[pd.DataFrame] = []
    calibration_json: dict = {}
    calibration_combo_index = 0
    for transport in (mimic3, eicu):
        dataset_name = "MIMIC-III" if transport.dataset == "mimic3" else "eICU direct transport"
        calibration_json[dataset_name] = {
            "validation_design": transport.validation_design,
            "grouping_unit": transport.group_label,
            "n_folds": transport.n_folds,
            "models": {},
        }
        for model_index, model_key in enumerate(MODELS):
            model_json = {}
            for target_index, target_key in enumerate(TARGETS):
                raw_probability = _target_probability(
                    transport.probabilities[model_index], target_key
                )
                target = (
                    (transport.labels >= 2).astype(int)
                    if target_key == "high_need"
                    else (transport.labels == 3).astype(int)
                )
                in_sample_affine = fit_logistic_affine(target, raw_probability)
                if not np.isfinite(in_sample_affine["intercept"]):
                    raise ValueError(
                        f"In-sample affine map failed: {dataset_name}, "
                        f"{model_key}, {target_key}"
                    )
                if transport.dataset == "mimic3":
                    oof_probability, fold_maps = (
                        mimic3_patient_grouped_recalibration(
                            target, raw_probability, transport.groups
                        )
                    )
                else:
                    oof_probability, fold_maps = (
                        patient_safe_leave_one_hospital_cluster_out_recalibration(
                            target, raw_probability, transport.groups
                        )
                    )
                seed_base = (
                    BOOTSTRAP_SEED
                    + 1000 * calibration_combo_index
                )
                raw_bootstrap = grouped_calibration_bootstrap(
                    target,
                    raw_probability,
                    transport.groups,
                    replicates=replicates,
                    seed=seed_base + 1,
                )
                oof_bootstrap = grouped_calibration_bootstrap(
                    target,
                    oof_probability,
                    transport.groups,
                    replicates=replicates,
                    seed=seed_base + 2,
                )
                row = _serialize_calibration_rows(
                    dataset_name,
                    transport,
                    model_key,
                    target_key,
                    target,
                    raw_probability,
                    oof_probability,
                    in_sample_affine,
                    raw_bootstrap,
                    oof_bootstrap,
                    fold_maps,
                    bootstrap_replicates=replicates,
                )
                row["model_fold_map_count"] = int(len(fold_maps))
                calibration_rows.append(row)
                model_json[target_key] = {
                    key: value
                    for key, value in row.items()
                    if key not in {"dataset", "target", "model", "model_key"}
                }

                calibration_prediction_frames.append(
                    pd.DataFrame(
                        {
                            "dataset": dataset_name,
                            "model": MODELS[model_key],
                            "target": target_key,
                            "stay_id": transport.ids,
                            "group_id": transport.groups,
                            "y": target,
                            "raw_probability": raw_probability,
                            "in_sample_recal_probability": apply_logistic_affine(
                                raw_probability,
                                in_sample_affine["intercept"],
                                in_sample_affine["slope"],
                            ),
                            "oof_recal_probability": oof_probability,
                        }
                    )
                )
                calibration_combo_index += 1
            calibration_json[dataset_name]["models"][MODELS[model_key]] = (
                model_json
            )

    calibration_frame = pd.DataFrame(calibration_rows)
    expected_calibration_rows = 2 * len(MODELS) * len(TARGETS)
    if len(calibration_frame) != expected_calibration_rows:
        raise ValueError(
            f"Calibration row check failed: {len(calibration_frame)} != "
            f"{expected_calibration_rows}"
        )
    calibration_ci_checks = _validate_ci_columns(
        calibration_frame,
        prefixes=(
            "raw_slope",
            "raw_intercept",
            "raw_brier",
            "raw_log_loss",
            "oof_recal_slope",
            "oof_recal_intercept",
            "oof_recal_brier",
            "oof_recal_log_loss",
        ),
    )
    if not calibration_frame["in_sample_recal_converged"].all():
        raise ValueError("At least one in-sample affine recalibration did not converge")
    calibration_frame.drop(columns=["fold_maps"]).to_csv(
        CALIBRATION_CSV, index=False
    )
    calibration_predictions = pd.concat(
        calibration_prediction_frames, ignore_index=True
    )
    calibration_predictions.to_csv(CALIBRATION_PREDICTIONS_CSV, index=False)
    external_calibration_curves = build_external_calibration_curves(
        calibration_predictions
    )
    external_calibration_curves.to_csv(
        EXTERNAL_CALIBRATION_CURVES_CSV, index=False
    )
    CALIBRATION_MD.write_text(
        _build_markdown_calibration(calibration_frame),
        encoding="utf-8",
    )

    common3_rows: list[dict] = []
    common3_prediction_frames: list[pd.DataFrame] = []
    common3_json: dict = {}
    common3_combo_index = 0

    mimic4_common_labels = _map_mimic4_common3(mimic4_labels)
    mimic4_groups = mimic4["subject_key"].astype(str).to_numpy()
    common3_json["MIMIC-IV"] = {
        "validation_design": "five-fold out-of-fold predictions; five-seed average",
        "grouping_unit": "patient (subject_key)",
        "models": {},
    }
    for model_key, source_probability in mimic4_probabilities.items():
        probabilities = _common3_probabilities(
            source_probability, source_classes=5
        )
        bootstrap = grouped_common3_bootstrap(
            mimic4_common_labels,
            probabilities,
            mimic4_groups,
            replicates=replicates,
            seed=BOOTSTRAP_SEED + 1000 * common3_combo_index + 1,
        )
        row = _serialize_common3_row(
            "MIMIC-IV",
            "five-fold out-of-fold predictions; five-seed average",
            "patient (subject_key)",
            model_key,
            mimic4_common_labels,
            probabilities,
            bootstrap,
        )
        common3_rows.append(row)
        common3_json["MIMIC-IV"]["models"][MODELS[model_key]] = {
            key: value for key, value in row.items() if key not in {"dataset"}
        }
        common3_prediction_frames.append(
            pd.DataFrame(
                {
                    "dataset": "MIMIC-IV",
                    "model": MODELS[model_key],
                    "stay_id": mimic4["stay_id"].to_numpy(dtype=np.int64),
                    "group_id": mimic4_groups,
                    "y_original": mimic4_labels,
                    "y_common3": mimic4_common_labels,
                    "p_class_0": probabilities[:, 0],
                    "p_class_1": probabilities[:, 1],
                    "p_class_2": probabilities[:, 2],
                    "predicted_class": probabilities.argmax(axis=1),
                }
            )
        )
        common3_combo_index += 1

    for transport in (mimic3, eicu):
        dataset_name = "MIMIC-III" if transport.dataset == "mimic3" else "eICU direct transport"
        common_labels = _map_four_to_common3(transport.labels)
        common3_json[dataset_name] = {
            "validation_design": "direct external transport",
            "grouping_unit": transport.group_label,
            "models": {},
        }
        for model_index, model_key in enumerate(MODELS):
            probabilities = _common3_probabilities(
                transport.probabilities[model_index], source_classes=4
            )
            bootstrap = grouped_common3_bootstrap(
                common_labels,
                probabilities,
                transport.groups,
                replicates=replicates,
                seed=BOOTSTRAP_SEED + 1000 * common3_combo_index + 1,
            )
            row = _serialize_common3_row(
                dataset_name,
                "direct external transport",
                transport.group_label,
                model_key,
                common_labels,
                probabilities,
                bootstrap,
            )
            common3_rows.append(row)
            common3_json[dataset_name]["models"][MODELS[model_key]] = {
                key: value for key, value in row.items() if key not in {"dataset"}
            }
            common3_prediction_frames.append(
                pd.DataFrame(
                    {
                        "dataset": dataset_name,
                        "model": MODELS[model_key],
                        "stay_id": transport.ids,
                        "group_id": transport.groups,
                        "y_original": transport.labels,
                        "y_common3": common_labels,
                        "p_class_0": probabilities[:, 0],
                        "p_class_1": probabilities[:, 1],
                        "p_class_2": probabilities[:, 2],
                        "predicted_class": probabilities.argmax(axis=1),
                    }
                )
            )
            common3_combo_index += 1

    common3_frame = pd.DataFrame(common3_rows)
    expected_common3_rows = 3 * len(MODELS)
    if len(common3_frame) != expected_common3_rows:
        raise ValueError(
            f"Common-three-class row check failed: {len(common3_frame)} != "
            f"{expected_common3_rows}"
        )
    common3_ci_checks = _validate_ci_columns(
        common3_frame,
        prefixes=(
            "macro_auc",
            "ordinal_mae",
            "quadratic_weighted_kappa",
        ),
    )
    common3_frame.to_csv(COMMON3_CSV, index=False)
    common3_predictions = pd.concat(
        common3_prediction_frames, ignore_index=True
    )
    common3_predictions.to_csv(COMMON3_PREDICTIONS_CSV, index=False)
    COMMON3_MD.write_text(_build_markdown_common3(common3_frame), encoding="utf-8")

    source_paths = [
        PROCESSED / "mimic4" / "cohort.csv",
        PROCESSED / "mimic3" / "cohort.csv",
        PROCESSED / "eicu" / "cohort.csv",
    ]
    for dataset, models in (
        ("mimic4", tuple(MODELS)),
        ("mimic3", tuple(MODELS)),
        ("eicu", tuple(MODELS)),
    ):
        for model_key in models:
            if dataset == "mimic4":
                source_paths.extend(
                    PREDICTIONS / f"mimic4_{model_key}_seed{seed}.npz"
                    for seed in SEEDS
                )
            else:
                source_paths.append(
                    PREDICTIONS / f"{dataset}_{model_key}_transport.npz"
                )

    source_hashes = {
        _relative(path): _sha256(path)
        for path in sorted(set(source_paths))
    }
    calibration_counts = {
        dataset: {
            "rows": int(
                (
                    calibration_predictions["dataset"]
                    == dataset
                ).sum()
            ),
            "unique_stay_id": int(
                calibration_predictions.loc[
                    calibration_predictions["dataset"] == dataset, "stay_id"
                ].nunique()
            ),
        }
        for dataset in ("MIMIC-III", "eICU direct transport")
    }
    expected_prediction_rows = {
        "MIMIC-III": len(mimic3.ids) * len(MODELS) * len(TARGETS),
        "eICU direct transport": len(eicu.ids) * len(MODELS) * len(TARGETS),
    }
    for dataset, expected in expected_prediction_rows.items():
        if calibration_counts[dataset]["rows"] != expected:
            raise ValueError(f"Calibration prediction row count failed for {dataset}")
        if calibration_counts[dataset]["unique_stay_id"] != (
            len(mimic3.ids) if dataset == "MIMIC-III" else len(eicu.ids)
        ):
            raise ValueError(f"Calibration patient alignment failed for {dataset}")
    curve_counts = (
        external_calibration_curves.groupby(
            ["dataset", "model", "target", "score_type"], observed=True
        )
        .size()
        .to_dict()
    )
    if len(curve_counts) != 2 * len(MODELS) * len(TARGETS) * 2:
        raise ValueError("External calibration curve combination count failed")
    if not all(value == 10 for value in curve_counts.values()):
        raise ValueError("Every external calibration curve must contain 10 bins")
    expected_curve_n = {
        "MIMIC-III": len(mimic3.ids),
        "eICU direct transport": len(eicu.ids),
    }
    for (dataset, model, target_key, score_type), group in (
        external_calibration_curves.groupby(
            ["dataset", "model", "target", "score_type"], observed=True
        )
    ):
        if int(group["n"].sum()) != expected_curve_n[dataset]:
            raise ValueError(
                "External calibration curve row totals failed for "
                f"{dataset}/{model}/{target_key}/{score_type}"
            )

    common3_counts = {
        dataset: {
            "rows": int((common3_predictions["dataset"] == dataset).sum()),
            "unique_stay_id": int(
                common3_predictions.loc[
                    common3_predictions["dataset"] == dataset, "stay_id"
                ].nunique()
            ),
        }
        for dataset in ("MIMIC-IV", "MIMIC-III", "eICU direct transport")
    }
    expected_common3_counts = {
        "MIMIC-IV": len(mimic4) * len(MODELS),
        "MIMIC-III": len(mimic3.ids) * len(MODELS),
        "eICU direct transport": len(eicu.ids) * len(MODELS),
    }
    for dataset, expected in expected_common3_counts.items():
        if common3_counts[dataset]["rows"] != expected:
            raise ValueError(f"Common-three-class row count failed for {dataset}")
        expected_ids = {
            "MIMIC-IV": len(mimic4),
            "MIMIC-III": len(mimic3.ids),
            "eICU direct transport": len(eicu.ids),
        }[dataset]
        if common3_counts[dataset]["unique_stay_id"] != expected_ids:
            raise ValueError(f"Common-three-class patient alignment failed for {dataset}")

    report = {
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": _relative(Path(__file__).resolve()),
        "analysis": {
            "bootstrap_replicates": int(replicates),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "grouped_bootstrap_method": (
                "cluster resampling with replacement; eICU by patient-safe "
                "hospital cluster and MIMIC-III/MIMIC-IV by subject_key"
            ),
            "cross_validated_ci_note": (
                "Out-of-fold recalibration maps were held fixed during cluster "
                "bootstrap resampling; intervals are conditional on the fitted "
                "maps and do not include recalibration-map estimation variance."
            ),
            "patient_safe_hospital_clusters": (
                "Connected components of the eICU patient-hospital graph. "
                "Hospitals sharing any patient are held out together, so no "
                "patient contributes to both training and test folds."
            ),
            "affine_map": "logit(q) = intercept + slope * logit(p)",
            "recalibration_fit": (
                "unpenalized logistic regression via Newton-Raphson"
            ),
            "ordinal_prediction_rule": "probability-maximizing class",
            "sign_convention": "original model positive direction; no absolute values",
            "calibration_metric_note": (
                "Raw calibration slope/intercept calibrate the original "
                "probability; OOF-recalibrated calibration slope/intercept "
                "calibrate the cross-fitted recalibrated probability. They are "
                "not the affine-map coefficients."
            ),
            "cross_model_calibration_comparison_note": (
                "All transported models were trained on the same five-class "
                "development outcome and then collapsed to the shared "
                "four-class external scale. Cross-model calibration "
                "differences still compare complete modeling pipelines rather "
                "than graph structure alone. Within-model raw versus "
                "OOF-recalibrated comparisons are the primary calibration "
                "evidence."
            ),
            "target_definitions": {
                "high_need": "P(y>=2)",
                "death": "P(y=3)",
            },
            "target_probability_construction": {
                "high_need": "probability class 2 + probability class 3",
                "death": "probability class 3",
            },
            "common3_mapping": {
                "MIMIC-IV": {"0": 0, "1": 1, "2": 1, "3": 1, "4": 2},
                "MIMIC-III": {"0": 0, "1": 1, "2": 1, "3": 2},
                "eICU": {"0": 0, "1": 1, "2": 1, "3": 2},
            },
            "common3_labels": {
                "0": "home or against advice",
                "1": "non-home post-acute or other care",
                "2": "hospice or in-hospital death",
            },
            "common3_harmonization_note": (
                "Class 1 is a pragmatic harmonization. It combines "
                "non-home post-acute and other care categories that are not "
                "construct-identical across databases."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "source_hashes_sha256": source_hashes,
        "alignment": {
            "MIMIC-IV": mimic4_alignment,
            "MIMIC-III": mimic3_alignment,
            "eICU": eicu_alignment,
        },
        "calibration": calibration_json,
        "common_three_class": common3_json,
        "row_checks": {
            "table7_rows": int(len(calibration_frame)),
            "table7_expected": int(expected_calibration_rows),
            "table8_rows": int(len(common3_frame)),
            "table8_expected": int(expected_common3_rows),
            "calibration_prediction_rows": calibration_counts,
            "calibration_prediction_rows_expected": expected_prediction_rows,
            "common3_prediction_rows": common3_counts,
            "common3_prediction_rows_expected": expected_common3_counts,
            "external_calibration_curve_rows": int(
                len(external_calibration_curves)
            ),
            "external_calibration_curve_rows_expected": int(
                2 * len(MODELS) * len(TARGETS) * 2 * 10
            ),
            "external_calibration_curve_bins_per_combination": 10,
        },
        "ci_checks": {
            "table7": calibration_ci_checks,
            "table8": common3_ci_checks,
        },
        "outputs": {
            "table7_csv": _relative(CALIBRATION_CSV),
            "table7_markdown": _relative(CALIBRATION_MD),
            "table8_csv": _relative(COMMON3_CSV),
            "table8_markdown": _relative(COMMON3_MD),
            "calibration_predictions_csv": _relative(
                CALIBRATION_PREDICTIONS_CSV
            ),
            "common3_predictions_csv": _relative(COMMON3_PREDICTIONS_CSV),
            "external_calibration_curves_csv": _relative(
                EXTERNAL_CALIBRATION_CURVES_CSV
            ),
            "report_json": _relative(REPORT_JSON),
        },
    }
    REPORT_JSON.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def verify_outputs() -> dict:
    if not REPORT_JSON.is_file():
        raise FileNotFoundError(REPORT_JSON)
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    calibration = pd.read_csv(CALIBRATION_CSV)
    common3 = pd.read_csv(COMMON3_CSV)
    calibration_predictions = pd.read_csv(CALIBRATION_PREDICTIONS_CSV)
    common3_predictions = pd.read_csv(COMMON3_PREDICTIONS_CSV)
    calibration_curves = pd.read_csv(EXTERNAL_CALIBRATION_CURVES_CSV)
    if report["status"] != "passed":
        raise ValueError("Report status is not passed")
    if len(calibration) != 12 or len(common3) != 9:
        raise ValueError("Output table row counts are incorrect")
    _validate_ci_columns(
        calibration,
        prefixes=(
            "raw_slope",
            "raw_intercept",
            "raw_brier",
            "raw_log_loss",
            "oof_recal_slope",
            "oof_recal_intercept",
            "oof_recal_brier",
            "oof_recal_log_loss",
        ),
    )
    _validate_ci_columns(
        common3,
        prefixes=("macro_auc", "ordinal_mae", "quadratic_weighted_kappa"),
    )
    expected_calibration_predictions = (
        len(mimic3_patient_ids := pd.read_csv(
            PROCESSED / "mimic3" / "cohort.csv", usecols=["stay_id"]
        ))
        + len(pd.read_csv(PROCESSED / "eicu" / "cohort.csv", usecols=["stay_id"]))
    ) * len(MODELS) * len(TARGETS)
    if len(calibration_predictions) != expected_calibration_predictions:
        raise ValueError("Calibration prediction row count is incorrect")
    expected_common3_predictions = (
        len(pd.read_csv(PROCESSED / "mimic4" / "cohort.csv", usecols=["stay_id"]))
        + len(mimic3_patient_ids)
        + len(pd.read_csv(PROCESSED / "eicu" / "cohort.csv", usecols=["stay_id"]))
    ) * len(MODELS)
    if len(common3_predictions) != expected_common3_predictions:
        raise ValueError("Common-three-class prediction row count is incorrect")
    expected_curve_rows = 2 * len(MODELS) * len(TARGETS) * 2 * 10
    if len(calibration_curves) != expected_curve_rows:
        raise ValueError("External calibration curve row count is incorrect")
    required_curve_columns = {
        "dataset",
        "model",
        "target",
        "score_type",
        "bin_id",
        "n",
        "predicted_mean",
        "observed_rate",
        "observed_ci_low",
        "observed_ci_high",
    }
    if not required_curve_columns.issubset(calibration_curves.columns):
        raise ValueError("External calibration curve columns are incomplete")
    curve_counts = calibration_curves.groupby(
        ["dataset", "model", "target", "score_type"], observed=True
    ).size()
    if len(curve_counts) != 24 or not (curve_counts == 10).all():
        raise ValueError("External calibration curves must have 10 bins each")
    curve_ci_low = calibration_curves["observed_ci_low"].to_numpy(dtype=float)
    curve_ci_high = calibration_curves["observed_ci_high"].to_numpy(dtype=float)
    if not (
        np.isfinite(curve_ci_low).all()
        and np.isfinite(curve_ci_high).all()
        and np.all(curve_ci_low <= curve_ci_high)
    ):
        raise ValueError("External calibration curve CI check failed")
    return {
        "status": "passed",
        "table7_rows": int(len(calibration)),
        "table8_rows": int(len(common3)),
        "calibration_prediction_rows": int(len(calibration_predictions)),
        "common3_prediction_rows": int(len(common3_predictions)),
        "external_calibration_curve_rows": int(len(calibration_curves)),
        "ci_checks": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate external affine recalibration and a common three-class "
            "outcome using locked prediction caches."
        )
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=DEFAULT_REPLICATES,
        help="Grouped bootstrap replicates (default: 2000).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate existing outputs without recomputing them.",
    )
    args = parser.parse_args()
    if args.replicates < 200:
        raise ValueError("At least 200 bootstrap replicates are required")
    result = verify_outputs() if args.verify_only else run(args.replicates)
    if args.verify_only:
        concise = result
    else:
        concise = {
            "status": result["status"],
            "bootstrap_replicates": result["analysis"]["bootstrap_replicates"],
            "table7_rows": result["row_checks"]["table7_rows"],
            "table8_rows": result["row_checks"]["table8_rows"],
            "external_calibration_curve_rows": result["row_checks"][
                "external_calibration_curve_rows"
            ],
            "outputs": result["outputs"],
        }
    print(json.dumps(_json_ready(concise), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

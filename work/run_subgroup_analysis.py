"""Evaluate prespecified development-cohort subgroups from frozen OOF predictions."""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import special
from sklearn.exceptions import UndefinedMetricWarning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import PROCESSED, RESULTS  # noqa: E402
from src.ami_dynamic.evaluation import (  # noqa: E402
    paired_bootstrap_delta,
    point_metrics,
)

BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 20_261_106


def _load_average(model: str) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(
        (RESULTS / "predictions").glob(f"mimic4_{model}_seed*.npz")
    )
    if len(paths) != 5:
        raise FileNotFoundError(f"Expected five seed files for {model}")
    probabilities = []
    reference_labels = None
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            labels = data["labels"].astype(int)
            if reference_labels is None:
                reference_labels = labels
            elif not np.array_equal(reference_labels, labels):
                raise ValueError(f"{path.name}: labels differ")
            probabilities.append(data["probabilities"].astype(float))
    return reference_labels, np.mean(np.stack(probabilities), axis=0)


def _strata(cohort: pd.DataFrame) -> dict[str, np.ndarray]:
    flags = pd.read_csv(PROCESSED / "mimic4" / "diagnosis_flags.csv")
    frame = cohort.merge(flags, on=["stay_id", "hadm_id"], how="left")
    age = pd.to_numeric(frame["age"], errors="coerce")
    strata = {
        "overall": np.ones(len(frame), dtype=bool),
        "female": frame["sex_male"].eq(0),
        "male": frame["sex_male"].eq(1),
        "age_18_64": age.lt(65),
        "age_65_79": age.between(65, 79.999),
        "age_80_plus": age.ge(80),
        "STEMI": frame["ami_subtype"].eq("STEMI"),
        "NSTEMI": frame["ami_subtype"].eq("NSTEMI"),
        "type_2_MI": frame["ami_subtype"].eq("Type 2 MI"),
        "other_or_unspecified_MI": frame["ami_subtype"].isin(
            ["Other", "Unspecified"]
        ),
        "primary_position_AMI": frame["ami_primary_position"].eq(1),
        "non_primary_position_AMI": frame["ami_primary_position"].eq(0),
        "emergency_admission": frame["emergency_admission"].eq(1),
        "non_emergency_admission": frame["emergency_admission"].eq(0),
        "CCU_or_CVICU": (
            frame["care_unit_ccu"].eq(1) | frame["care_unit_cvicu"].eq(1)
        ),
        "other_ICU": ~(
            frame["care_unit_ccu"].eq(1) | frame["care_unit_cvicu"].eq(1)
        ),
    }
    return {
        name: np.asarray(mask, dtype=bool)
        for name, mask in strata.items()
    }


def _binary_brier(target: np.ndarray, probability: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if len(target) == 0:
        return float("nan")
    return float(np.mean((target - probability) ** 2))


def _calibration_fit(
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float]:
    """Near-unpenalized calibration fit: logit(y)=intercept+slope*logit(p)."""
    target = np.asarray(target, dtype=float)
    probability = np.clip(
        np.asarray(probability, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )
    if len(target) < 2 or len(np.unique(target)) < 2:
        return float("nan"), float("nan")

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


def _bootstrap_calibration(
    target: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    *,
    task_seed: int,
) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    events = int(np.sum(target))
    non_events = int(len(target) - events)
    estimable = len(target) >= 30 and events >= 10 and non_events >= 10
    if not estimable:
        output = {
            "estimable": False,
            "events": events,
            "prevalence": float(target.mean()) if len(target) else float("nan"),
            "brier": float("nan"),
            "calibration_intercept": float("nan"),
            "calibration_slope": float("nan"),
        }
        for metric in ("brier", "calibration_intercept", "calibration_slope"):
            output[f"{metric}_ci_low"] = float("nan")
            output[f"{metric}_ci_high"] = float("nan")
            output[f"{metric}_valid_replicates"] = 0
        return output

    group_values, group_codes = np.unique(np.asarray(groups), return_inverse=True)
    group_indices = [
        np.flatnonzero(group_codes == group_index)
        for group_index in range(len(group_values))
    ]
    rng = np.random.default_rng(task_seed)
    samples = {"slope": [], "intercept": [], "brier": []}
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(group_indices), len(group_indices))
        rows = np.concatenate([group_indices[index] for index in selected])
        sample_target = target[rows]
        sample_probability = probability[rows]
        intercept, slope = _calibration_fit(sample_target, sample_probability)
        if np.isfinite(intercept) and np.isfinite(slope):
            samples["intercept"].append(intercept)
            samples["slope"].append(slope)
            samples["brier"].append(
                _binary_brier(sample_target, sample_probability)
            )

    intercept, slope = _calibration_fit(target, probability)
    output = {
        "estimable": True,
        "events": events,
        "prevalence": float(target.mean()) if len(target) else float("nan"),
        "brier": _binary_brier(target, probability),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
    for output_name, sample_name in (
        ("brier", "brier"),
        ("calibration_intercept", "intercept"),
        ("calibration_slope", "slope"),
    ):
        values = np.asarray(samples[sample_name], dtype=float)
        values = values[np.isfinite(values)]
        output[f"{output_name}_ci_low"] = (
            float(np.percentile(values, 2.5)) if len(values) else float("nan")
        )
        output[f"{output_name}_ci_high"] = (
            float(np.percentile(values, 97.5)) if len(values) else float("nan")
        )
        output[f"{output_name}_valid_replicates"] = int(len(values))
    return output


def _calibration_task(
    *,
    subgroup: str,
    model: str,
    target: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    task_seed: int,
) -> tuple[str, str, dict[str, float]]:
    return (
        subgroup,
        model,
        _bootstrap_calibration(
            target,
            probability,
            groups,
            task_seed=task_seed,
        ),
    )


def main() -> None:
    cohort = pd.read_csv(PROCESSED / "mimic4" / "cohort.csv")
    groups = cohort["subject_key"].astype(str).to_numpy()
    labels, xgb_probability = _load_average("xgb_static_ts")
    deep_labels, deep_probability = _load_average("dggru_tr")
    if not np.array_equal(labels, deep_labels):
        raise ValueError("Model prediction labels differ")
    strata = _strata(cohort)
    high_need = labels >= 3
    high_need_probabilities = {
        "xgb_static_ts": xgb_probability[:, 3:].sum(axis=1),
        "dggru_tr": deep_probability[:, 3:].sum(axis=1),
    }

    calibration_tasks = []
    subgroup_items = list(strata.items())
    for subgroup_index, (name, mask) in enumerate(subgroup_items):
        for model_index, model in enumerate(("xgb_static_ts", "dggru_tr")):
            calibration_tasks.append(
                delayed(_calibration_task)(
                    subgroup=name,
                    model=model,
                    target=high_need[mask],
                    probability=high_need_probabilities[model][mask],
                    groups=groups[mask],
                    task_seed=(
                        BOOTSTRAP_SEED
                        + subgroup_index * 10
                        + model_index
                    ),
                )
            )
    calibration_results = Parallel(
        n_jobs=min(os.cpu_count() or 1, 16),
        prefer="processes",
        verbose=5,
    )(calibration_tasks)
    calibration_by_group = {
        (subgroup, model): payload
        for subgroup, model, payload in calibration_results
    }

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for name, mask in strata.items():
            row = {
                "subgroup": name,
                "n": int(mask.sum()),
                "class_counts": {
                    str(int(key)): int(value)
                    for key, value in pd.Series(labels[mask])
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "xgb_static_ts_macro_auc": point_metrics(
                    labels[mask], xgb_probability[mask]
                )["macro_auc"],
                "dggru_tr_macro_auc": point_metrics(
                    labels[mask], deep_probability[mask]
                )["macro_auc"],
                "dggru_tr_minus_xgb": paired_bootstrap_delta(
                    labels[mask],
                    deep_probability[mask],
                    xgb_probability[mask],
                    groups=groups[mask],
                    replicates=1000,
                ),
            }
            for model in ("xgb_static_ts", "dggru_tr"):
                payload = calibration_by_group[(name, model)]
                for metric, value in payload.items():
                    row[f"{model}_high_need_{metric}"] = value
            rows.append(row)
    output = {
        "n": int(len(cohort)),
        "note": (
            "Subgroup estimates reuse the prespecified outer-fold predictions; "
            "they are descriptive and are not models refitted within subgroups. "
            "High-need calibration uses y>=3 and patient-level bootstrap."
        ),
        "high_need_definition": "five-level outcome y>=3",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "subgroups": rows,
    }
    path = RESULTS / "audits" / "mimic4_subgroup_analysis.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows).drop(columns=["class_counts"])
    frame.to_csv(
        RESULTS / "tables" / "mimic4_subgroups.csv",
        index=False,
    )
    calibration_columns = [
        "subgroup",
        "n",
        "xgb_static_ts_high_need_estimable",
        "xgb_static_ts_high_need_events",
        "xgb_static_ts_high_need_prevalence",
        "xgb_static_ts_high_need_brier",
        "xgb_static_ts_high_need_brier_ci_low",
        "xgb_static_ts_high_need_brier_ci_high",
        "xgb_static_ts_high_need_calibration_slope",
        "xgb_static_ts_high_need_calibration_slope_ci_low",
        "xgb_static_ts_high_need_calibration_slope_ci_high",
        "xgb_static_ts_high_need_calibration_intercept",
        "xgb_static_ts_high_need_calibration_intercept_ci_low",
        "xgb_static_ts_high_need_calibration_intercept_ci_high",
        "dggru_tr_high_need_estimable",
        "dggru_tr_high_need_brier",
        "dggru_tr_high_need_brier_ci_low",
        "dggru_tr_high_need_brier_ci_high",
        "dggru_tr_high_need_calibration_slope",
        "dggru_tr_high_need_calibration_slope_ci_low",
        "dggru_tr_high_need_calibration_slope_ci_high",
        "dggru_tr_high_need_calibration_intercept",
        "dggru_tr_high_need_calibration_intercept_ci_low",
        "dggru_tr_high_need_calibration_intercept_ci_high",
    ]
    frame[calibration_columns].to_csv(
        RESULTS / "tables" / "tableS16_subgroup_calibration.csv",
        index=False,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

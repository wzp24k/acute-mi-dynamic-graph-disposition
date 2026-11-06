"""Evaluate prespecified development-cohort subgroups from frozen OOF predictions."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import PROCESSED, RESULTS  # noqa: E402
from src.ami_dynamic.evaluation import (  # noqa: E402
    bootstrap_metric_ci,
    paired_bootstrap_delta,
    point_metrics,
)


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


def main() -> None:
    cohort = pd.read_csv(PROCESSED / "mimic4" / "cohort.csv")
    groups = cohort["subject_key"].astype(str).to_numpy()
    labels, xgb_probability = _load_average("xgb_static_ts")
    deep_labels, deep_probability = _load_average("dggru_tr")
    if not np.array_equal(labels, deep_labels):
        raise ValueError("Model prediction labels differ")
    strata = _strata(cohort)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for name, mask in strata.items():
            rows.append(
                {
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
            )
    output = {
        "n": int(len(cohort)),
        "note": (
            "Subgroup estimates reuse the prespecified outer-fold predictions; "
            "they are descriptive and are not models refitted within subgroups."
        ),
        "subgroups": rows,
    }
    path = RESULTS / "audits" / "mimic4_subgroup_analysis.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    pd.DataFrame(rows).drop(columns=["class_counts"]).to_csv(
        RESULTS / "tables" / "mimic4_subgroups.csv",
        index=False,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

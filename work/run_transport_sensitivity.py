"""Evaluation-only sensitivity analyses for frozen eICU transport predictions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import PROCESSED, RESULTS  # noqa: E402
from src.ami_dynamic.evaluation import (  # noqa: E402
    bootstrap_metric_ci,
    calibration_slope_intercept,
    point_metrics,
)
from work.run_external_calibration_common3 import (  # noqa: E402
    patient_safe_hospital_clusters,
)


def _metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
) -> dict:
    point = point_metrics(labels, probabilities)
    point.pop("cumulative_probabilities", None)
    return {
        "n": int(len(labels)),
        "outcome_counts": {
            str(int(key)): int(value)
            for key, value in pd.Series(labels).value_counts().sort_index().items()
        },
        "metrics": point,
        "macro_auc_ci": bootstrap_metric_ci(
            labels,
            probabilities,
            groups=groups,
            replicates=2000,
        ),
        "highest_class_auc_ci": bootstrap_metric_ci(
            labels,
            probabilities,
            metric="highest_class_auc",
            groups=groups,
            replicates=2000,
        ),
        "calibration": {
            "highest_class_y_3": calibration_slope_intercept(
                (labels == 3).astype(int), probabilities[:, 3]
            ),
            "high_need_y_ge_2": calibration_slope_intercept(
                (labels >= 2).astype(int), probabilities[:, 2:].sum(axis=1)
            ),
        },
    }


def main() -> None:
    cohort = pd.read_csv(PROCESSED / "eicu" / "cohort.csv")
    hospital_clusters, cluster_metadata = patient_safe_hospital_clusters(
        cohort["subject_key"].astype(str).to_numpy(),
        cohort["hospital_key"].astype(str).to_numpy(),
    )
    output = {
        "patient_safe_hospital_clusters": cluster_metadata,
        "one_patient_per_subject": {"models": {}},
        "exclude_other_disposition": {"models": {}},
    }
    order = np.arange(len(cohort))
    first_rows = (
        cohort.assign(_row=order)
        .sort_values(["subject_key", "stay_id"])
        .drop_duplicates("subject_key", keep="first")["_row"]
        .to_numpy()
    )
    non_other_rows = np.flatnonzero(
        ~cohort["disposition_raw"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("OTHER")
    )
    subsets = {
        "one_patient_per_subject": first_rows,
        "exclude_other_disposition": non_other_rows,
    }
    model_files = sorted(
        (RESULTS / "predictions").glob("eicu_*_transport.npz")
    )
    if not model_files:
        raise FileNotFoundError("Run external transport first")
    for path in model_files:
        model_name = path.stem.removeprefix("eicu_").removesuffix("_transport")
        with np.load(path, allow_pickle=True) as data:
            labels = data["labels"].astype(int)
            probabilities = data["probabilities"].astype(float)
            stay_id = data["stay_id"].astype(np.int64)
        if not np.array_equal(
            stay_id, cohort["stay_id"].to_numpy(dtype=np.int64)
        ):
            raise ValueError(f"{path.name}: cohort order differs")
        for name, rows in subsets.items():
            groups = (
                hospital_clusters[rows]
                if name == "exclude_other_disposition"
                else cohort.iloc[rows]["subject_key"].astype(str).to_numpy()
            )
            output[name]["models"][model_name] = _metrics(
                labels[rows],
                probabilities[rows],
                groups,
            )
    path = RESULTS / "metrics_transport_sensitivity.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

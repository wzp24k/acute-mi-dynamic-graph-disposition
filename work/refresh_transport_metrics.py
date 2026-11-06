"""Recompute transport metrics from frozen prediction caches.

This script is evaluation-only.  It exists so that the highest-class target
is always anchored to the model's highest probability column, even when a
bootstrap resample does not contain that class.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import PROCESSED, RESULTS  # noqa: E402
from src.ami_dynamic.evaluation import (  # noqa: E402
    bootstrap_binary_auc_ci,
    bootstrap_metric_ci,
    calibration_slope_intercept,
    point_metrics,
)


def refresh() -> dict:
    path = RESULTS / "metrics_transport.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    replicates = int(os.environ.get("AMI_BOOTSTRAP_REPLICATES", "2000"))
    refreshed = []
    expected_contract = (
        "five-class development outcome collapsed to four classes"
    )
    for dataset in ("mimic3", "eicu"):
        cohort = pd.read_csv(PROCESSED / dataset / "cohort.csv")
        for model, result in payload[dataset]["models"].items():
            prediction_path = (
                RESULTS / "predictions" / f"{dataset}_{model}_transport.npz"
            )
            with np.load(prediction_path, allow_pickle=True) as data:
                if "training_label_contract" not in data.files:
                    raise ValueError(
                        f"{prediction_path.name}: missing training-label "
                        "contract; rerun transport with current code"
                    )
                contract = str(data["training_label_contract"].item())
                if contract != expected_contract:
                    raise ValueError(
                        f"{prediction_path.name}: unexpected training-label "
                        f"contract {contract!r}"
                    )
                labels = data["labels"].astype(int)
                probabilities = data["probabilities"].astype(float)
                stay_id = data["stay_id"].astype(np.int64)
                if "subject_key" in data.files:
                    groups = data["subject_key"].astype(str)
                else:
                    groups = (
                        cohort.set_index("stay_id")
                        .loc[stay_id, "subject_key"]
                        .astype(str)
                        .to_numpy()
                    )
                if "strict_death_labels" in data.files:
                    strict_death = data["strict_death_labels"].astype(int)
                else:
                    aligned = cohort.set_index("stay_id").loc[stay_id]
                    strict_death = (
                        pd.to_numeric(
                            aligned["hospital_expire_flag"],
                            errors="coerce",
                        )
                        .fillna(0)
                        .astype(int)
                        .to_numpy()
                    )
            if len(stay_id) != len(labels):
                raise ValueError(f"{prediction_path.name}: row count mismatch")
            expected = cohort.set_index("stay_id").loc[stay_id]
            if not np.array_equal(
                labels,
                expected["outcome4"].astype(int).to_numpy(),
            ):
                raise ValueError(f"{prediction_path.name}: label order mismatch")

            point = point_metrics(labels, probabilities)
            point.pop("cumulative_probabilities", None)
            result["metrics"] = point
            result["training_label_contract"] = contract
            result["macro_auc_ci"] = bootstrap_metric_ci(
                labels,
                probabilities,
                groups=groups,
                replicates=replicates,
            )
            result["highest_class_auc_ci"] = bootstrap_metric_ci(
                labels,
                probabilities,
                metric="highest_class_auc",
                groups=groups,
                replicates=replicates,
            )
            result["strict_mortality_auc_ci"] = bootstrap_binary_auc_ci(
                strict_death,
                probabilities[:, -1],
                groups=groups,
                replicates=replicates,
            )
            result.pop("mortality_auc_ci", None)
            calibration = result.setdefault("calibration", {})
            calibration.pop("mortality", None)
            calibration["highest_class_y_3"] = (
                calibration_slope_intercept(
                    (labels == probabilities.shape[1] - 1).astype(int),
                    probabilities[:, -1],
                )
            )
            calibration["high_need_y_ge_2"] = (
                calibration_slope_intercept(
                    (labels >= 2).astype(int),
                    probabilities[:, 2:].sum(axis=1),
                )
            )
            refreshed.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "prediction": prediction_path.name,
                    "n": int(len(labels)),
                }
            )
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = {
        "status": "complete",
        "source": path.name,
        "bootstrap_replicates": replicates,
        "highest_class_target": "probabilities.shape[1] - 1",
        "training_label_contract": expected_contract,
        "aliases_removed": ["mortality_auc_ci", "calibration.mortality"],
        "models": refreshed,
    }
    audit_path = RESULTS / "audits" / "transport_metrics_refresh.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return payload


if __name__ == "__main__":
    refresh()

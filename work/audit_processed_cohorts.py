"""Independent audit of the derived AMI cohorts and sequence tensors."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import (  # noqa: E402
    CHANNELS,
    LAB_NAMES,
    PROCESSED,
    RESULTS,
    load_config,
    sha256_file,
)
from src.ami_dynamic.train_evaluate import _feature_columns  # noqa: E402


FORBIDDEN = {
    "outcome",
    "outcome4",
    "outcome_six",
    "disposition_raw",
    "death_at_hospital",
    "icu_out_hour",
    "death_hour_from_icu",
    "hospital_expire_flag",
}


def audit_dataset(name: str, config: dict) -> dict:
    directory = PROCESSED / name
    cohort = pd.read_csv(directory / "cohort.csv")
    sequence = np.load(directory / "sequences.npz")
    values = sequence["values"]
    masks = sequence["masks"]
    stay_ids = sequence["stay_id"]
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    flow = pd.read_csv(directory / "cohort_flow.csv")

    hash_checks = {
        filename: {
            "recorded": digest,
            "current": sha256_file(directory / filename),
            "match": digest == sha256_file(directory / filename),
        }
        for filename, digest in audit["files"].items()
    }
    channel_presence = masks.sum(axis=1) > 0
    channel_count = channel_presence.sum(axis=1)
    static_columns, full_columns = _feature_columns(cohort)
    lab_coverage = {
        lab: {
            "n": int(cohort[f"lab_{lab}_missing"].eq(0).sum()),
            "fraction": float(cohort[f"lab_{lab}_missing"].eq(0).mean()),
        }
        for lab in LAB_NAMES
    }
    carried_cells = int(
        (np.isfinite(values) & (masks == 0)).sum()
    )
    result = {
        "dataset": name,
        "n": int(len(cohort)),
        "unique_stay_id": int(cohort["stay_id"].nunique()),
        "sequence_alignment": bool(
            np.array_equal(
                stay_ids.astype(np.int64),
                cohort["stay_id"].to_numpy(dtype=np.int64),
            )
        ),
        "sequence_shape": list(values.shape),
        "outcome_counts": {
            str(int(key)): int(value)
            for key, value in cohort["outcome"].value_counts().sort_index().items()
        },
        "four_class_counts": {
            str(int(key)): int(value)
            for key, value in cohort["outcome4"].value_counts().sort_index().items()
        },
        "six_class_available": bool(cohort["outcome_six"].notna().any()),
        "unique_subject_key": int(cohort["subject_key"].nunique()),
        "subject_duplicates": int(
            len(cohort) - cohort["subject_key"].nunique()
        ),
        "unique_hospital_key": int(cohort["hospital_key"].nunique()),
        "flow": flow.to_dict(orient="records"),
        "channel_counts": {
            str(int(key)): int(value)
            for key, value in pd.Series(channel_count).value_counts().sort_index().items()
        },
        "at_least_4_of_8": int((channel_count >= 4).sum()),
        "per_channel": {
            channel: int(channel_presence[:, index].sum())
            for index, channel in enumerate(CHANNELS)
        },
        "observed_fraction": float(masks.mean()),
        "carried_cells": carried_cells,
        "nonfinite_values": int((~np.isfinite(values)).sum()),
        "lab_coverage": lab_coverage,
        "troponin_available": int(
            cohort["lab_troponin_missing"].eq(0).sum()
        ),
        "forbidden_features": sorted(
            FORBIDDEN.intersection(static_columns + full_columns)
        ),
        "n_static_features": len(static_columns),
        "n_full_features": len(full_columns),
        "hash_checks": hash_checks,
    }
    if name == "mimic4":
        landmark_cfg = config["landmarks_hours"]["primary"]
        result["landmark_hours"] = landmark_cfg
        result["duplicate_development_subjects"] = int(
            len(cohort) - cohort["subject_id"].nunique()
        )
    if name == "eicu":
        site_counts = cohort["hospital_key"].value_counts()
        result["sites"] = {
            "n_total": int(len(site_counts)),
            "n_at_least_30": int((site_counts >= 30).sum()),
            "at_least_30_admissions": int(site_counts[site_counts >= 30].sum()),
            "largest": site_counts.head(10).to_dict(),
        }
        raw_audit_path = PROJECT_ROOT / "work" / "audit_transport_coverage.json"
        if raw_audit_path.is_file():
            raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
            result["raw_nonnull_audit_at_least_4"] = int(
                raw_audit["eicu"]["at_least_4_of_8"]["n"]
            )
            result["valid_physiology_at_least_4"] = int(
                (channel_count >= 4).sum()
            )
            result["coverage_difference_note"] = (
                "The raw audit counted any non-null value. The analytic tensor "
                "also applies prespecified physiologic ranges, time windows, "
                "source precedence, and supported nurseCharting channels. "
                "The small difference reflects values removed by those rules."
            )
    return result


def main() -> None:
    config = load_config()
    (RESULTS / "audits").mkdir(parents=True, exist_ok=True)
    report = {
        name: audit_dataset(name, config)
        for name in ("mimic4", "mimic3", "eicu")
    }
    path = RESULTS / "audits" / "cohort_audit.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

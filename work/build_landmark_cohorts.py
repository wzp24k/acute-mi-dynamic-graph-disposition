"""Rebuild prespecified 6-hour and 72-hour MIMIC-IV landmark cohorts."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.build_cohorts import (  # noqa: E402
    MIMIC4_ROOT,
    _extract_mimic_labs,
    _mimic4_base,
    _mimic4_events,
    connect_duckdb,
)
from src.ami_dynamic.common import PROCESSED, save_processed  # noqa: E402


def build(landmark_hours: int) -> dict:
    if landmark_hours not in {6, 72}:
        raise ValueError("Secondary landmark must be 6 or 72 hours")
    con = connect_duckdb()
    try:
        cohort, metadata = _mimic4_base(con, landmark_hours=landmark_hours)
        hourly = _mimic4_events(
            con, cohort, landmark_hours=landmark_hours
        )
        labs = _extract_mimic_labs(
            con,
            cohort,
            MIMIC4_ROOT / "hosp/labevents.csv.gz",
            landmark_hours=landmark_hours,
        )
    finally:
        con.close()

    dataset_name = f"mimic4_{landmark_hours}h"
    save_processed(
        dataset_name,
        cohort,
        hourly,
        labs,
        metadata["flow"],
        max_hour=landmark_hours,
    )
    audit_path = PROCESSED / dataset_name / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["landmark_hours"] = landmark_hours
    audit["eligibility"] = {
        "alive_at_landmark": True,
        "still_in_icu_at_landmark": True,
        "first_icu_stay_per_hospital_admission": True,
        "one_hospitalization_per_patient": True,
        "post_landmark_values_excluded": True,
    }
    audit["sequence_hours"] = landmark_hours
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(
        f"[{dataset_name}] n={audit['n']} "
        f"shape={audit['sequence_shape']} "
        f"outcomes={audit['outcome_counts']}",
        flush=True,
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--landmarks", nargs="+", type=int, default=[6, 72]
    )
    args = parser.parse_args()
    os.environ.setdefault("AMI_DUCKDB_THREADS", "12")
    os.environ.setdefault("AMI_DUCKDB_MEMORY", "10GB")
    results = [build(hours) for hours in args.landmarks]
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

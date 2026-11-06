"""Derive AMI position and subtype flags without changing the processed cohort."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.build_cohorts import (  # noqa: E402
    MIMIC3_ROOT,
    MIMIC4_ROOT,
    connect_duckdb,
    create_view,
)
from src.ami_dynamic.common import PROCESSED  # noqa: E402


def _subtype(code: str, version: int) -> str:
    code = str(code or "").upper().replace(".", "")
    if version == 10:
        if code.startswith(("I210", "I211", "I212", "I213")):
            return "STEMI"
        if code.startswith("I214"):
            return "NSTEMI"
        if code.startswith("I21A1"):
            return "Type 2 MI"
        if code.startswith("I219"):
            return "Unspecified"
    else:
        if code.startswith(("4100", "4101", "4102", "4103", "4104", "4105", "4106")):
            return "STEMI"
        if code.startswith("4107"):
            return "NSTEMI"
        if code.startswith("4109"):
            return "Unspecified"
    return "Other"


def derive_mimic4() -> pd.DataFrame:
    con = connect_duckdb()
    create_view(
        con,
        "diagnoses",
        MIMIC4_ROOT / "hosp/diagnoses_icd.csv.gz",
    )
    frame = con.execute(
        """
        SELECT hadm_id, seq_num, icd_version, icd_code
        FROM diagnoses
        WHERE (icd_version = 10 AND icd_code LIKE 'I21%')
           OR (icd_version = 9 AND icd_code LIKE '410%')
        ORDER BY hadm_id, seq_num
        """
    ).fetchdf()
    frame["subtype"] = [
        _subtype(code, int(version))
        for code, version in zip(frame["icd_code"], frame["icd_version"])
    ]
    grouped = (
        frame.groupby("hadm_id")
        .agg(
            ami_primary_position=("seq_num", lambda values: int(values.min() == 1)),
            ami_type2=("subtype", lambda values: int("Type 2 MI" in set(values))),
            ami_subtype=("subtype", lambda values: set(values)),
            ami_diagnosis_count=("icd_code", "size"),
        )
        .reset_index()
    )
    priority = ["STEMI", "NSTEMI", "Type 2 MI", "Unspecified", "Other"]
    grouped["ami_subtype"] = grouped["ami_subtype"].map(
        lambda values: next(value for value in priority if value in values)
    )
    cohort = pd.read_csv(PROCESSED / "mimic4" / "cohort.csv", usecols=["stay_id", "hadm_id"])
    return cohort.merge(grouped, on="hadm_id", how="left", validate="one_to_one")


def derive_mimic3() -> pd.DataFrame:
    con = connect_duckdb()
    create_view(
        con,
        "diagnoses",
        MIMIC3_ROOT / "DIAGNOSES_ICD.csv.gz",
    )
    frame = con.execute(
        """
        SELECT "HADM_ID" AS hadm_id,
               "SEQ_NUM" AS seq_num,
               CAST("ICD9_CODE" AS VARCHAR) AS icd_code
        FROM diagnoses
        WHERE CAST("ICD9_CODE" AS VARCHAR) LIKE '410%'
        ORDER BY hadm_id, seq_num
        """
    ).fetchdf()
    frame["subtype"] = [
        _subtype(code, 9) for code in frame["icd_code"]
    ]
    grouped = (
        frame.groupby("hadm_id")
        .agg(
            ami_primary_position=("seq_num", lambda values: int(values.min() == 1)),
            ami_type2=("subtype", lambda values: 0),
            ami_subtype=("subtype", lambda values: set(values)),
            ami_diagnosis_count=("icd_code", "size"),
        )
        .reset_index()
    )
    priority = ["STEMI", "NSTEMI", "Unspecified", "Other"]
    grouped["ami_subtype"] = grouped["ami_subtype"].map(
        lambda values: next(value for value in priority if value in values)
    )
    cohort = pd.read_csv(PROCESSED / "mimic3" / "cohort.csv", usecols=["stay_id", "hadm_id"])
    return cohort.merge(grouped, on="hadm_id", how="left", validate="one_to_one")


def main() -> None:
    for dataset, derive in (("mimic4", derive_mimic4), ("mimic3", derive_mimic3)):
        frame = derive()
        path = PROCESSED / dataset / "diagnosis_flags.csv"
        frame.to_csv(path, index=False)
        print(
            dataset,
            "n=",
            len(frame),
            "primary=",
            int(frame["ami_primary_position"].fillna(0).sum()),
            "type2=",
            int(frame["ami_type2"].fillna(0).sum()),
            frame["ami_subtype"].value_counts().to_dict(),
            flush=True,
        )


if __name__ == "__main__":
    main()

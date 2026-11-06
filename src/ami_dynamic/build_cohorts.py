"""Build harmonized AMI cohorts, hourly sequences, and static features."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .common import (
    EICU_DISPOSITION_4,
    EICU_LABS,
    LAB_NAMES,
    LAB_RANGES,
    MIMIC4_LABS,
    PROCESSED,
    SOURCE_ROOT,
    add_demographic_flags,
    clean_events,
    ensure_dirs,
    five_to_four,
    hourly_last_by_priority,
    map_five_level,
    map_six_level,
    parse_eicu_age,
    save_processed,
)


MIMIC4_ROOT = SOURCE_ROOT / "mimic-iv-3.1"
MIMIC3_ROOT = (
    SOURCE_ROOT
    / "mimic-iii-clinical-database-1.4_1"
    / "mimic-iii-clinical-database-1.4"
)
EICU_ROOT = SOURCE_ROOT / "eicu"


def connect_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={int(os.environ.get('AMI_DUCKDB_THREADS', '6'))}")
    con.execute(
        f"SET memory_limit='{os.environ.get('AMI_DUCKDB_MEMORY', '8GB')}'"
    )
    return con


def create_view(
    con: duckdb.DuckDBPyConnection,
    name: str,
    path: Path,
    *,
    all_varchar: bool = False,
) -> None:
    suffix = ", all_varchar=true" if all_varchar else ""
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_csv_auto('{path.as_posix()}', header=true{suffix})"
    )


def _as_float(value: object) -> float:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(result) if pd.notna(result) else np.nan


def _attach_outcomes(
    frame: pd.DataFrame,
    dataset: str,
    disposition_col: str,
    expire_col: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["disposition_raw"] = out[disposition_col].fillna("").astype(str)
    hospital_death = (
        pd.to_numeric(out[expire_col], errors="coerce").fillna(0).eq(1)
    )
    if dataset in {"mimic4", "mimic3"}:
        mapped_five = pd.Series(
            [
                map_five_level(value, dataset)
                for value in out["disposition_raw"]
            ],
            index=out.index,
            dtype=float,
        )
        mapped_six = pd.Series(
            [
                map_six_level(value, dataset)
                for value in out["disposition_raw"]
            ],
            index=out.index,
            dtype=float,
        )
        out["outcome"] = mapped_five.mask(
            hospital_death & mapped_five.notna(), 4.0
        )
        out["outcome_six"] = mapped_six.mask(
            hospital_death & mapped_six.notna(), 5.0
        )
        out["outcome4"] = out["outcome"].map(five_to_four)
    else:
        text = out["disposition_raw"].str.strip().str.upper()
        mapped = text.map(EICU_DISPOSITION_4).astype(float)
        out["outcome"] = mapped.mask(hospital_death, 3.0)
        out["outcome_six"] = np.nan
        out["outcome4"] = out["outcome"]
    out["death_at_hospital"] = (
        hospital_death
        | out["disposition_raw"]
        .str.upper()
        .isin({"DIED", "DEAD/EXPIRED", "DEATH", "EXPIRED", "HOSPICE"})
    ).astype(float)
    return out


def _flow_rows(
    dataset: str,
    source_n: int,
    first_unit_n: int,
    adult_n: int,
    landmark_n: int,
    mapped_n: int,
    final_n: int,
    landmark_hours: int = 24,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("source_ami_admissions_or_units", source_n),
            ("first_icu_or_unit", first_unit_n),
            ("adult", adult_n),
            (f"alive_and_in_icu_at_{int(landmark_hours)}h", landmark_n),
            ("mapped_disposition", mapped_n),
            ("deduplicated_analysis_unit", final_n),
        ],
        columns=["reason_code", "n"],
    ).assign(dataset=dataset)


def _gcs_from_components(
    frame: pd.DataFrame,
    id_col: str,
    time_col: str,
) -> pd.DataFrame:
    components = frame[
        frame["itemid"].isin([220739, 223900, 223901])
    ].copy()
    if components.empty:
        return pd.DataFrame(columns=[id_col, time_col, "gcs"])
    components["value"] = pd.to_numeric(components["valuenum"], errors="coerce")
    ranges = {220739: (1, 4), 223900: (1, 5), 223901: (1, 6)}
    for itemid, (low, high) in ranges.items():
        mask = components["itemid"].eq(itemid)
        components.loc[mask, "value"] = components.loc[mask, "value"].where(
            components.loc[mask, "value"].between(low, high)
        )
    pivot = components.pivot_table(
        index=[id_col, time_col],
        columns="itemid",
        values="value",
        aggfunc="last",
    )
    pivot = pivot.reindex(columns=[220739, 223900, 223901])
    pivot = pivot[pivot.notna().all(axis=1)]
    return (
        pivot.sum(axis=1)
        .rename("gcs")
        .reset_index()
    )


def _assemble_event_long(
    vitals: pd.DataFrame,
    gcs: pd.DataFrame,
) -> pd.DataFrame:
    parts = []
    if not vitals.empty:
        parts.append(
            vitals[["stay_id", "time_hour", "channel", "value", "priority", "event_time"]]
        )
    if not gcs.empty:
        gcs_part = gcs.copy()
        gcs_part["channel"] = "gcs"
        gcs_part["value"] = gcs_part["gcs"]
        gcs_part["priority"] = 0
        gcs_part["event_time"] = gcs_part["time_hour"]
        parts.append(
            gcs_part[["stay_id", "time_hour", "channel", "value", "priority", "event_time"]]
        )
    if not parts:
        return pd.DataFrame(
            columns=[
                "stay_id",
                "time_hour",
                "channel",
                "value",
                "priority",
                "event_time",
            ]
        )
    return pd.concat(parts, ignore_index=True)


def _mimic4_base(
    con: duckdb.DuckDBPyConnection,
    landmark_hours: int = 24,
) -> tuple[pd.DataFrame, dict]:
    for view, rel in {
        "admissions": "hosp/admissions.csv.gz",
        "patients": "hosp/patients.csv.gz",
        "diagnoses": "hosp/diagnoses_icd.csv.gz",
        "icustays": "icu/icustays.csv.gz",
    }.items():
        create_view(con, view, MIMIC4_ROOT / rel)
    source_n = int(
        con.execute(
            """
            SELECT COUNT(DISTINCT hadm_id)
            FROM diagnoses
            WHERE (icd_version = 10 AND icd_code LIKE 'I21%')
               OR (icd_version = 9 AND icd_code LIKE '410%')
            """
        ).fetchone()[0]
    )
    first_n = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT i.stay_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY i.hadm_id ORDER BY i.intime, i.stay_id
                       ) AS rn
                FROM icustays i
                WHERE EXISTS (
                    SELECT 1 FROM diagnoses d
                    WHERE d.hadm_id = i.hadm_id
                      AND ((d.icd_version = 10 AND d.icd_code LIKE 'I21%')
                           OR (d.icd_version = 9 AND d.icd_code LIKE '410%'))
                )
            ) WHERE rn = 1
            """
        ).fetchone()[0]
    )
    adult_n = int(
        con.execute(
            """
            WITH ranked AS (
                SELECT i.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY i.hadm_id ORDER BY i.intime, i.stay_id
                       ) AS rn
                FROM icustays i
                WHERE EXISTS (
                    SELECT 1 FROM diagnoses d
                    WHERE d.hadm_id = i.hadm_id
                      AND ((d.icd_version = 10 AND d.icd_code LIKE 'I21%')
                           OR (d.icd_version = 9 AND d.icd_code LIKE '410%'))
                )
            )
            SELECT COUNT(*)
            FROM ranked r
            JOIN admissions a ON a.hadm_id = r.hadm_id
            JOIN patients p ON p.subject_id = r.subject_id
            WHERE r.rn = 1
              AND p.anchor_age + date_part('year', r.intime) - p.anchor_year >= 18
            """
        ).fetchone()[0]
    )
    frame = con.execute(
        """
        WITH ami_hadm AS (
            SELECT DISTINCT hadm_id
            FROM diagnoses
            WHERE (icd_version = 10 AND icd_code LIKE 'I21%')
               OR (icd_version = 9 AND icd_code LIKE '410%')
        ),
        ranked_icu AS (
            SELECT i.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY i.hadm_id ORDER BY i.intime, i.stay_id
                   ) AS icu_rank
            FROM icustays i
            JOIN ami_hadm a USING (hadm_id)
        ),
        candidates AS (
            SELECT r.stay_id, r.hadm_id, r.subject_id, r.intime, r.outtime,
                   r.first_careunit,
                   a.admittime, a.dischtime, a.deathtime, a.race,
                   a.admission_type, a.admission_location AS admission_source,
                   a.discharge_location, a.hospital_expire_flag,
                   p.gender AS sex,
                   p.anchor_age + date_part('year', r.intime) - p.anchor_year AS age,
                   date_diff('minute', r.intime, r.outtime) / 60.0 AS icu_out_hour,
                   CASE WHEN a.deathtime IS NOT NULL
                        THEN date_diff('minute', r.intime, a.deathtime) / 60.0
                        ELSE NULL END AS death_hour_from_icu
            FROM ranked_icu r
            JOIN admissions a ON a.hadm_id = r.hadm_id
            JOIN patients p ON p.subject_id = r.subject_id
            WHERE r.icu_rank = 1
        )
        SELECT *
        FROM candidates
        """
    ).fetchdf()
    frame.columns = [column.lower() for column in frame.columns]
    frame["dataset"] = "mimic4"
    frame["subject_key"] = "mimic4_" + frame["subject_id"].astype(str)
    frame["hospital_key"] = "mimic4"
    frame = _attach_outcomes(
        frame, "mimic4", "discharge_location", "hospital_expire_flag"
    )
    frame = frame[
        pd.to_numeric(frame["age"], errors="coerce").ge(18)
    ].copy()
    landmark = (
        pd.to_numeric(frame["icu_out_hour"], errors="coerce").ge(landmark_hours)
        & (
            pd.to_numeric(frame["death_hour_from_icu"], errors="coerce").isna()
            | pd.to_numeric(frame["death_hour_from_icu"], errors="coerce").gt(
                landmark_hours
            )
        )
    )
    landmark_n = int(landmark.sum())
    mapped_n = int((landmark & frame["outcome"].notna()).sum())
    frame = frame[landmark & frame["outcome"].notna()].copy()
    frame = frame.sort_values(["intime", "stay_id"]).drop_duplicates(
        "subject_id", keep="first"
    )
    frame = add_demographic_flags(frame)
    flow = _flow_rows(
        "mimic4",
        source_n,
        first_n,
        adult_n,
        landmark_n,
        mapped_n,
        len(frame),
        landmark_hours=landmark_hours,
    )
    return frame.reset_index(drop=True), {"flow": flow}


def _mimic3_base(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    for view, rel in {
        "admissions": "ADMISSIONS.csv.gz",
        "patients": "PATIENTS.csv.gz",
        "diagnoses": "DIAGNOSES_ICD.csv.gz",
        "icustays": "ICUSTAYS.csv.gz",
    }.items():
        create_view(con, view, MIMIC3_ROOT / rel)
    source_n = int(
        con.execute(
            "SELECT COUNT(DISTINCT hadm_id) FROM diagnoses WHERE icd9_code LIKE '410%'"
        ).fetchone()[0]
    )
    first_n = int(
        con.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT i.icustay_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY i.hadm_id ORDER BY i.intime, i.icustay_id
                       ) AS rn
                FROM icustays i
                WHERE EXISTS (
                    SELECT 1 FROM diagnoses d
                    WHERE d.hadm_id = i.hadm_id AND d.icd9_code LIKE '410%'
                )
            ) WHERE rn = 1
            """
        ).fetchone()[0]
    )
    adult_n = int(
        con.execute(
            """
            WITH ranked AS (
                SELECT i.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY i.hadm_id ORDER BY i.intime, i.icustay_id
                       ) AS rn
                FROM icustays i
                WHERE EXISTS (
                    SELECT 1 FROM diagnoses d
                    WHERE d.hadm_id = i.hadm_id AND d.icd9_code LIKE '410%'
                )
            )
            SELECT COUNT(*)
            FROM ranked r
            JOIN patients p ON p.subject_id = r.subject_id
            WHERE r.rn = 1
              AND date_diff('year', p.dob, r.intime) >= 18
            """
        ).fetchone()[0]
    )
    frame = con.execute(
        """
        WITH ami_hadm AS (
            SELECT DISTINCT hadm_id FROM diagnoses WHERE icd9_code LIKE '410%'
        ),
        ranked AS (
            SELECT i.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY i.hadm_id ORDER BY i.intime, i.icustay_id
                   ) AS icu_rank
            FROM icustays i
            JOIN ami_hadm USING (hadm_id)
        )
        SELECT r.icustay_id AS stay_id, r.hadm_id, r.subject_id,
               r.intime, r.outtime, r.first_careunit,
               a.admittime, a.dischtime, a.deathtime, a.ethnicity AS race,
               a.admission_type, a.admission_location AS admission_source,
               a.discharge_location, a.hospital_expire_flag,
               p.gender AS sex,
               CASE
                   WHEN date_diff('year', p.dob, a.admittime) > 89
                   THEN 90
                   ELSE date_diff('year', p.dob, a.admittime)
               END AS age,
               date_diff('minute', r.intime, r.outtime) / 60.0 AS icu_out_hour,
               CASE WHEN a.deathtime IS NOT NULL
                    THEN date_diff('minute', r.intime, a.deathtime) / 60.0
                    ELSE NULL END AS death_hour_from_icu
        FROM ranked r
        JOIN admissions a ON a.hadm_id = r.hadm_id
        JOIN patients p ON p.subject_id = r.subject_id
        WHERE r.icu_rank = 1
        """
    ).fetchdf()
    frame.columns = [column.lower() for column in frame.columns]
    frame["dataset"] = "mimic3"
    frame["subject_key"] = "mimic3_" + frame["subject_id"].astype(str)
    frame["hospital_key"] = "mimic3"
    frame = _attach_outcomes(
        frame, "mimic3", "discharge_location", "hospital_expire_flag"
    )
    frame = frame[
        pd.to_numeric(frame["age"], errors="coerce").ge(18)
    ].copy()
    landmark = (
        pd.to_numeric(frame["icu_out_hour"], errors="coerce").ge(24)
        & (
            pd.to_numeric(frame["death_hour_from_icu"], errors="coerce").isna()
            | pd.to_numeric(frame["death_hour_from_icu"], errors="coerce").gt(24)
        )
    )
    landmark_n = int(landmark.sum())
    frame = frame[landmark & frame["outcome"].notna()].copy()
    mapped_n = len(frame)
    frame = add_demographic_flags(frame)
    flow = _flow_rows(
        "mimic3",
        source_n,
        first_n,
        adult_n,
        landmark_n,
        mapped_n,
        len(frame),
    )
    return frame.reset_index(drop=True), {"flow": flow}


def _eicu_base(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    create_view(con, "patient", EICU_ROOT / "patient.csv.gz")
    create_view(con, "diagnosis", EICU_ROOT / "diagnosis.csv.gz")
    source_n = int(
        con.execute(
            """
            SELECT COUNT(DISTINCT patientunitstayid)
            FROM diagnosis
            WHERE (icd9code LIKE '410%'
                   OR diagnosisstring ILIKE '%acute myocardial infarction%')
              AND COALESCE(diagnosisstring, '') NOT ILIKE '%ruled out%'
              AND COALESCE(diagnosisstring, '') NOT ILIKE '%history of%'
            """
        ).fetchone()[0]
    )
    first_n = int(
        con.execute(
            """
            WITH ami AS (
                SELECT DISTINCT patientunitstayid
                FROM diagnosis
                WHERE (icd9code LIKE '410%'
                       OR diagnosisstring ILIKE '%acute myocardial infarction%')
                  AND COALESCE(diagnosisstring, '') NOT ILIKE '%ruled out%'
                  AND COALESCE(diagnosisstring, '') NOT ILIKE '%history of%'
            )
            SELECT COUNT(*)
            FROM (
                SELECT p.patientunitstayid,
                       ROW_NUMBER() OVER (
                           PARTITION BY p.patienthealthsystemstayid
                           ORDER BY p.hospitaladmitoffset, p.patientunitstayid
                       ) AS rn
                FROM patient p JOIN ami USING (patientunitstayid)
            ) WHERE rn = 1
            """
        ).fetchone()[0]
    )
    frame = con.execute(
        """
        WITH ami AS (
            SELECT DISTINCT patientunitstayid
            FROM diagnosis
            WHERE (icd9code LIKE '410%'
                   OR diagnosisstring ILIKE '%acute myocardial infarction%')
              AND COALESCE(diagnosisstring, '') NOT ILIKE '%ruled out%'
              AND COALESCE(diagnosisstring, '') NOT ILIKE '%history of%'
        ),
        ranked AS (
            SELECT p.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.patienthealthsystemstayid
                       ORDER BY p.hospitaladmitoffset, p.patientunitstayid
                   ) AS unit_rank
            FROM patient p
            JOIN ami USING (patientunitstayid)
        )
        SELECT patientunitstayid AS stay_id,
               uniquepid AS subject_id,
               patienthealthsystemstayid,
               hospitalid, unitvisitnumber, unittype AS first_careunit,
               gender AS sex, age, ethnicity AS race,
               hospitaladmitsource AS admission_source,
               unitadmitsource, hospitaldischargeyear,
               unitdischargeoffset, hospitaldischargeoffset,
               unitdischargestatus, hospitaldischargestatus,
               hospitaldischargelocation AS discharge_location
        FROM ranked
        WHERE unit_rank = 1
        """
    ).fetchdf()
    frame.columns = [column.lower() for column in frame.columns]
    frame["age"] = frame["age"].map(parse_eicu_age)
    frame["admission_type"] = frame["unitadmitsource"].fillna("").astype(str)
    frame["admission_source"] = frame["admission_source"].fillna("").astype(str)
    frame["icu_out_hour"] = pd.to_numeric(
        frame["unitdischargeoffset"], errors="coerce"
    ) / 60.0
    frame["death_hour_from_icu"] = np.where(
        frame["unitdischargestatus"].fillna("").astype(str).str.lower().eq("expired"),
        frame["icu_out_hour"],
        np.nan,
    )
    frame["dataset"] = "eicu"
    frame["subject_key"] = "eicu_" + frame["subject_id"].astype(str)
    frame["hospital_key"] = "eicu_" + frame["hospitalid"].astype(str)
    frame["hospital_expire_flag"] = (
        frame["hospitaldischargestatus"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("expired")
        .astype(int)
    )
    frame = _attach_outcomes(
        frame, "eicu", "discharge_location", "hospital_expire_flag"
    )
    adult_n = int(pd.to_numeric(frame["age"], errors="coerce").ge(18).sum())
    frame = frame[
        pd.to_numeric(frame["age"], errors="coerce").ge(18)
    ].copy()
    landmark = pd.to_numeric(frame["icu_out_hour"], errors="coerce").ge(24)
    landmark_n = int(landmark.sum())
    frame = frame[landmark & frame["outcome"].notna()].copy()
    mapped_n = len(frame)
    frame = add_demographic_flags(frame)
    flow = _flow_rows(
        "eicu",
        source_n,
        first_n,
        adult_n,
        landmark_n,
        mapped_n,
        len(frame),
    )
    return frame.reset_index(drop=True), {"flow": flow}


def _extract_mimic_labs(
    con: duckdb.DuckDBPyConnection,
    cohort: pd.DataFrame,
    labevents: Path,
    *,
    landmark_hours: int = 24,
) -> pd.DataFrame:
    create_view(con, "labevents", labevents)
    con.register(
        "cohort_labs",
        cohort[["stay_id", "hadm_id", "intime"]].copy(),
    )
    item_map = {item: name for name, item in MIMIC4_LABS.items()}
    items = ",".join(str(item) for item in item_map)
    frame = con.execute(
        f"""
        SELECT c.stay_id AS stay_id, l.itemid AS itemid,
               l.charttime AS charttime, l.valuenum AS valuenum,
               date_diff('second', c.intime, l.charttime) / 3600.0 AS time_hour
        FROM labevents l
        JOIN cohort_labs c USING (hadm_id)
        WHERE l.itemid IN ({items})
          AND l.charttime >= c.intime
          AND l.charttime < c.intime + INTERVAL {int(landmark_hours)} HOUR
          AND l.valuenum IS NOT NULL
        """
    ).fetchdf()
    if frame.empty:
        return pd.DataFrame(
            columns=["stay_id", "lab", "value", "time_hour", "assay_t"]
        )
    frame["itemid"] = pd.to_numeric(frame["itemid"], errors="coerce")
    raw_name = frame["itemid"].map(item_map)
    frame["assay_t"] = np.where(raw_name.eq("troponin_t"), 1, 0)
    frame["lab"] = raw_name.replace(
        {"troponin_i": "troponin", "troponin_t": "troponin"}
    )
    frame["value"] = pd.to_numeric(frame["valuenum"], errors="coerce")
    for lab, (low, high) in LAB_RANGES.items():
        mask = frame["lab"].eq(lab)
        frame.loc[mask, "value"] = frame.loc[mask, "value"].where(
            frame.loc[mask, "value"].between(low, high)
        )
    return (
        frame.dropna(subset=["lab", "value"])
        .sort_values(["stay_id", "time_hour", "lab"])
        [["stay_id", "lab", "value", "time_hour", "assay_t"]]
        .reset_index(drop=True)
    )


def _extract_eicu_labs(
    con: duckdb.DuckDBPyConnection,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    create_view(con, "lab", EICU_ROOT / "lab.csv.gz")
    con.register("cohort_labs", cohort[["stay_id"]].copy())
    raw_names = [name for names in EICU_LABS.values() for name in names]
    quoted = ",".join("'" + name.replace("'", "''") + "'" for name in raw_names)
    frame = con.execute(
        f"""
        SELECT c.stay_id, l.labname, l.labresultoffset / 60.0 AS time_hour,
               TRY_CAST(l.labresult AS DOUBLE) AS valuenum
        FROM lab l
        JOIN cohort_labs c ON l.patientunitstayid = c.stay_id
        WHERE l.labresultoffset >= 0
          AND l.labresultoffset < 1440
          AND l.labname IN ({quoted})
        """
    ).fetchdf()
    if frame.empty:
        return pd.DataFrame(
            columns=["stay_id", "lab", "value", "time_hour", "assay_t"]
        )
    name_map = {
        raw: lab_name
        for lab_name, names in EICU_LABS.items()
        for raw in names
    }
    frame["raw_lab"] = frame["labname"].map(name_map)
    frame["assay_t"] = frame["raw_lab"].eq("troponin_t").astype(int)
    frame["lab"] = frame["raw_lab"].replace(
        {"troponin_i": "troponin", "troponin_t": "troponin"}
    )
    frame["value"] = pd.to_numeric(frame["valuenum"], errors="coerce")
    for lab, (low, high) in LAB_RANGES.items():
        mask = frame["lab"].eq(lab)
        frame.loc[mask, "value"] = frame.loc[mask, "value"].where(
            frame.loc[mask, "value"].between(low, high)
        )
    return (
        frame.dropna(subset=["lab", "value"])
        .sort_values(["stay_id", "time_hour", "lab"])
        [["stay_id", "lab", "value", "time_hour", "assay_t"]]
        .reset_index(drop=True)
    )


def _mimic4_events(
    con: duckdb.DuckDBPyConnection,
    cohort: pd.DataFrame,
    *,
    landmark_hours: int = 24,
) -> pd.DataFrame:
    create_view(con, "chartevents", MIMIC4_ROOT / "icu/chartevents.csv.gz")
    con.register("cohort_chart", cohort[["stay_id", "intime"]].copy())
    chart = con.execute(
        f"""
        SELECT c.stay_id, ce.charttime, ce.itemid, ce.valuenum, ce.value,
               date_diff('second', c.intime, ce.charttime) / 3600.0 AS time_hour
        FROM chartevents ce
        JOIN cohort_chart c USING (stay_id)
        WHERE ce.itemid IN (
            220045, 220210, 224690, 220277,
            220050, 220051, 220052,
            220179, 220180, 220181,
            223761, 223762, 220739, 223900, 223901
        )
          AND ce.charttime >= c.intime
          AND ce.charttime < c.intime + INTERVAL {int(landmark_hours)} HOUR
        """
    ).fetchdf()
    if chart.empty:
        return chart
    chart["itemid"] = pd.to_numeric(chart["itemid"], errors="coerce")
    chart["valuenum"] = pd.to_numeric(chart["valuenum"], errors="coerce")
    gcs = _gcs_from_components(chart, "stay_id", "charttime")
    gcs = gcs.merge(
        cohort[["stay_id", "intime"]], on="stay_id", how="left"
    )
    gcs["time_hour"] = (
        pd.to_datetime(gcs["charttime"]) - pd.to_datetime(gcs["intime"])
    ).dt.total_seconds() / 3600.0
    item_map = {
        220045: ("hr", 0),
        220210: ("rr", 0),
        224690: ("rr", 1),
        220277: ("spo2", 0),
        220050: ("sbp", 0),
        220179: ("sbp", 1),
        220051: ("dbp", 0),
        220180: ("dbp", 1),
        220052: ("mbp", 0),
        220181: ("mbp", 1),
        223762: ("temp", 0),
        223761: ("temp", 1),
    }
    work = chart[chart["itemid"].isin(item_map)].copy()
    work["channel"] = work["itemid"].map(lambda x: item_map[int(x)][0])
    work["priority"] = work["itemid"].map(lambda x: item_map[int(x)][1])
    work["value"] = work["valuenum"]
    fahrenheit = work["itemid"].eq(223761)
    work.loc[fahrenheit, "value"] = (
        pd.to_numeric(work.loc[fahrenheit, "valuenum"], errors="coerce") - 32.0
    ) * 5.0 / 9.0
    work["event_time"] = pd.to_datetime(work["charttime"], errors="coerce").astype("int64")
    long = _assemble_event_long(work, gcs)
    long = clean_events(long, max_hour=landmark_hours)
    return hourly_last_by_priority(long)


def _mimic3_events(
    con: duckdb.DuckDBPyConnection,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    create_view(
        con, "chartevents", MIMIC3_ROOT / "CHARTEVENTS.csv.gz", all_varchar=True
    )
    con.register("cohort_chart", cohort[["stay_id", "intime"]].copy())
    chart = con.execute(
        """
        SELECT c.stay_id, TRY_CAST(ce.charttime AS TIMESTAMP) AS charttime,
               TRY_CAST(ce.itemid AS INTEGER) AS itemid,
               TRY_CAST(ce.valuenum AS DOUBLE) AS valuenum,
               ce.value,
               date_diff(
                   'second', c.intime, TRY_CAST(ce.charttime AS TIMESTAMP)
               ) / 3600.0 AS time_hour
        FROM chartevents ce
        JOIN cohort_chart c ON TRY_CAST(ce.icustay_id AS BIGINT) = c.stay_id
        WHERE TRY_CAST(ce.itemid AS INTEGER) IN (
            211, 220045, 618, 220210, 224690, 646, 220277,
            51, 455, 220050, 220179,
            8368, 8440, 8441, 220051, 220180,
            52, 456, 220052, 220181,
            676, 677, 223762, 678, 679, 223761,
            198, 220739, 223900, 223901
        )
          AND TRY_CAST(ce.charttime AS TIMESTAMP) >= c.intime
          AND TRY_CAST(ce.charttime AS TIMESTAMP) < c.intime + INTERVAL 24 HOUR
        """
    ).fetchdf()
    if chart.empty:
        return chart
    chart["itemid"] = pd.to_numeric(chart["itemid"], errors="coerce")
    chart["valuenum"] = pd.to_numeric(chart["valuenum"], errors="coerce")
    components = _gcs_from_components(chart, "stay_id", "charttime")
    totals = chart[chart["itemid"].eq(198)].copy()
    if not totals.empty:
        totals = totals.rename(columns={"valuenum": "gcs"})[
            ["stay_id", "charttime", "gcs"]
        ]
        totals["gcs"] = pd.to_numeric(totals["gcs"], errors="coerce").where(
            lambda values: values.between(3, 15)
        )
        gcs = pd.concat([components, totals], ignore_index=True).dropna(
            subset=["gcs"]
        )
    else:
        gcs = components
    gcs = gcs.merge(cohort[["stay_id", "intime"]], on="stay_id", how="left")
    gcs["time_hour"] = (
        pd.to_datetime(gcs["charttime"]) - pd.to_datetime(gcs["intime"])
    ).dt.total_seconds() / 3600.0
    item_map = {
        211: ("hr", 0),
        220045: ("hr", 1),
        618: ("rr", 0),
        220210: ("rr", 1),
        224690: ("rr", 2),
        646: ("spo2", 0),
        220277: ("spo2", 1),
        51: ("sbp", 0),
        455: ("sbp", 1),
        220050: ("sbp", 2),
        220179: ("sbp", 3),
        8368: ("dbp", 0),
        8440: ("dbp", 1),
        8441: ("dbp", 2),
        220051: ("dbp", 3),
        220180: ("dbp", 4),
        52: ("mbp", 0),
        456: ("mbp", 1),
        220052: ("mbp", 2),
        220181: ("mbp", 3),
        676: ("temp", 0),
        677: ("temp", 1),
        223762: ("temp", 2),
        678: ("temp", 3),
        679: ("temp", 4),
        223761: ("temp", 5),
    }
    work = chart[chart["itemid"].isin(item_map)].copy()
    work["channel"] = work["itemid"].map(lambda x: item_map[int(x)][0])
    work["priority"] = work["itemid"].map(lambda x: item_map[int(x)][1])
    work["value"] = work["valuenum"]
    fahrenheit = work["itemid"].isin([678, 679, 223761])
    work.loc[fahrenheit, "value"] = (
        pd.to_numeric(work.loc[fahrenheit, "valuenum"], errors="coerce") - 32.0
    ) * 5.0 / 9.0
    work["event_time"] = pd.to_datetime(work["charttime"], errors="coerce").astype("int64")
    long = _assemble_event_long(work, gcs)
    long = clean_events(long)
    return hourly_last_by_priority(long)


def _eicu_events(
    con: duckdb.DuckDBPyConnection,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    create_view(con, "vitalperiodic", EICU_ROOT / "vitalPeriodic.csv.gz")
    create_view(con, "nursecharting", EICU_ROOT / "nurseCharting.csv.gz")
    con.register("cohort_chart", cohort[["stay_id"]].copy())
    vital = con.execute(
        """
        SELECT c.stay_id, v.observationoffset / 60.0 AS time_hour,
               v.heartrate, v.respiration, v.sao2,
               v.systemicsystolic, v.systemicdiastolic, v.systemicmean,
               v.temperature
        FROM vitalperiodic v
        JOIN cohort_chart c ON v.patientunitstayid = c.stay_id
        WHERE v.observationoffset >= 0 AND v.observationoffset < 1440
        """
    ).fetchdf()
    vital_long = vital.melt(
        id_vars=["stay_id", "time_hour"],
        value_vars=[
            "heartrate",
            "respiration",
            "sao2",
            "systemicsystolic",
            "systemicdiastolic",
            "systemicmean",
            "temperature",
        ],
        var_name="channel",
        value_name="value",
    )
    vital_long["channel"] = vital_long["channel"].map(
        {
            "heartrate": "hr",
            "respiration": "rr",
            "sao2": "spo2",
            "systemicsystolic": "sbp",
            "systemicdiastolic": "dbp",
            "systemicmean": "mbp",
            "temperature": "temp",
        }
    )
    vital_long["priority"] = 0
    vital_long["event_time"] = vital_long["time_hour"]
    nurse = con.execute(
        """
        SELECT c.stay_id, n.nursingchartoffset / 60.0 AS time_hour,
               n.nursingchartcelltypevallabel AS label,
               n.nursingchartcelltypevalname AS item,
               TRY_CAST(n.nursingchartvalue AS DOUBLE) AS value
        FROM nursecharting n
        JOIN cohort_chart c ON n.patientunitstayid = c.stay_id
        WHERE n.nursingchartoffset >= 0 AND n.nursingchartoffset < 1440
          AND (
              lower(n.nursingchartcelltypevalname) IN (
                  'non-invasive bp systolic', 'invasive bp systolic',
                  'non-invasive bp diastolic', 'invasive bp diastolic',
                  'non-invasive bp mean', 'invasive bp mean',
                  'temperature (c)', 'temperature (f)', 'gcs total'
              )
              OR lower(n.nursingchartcelltypevallabel) = 'glasgow coma score'
          )
        """
    ).fetchdf()
    if not nurse.empty:
        label = nurse["item"].fillna("").astype(str).str.lower()
        nurse["channel"] = np.select(
            [
                label.str.contains("systolic", regex=False),
                label.str.contains("diastolic", regex=False),
                label.str.contains("bp mean", regex=False),
                label.isin(["temperature (c)", "temperature (f)"]),
                label.eq("gcs total"),
            ],
            ["sbp", "dbp", "mbp", "temp", "gcs"],
            default="",
        )
        nurse = nurse[nurse["channel"].ne("")].copy()
        fahrenheit = label.loc[nurse.index].eq("temperature (f)")
        nurse.loc[fahrenheit, "value"] = (
            pd.to_numeric(nurse.loc[fahrenheit, "value"], errors="coerce") - 32.0
        ) * 5.0 / 9.0
        nurse["priority"] = 1
        nurse["event_time"] = nurse["time_hour"]
        long = pd.concat(
            [
                vital_long[["stay_id", "time_hour", "channel", "value", "priority", "event_time"]],
                nurse[["stay_id", "time_hour", "channel", "value", "priority", "event_time"]],
            ],
            ignore_index=True,
        )
    else:
        long = vital_long
    long = clean_events(long)
    return hourly_last_by_priority(long)


def build_mimic4() -> None:
    con = connect_duckdb()
    cohort, meta = _mimic4_base(con)
    hourly = _mimic4_events(con, cohort)
    labs = _extract_mimic_labs(
        con, cohort, MIMIC4_ROOT / "hosp/labevents.csv.gz"
    )
    save_processed("mimic4", cohort, hourly, labs, meta["flow"])
    print(
        f"[mimic4] n={len(cohort)} outcome={cohort['outcome'].value_counts().sort_index().to_dict()}",
        flush=True,
    )


def build_mimic3() -> None:
    con = connect_duckdb()
    cohort, meta = _mimic3_base(con)
    hourly = _mimic3_events(con, cohort)
    labs = _extract_mimic_labs(
        con, cohort, MIMIC3_ROOT / "LABEVENTS.csv.gz"
    )
    save_processed("mimic3", cohort, hourly, labs, meta["flow"])
    print(
        f"[mimic3] n={len(cohort)} outcome={cohort['outcome'].value_counts().sort_index().to_dict()}",
        flush=True,
    )


def build_eicu() -> None:
    con = connect_duckdb()
    cohort, meta = _eicu_base(con)
    hourly = _eicu_events(con, cohort)
    labs = _extract_eicu_labs(con, cohort)
    save_processed("eicu", cohort, hourly, labs, meta["flow"])
    print(
        f"[eicu] n={len(cohort)} outcome={cohort['outcome'].value_counts().sort_index().to_dict()}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["mimic4", "mimic3", "eicu"],
        default=["mimic4", "mimic3", "eicu"],
    )
    args = parser.parse_args()
    ensure_dirs()
    builders = {
        "mimic4": build_mimic4,
        "mimic3": build_mimic3,
        "eicu": build_eicu,
    }
    for dataset in args.datasets:
        builders[dataset]()


if __name__ == "__main__":
    main()

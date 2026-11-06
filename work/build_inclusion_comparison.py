"""Rebuild the MIMIC-IV AMI 24 h landmark and compare included vs excluded."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(
    os.environ.get("AMI_SOURCE_ROOT", r"D:\TT paper\0811Temporal Pathway")
)
MIMIC4 = SOURCE_ROOT / "mimic-iv-3.1"

PROCESSED_COHORT = PROJECT_ROOT / "data" / "processed" / "mimic4" / "cohort.csv"
TABLE_CSV = (
    PROJECT_ROOT / "results" / "tables" / "tableS13_included_vs_excluded.csv"
)
TABLE_MD = (
    PROJECT_ROOT / "results" / "tables" / "tableS13_included_vs_excluded.md"
)
AUDIT_JSON = (
    PROJECT_ROOT / "results" / "audits" / "inclusion_comparison.json"
)

EXPECTED_INCLUDED_N = 6487
LANDMARK_HOURS = 24

MIMIC4_DISPOSITIONS = {
    "HOME",
    "AGAINST ADVICE",
    "HOME HEALTH CARE",
    "REHAB",
    "ASSISTED LIVING",
    "SKILLED NURSING FACILITY",
    "CHRONIC/LONG TERM ACUTE CARE",
    "ACUTE HOSPITAL",
    "PSYCH FACILITY",
    "OTHER FACILITY",
    "HEALTHCARE FACILITY",
    "HOSPICE",
    "DIED",
}

REASON_LABELS = {
    "included": "Included in the analytic cohort",
    "death_on_or_before_24h": "Died on or before ICU hour 24",
    "icu_departure_before_24h": "Left ICU alive before hour 24",
    "missing_icu_outtime": "Missing ICU discharge time",
    "outcome_unmappable": "Survived landmark but outcome could not be mapped",
    "later_hospitalization_same_patient": (
        "Eligible landmark but later hospitalization for the same patient"
    ),
}

CONTINUOUS_VARIABLES = [
    ("Demographics", "Age, years", "age", "years", 0),
    (
        "Pre-landmark vital signs",
        "Heart rate, first",
        "hr",
        "beats/min",
        1,
    ),
    (
        "Pre-landmark vital signs",
        "Respiratory rate, first",
        "rr",
        "breaths/min",
        1,
    ),
    (
        "Pre-landmark vital signs",
        "Peripheral oxygen saturation, first",
        "spo2",
        "%",
        1,
    ),
    (
        "Pre-landmark vital signs",
        "Systolic blood pressure, first",
        "sbp",
        "mmHg",
        1,
    ),
    (
        "Pre-landmark vital signs",
        "Diastolic blood pressure, first",
        "dbp",
        "mmHg",
        1,
    ),
    (
        "Pre-landmark vital signs",
        "Mean arterial pressure, first",
        "mbp",
        "mmHg",
        1,
    ),
    (
        "Pre-landmark vital signs",
        "Temperature, first",
        "temp",
        "C",
        1,
    ),
    (
        "Pre-landmark laboratory values",
        "Troponin I or T, first",
        "troponin",
        "assay-specific units",
        3,
    ),
    (
        "Pre-landmark laboratory values",
        "CK-MB, first",
        "ck_mb",
        "ng/mL",
        2,
    ),
    (
        "Pre-landmark laboratory values",
        "Creatinine, first",
        "creatinine",
        "mg/dL",
        2,
    ),
    (
        "Pre-landmark laboratory values",
        "Blood urea nitrogen, first",
        "blood_urea_nitrogen",
        "mg/dL",
        1,
    ),
    (
        "Pre-landmark laboratory values",
        "Lactate, first",
        "lactate",
        "mmol/L",
        2,
    ),
    (
        "Pre-landmark laboratory values",
        "White blood cell count, first",
        "white_blood_cell_count",
        "x10^9/L",
        1,
    ),
    (
        "Pre-landmark laboratory values",
        "Hemoglobin, first",
        "hemoglobin",
        "g/dL",
        1,
    ),
    (
        "Pre-landmark laboratory values",
        "Platelets, first",
        "platelets",
        "x10^9/L",
        1,
    ),
    (
        "Pre-landmark laboratory values",
        "Sodium, first",
        "sodium",
        "mmol/L",
        0,
    ),
    (
        "Pre-landmark laboratory values",
        "Potassium, first",
        "potassium",
        "mmol/L",
        2,
    ),
    (
        "Pre-landmark laboratory values",
        "Glucose, first",
        "glucose",
        "mg/dL",
        0,
    ),
    (
        "Pre-landmark laboratory values",
        "INR, first",
        "inr",
        "ratio",
        2,
    ),
    (
        "Pre-landmark laboratory values",
        "Total bilirubin, first",
        "total_bilirubin",
        "mg/dL",
        2,
    ),
    (
        "Pre-landmark laboratory values",
        "ALT, first",
        "alt",
        "U/L",
        0,
    ),
    (
        "Pre-landmark laboratory values",
        "AST, first",
        "ast",
        "U/L",
        0,
    ),
    (
        "Pre-landmark laboratory values",
        "Albumin, first",
        "albumin",
        "g/dL",
        1,
    ),
    (
        "Pre-landmark laboratory values",
        "Bicarbonate, first",
        "bicarbonate",
        "mmol/L",
        0,
    ),
    (
        "Pre-landmark laboratory values",
        "Chloride, first",
        "chloride",
        "mmol/L",
        0,
    ),
]


def quote_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def connect_duckdb() -> tuple[duckdb.DuckDBPyConnection, tempfile.TemporaryDirectory]:
    temp_dir = tempfile.TemporaryDirectory(prefix="ami_inclusion_duckdb_")
    con = duckdb.connect()
    threads = int(os.environ.get("AMI_DUCKDB_THREADS", os.cpu_count() or 1))
    memory = os.environ.get("AMI_DUCKDB_MEMORY", "10GB")
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{memory}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute(
        "SET temp_directory='"
        + str(Path(temp_dir.name).as_posix()).replace("'", "''")
        + "'"
    )
    return con, temp_dir


def create_views(con: duckdb.DuckDBPyConnection) -> None:
    paths = {
        "admissions": MIMIC4 / "hosp" / "admissions.csv.gz",
        "diagnoses": MIMIC4 / "hosp" / "diagnoses_icd.csv.gz",
        "icustays": MIMIC4 / "icu" / "icustays.csv.gz",
        "patients": MIMIC4 / "hosp" / "patients.csv.gz",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        con.execute(
            f"CREATE OR REPLACE VIEW {name} AS "
            f"SELECT * FROM read_csv_auto('{quote_path(path)}', header=true)"
        )


def first_icu_cohort(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    disposition_sql = ", ".join(
        "'" + value.replace("'", "''") + "'" for value in MIMIC4_DISPOSITIONS
    )
    query = f"""
        WITH ami_hadm AS (
            SELECT DISTINCT hadm_id
            FROM diagnoses
            WHERE (TRY_CAST(icd_version AS INTEGER) = 10
                   AND upper(replace(icd_code, '.', '')) LIKE 'I21%')
               OR (TRY_CAST(icd_version AS INTEGER) = 9
                   AND upper(replace(icd_code, '.', '')) LIKE '410%')
        ),
        ranked_icu AS (
            SELECT i.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY i.hadm_id
                       ORDER BY i.intime, i.stay_id
                   ) AS icu_rank
            FROM icustays i
            JOIN ami_hadm a USING (hadm_id)
        ),
        dx AS (
            SELECT hadm_id,
                   seq_num,
                   TRY_CAST(icd_version AS INTEGER) AS icd_version,
                   upper(replace(icd_code, '.', '')) AS code
            FROM diagnoses
            WHERE (TRY_CAST(icd_version AS INTEGER) = 10
                   AND upper(replace(icd_code, '.', '')) LIKE 'I21%')
               OR (TRY_CAST(icd_version AS INTEGER) = 9
                   AND upper(replace(icd_code, '.', '')) LIKE '410%')
        ),
        dx_flags AS (
            SELECT hadm_id,
                   MAX(
                       CASE WHEN TRY_CAST(seq_num AS INTEGER) = 1 THEN 1 ELSE 0 END
                   ) AS ami_primary_position,
                   MAX(
                       CASE
                           WHEN (icd_version = 10 AND (
                                   code LIKE 'I210%' OR code LIKE 'I211%'
                                   OR code LIKE 'I212%' OR code LIKE 'I213%'
                               ))
                             OR (icd_version = 9 AND (
                                   code LIKE '4100%' OR code LIKE '4101%'
                                   OR code LIKE '4102%' OR code LIKE '4103%'
                                   OR code LIKE '4104%' OR code LIKE '4105%'
                                   OR code LIKE '4106%'
                               ))
                           THEN 1 ELSE 0
                       END
                   ) AS is_stemi,
                   MAX(
                       CASE
                           WHEN (icd_version = 10 AND code LIKE 'I214%')
                             OR (icd_version = 9 AND code LIKE '4107%')
                           THEN 1 ELSE 0
                       END
                   ) AS is_nstemi,
                   MAX(
                       CASE WHEN icd_version = 10 AND code LIKE 'I21A1%'
                            THEN 1 ELSE 0 END
                   ) AS is_type2,
                   MAX(
                       CASE
                           WHEN (icd_version = 10 AND code LIKE 'I219%')
                             OR (icd_version = 9 AND code LIKE '4109%')
                           THEN 1 ELSE 0
                       END
                   ) AS is_unspecified
            FROM dx
            GROUP BY hadm_id
        ),
        base AS (
            SELECT r.stay_id, r.hadm_id, r.subject_id, r.intime, r.outtime,
                   r.first_careunit,
                   a.admittime, a.dischtime, a.deathtime,
                   a.race, a.admission_type, a.admission_location,
                   a.discharge_location, a.hospital_expire_flag,
                   p.gender AS sex,
                   p.anchor_age + date_part('year', r.intime)
                       - p.anchor_year AS age,
                   date_diff('minute', r.intime, r.outtime) / 60.0
                       AS icu_out_hour,
                   CASE WHEN a.deathtime IS NOT NULL
                        THEN date_diff('minute', r.intime, a.deathtime) / 60.0
                        ELSE NULL END AS death_hour_from_icu,
                   coalesce(d.ami_primary_position, 0) AS ami_primary_position,
                   d.is_stemi, d.is_nstemi, d.is_type2, d.is_unspecified
            FROM ranked_icu r
            JOIN admissions a ON a.hadm_id = r.hadm_id
            JOIN patients p ON p.subject_id = r.subject_id
            LEFT JOIN dx_flags d ON d.hadm_id = r.hadm_id
            WHERE r.icu_rank = 1
        ),
        staged AS (
            SELECT *,
                   CASE
                       WHEN age >= 18 THEN 1 ELSE 0
                   END AS adult,
                   CASE
                       WHEN deathtime IS NOT NULL
                        AND deathtime <= intime + INTERVAL 24 HOUR THEN 1
                       ELSE 0
                   END AS death_on_or_before_24h,
                   CASE
                       WHEN outtime IS NOT NULL
                        AND outtime >= intime + INTERVAL 24 HOUR
                        AND (deathtime IS NULL
                             OR deathtime > intime + INTERVAL 24 HOUR)
                       THEN 1 ELSE 0
                   END AS landmark_pass,
                   CASE
                       WHEN trim(upper(coalesce(discharge_location, ''))) IN (
                           {disposition_sql}
                       ) THEN 1 ELSE 0
                   END AS outcome_mapped
            FROM base
        ),
        eligible AS (
            SELECT *,
                   SUM(
                       CASE WHEN landmark_pass = 1 AND outcome_mapped = 1
                            THEN 1 ELSE 0 END
                   ) OVER (
                       PARTITION BY subject_id
                       ORDER BY intime, stay_id
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS eligible_subject_rank
            FROM staged
            WHERE adult = 1
        )
        SELECT *,
               CASE
                   WHEN death_on_or_before_24h = 1
                       THEN 'death_on_or_before_24h'
                   WHEN outtime IS NULL
                       THEN 'missing_icu_outtime'
                   WHEN landmark_pass = 0
                       THEN 'icu_departure_before_24h'
                   WHEN outcome_mapped = 0
                       THEN 'outcome_unmappable'
                   WHEN eligible_subject_rank > 1
                       THEN 'later_hospitalization_same_patient'
                   ELSE 'included'
               END AS inclusion_status,
               least(
                   intime + INTERVAL 24 HOUR,
                   coalesce(outtime, intime + INTERVAL 24 HOUR),
                   coalesce(deathtime, intime + INTERVAL 24 HOUR)
               ) AS window_end,
               CASE
                   WHEN is_stemi = 1 THEN 'STEMI'
                   WHEN is_nstemi = 1 THEN 'NSTEMI'
                   WHEN is_type2 = 1 THEN 'Type 2 MI'
                   WHEN is_unspecified = 1 THEN 'Unspecified'
                   ELSE 'Other'
               END AS ami_subtype
        FROM eligible
    """
    frame = con.execute(query).fetchdf()
    frame.columns = [column.lower() for column in frame.columns]
    if frame.empty:
        raise RuntimeError("The source reconstruction returned no patients.")
    if int(frame["adult"].sum()) != len(frame):
        raise RuntimeError("Adult filtering was not applied consistently.")
    return frame


def source_flow_counts(con: duckdb.DuckDBPyConnection) -> dict:
    source_n = int(
        con.execute(
            """
            SELECT COUNT(DISTINCT hadm_id)
            FROM diagnoses
            WHERE (TRY_CAST(icd_version AS INTEGER) = 10
                   AND upper(replace(icd_code, '.', '')) LIKE 'I21%')
               OR (TRY_CAST(icd_version AS INTEGER) = 9
                   AND upper(replace(icd_code, '.', '')) LIKE '410%')
            """
        ).fetchone()[0]
    )
    first_icu_n = int(
        con.execute(
            """
            WITH ami_hadm AS (
                SELECT DISTINCT hadm_id
                FROM diagnoses
                WHERE (TRY_CAST(icd_version AS INTEGER) = 10
                       AND upper(replace(icd_code, '.', '')) LIKE 'I21%')
                   OR (TRY_CAST(icd_version AS INTEGER) = 9
                       AND upper(replace(icd_code, '.', '')) LIKE '410%')
            ),
            ranked AS (
                SELECT i.stay_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY i.hadm_id
                           ORDER BY i.intime, i.stay_id
                       ) AS icu_rank
                FROM icustays i
                JOIN ami_hadm a USING (hadm_id)
            )
            SELECT COUNT(*) FROM ranked WHERE icu_rank = 1
            """
        ).fetchone()[0]
    )
    return {
        "source_ami_admissions": source_n,
        "ami_admissions_without_first_icu_stay": source_n - first_icu_n,
        "first_icu_stays": first_icu_n,
    }


def first_vitals(con: duckdb.DuckDBPyConnection, cohort: pd.DataFrame) -> pd.DataFrame:
    con.register(
        "cohort_window",
        cohort[
            ["stay_id", "hadm_id", "intime", "outtime", "deathtime", "window_end"]
        ].copy(),
    )
    chartevents = MIMIC4 / "icu" / "chartevents.csv.gz"
    if not chartevents.is_file():
        raise FileNotFoundError(chartevents)
    con.execute(
        f"CREATE OR REPLACE VIEW chartevents AS "
        f"SELECT * FROM read_csv_auto('{quote_path(chartevents)}', header=true)"
    )
    query = """
        WITH events AS (
            SELECT c.stay_id,
                   ce.charttime,
                   CASE
                       WHEN ce.itemid = 220045 THEN 'hr'
                       WHEN ce.itemid IN (220210, 224690) THEN 'rr'
                       WHEN ce.itemid = 220277 THEN 'spo2'
                       WHEN ce.itemid IN (220050, 220179) THEN 'sbp'
                       WHEN ce.itemid IN (220051, 220180) THEN 'dbp'
                       WHEN ce.itemid IN (220052, 220181) THEN 'mbp'
                       WHEN ce.itemid IN (223761, 223762) THEN 'temp'
                   END AS channel,
                   CASE
                       WHEN ce.itemid = 223761
                           THEN (TRY_CAST(ce.valuenum AS DOUBLE) - 32.0) * 5.0 / 9.0
                       ELSE TRY_CAST(ce.valuenum AS DOUBLE)
                   END AS value,
                   CASE
                       WHEN ce.itemid IN (
                           220045, 220210, 220277, 220050, 220051, 220052,
                           223762
                       ) THEN 0
                       ELSE 1
                   END AS source_priority
            FROM chartevents ce
            JOIN cohort_window c USING (stay_id)
            WHERE ce.itemid IN (
                220045, 220210, 224690, 220277, 220050, 220051, 220052,
                220179, 220180, 220181, 223761, 223762
            )
              AND ce.charttime >= c.intime
              AND ce.charttime < c.window_end
              AND TRY_CAST(ce.valuenum AS DOUBLE) IS NOT NULL
        ),
        valid AS (
            SELECT *
            FROM events
            WHERE (channel = 'hr' AND value BETWEEN 20 AND 250)
               OR (channel = 'rr' AND value BETWEEN 0 AND 80)
               OR (channel = 'spo2' AND value BETWEEN 50 AND 100)
               OR (channel = 'sbp' AND value BETWEEN 40 AND 300)
               OR (channel = 'dbp' AND value BETWEEN 20 AND 200)
               OR (channel = 'mbp' AND value BETWEEN 20 AND 250)
               OR (channel = 'temp' AND value BETWEEN 25 AND 45)
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY stay_id, channel
                       ORDER BY charttime, source_priority
                   ) AS rn
            FROM valid
        )
        SELECT stay_id, channel, value, charttime
        FROM ranked
        WHERE rn = 1
    """
    return con.execute(query).fetchdf()


def first_labs(con: duckdb.DuckDBPyConnection, cohort: pd.DataFrame) -> pd.DataFrame:
    labevents = MIMIC4 / "hosp" / "labevents.csv.gz"
    if not labevents.is_file():
        raise FileNotFoundError(labevents)
    con.execute(
        f"CREATE OR REPLACE VIEW labevents AS "
        f"SELECT * FROM read_csv_auto('{quote_path(labevents)}', header=true)"
    )
    query = """
        WITH events AS (
            SELECT c.stay_id,
                   l.charttime,
                   CASE
                       WHEN l.itemid IN (51002, 51003) THEN 'troponin'
                       WHEN l.itemid = 50911 THEN 'ck_mb'
                       WHEN l.itemid = 50912 THEN 'creatinine'
                       WHEN l.itemid = 51006 THEN 'blood_urea_nitrogen'
                       WHEN l.itemid = 50813 THEN 'lactate'
                       WHEN l.itemid = 51301 THEN 'white_blood_cell_count'
                       WHEN l.itemid = 51222 THEN 'hemoglobin'
                       WHEN l.itemid = 51265 THEN 'platelets'
                       WHEN l.itemid = 50983 THEN 'sodium'
                       WHEN l.itemid = 50971 THEN 'potassium'
                       WHEN l.itemid = 50931 THEN 'glucose'
                       WHEN l.itemid = 51237 THEN 'inr'
                       WHEN l.itemid = 50885 THEN 'total_bilirubin'
                       WHEN l.itemid = 50861 THEN 'alt'
                       WHEN l.itemid = 50878 THEN 'ast'
                       WHEN l.itemid = 50862 THEN 'albumin'
                       WHEN l.itemid = 50882 THEN 'bicarbonate'
                       WHEN l.itemid = 50902 THEN 'chloride'
                   END AS lab,
                   TRY_CAST(l.valuenum AS DOUBLE) AS value
            FROM labevents l
            JOIN cohort_window c USING (hadm_id)
            WHERE l.itemid IN (
                51002, 51003, 50911, 50912, 51006, 50813, 51301, 51222,
                51265, 50983, 50971, 50931, 51237, 50885, 50861, 50878,
                50862, 50882, 50902
            )
              AND l.charttime >= c.intime
              AND l.charttime < c.window_end
              AND TRY_CAST(l.valuenum AS DOUBLE) IS NOT NULL
        ),
        valid AS (
            SELECT *
            FROM events
            WHERE (lab = 'troponin' AND value BETWEEN 0 AND 500000)
               OR (lab = 'ck_mb' AND value BETWEEN 0 AND 100000)
               OR (lab = 'creatinine' AND value BETWEEN 0.1 AND 30)
               OR (lab = 'blood_urea_nitrogen' AND value BETWEEN 1 AND 300)
               OR (lab = 'lactate' AND value BETWEEN 0.1 AND 50)
               OR (lab = 'white_blood_cell_count' AND value BETWEEN 0 AND 300)
               OR (lab = 'hemoglobin' AND value BETWEEN 1 AND 30)
               OR (lab = 'platelets' AND value BETWEEN 0 AND 2000)
               OR (lab = 'sodium' AND value BETWEEN 80 AND 200)
               OR (lab = 'potassium' AND value BETWEEN 1 AND 15)
               OR (lab = 'glucose' AND value BETWEEN 10 AND 1000)
               OR (lab = 'inr' AND value BETWEEN 0.1 AND 30)
               OR (lab = 'total_bilirubin' AND value BETWEEN 0.1 AND 80)
               OR (lab = 'alt' AND value BETWEEN 1 AND 20000)
               OR (lab = 'ast' AND value BETWEEN 1 AND 20000)
               OR (lab = 'albumin' AND value BETWEEN 0.5 AND 8)
               OR (lab = 'bicarbonate' AND value BETWEEN 1 AND 80)
               OR (lab = 'chloride' AND value BETWEEN 40 AND 200)
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY stay_id, lab
                       ORDER BY charttime
                   ) AS rn
            FROM valid
        )
        SELECT stay_id, lab, value, charttime
        FROM ranked
        WHERE rn = 1
    """
    return con.execute(query).fetchdf()


def grouped_race(value: object) -> str:
    text = str(value or "").strip().upper()
    if "BLACK" in text:
        return "Black"
    if "HISPANIC" in text or "LATINO" in text:
        return "Hispanic or Latino"
    if "ASIAN" in text:
        return "Asian"
    if "WHITE" in text:
        return "White"
    return "Other or unknown"


def grouped_admission_type(value: object) -> str:
    text = str(value or "").strip().upper()
    if text == "EMERGENCY":
        return "Emergency"
    if text == "URGENT":
        return "Urgent"
    if text in {"ELECTIVE", "SURGICAL SAME DAY ADMISSION"}:
        return "Elective or surgical same-day"
    if "OBSERVATION" in text:
        return "Observation"
    return "Other or unknown"


def grouped_sex(value: object) -> str:
    text = str(value or "").strip().upper()
    if text == "M":
        return "Male"
    if text == "F":
        return "Female"
    return "Other or unknown"


def prepare_patient_data(
    cohort: pd.DataFrame,
    vitals: pd.DataFrame,
    labs: pd.DataFrame,
) -> pd.DataFrame:
    frame = cohort.copy()
    if not vitals.empty:
        vital_wide = vitals.pivot(
            index="stay_id", columns="channel", values="value"
        ).reset_index()
        vital_wide.columns.name = None
        frame = frame.merge(vital_wide, on="stay_id", how="left")
    for channel in ("hr", "rr", "spo2", "sbp", "dbp", "mbp", "temp"):
        if channel not in frame:
            frame[channel] = np.nan

    if not labs.empty:
        lab_wide = labs.pivot(
            index="stay_id", columns="lab", values="value"
        ).reset_index()
        lab_wide.columns.name = None
        frame = frame.merge(lab_wide, on="stay_id", how="left")
    for _, _, column, _, _ in CONTINUOUS_VARIABLES:
        if column not in frame:
            frame[column] = np.nan

    frame["race_group"] = frame["race"].map(grouped_race)
    frame["admission_type_group"] = frame["admission_type"].map(
        grouped_admission_type
    )
    frame["sex_group"] = frame["sex"].map(grouped_sex)
    frame["primary_position"] = frame["ami_primary_position"].map(
        lambda value: "Primary-position AMI" if int(value) == 1 else "Not primary"
    )
    frame["comparison_group"] = np.where(
        frame["inclusion_status"].eq("included"),
        "included",
        np.where(
            frame["inclusion_status"].eq("later_hospitalization_same_patient"),
            "deduplicated",
            "excluded_landmark_or_mapping",
        ),
    )
    return frame


def robust_smd(included: pd.Series, excluded: pd.Series) -> tuple[float, float]:
    included = pd.to_numeric(included, errors="coerce").dropna()
    excluded = pd.to_numeric(excluded, errors="coerce").dropna()
    if len(included) < 2 or len(excluded) < 2:
        return float("nan"), float("nan")
    mean_difference = float(included.mean() - excluded.mean())
    denominator = float(
        np.sqrt(
            (
                float(included.var(ddof=1))
                + float(excluded.var(ddof=1))
            )
            / 2.0
        )
    )
    if not np.isfinite(denominator) or denominator < 1e-12:
        return (0.0 if abs(mean_difference) < 1e-12 else float("nan")), mean_difference
    return mean_difference / denominator, mean_difference


def binary_smd(included_n: int, included_total: int, excluded_n: int, excluded_total: int) -> float:
    if included_total <= 0 or excluded_total <= 0:
        return float("nan")
    p1 = included_n / included_total
    p2 = excluded_n / excluded_total
    denominator = np.sqrt(
        (p1 * (1.0 - p1) + p2 * (1.0 - p2)) / 2.0
    )
    if denominator < 1e-12:
        return 0.0 if abs(p1 - p2) < 1e-12 else float("nan")
    return float((p1 - p2) / denominator)


def median_iqr(values: pd.Series, decimals: int) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "Not available"
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return (
        f"{float(median):.{decimals}f} "
        f"({float(q1):.{decimals}f}-{float(q3):.{decimals}f})"
    )


def n_percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "Not available"
    return f"{numerator:,} ({100.0 * numerator / denominator:.1f}%)"


def comparison_rows(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict], list[dict], list[dict]]:
    included = frame[frame["comparison_group"].eq("included")]
    excluded = frame[
        frame["comparison_group"].eq("excluded_landmark_or_mapping")
    ]
    if len(included) != EXPECTED_INCLUDED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_INCLUDED_N:,} included patients, found {len(included):,}."
        )

    rows: list[dict] = []
    for section, label, column, unit, decimals in CONTINUOUS_VARIABLES:
        overall_values = pd.to_numeric(frame[column], errors="coerce")
        included_values = pd.to_numeric(included[column], errors="coerce")
        excluded_values = pd.to_numeric(excluded[column], errors="coerce")
        overall_available = int(overall_values.notna().sum())
        included_available = int(included_values.notna().sum())
        excluded_available = int(excluded_values.notna().sum())
        availability_smd = binary_smd(
            included_available,
            len(included),
            excluded_available,
            len(excluded),
        )
        value_smd, mean_difference = robust_smd(
            included_values, excluded_values
        )
        rows.append(
            {
                "section": section,
                "characteristic": label,
                "level": f"Available ({unit})",
                "summary": "n (%)",
                "overall_statistic": n_percent(overall_available, len(frame)),
                "included_statistic": n_percent(
                    included_available, len(included)
                ),
                "excluded_statistic": n_percent(
                    excluded_available, len(excluded)
                ),
                "overall_n": overall_available,
                "overall_denominator": len(frame),
                "included_n": included_available,
                "included_denominator": len(included),
                "excluded_n": excluded_available,
                "excluded_denominator": len(excluded),
                "standardized_difference": availability_smd,
                "absolute_standardized_difference": abs(availability_smd),
            }
        )
        rows.append(
            {
                "section": section,
                "characteristic": label,
                "level": f"Median (IQR), {unit}",
                "summary": "median (IQR)",
                "overall_statistic": median_iqr(overall_values, decimals),
                "included_statistic": median_iqr(included_values, decimals),
                "excluded_statistic": median_iqr(excluded_values, decimals),
                "overall_n": overall_available,
                "overall_denominator": len(frame),
                "included_n": included_available,
                "included_denominator": len(included),
                "excluded_n": excluded_available,
                "excluded_denominator": len(excluded),
                "standardized_difference": value_smd,
                "absolute_standardized_difference": abs(value_smd),
            }
        )
    categorical = [
        ("Demographics", "Sex", "sex_group", ["Female", "Male"]),
        (
            "Demographics",
            "Race",
            "race_group",
            ["White", "Black", "Hispanic or Latino", "Asian", "Other or unknown"],
        ),
        (
            "Admission and AMI characteristics",
            "Admission type",
            "admission_type_group",
            [
                "Urgent",
                "Elective or surgical same-day",
                "Observation",
                "Other or unknown",
            ],
        ),
        (
            "Admission and AMI characteristics",
            "AMI subtype",
            "ami_subtype",
            ["STEMI", "NSTEMI", "Type 2 MI", "Unspecified", "Other"],
        ),
        (
            "Admission and AMI characteristics",
            "AMI primary position",
            "primary_position",
            ["Primary-position AMI", "Not primary"],
        ),
    ]
    for section, label, column, levels in categorical:
        for level in levels:
            overall_n = int(frame[column].eq(level).sum())
            included_n = int(included[column].eq(level).sum())
            excluded_n = int(excluded[column].eq(level).sum())
            smd = binary_smd(
                included_n, len(included), excluded_n, len(excluded)
            )
            rows.append(
                {
                    "section": section,
                    "characteristic": label,
                    "level": level,
                    "summary": "n (%)",
                    "overall_statistic": n_percent(overall_n, len(frame)),
                    "included_statistic": n_percent(
                        included_n, len(included)
                    ),
                    "excluded_statistic": n_percent(
                        excluded_n, len(excluded)
                    ),
                    "overall_n": overall_n,
                    "overall_denominator": len(frame),
                    "included_n": included_n,
                    "included_denominator": len(included),
                    "excluded_n": excluded_n,
                    "excluded_denominator": len(excluded),
                    "standardized_difference": smd,
                    "absolute_standardized_difference": abs(smd),
                }
            )

    table = pd.DataFrame(rows)
    table["_section_order"] = table["section"].map(
        {
            "Demographics": 0,
            "Admission and AMI characteristics": 1,
            "Pre-landmark vital signs": 2,
            "Pre-landmark laboratory values": 3,
        }
    )
    table["_characteristic_order"] = table.groupby(
        ["section", "characteristic"], sort=False
    ).ngroup()
    table["_level_order"] = table.groupby(
        ["section", "characteristic"], sort=False
    ).cumcount()
    table = (
        table.sort_values(
            [
                "_section_order",
                "_characteristic_order",
                "_level_order",
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_section_order",
                "_characteristic_order",
                "_level_order",
            ]
        )
        .reset_index(drop=True)
    )

    def records_for(records: pd.DataFrame) -> list[dict]:
        records = records[
            records["absolute_standardized_difference"].notna()
        ].sort_values(
            "absolute_standardized_difference", ascending=False
        )
        output = []
        for record in records.to_dict("records"):
            output.append(
                {
                    "characteristic": record["characteristic"],
                    "level": record["level"],
                    "included": record["included_statistic"],
                    "excluded": record["excluded_statistic"],
                    "standardized_difference_included_minus_excluded": round(
                        float(record["standardized_difference"]), 4
                    ),
                }
            )
        return output

    availability = records_for(
        table[
            table["summary"].eq("n (%)")
            & table["level"].str.startswith("Available")
        ].copy()
    )[:8]
    clinical = records_for(
        pd.concat(
            [
                table[
                    table["summary"].eq("n (%)")
                    & ~table["level"].str.startswith("Available")
                ],
                table[table["summary"].eq("median (IQR)")],
            ]
        )
    )[:12]
    key_differences = availability + clinical
    return table, key_differences, availability, clinical


def flow_summary(cohort: pd.DataFrame) -> dict:
    counts = {
        key: int(value)
        for key, value in cohort["inclusion_status"].value_counts().items()
    }
    total = len(cohort)
    if sum(counts.values()) != total:
        raise RuntimeError("Inclusion status counts do not sum to the cohort.")
    if counts.get("included", 0) != EXPECTED_INCLUDED_N:
        raise RuntimeError("Included count does not match the locked cohort.")
    excluded_primary = (
        counts.get("death_on_or_before_24h", 0)
        + counts.get("icu_departure_before_24h", 0)
        + counts.get("missing_icu_outtime", 0)
        + counts.get("outcome_unmappable", 0)
    )
    if total != (
        counts.get("included", 0)
        + excluded_primary
        + counts.get("later_hospitalization_same_patient", 0)
    ):
        raise RuntimeError("Landmark flow does not reconcile.")
    return {
        "first_icu_adults": total,
        "included": counts.get("included", 0),
        "excluded_at_landmark_or_unmappable": excluded_primary,
        "later_hospitalization_same_patient": counts.get(
            "later_hospitalization_same_patient", 0
        ),
        "breakdown": counts,
    }


def rendered_markdown(
    table: pd.DataFrame,
    flow: dict,
    availability_differences: list[dict],
    clinical_differences: list[dict],
) -> str:
    included_n = int(
        table.loc[
            table["included_denominator"].gt(0), "included_denominator"
        ].iloc[0]
    )
    excluded_n = int(
        table.loc[
            table["excluded_denominator"].gt(0), "excluded_denominator"
        ].iloc[0]
    )
    lines = [
        "# Table S13. Included versus excluded patients at the MIMIC-IV AMI 24-hour landmark",
        "",
        "The source tables were rebuilt read-only. The primary comparison is "
        f"between the {included_n:,} included patients and the {excluded_n:,} "
        "patients who died on or before ICU hour 24, left the ICU alive before "
        "hour 24, had no ICU discharge time, or had an unmappable discharge "
        "disposition after surviving the landmark.",
        "",
        "## Cohort reconciliation",
        "",
        "| Disposition | n |",
        "|---|---:|",
        (
            "| AMI hospital admissions in the source table | "
            f"{flow['source_ami_admissions']:,} |"
        ),
        (
            "| AMI admissions without a first ICU stay | "
            f"{flow['ami_admissions_without_first_icu_stay']:,} |"
        ),
        f"| First ICU stay | {flow['first_icu_stays']:,} |",
        f"| First ICU stay in an adult AMI admission | {flow['first_icu_adults']:,} |",
        (
            "| Died on or before ICU hour 24 | "
            f"{flow['breakdown'].get('death_on_or_before_24h', 0):,} |"
        ),
        (
            "| Left ICU alive before hour 24 | "
            f"{flow['breakdown'].get('icu_departure_before_24h', 0):,} |"
        ),
        (
            "| Missing ICU discharge time | "
            f"{flow['breakdown'].get('missing_icu_outtime', 0):,} |"
        ),
        (
            "| Survived landmark but outcome was unmappable | "
            f"{flow['breakdown'].get('outcome_unmappable', 0):,} |"
        ),
        f"| Included in the analytic cohort | {flow['included']:,} |",
        (
            "| Eligible landmark but later hospitalization for the same patient "
            f"(not in Table S13 comparison column) | "
            f"{flow['later_hospitalization_same_patient']:,} |"
        ),
        "",
        "The last row is a patient-level deduplication step, not a clinical "
        "landmark exclusion. Adding it to the included and primary excluded "
        "groups returns the first-ICU adult cohort exactly.",
        "",
        "## Included versus excluded",
        "",
        "Values are median (IQR) for continuous variables and n (%) for "
        "categorical variables. Continuous-variable availability is shown as "
        "n (%) in the preceding row. Standardized differences use included "
        "minus excluded; no P values are reported.",
        "",
        "| Characteristic | Included (n="
        + f"{included_n:,}) | Excluded (n={excluded_n:,}) | SMD |",
        "|---|---:|---:|---:|",
    ]
    current_section = None
    for record in table.to_dict("records"):
        section = str(record["section"])
        if section != current_section:
            lines.append(f"| **{section}** |  |  |  |")
            current_section = section
        characteristic = str(record["characteristic"])
        level = str(record["level"])
        if level.startswith("Available"):
            characteristic = f"{characteristic}: available"
        elif record["summary"] == "median (IQR)":
            characteristic = f"{characteristic}: median (IQR)"
        else:
            characteristic = f"{characteristic}: {level}"
        smd = record["standardized_difference"]
        smd_text = "" if pd.isna(smd) else f"{float(smd):+.3f}"
        lines.append(
            f"| {characteristic} | {record['included_statistic']} | "
            f"{record['excluded_statistic']} | {smd_text} |"
        )

    lines.extend(
        [
            "",
            "## Key differences",
            "",
            "### Measurement availability",
            "",
            "Excluded patients had fewer routine laboratory results because their "
            "observation window ended before hour 24. Availability therefore "
            "contributes to selection and is reported as an effect size, not as "
            "a significance test.",
            "",
        ]
    )
    for diff in availability_differences:
        lines.append(
            f"- {diff['characteristic']}: included {diff['included']}; excluded "
            f"{diff['excluded']}; SMD "
            f"{diff['standardized_difference_included_minus_excluded']:+.3f}."
        )
    lines.extend(
        [
            "",
            "### Clinical and demographic composition",
            "",
        ]
    )
    for diff in clinical_differences:
        label = diff["characteristic"]
        if not diff["level"].startswith("Available"):
            if diff["level"].startswith("Median"):
                label = f"{label}, median"
            else:
                label = f"{label}: {diff['level']}"
        lines.append(
            f"- {label}: included {diff['included']}; excluded "
            f"{diff['excluded']}; SMD "
            f"{diff['standardized_difference_included_minus_excluded']:+.3f}."
        )
    lines.extend(
        [
            "",
            "These differences describe selection at the landmark and must not "
            "be interpreted as causal effects.",
        ]
    )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The observation window for excluded patients was censored at the "
            "earlier of ICU hour 24, ICU discharge, or death. Values first "
            "recorded in that window were compared.",
            "- Temperature was converted from Fahrenheit to Celsius when needed. "
            "Troponin I and T were combined; their units are assay dependent.",
            "- Standardized differences for observed continuous values use the "
            "difference in means divided by the average pooled standard "
            "deviation. The table reports medians and IQRs for distributional "
            "transparency.",
            "- Missing values were not imputed. Availability is reported "
            "separately for every vital sign and laboratory variable.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    con, temp_dir = connect_duckdb()
    try:
        create_views(con)
        source_flow = source_flow_counts(con)
        cohort = first_icu_cohort(con)
        flow = {**source_flow, **flow_summary(cohort)}
        vitals = first_vitals(con, cohort)
        labs = first_labs(con, cohort)
        frame = prepare_patient_data(cohort, vitals, labs)
        (
            table,
            key_differences,
            availability_differences,
            clinical_differences,
        ) = comparison_rows(frame)

        processed = pd.read_csv(PROCESSED_COHORT, usecols=["stay_id"])
        reconstructed_ids = set(
            frame.loc[frame["inclusion_status"].eq("included"), "stay_id"]
            .astype(int)
            .tolist()
        )
        processed_ids = set(processed["stay_id"].astype(int).tolist())
        overlap_n = len(reconstructed_ids & processed_ids)
        processed_only = sorted(processed_ids - reconstructed_ids)
        reconstructed_only = sorted(reconstructed_ids - processed_ids)
        ids_match = not processed_only and not reconstructed_only
        if not ids_match:
            raise RuntimeError(
                "Reconstructed included patients do not match the locked cohort."
            )

        table.to_csv(TABLE_CSV, index=False, encoding="utf-8")
        TABLE_MD.write_text(
            rendered_markdown(
                table,
                flow,
                availability_differences,
                clinical_differences,
            ),
            encoding="utf-8",
        )

        continuous_audit = []
        for section, label, column, unit, _ in CONTINUOUS_VARIABLES:
            included_values = pd.to_numeric(
                frame.loc[frame["comparison_group"].eq("included"), column],
                errors="coerce",
            )
            excluded_values = pd.to_numeric(
                frame.loc[
                    frame["comparison_group"].eq(
                        "excluded_landmark_or_mapping"
                    ),
                    column,
                ],
                errors="coerce",
            )
            smd, mean_difference = robust_smd(
                included_values, excluded_values
            )
            continuous_audit.append(
                {
                    "section": section,
                    "variable": column,
                    "label": label,
                    "unit": unit,
                    "included_available_n": int(
                        included_values.notna().sum()
                    ),
                    "excluded_available_n": int(
                        excluded_values.notna().sum()
                    ),
                    "value_standardized_difference": (
                        None if not np.isfinite(smd) else round(float(smd), 6)
                    ),
                    "included_minus_excluded_mean": (
                        None
                        if not np.isfinite(mean_difference)
                        else round(float(mean_difference), 6)
                    ),
                }
            )

        audit = {
            "table": "Table S13",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(SOURCE_ROOT),
            "source_files": {
                "patients": str(
                    MIMIC4 / "hosp" / "patients.csv.gz"
                ),
                "admissions": str(
                    MIMIC4 / "hosp" / "admissions.csv.gz"
                ),
                "icustays": str(
                    MIMIC4 / "icu" / "icustays.csv.gz"
                ),
                "diagnoses_icd": str(
                    MIMIC4 / "hosp" / "diagnoses_icd.csv.gz"
                ),
                "chartevents": str(
                    MIMIC4 / "icu" / "chartevents.csv.gz"
                ),
                "labevents": str(
                    MIMIC4 / "hosp" / "labevents.csv.gz"
                ),
            },
            "cohort_definition": {
                "database": "MIMIC-IV v3.1",
                "ami_definition": (
                    "ICD-9 410% or ICD-10 I21% diagnosis in any position"
                ),
                "unit": "first ICU stay per hospital admission",
                "age_minimum": 18,
                "landmark_hours": LANDMARK_HOURS,
                "landmark_rule": (
                    "ICU outtime >= ICU intime + 24 h and no hospital death "
                    "on or before ICU hour 24"
                ),
                "outcome_mapping": (
                    "MIMIC-IV hospital discharge_location mapped to the "
                    "five-level ordinal outcome"
                ),
                "deduplication": (
                    "after landmark and outcome eligibility, retain the first "
                    "eligible hospitalization per patient"
                ),
            },
            "reconstruction": {
                **source_flow,
                **flow,
                "ami_admissions_without_first_icu_stay": source_flow[
                    "ami_admissions_without_first_icu_stay"
                ],
                "first_icu_stays": source_flow["first_icu_stays"],
                "expected_included_n": EXPECTED_INCLUDED_N,
                "processed_cohort_n": len(processed_ids),
                "overlap_n": overlap_n,
                "processed_only_n": len(processed_only),
                "reconstructed_only_n": len(reconstructed_only),
                "included_stay_ids_match_processed_cohort": ids_match,
                "primary_comparison_included_n": EXPECTED_INCLUDED_N,
                "primary_comparison_excluded_n": flow[
                    "excluded_at_landmark_or_unmappable"
                ],
                "status_labels": REASON_LABELS,
            },
            "comparison_method": {
                "continuous_summary": "median (IQR)",
                "categorical_summary": "n (%)",
                "standardized_difference_direction": (
                    "included minus excluded"
                ),
                "continuous_standardized_difference": (
                    "(mean included - mean excluded) divided by "
                    "sqrt((variance included + variance excluded) / 2)"
                ),
                "binary_standardized_difference": (
                    "(p included - p excluded) divided by "
                    "sqrt((p1(1-p1) + p2(1-p2)) / 2)"
                ),
                "significance_testing": "none; effect sizes only",
                "missing_data": "not imputed; availability reported",
                "event_window": (
                    "ICU admission to the earlier of ICU hour 24, ICU "
                    "discharge, or death"
                ),
            },
            "variables": {
                "categorical": [
                    "sex",
                    "race",
                    "admission_type",
                    "AMI subtype",
                    "AMI primary position",
                ],
                "continuous": continuous_audit,
            },
            "key_differences": key_differences,
            "key_availability_differences": availability_differences,
            "key_clinical_differences": clinical_differences,
            "output_files": {
                "csv": str(TABLE_CSV),
                "markdown": str(TABLE_MD),
                "audit": str(AUDIT_JSON),
            },
            "raw_data_modified": False,
        }
        AUDIT_JSON.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            json.dumps(
                {
                    "flow": flow,
                    "source_flow": source_flow,
                    "included_ids_match": ids_match,
                    "table_rows": len(table),
                    "availability_differences": availability_differences,
                    "clinical_differences": clinical_differences,
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        con.close()
        temp_dir.cleanup()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Read-only feasibility audit for the AMI discharge-disposition project.

The script reads the licensed source files in place and writes only aggregate
counts to work/audit_ami_feasibility.json. It does not export patient-level rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb


SOURCE = Path(r"D:\TT paper\0811Temporal Pathway")
MIMIC4 = SOURCE / "mimic-iv-3.1"
MIMIC3 = SOURCE / "mimic-iii-clinical-database-1.4_1" / "mimic-iii-clinical-database-1.4"
EICU = SOURCE / "eicu"
OUT = Path(__file__).resolve().parent / "audit_ami_feasibility.json"


def q(path: Path) -> str:
    return str(path).replace("\\", "/")


def mimic4(con: duckdb.DuckDBPyConnection) -> dict:
    con.execute(
        f"""
        CREATE OR REPLACE VIEW m4_adm AS
        SELECT * FROM read_csv_auto('{q(MIMIC4 / "hosp/admissions.csv.gz")}', header=true)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW m4_dx AS
        SELECT * FROM read_csv_auto('{q(MIMIC4 / "hosp/diagnoses_icd.csv.gz")}', header=true)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW m4_icu AS
        SELECT * FROM read_csv_auto('{q(MIMIC4 / "icu/icustays.csv.gz")}', header=true)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW m4_pat AS
        SELECT * FROM read_csv_auto('{q(MIMIC4 / "hosp/patients.csv.gz")}', header=true)
        """
    )

    ami_hadm = """
        SELECT DISTINCT hadm_id
        FROM m4_dx
        WHERE (icd_version = '10' AND icd_code LIKE 'I21%')
           OR (icd_version = '9' AND icd_code LIKE '410%')
    """
    counts = {}
    counts["ami_admissions"] = con.execute(
        f"SELECT COUNT(*) FROM ({ami_hadm})"
    ).fetchone()[0]

    counts["ami_type_at_admission"] = {
        row[0]: row[1]
        for row in con.execute(
            """
            WITH admission_type AS (
                SELECT
                    hadm_id,
                    MIN(CASE
                        WHEN icd_version = '10' AND replace(icd_code, '.', '') = 'I21A1' THEN 4
                        WHEN icd_version = '9' AND replace(icd_code, '.', '') LIKE '4107%' THEN 2
                        WHEN icd_version = '10' AND replace(icd_code, '.', '') LIKE 'I214%' THEN 2
                        WHEN icd_version = '10' AND (
                            replace(icd_code, '.', '') LIKE 'I210%'
                            OR replace(icd_code, '.', '') LIKE 'I211%'
                            OR replace(icd_code, '.', '') LIKE 'I212%'
                            OR replace(icd_code, '.', '') = 'I213'
                        ) THEN 1
                        WHEN icd_version = '9' AND (
                            replace(icd_code, '.', '') LIKE '4100%'
                            OR replace(icd_code, '.', '') LIKE '4101%'
                            OR replace(icd_code, '.', '') LIKE '4102%'
                            OR replace(icd_code, '.', '') LIKE '4103%'
                            OR replace(icd_code, '.', '') LIKE '4104%'
                            OR replace(icd_code, '.', '') LIKE '4105%'
                            OR replace(icd_code, '.', '') LIKE '4106%'
                        ) THEN 1
                        WHEN icd_version = '10' AND replace(icd_code, '.', '') = 'I219' THEN 3
                        WHEN icd_version = '9' AND replace(icd_code, '.', '') LIKE '4109%' THEN 3
                        ELSE 5
                    END) AS type_code
                FROM m4_dx
                WHERE (icd_version = '10' AND icd_code LIKE 'I21%')
                   OR (icd_version = '9' AND icd_code LIKE '410%')
                GROUP BY hadm_id
            )
            SELECT
                CASE type_code
                    WHEN 1 THEN 'STEMI'
                    WHEN 2 THEN 'NSTEMI'
                    WHEN 3 THEN 'Unspecified'
                    WHEN 4 THEN 'Type 2'
                    ELSE 'Other'
                END AS grp,
                COUNT(*)
            FROM admission_type
            GROUP BY 1 ORDER BY 2 DESC
            """
        ).fetchall()
    }
    counts["top_codes"] = [
        {"version": r[0], "code": r[1], "n": r[2]}
        for r in con.execute(
            """
            SELECT icd_version, icd_code, COUNT(DISTINCT hadm_id)
            FROM m4_dx
            WHERE (icd_version = '10' AND icd_code LIKE 'I21%')
               OR (icd_version = '9' AND icd_code LIKE '410%')
            GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 20
            """
        ).fetchall()
    ]

    first_icu = """
        WITH ranked AS (
            SELECT i.*,
                   ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime, i.stay_id) AS rn
            FROM m4_icu i
            JOIN (
                SELECT DISTINCT hadm_id
                FROM m4_dx
                WHERE (icd_version = '10' AND icd_code LIKE 'I21%')
                   OR (icd_version = '9' AND icd_code LIKE '410%')
            ) a ON i.hadm_id = a.hadm_id
        )
        SELECT r.*, a.discharge_location, a.hospital_expire_flag, a.deathtime
        FROM ranked r
        JOIN m4_adm a ON r.hadm_id = a.hadm_id
        WHERE r.rn = 1
    """
    counts["first_icu_stays"] = con.execute(
        f"SELECT COUNT(*) FROM ({first_icu})"
    ).fetchone()[0]

    counts["in_icu_at_24h"] = con.execute(
        f"""
        SELECT COUNT(*)
        FROM ({first_icu}) t
        WHERE t.outtime >= t.intime + INTERVAL 24 HOUR
          AND (t.deathtime IS NULL OR t.deathtime > t.intime + INTERVAL 24 HOUR)
        """
    ).fetchone()[0]

    counts["adult_in_icu_at_24h"] = con.execute(
        f"""
        SELECT COUNT(*)
        FROM ({first_icu}) t
        JOIN m4_pat p ON t.subject_id = p.subject_id
        WHERE t.outtime >= t.intime + INTERVAL 24 HOUR
          AND (t.deathtime IS NULL OR t.deathtime > t.intime + INTERVAL 24 HOUR)
          AND p.anchor_age >= 18
        """
    ).fetchone()[0]

    counts["discharge_location_all_ami"] = [
        {"value": r[0], "n": r[1]}
        for r in con.execute(
            f"""
            SELECT a.discharge_location, COUNT(*)
            FROM m4_adm a
            JOIN ({ami_hadm}) x ON a.hadm_id = x.hadm_id
            GROUP BY 1 ORDER BY 2 DESC
            """
        ).fetchall()
    ]
    counts["discharge_location_24h"] = [
        {"value": r[0], "n": r[1]}
        for r in con.execute(
            f"""
            SELECT t.discharge_location, COUNT(*)
            FROM ({first_icu}) t
            JOIN m4_pat p ON t.subject_id = p.subject_id
            WHERE t.outtime >= t.intime + INTERVAL 24 HOUR
              AND (t.deathtime IS NULL OR t.deathtime > t.intime + INTERVAL 24 HOUR)
              AND p.anchor_age >= 18
            GROUP BY 1 ORDER BY 2 DESC
            """
        ).fetchall()
    ]
    counts["five_level_24h"] = [
        {"level": r[0], "label": r[1], "n": r[2]}
        for r in con.execute(
            f"""
            SELECT level, label, COUNT(*)
            FROM (
                SELECT
                    CASE
                        WHEN hospital_expire_flag = 1
                         AND discharge_location IN (
                            'HOME','AGAINST ADVICE','HOME HEALTH CARE','REHAB',
                            'ASSISTED LIVING','SKILLED NURSING FACILITY',
                            'CHRONIC/LONG TERM ACUTE CARE','ACUTE HOSPITAL',
                            'PSYCH FACILITY','OTHER FACILITY',
                            'HEALTHCARE FACILITY','HOSPICE','DIED'
                         ) THEN 4
                        WHEN discharge_location IN ('HOME','AGAINST ADVICE') THEN 0
                        WHEN discharge_location = 'HOME HEALTH CARE' THEN 1
                        WHEN discharge_location IN ('REHAB','ASSISTED LIVING') THEN 2
                        WHEN discharge_location IN (
                            'SKILLED NURSING FACILITY',
                            'CHRONIC/LONG TERM ACUTE CARE',
                            'ACUTE HOSPITAL',
                            'PSYCH FACILITY',
                            'OTHER FACILITY',
                            'HEALTHCARE FACILITY'
                        ) THEN 3
                        WHEN discharge_location IN ('HOSPICE','DIED') THEN 4
                        ELSE NULL
                    END AS level,
                    CASE
                        WHEN hospital_expire_flag = 1
                         AND discharge_location IN (
                            'HOME','AGAINST ADVICE','HOME HEALTH CARE','REHAB',
                            'ASSISTED LIVING','SKILLED NURSING FACILITY',
                            'CHRONIC/LONG TERM ACUTE CARE','ACUTE HOSPITAL',
                            'PSYCH FACILITY','OTHER FACILITY',
                            'HEALTHCARE FACILITY','HOSPICE','DIED'
                         ) THEN 'Hospice or in-hospital death'
                        WHEN discharge_location IN ('HOME','AGAINST ADVICE') THEN 'Home or against advice'
                        WHEN discharge_location = 'HOME HEALTH CARE' THEN 'Home health care'
                        WHEN discharge_location IN ('REHAB','ASSISTED LIVING') THEN 'Rehabilitation or assisted living'
                        WHEN discharge_location IN (
                            'SKILLED NURSING FACILITY',
                            'CHRONIC/LONG TERM ACUTE CARE',
                            'ACUTE HOSPITAL',
                            'PSYCH FACILITY',
                            'OTHER FACILITY',
                            'HEALTHCARE FACILITY'
                        ) THEN 'Skilled nursing or other facility'
                        WHEN discharge_location IN ('HOSPICE','DIED') THEN 'Hospice or in-hospital death'
                        ELSE NULL
                    END AS label
                FROM ({first_icu}) t
                JOIN m4_pat p ON t.subject_id = p.subject_id
                WHERE t.outtime >= t.intime + INTERVAL 24 HOUR
                  AND (t.deathtime IS NULL OR t.deathtime > t.intime + INTERVAL 24 HOUR)
                  AND p.anchor_age >= 18
            )
            WHERE level IS NOT NULL
            GROUP BY 1, 2 ORDER BY 1
            """
        ).fetchall()
    ]
    return counts


def mimic3(con: duckdb.DuckDBPyConnection) -> dict:
    con.execute(
        f"""
        CREATE OR REPLACE VIEW m3_adm AS
        SELECT * FROM read_csv_auto('{q(MIMIC3 / "ADMISSIONS.csv.gz")}', header=true)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW m3_dx AS
        SELECT * FROM read_csv_auto('{q(MIMIC3 / "DIAGNOSES_ICD.csv.gz")}', header=true)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW m3_icu AS
        SELECT * FROM read_csv_auto('{q(MIMIC3 / "ICUSTAYS.csv.gz")}', header=true)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW m3_pat AS
        SELECT * FROM read_csv_auto('{q(MIMIC3 / "PATIENTS.csv.gz")}', header=true)
        """
    )

    ami_hadm = """
        SELECT DISTINCT hadm_id
        FROM m3_dx
        WHERE icd9_code LIKE '410%'
    """
    counts = {}
    counts["ami_admissions"] = con.execute(
        f"SELECT COUNT(*) FROM ({ami_hadm})"
    ).fetchone()[0]

    first_icu = """
        WITH ranked AS (
            SELECT i.*,
                   ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime, i.icustay_id) AS rn
            FROM m3_icu i
            JOIN (
                SELECT DISTINCT hadm_id
                FROM m3_dx
                WHERE icd9_code LIKE '410%'
            ) a ON i.hadm_id = a.hadm_id
        )
        SELECT r.*, a.discharge_location, a.hospital_expire_flag, a.deathtime
        FROM ranked r
        JOIN m3_adm a ON r.hadm_id = a.hadm_id
        WHERE r.rn = 1
    """
    counts["first_icu_stays"] = con.execute(
        f"SELECT COUNT(*) FROM ({first_icu})"
    ).fetchone()[0]
    counts["in_icu_at_24h"] = con.execute(
        f"""
        SELECT COUNT(*)
        FROM ({first_icu}) t
        WHERE t.outtime >= t.intime + INTERVAL 24 HOUR
          AND (t.deathtime IS NULL OR t.deathtime > t.intime + INTERVAL 24 HOUR)
        """
    ).fetchone()[0]
    counts["adult_in_icu_at_24h"] = con.execute(
        f"""
        SELECT COUNT(*)
        FROM ({first_icu}) t
        JOIN m3_pat p ON t.subject_id = p.subject_id
        WHERE t.outtime >= t.intime + INTERVAL 24 HOUR
          AND (t.deathtime IS NULL OR t.deathtime > t.intime + INTERVAL 24 HOUR)
          AND date_diff('year', p.dob, t.intime) >= 18
        """
    ).fetchone()[0]
    counts["discharge_location_all_ami"] = [
        {"value": r[0], "n": r[1]}
        for r in con.execute(
            f"""
            SELECT a.discharge_location, COUNT(*)
            FROM m3_adm a
            JOIN ({ami_hadm}) x ON a.hadm_id = x.hadm_id
            GROUP BY 1 ORDER BY 2 DESC
            """
        ).fetchall()
    ]
    counts["discharge_location_24h"] = [
        {"value": r[0], "n": r[1]}
        for r in con.execute(
            f"""
            SELECT t.discharge_location, COUNT(*)
            FROM ({first_icu}) t
            JOIN m3_pat p ON t.subject_id = p.subject_id
            WHERE t.outtime >= t.intime + INTERVAL 24 HOUR
              AND (t.deathtime IS NULL OR t.deathtime > t.intime + INTERVAL 24 HOUR)
              AND date_diff('year', p.dob, t.intime) >= 18
            GROUP BY 1 ORDER BY 2 DESC
            """
        ).fetchall()
    ]
    counts["five_level_24h"] = [
        {"level": r[0], "label": r[1], "n": r[2]}
        for r in con.execute(
            f"""
            SELECT level, label, COUNT(*)
            FROM (
                SELECT
                    CASE
                        WHEN hospital_expire_flag = 1
                         AND discharge_location IN (
                            'HOME','LEFT AGAINST MEDICAL ADVI','HOME HEALTH CARE',
                            'HOME WITH HOME IV PROVIDR','REHAB/DISTINCT PART HOSP',
                            'SNF','LONG TERM CARE HOSPITAL',
                            'DISC-TRAN CANCER/CHLDRN H','SHORT TERM HOSPITAL',
                            'DISCH-TRAN TO PSYCH HOSP','OTHER FACILITY',
                            'HOSPICE-HOME','HOSPICE-MEDICAL FACILITY',
                            'DEAD/EXPIRED'
                         ) THEN 4
                        WHEN discharge_location IN ('HOME','LEFT AGAINST MEDICAL ADVI') THEN 0
                        WHEN discharge_location IN ('HOME HEALTH CARE','HOME WITH HOME IV PROVIDR') THEN 1
                        WHEN discharge_location LIKE 'REHAB%' THEN 2
                        WHEN discharge_location IN (
                            'SNF',
                            'LONG TERM CARE HOSPITAL',
                            'DISC-TRAN CANCER/CHLDRN H',
                            'SHORT TERM HOSPITAL',
                            'DISCH-TRAN TO PSYCH HOSP',
                            'OTHER FACILITY'
                        ) THEN 3
                        WHEN discharge_location IN (
                            'HOSPICE-HOME',
                            'HOSPICE-MEDICAL FACILITY',
                            'DEAD/EXPIRED'
                        ) THEN 4
                        ELSE NULL
                    END AS level,
                    CASE
                        WHEN hospital_expire_flag = 1
                         AND discharge_location IN (
                            'HOME','LEFT AGAINST MEDICAL ADVI','HOME HEALTH CARE',
                            'HOME WITH HOME IV PROVIDR','REHAB/DISTINCT PART HOSP',
                            'SNF','LONG TERM CARE HOSPITAL',
                            'DISC-TRAN CANCER/CHLDRN H','SHORT TERM HOSPITAL',
                            'DISCH-TRAN TO PSYCH HOSP','OTHER FACILITY',
                            'HOSPICE-HOME','HOSPICE-MEDICAL FACILITY',
                            'DEAD/EXPIRED'
                         ) THEN 'Hospice or in-hospital death'
                        WHEN discharge_location IN ('HOME','LEFT AGAINST MEDICAL ADVI') THEN 'Home or against advice'
                        WHEN discharge_location IN ('HOME HEALTH CARE','HOME WITH HOME IV PROVIDR') THEN 'Home health care'
                        WHEN discharge_location LIKE 'REHAB%' THEN 'Rehabilitation or assisted living'
                        WHEN discharge_location IN (
                            'SNF',
                            'LONG TERM CARE HOSPITAL',
                            'DISC-TRAN CANCER/CHLDRN H',
                            'SHORT TERM HOSPITAL',
                            'DISCH-TRAN TO PSYCH HOSP',
                            'OTHER FACILITY'
                        ) THEN 'Skilled nursing or other facility'
                        WHEN discharge_location IN (
                            'HOSPICE-HOME',
                            'HOSPICE-MEDICAL FACILITY',
                            'DEAD/EXPIRED'
                        ) THEN 'Hospice or in-hospital death'
                        ELSE NULL
                    END AS label
                FROM ({first_icu}) t
                JOIN m3_pat p ON t.subject_id = p.subject_id
                WHERE t.outtime >= t.intime + INTERVAL 24 HOUR
                  AND (t.deathtime IS NULL OR t.deathtime > t.intime + INTERVAL 24 HOUR)
                  AND date_diff('year', p.dob, t.intime) >= 18
            )
            WHERE level IS NOT NULL
            GROUP BY 1, 2 ORDER BY 1
            """
        ).fetchall()
    ]
    return counts


def eicu(con: duckdb.DuckDBPyConnection) -> dict:
    con.execute(
        f"""
        CREATE OR REPLACE VIEW e_pat AS
        SELECT * FROM read_csv_auto('{q(EICU / "patient.csv.gz")}', header=true)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW e_dx AS
        SELECT * FROM read_csv_auto('{q(EICU / "diagnosis.csv.gz")}', header=true)
        """
    )

    counts = {}
    counts["diagnosis_410_unit_stays"] = con.execute(
        "SELECT COUNT(DISTINCT patientunitstayid) FROM e_dx WHERE icd9code LIKE '410%'"
    ).fetchone()[0]
    counts["diagnosis_string_ami_unit_stays"] = con.execute(
        """
        SELECT COUNT(DISTINCT patientunitstayid)
        FROM e_dx
        WHERE diagnosisstring ILIKE '%myocardial infarction%'
        """
    ).fetchone()[0]
    counts["diagnosis_either_unit_stays"] = con.execute(
        """
        SELECT COUNT(DISTINCT patientunitstayid)
        FROM e_dx
        WHERE (icd9code LIKE '410%' OR diagnosisstring ILIKE '%acute myocardial infarction%')
          AND coalesce(diagnosisstring, '') NOT ILIKE '%ruled out%'
          AND coalesce(diagnosisstring, '') NOT ILIKE '%history of%'
        """
    ).fetchone()[0]
    counts["diagnosis_ruled_out_unit_stays"] = con.execute(
        """
        SELECT COUNT(DISTINCT patientunitstayid)
        FROM e_dx
        WHERE diagnosisstring ILIKE '%myocardial infarction ruled out%'
        """
    ).fetchone()[0]
    counts["diagnosis_410_primary_rank"] = [
        {"rank": r[0], "n": r[1]}
        for r in con.execute(
            """
            SELECT diagnosispriority, COUNT(DISTINCT patientunitstayid)
            FROM e_dx
            WHERE (icd9code LIKE '410%' OR diagnosisstring ILIKE '%acute myocardial infarction%')
              AND coalesce(diagnosisstring, '') NOT ILIKE '%ruled out%'
              AND coalesce(diagnosisstring, '') NOT ILIKE '%history of%'
            GROUP BY 1 ORDER BY 1 NULLS LAST
            """
        ).fetchall()
    ]
    counts["diagnosis_top_strings"] = [
        {"value": r[0], "n": r[1]}
        for r in con.execute(
            """
            SELECT diagnosisstring, COUNT(DISTINCT patientunitstayid)
            FROM e_dx
            WHERE (icd9code LIKE '410%' OR diagnosisstring ILIKE '%acute myocardial infarction%')
              AND coalesce(diagnosisstring, '') NOT ILIKE '%ruled out%'
              AND coalesce(diagnosisstring, '') NOT ILIKE '%history of%'
            GROUP BY 1 ORDER BY 2 DESC LIMIT 25
            """
        ).fetchall()
    ]

    base = """
        WITH ami AS (
            SELECT DISTINCT patientunitstayid
            FROM e_dx
            WHERE (icd9code LIKE '410%' OR diagnosisstring ILIKE '%acute myocardial infarction%')
              AND coalesce(diagnosisstring, '') NOT ILIKE '%ruled out%'
              AND coalesce(diagnosisstring, '') NOT ILIKE '%history of%'
        ),
        first_unit AS (
            SELECT p.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.patienthealthsystemstayid
                       ORDER BY p.hospitaladmitoffset, p.patientunitstayid
                   ) AS rn
            FROM e_pat p
            JOIN ami a ON p.patientunitstayid = a.patientunitstayid
        )
        SELECT * FROM first_unit WHERE rn = 1
    """
    counts["first_unit_stays"] = con.execute(
        f"SELECT COUNT(*) FROM ({base})"
    ).fetchone()[0]
    counts["first_unit_in_unit_at_24h"] = con.execute(
        f"""
        SELECT COUNT(*)
        FROM ({base}) t
        WHERE t.unitdischargeoffset >= 60 * 24
          AND (t.unitdischargestatus IS NULL OR lower(t.unitdischargestatus) <> 'expired')
        """
    ).fetchone()[0]
    counts["first_unit_adult_in_unit_at_24h"] = con.execute(
        f"""
        SELECT COUNT(*)
        FROM ({base}) t
        WHERE t.unitdischargeoffset >= 60 * 24
          AND (t.unitdischargestatus IS NULL OR lower(t.unitdischargestatus) <> 'expired')
          AND CASE
                WHEN trim(CAST(t.age AS VARCHAR)) = '> 89' THEN 90
                ELSE TRY_CAST(NULLIF(trim(CAST(t.age AS VARCHAR)), '') AS DOUBLE)
              END >= 18
        """
    ).fetchone()[0]
    counts["hospital_discharge_location_24h"] = [
        {"value": r[0], "status": r[1], "n": r[2]}
        for r in con.execute(
            f"""
            SELECT t.hospitaldischargelocation, t.hospitaldischargestatus, COUNT(*)
            FROM ({base}) t
            WHERE t.unitdischargeoffset >= 60 * 24
              AND (t.unitdischargestatus IS NULL OR lower(t.unitdischargestatus) <> 'expired')
              AND CASE
                    WHEN trim(CAST(t.age AS VARCHAR)) = '> 89' THEN 90
                    ELSE TRY_CAST(NULLIF(trim(CAST(t.age AS VARCHAR)), '') AS DOUBLE)
                  END >= 18
            GROUP BY 1, 2 ORDER BY 3 DESC
            """
        ).fetchall()
    ]
    counts["unit_discharge_location_24h"] = [
        {"value": r[0], "status": r[1], "n": r[2]}
        for r in con.execute(
            f"""
            SELECT t.unitdischargelocation, t.unitdischargestatus, COUNT(*)
            FROM ({base}) t
            WHERE t.unitdischargeoffset >= 60 * 24
              AND (t.unitdischargestatus IS NULL OR lower(t.unitdischargestatus) <> 'expired')
              AND CASE
                    WHEN trim(CAST(t.age AS VARCHAR)) = '> 89' THEN 90
                    ELSE TRY_CAST(NULLIF(trim(CAST(t.age AS VARCHAR)), '') AS DOUBLE)
                  END >= 18
            GROUP BY 1, 2 ORDER BY 3 DESC
            """
        ).fetchall()
    ]
    counts["four_level_24h"] = [
        {"level": r[0], "label": r[1], "n": r[2]}
        for r in con.execute(
            f"""
            SELECT level, label, COUNT(*)
            FROM (
                SELECT
                    CASE
                        WHEN lower(coalesce(hospitaldischargestatus, '')) = 'expired'
                          OR lower(coalesce(hospitaldischargelocation, '')) = 'death' THEN 3
                        WHEN lower(coalesce(hospitaldischargelocation, '')) = 'home' THEN 0
                        WHEN lower(coalesce(hospitaldischargelocation, '')) = 'rehabilitation' THEN 1
                        WHEN lower(coalesce(hospitaldischargelocation, '')) IN (
                            'skilled nursing facility',
                            'nursing home',
                            'other',
                            'other hospital',
                            'other external'
                        ) THEN 2
                        ELSE NULL
                    END AS level,
                    CASE
                        WHEN lower(coalesce(hospitaldischargestatus, '')) = 'expired'
                          OR lower(coalesce(hospitaldischargelocation, '')) = 'death' THEN 'Death or hospice'
                        WHEN lower(coalesce(hospitaldischargelocation, '')) = 'home' THEN 'Home'
                        WHEN lower(coalesce(hospitaldischargelocation, '')) = 'rehabilitation' THEN 'Rehabilitation'
                        WHEN lower(coalesce(hospitaldischargelocation, '')) IN (
                            'skilled nursing facility',
                            'nursing home',
                            'other',
                            'other hospital',
                            'other external'
                        ) THEN 'Skilled nursing or other facility'
                        ELSE NULL
                    END AS label
                FROM ({base}) t
                WHERE t.unitdischargeoffset >= 60 * 24
                  AND (t.unitdischargestatus IS NULL OR lower(t.unitdischargestatus) <> 'expired')
                  AND CASE
                        WHEN trim(CAST(t.age AS VARCHAR)) = '> 89' THEN 90
                        ELSE TRY_CAST(NULLIF(trim(CAST(t.age AS VARCHAR)), '') AS DOUBLE)
                      END >= 18
            )
            WHERE level IS NOT NULL
            GROUP BY 1, 2 ORDER BY 1
            """
        ).fetchall()
    ]
    return counts


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads=8")
    result = {
        "source_root": str(SOURCE),
        "mimic4": mimic4(con),
        "mimic3": mimic3(con),
        "eicu": eicu(con),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

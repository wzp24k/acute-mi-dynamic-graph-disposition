# -*- coding: utf-8 -*-
"""Audit eight-channel coverage in MIMIC-III and eICU for AMI transport."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb


SOURCE = Path(r"D:\TT paper\0811Temporal Pathway")
MIMIC3 = SOURCE / "mimic-iii-clinical-database-1.4_1" / "mimic-iii-clinical-database-1.4"
EICU = SOURCE / "eicu"
OUT = Path(__file__).resolve().parent / "audit_transport_coverage.json"


def q(path: Path) -> str:
    return str(path).replace("\\", "/")


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
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE m3_cohort AS
        WITH ranked AS (
            SELECT i.*,
                   ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime, i.icustay_id) AS rn
            FROM m3_icu i
            JOIN (
                SELECT DISTINCT hadm_id FROM m3_dx WHERE icd9_code LIKE '410%'
            ) a ON i.hadm_id = a.hadm_id
        )
        SELECT r.icustay_id, r.hadm_id, r.subject_id, r.intime, r.outtime
        FROM ranked r
        JOIN m3_adm a ON r.hadm_id = a.hadm_id
        JOIN m3_pat p ON r.subject_id = p.subject_id
        WHERE r.rn = 1
          AND r.outtime >= r.intime + INTERVAL 24 HOUR
          AND (a.deathtime IS NULL OR a.deathtime > r.intime + INTERVAL 24 HOUR)
          AND date_diff('year', p.dob, r.intime) >= 18
        """
    )
    n = con.execute("SELECT COUNT(*) FROM m3_cohort").fetchone()[0]
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m3_chart24 AS
        SELECT c.icustay_id, c.charttime, c.itemid, c.valuenum
        FROM read_csv_auto('{q(MIMIC3 / "CHARTEVENTS.csv.gz")}', header=true) c
        JOIN m3_cohort h ON c.icustay_id = h.icustay_id
        WHERE c.itemid IN (
            211, 220045,
            618, 220210, 224690,
            646, 220277,
            51, 455, 220050, 220179, 224167, 227243,
            8368, 8441, 220051, 220180, 224643, 227242,
            52, 456, 220052, 220181,
            676, 677, 678, 679, 223761, 223762,
            198, 220739, 223900, 223901
        )
          AND c.charttime >= h.intime
          AND c.charttime < h.intime + INTERVAL 24 HOUR
          AND c.valuenum IS NOT NULL
        """
    )
    channel_expr = """
        CASE
            WHEN itemid IN (211, 220045) THEN 'hr'
            WHEN itemid IN (618, 220210, 224690) THEN 'rr'
            WHEN itemid IN (646, 220277) THEN 'spo2'
            WHEN itemid IN (51, 455, 220050, 220179, 224167, 227243) THEN 'sbp'
            WHEN itemid IN (8368, 8441, 220051, 220180, 224643, 227242) THEN 'dbp'
            WHEN itemid IN (52, 456, 220052, 220181) THEN 'mbp'
            WHEN itemid IN (676, 677, 678, 679, 223761, 223762) THEN 'temp'
            WHEN itemid IN (198, 220739, 223900, 223901) THEN 'gcs'
        END
    """
    dist = con.execute(
        f"""
        WITH c AS (
            SELECT icustay_id, {channel_expr} AS channel
            FROM m3_chart24
        ),
        per_stay AS (
            SELECT icustay_id, COUNT(DISTINCT channel) AS n_channels
            FROM c WHERE channel IS NOT NULL GROUP BY 1
        )
        SELECT COALESCE(n_channels, 0), COUNT(*)
        FROM m3_cohort h LEFT JOIN per_stay p USING (icustay_id)
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    per_channel = []
    for channel in ["hr", "rr", "spo2", "sbp", "dbp", "mbp", "temp", "gcs"]:
        k = con.execute(
            f"""
            SELECT COUNT(DISTINCT icustay_id)
            FROM m3_chart24
            WHERE {channel_expr} = '{channel}'
            """
        ).fetchone()[0]
        per_channel.append({"channel": channel, "n_stays": int(k), "fraction": round(k / n, 4)})
    return {
        "database": "MIMIC-III v1.4",
        "n": int(n),
        "channel_count_distribution": {int(r[0]): int(r[1]) for r in dist},
        "at_least_4_of_8": {
            "n": int(sum(r[1] for r in dist if int(r[0]) >= 4)),
            "fraction": round(sum(r[1] for r in dist if int(r[0]) >= 4) / n, 4),
        },
        "per_channel": per_channel,
    }


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
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE e_cohort AS
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
        SELECT patientunitstayid, hospitalid, unitdischargeoffset,
               hospitaldischargelocation, hospitaldischargestatus, age
        FROM first_unit
        WHERE rn = 1
          AND unitdischargeoffset >= 60 * 24
          AND (unitdischargestatus IS NULL OR lower(unitdischargestatus) <> 'expired')
          AND CASE
                WHEN trim(CAST(age AS VARCHAR)) = '> 89' THEN 90
                ELSE TRY_CAST(NULLIF(trim(CAST(age AS VARCHAR)), '') AS DOUBLE)
              END >= 18
          AND (
                lower(coalesce(hospitaldischargestatus, '')) = 'expired'
                OR lower(coalesce(hospitaldischargelocation, '')) IN (
                    'home', 'rehabilitation', 'skilled nursing facility',
                    'nursing home', 'other', 'other hospital', 'other external'
                )
          )
        """
    )
    n = con.execute("SELECT COUNT(*) FROM e_cohort").fetchone()[0]
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE e_vital24 AS
        SELECT v.patientunitstayid,
               MAX(CASE WHEN v.heartrate IS NOT NULL THEN 1 ELSE 0 END) AS hr,
               MAX(CASE WHEN v.respiration IS NOT NULL THEN 1 ELSE 0 END) AS rr,
               MAX(CASE WHEN v.sao2 IS NOT NULL THEN 1 ELSE 0 END) AS spo2,
               MAX(CASE WHEN v.systemicsystolic IS NOT NULL THEN 1 ELSE 0 END) AS sbp,
               MAX(CASE WHEN v.systemicdiastolic IS NOT NULL THEN 1 ELSE 0 END) AS dbp,
               MAX(CASE WHEN v.systemicmean IS NOT NULL THEN 1 ELSE 0 END) AS mbp,
               MAX(CASE WHEN v.temperature IS NOT NULL THEN 1 ELSE 0 END) AS temp,
               0 AS gcs
        FROM read_csv_auto('{q(EICU / "vitalPeriodic.csv.gz")}', header=true) v
        JOIN e_cohort h ON v.patientunitstayid = h.patientunitstayid
        WHERE v.observationoffset >= 0 AND v.observationoffset < 60 * 24
        GROUP BY 1
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE e_nurse24 AS
        SELECT n.patientunitstayid,
               MAX(CASE WHEN lower(n.nursingchartcelltypevalname) IN (
                   'heart rate'
               ) THEN 1 ELSE 0 END) AS hr,
               MAX(CASE WHEN lower(n.nursingchartcelltypevalname) IN (
                   'respiratory rate'
               ) THEN 1 ELSE 0 END) AS rr,
               MAX(CASE WHEN lower(n.nursingchartcelltypevalname) IN (
                   'o2 saturation'
               ) THEN 1 ELSE 0 END) AS spo2,
               MAX(CASE WHEN lower(n.nursingchartcelltypevalname) IN (
                   'non-invasive bp systolic', 'invasive bp systolic'
               ) THEN 1 ELSE 0 END) AS sbp,
               MAX(CASE WHEN lower(n.nursingchartcelltypevalname) IN (
                   'non-invasive bp diastolic', 'invasive bp diastolic'
               ) THEN 1 ELSE 0 END) AS dbp,
               MAX(CASE WHEN lower(n.nursingchartcelltypevalname) IN (
                   'non-invasive bp mean', 'invasive bp mean'
               ) THEN 1 ELSE 0 END) AS mbp,
               MAX(CASE WHEN lower(n.nursingchartcelltypevalname) IN (
                   'temperature (c)', 'temperature (f)'
               ) THEN 1 ELSE 0 END) AS temp,
               MAX(CASE WHEN lower(n.nursingchartcelltypevalname) = 'gcs total'
                   AND TRY_CAST(n.nursingchartvalue AS DOUBLE) IS NOT NULL
                   THEN 1 ELSE 0 END) AS gcs
        FROM read_csv_auto('{q(EICU / "nurseCharting.csv.gz")}', header=true) n
        JOIN e_cohort h ON n.patientunitstayid = h.patientunitstayid
        WHERE n.nursingchartoffset >= 0 AND n.nursingchartoffset < 60 * 24
        GROUP BY 1
        """
    )
    dist = con.execute(
        """
        WITH combined AS (
            SELECT h.patientunitstayid,
                   COALESCE(v.hr, 0) + COALESCE(n.hr, 0) AS hr,
                   COALESCE(v.rr, 0) + COALESCE(n.rr, 0) AS rr,
                   COALESCE(v.spo2, 0) + COALESCE(n.spo2, 0) AS spo2,
                   COALESCE(v.sbp, 0) + COALESCE(n.sbp, 0) AS sbp,
                   COALESCE(v.dbp, 0) + COALESCE(n.dbp, 0) AS dbp,
                   COALESCE(v.mbp, 0) + COALESCE(n.mbp, 0) AS mbp,
                   COALESCE(v.temp, 0) + COALESCE(n.temp, 0) AS temp,
                   COALESCE(v.gcs, 0) + COALESCE(n.gcs, 0) AS gcs
            FROM e_cohort h
            LEFT JOIN e_vital24 v USING (patientunitstayid)
            LEFT JOIN e_nurse24 n USING (patientunitstayid)
        )
        SELECT
            (hr > 0)::INT + (rr > 0)::INT + (spo2 > 0)::INT + (sbp > 0)::INT
            + (dbp > 0)::INT + (mbp > 0)::INT + (temp > 0)::INT + (gcs > 0)::INT AS n_channels,
            COUNT(*)
        FROM combined GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    per_channel = []
    for channel in ["hr", "rr", "spo2", "sbp", "dbp", "mbp", "temp", "gcs"]:
        k = con.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT h.patientunitstayid,
                       COALESCE(v.{channel}, 0) + COALESCE(n.{channel}, 0) AS flag
                FROM e_cohort h
                LEFT JOIN e_vital24 v USING (patientunitstayid)
                LEFT JOIN e_nurse24 n USING (patientunitstayid)
            )
            WHERE flag > 0
            """
        ).fetchone()[0]
        per_channel.append({"channel": channel, "n_stays": int(k), "fraction": round(k / n, 4)})
    return {
        "database": "eICU-CRD v2.0",
        "n": int(n),
        "channel_count_distribution": {int(r[0]): int(r[1]) for r in dist},
        "at_least_4_of_8": {
            "n": int(sum(r[1] for r in dist if int(r[0]) >= 4)),
            "fraction": round(sum(r[1] for r in dist if int(r[0]) >= 4) / n, 4),
        },
        "per_channel": per_channel,
    }


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads=8")
    result = {"mimic3": mimic3(con), "eicu": eicu(con)}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Audit cohort size and 7-day MACCE endpoint recoverability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .common import (
    EICU,
    MIMIC3,
    MIMIC4,
    RESULTS,
    TABLES,
    ensure_dirs,
    load_config,
)


def view(con: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_csv_auto('{path.as_posix()}', header=true)"
    )


def schema(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    return con.execute(f"DESCRIBE SELECT * FROM {name}").fetchdf()


def q(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    frame = con.execute(sql).fetchdf()
    frame.columns = [str(column).lower() for column in frame.columns]
    return frame


def mimic4_audit(con: duckdb.DuckDBPyConnection, horizon_hours: int) -> dict:
    view(con, "dx4", MIMIC4 / "hosp/diagnoses_icd.csv.gz")
    view(con, "adm4", MIMIC4 / "hosp/admissions.csv.gz")
    view(con, "pat4", MIMIC4 / "hosp/patients.csv.gz")
    view(con, "icu4", MIMIC4 / "icu/icustays.csv.gz")
    view(con, "proc4", MIMIC4 / "hosp/procedures_icd.csv.gz")

    cad = """
      (icd_version = 10 AND replace(icd_code, '.', '') LIKE 'I25%')
      OR (icd_version = 9 AND replace(icd_code, '.', '') LIKE '414%')
    """
    acute_mi = """
      (icd_version = 10 AND (
        replace(icd_code, '.', '') LIKE 'I21%'
        OR replace(icd_code, '.', '') LIKE 'I22%'))
      OR (icd_version = 9 AND replace(icd_code, '.', '') LIKE '410%')
    """
    stroke = """
      (icd_version = 10 AND (
        replace(icd_code, '.', '') LIKE 'I60%'
        OR replace(icd_code, '.', '') LIKE 'I61%'
        OR replace(icd_code, '.', '') LIKE 'I62%'
        OR replace(icd_code, '.', '') LIKE 'I63%'
        OR replace(icd_code, '.', '') LIKE 'I64%'))
      OR (icd_version = 9 AND (
        replace(icd_code, '.', '') LIKE '430%'
        OR replace(icd_code, '.', '') LIKE '431%'
        OR replace(icd_code, '.', '') LIKE '432%'
        OR replace(icd_code, '.', '') LIKE '433%'
        OR replace(icd_code, '.', '') LIKE '434%'
        OR replace(icd_code, '.', '') LIKE '436%'))
    """
    revasc = """
      (icd_version = 9 AND (
        replace(icd_code, '.', '') LIKE '0066%'
        OR replace(icd_code, '.', '') LIKE '361%'))
      OR (icd_version = 10 AND (
        replace(icd_code, '.', '') LIKE '027%'
        OR replace(icd_code, '.', '') LIKE '021%'
        OR replace(icd_code, '.', '') LIKE '02C%'))
    """
    base = q(
        con,
        f"""
        WITH cad_hadm AS (
          SELECT DISTINCT hadm_id
          FROM dx4
          WHERE {cad} AND try_cast(seq_num AS INTEGER) = 1
        ),
        ranked AS (
          SELECT i.*,
                 ROW_NUMBER() OVER (
                   PARTITION BY i.subject_id
                   ORDER BY i.intime, i.stay_id
                 ) AS patient_stay_rank,
                 ROW_NUMBER() OVER (
                   PARTITION BY i.hadm_id
                   ORDER BY i.intime, i.stay_id
                 ) AS admission_stay_rank
          FROM icu4 i
          JOIN cad_hadm USING (hadm_id)
        )
        SELECT r.stay_id, r.hadm_id, r.subject_id AS patient_id,
               r.intime, r.outtime,
               a.admittime, a.dischtime, a.deathtime, a.hospital_expire_flag,
               p.anchor_age + (year(a.admittime) - p.anchor_year) AS age
        FROM ranked r
        JOIN adm4 a USING (hadm_id)
        JOIN pat4 p ON p.subject_id = r.subject_id
        WHERE r.patient_stay_rank = 1 AND r.admission_stay_rank = 1
        """
    )
    if base.empty:
        return {"database": "mimic4", "eligible_n": 0}
    dt = pd.to_datetime
    base["intime"] = dt(base["intime"], errors="coerce")
    base["dischtime"] = dt(base["dischtime"], errors="coerce")
    base["deathtime"] = dt(base["deathtime"], errors="coerce")
    base["age"] = pd.to_numeric(base["age"], errors="coerce")
    base["death_hours"] = (
        base["deathtime"] - base["intime"]
    ).dt.total_seconds() / 3600
    base["eligible_24h"] = (
        base["age"].ge(18)
        & (base["death_hours"].isna() | base["death_hours"].gt(24))
    )
    ids = base.loc[base["eligible_24h"], ["stay_id", "hadm_id"]].copy()
    con.register("audit_ids4", ids)

    components = q(
        con,
        f"""
        SELECT c.stay_id,
          MAX(CASE WHEN {acute_mi} THEN 1 ELSE 0 END) AS coded_ami,
          MAX(CASE WHEN {stroke} THEN 1 ELSE 0 END) AS coded_stroke,
          MIN(CASE WHEN {acute_mi} THEN try_cast(seq_num AS INTEGER) END)
            AS ami_min_seq,
          MIN(CASE WHEN {stroke} THEN try_cast(seq_num AS INTEGER) END)
            AS stroke_min_seq
        FROM dx4 d JOIN audit_ids4 c USING (hadm_id)
        GROUP BY c.stay_id
        """
    )
    revasc_events = q(
        con,
        f"""
        SELECT c.stay_id, MIN(try_cast(p.chartdate AS DATE)) AS first_revasc_date
        FROM proc4 p JOIN audit_ids4 c USING (hadm_id)
        WHERE {revasc}
        GROUP BY c.stay_id
        """
    )
    base = base.merge(components, on="stay_id", how="left")
    base = base.merge(revasc_events, on="stay_id", how="left")
    for column in ("coded_ami", "coded_stroke"):
        base[column] = base[column].fillna(0).astype(int)
    base["first_revasc_date"] = pd.to_datetime(
        base["first_revasc_date"], errors="coerce"
    )
    base["revasc_hours"] = (
        base["first_revasc_date"].dt.tz_localize(None)
        - base["intime"].dt.tz_localize(None).dt.normalize()
    ).dt.total_seconds() / 3600
    base["death_7d"] = (base["death_hours"] > 24) & (
        base["death_hours"] <= horizon_hours
    )
    base["revasc_7d"] = (base["revasc_hours"] > 24) & (
        base["revasc_hours"] <= horizon_hours
    )
    base["coded_ami_or_stroke"] = base[["coded_ami", "coded_stroke"]].max(axis=1)
    base["macce_observed_primary"] = (
        base["death_7d"] | base["revasc_7d"]
    )
    base["macce_broad_sensitivity"] = (
        base["macce_observed_primary"] | base["coded_ami_or_stroke"].eq(1)
    )
    eligible = base[base["eligible_24h"]].copy()
    eligible.to_csv(
        RESULTS / "endpoint_audit_mimic4.csv",
        index=False,
    )
    return {
        "database": "mimic4",
        "principal_cad_admissions": int(
            con.execute(
                f"SELECT COUNT(DISTINCT hadm_id) FROM dx4 WHERE {cad} "
                "AND try_cast(seq_num AS INTEGER) = 1"
            ).fetchone()[0]
        ),
        "first_stay_patients": int(len(base)),
        "eligible_24h_n": int(len(eligible)),
        "age_range": [
            float(eligible["age"].min()),
            float(eligible["age"].max()),
        ],
        "death_24h_to_7d_n": int(eligible["death_7d"].sum()),
        "timed_revascularization_24h_to_7d_n": int(eligible["revasc_7d"].sum()),
        "coded_ami_n": int(eligible["coded_ami"].sum()),
        "coded_stroke_n": int(eligible["coded_stroke"].sum()),
        "timed_primary_macce_n": int(eligible["macce_observed_primary"].sum()),
        "timed_primary_macce_rate": float(
            eligible["macce_observed_primary"].mean()
        ),
        "broad_coded_macce_n": int(eligible["macce_broad_sensitivity"].sum()),
        "broad_coded_macce_rate": float(
            eligible["macce_broad_sensitivity"].mean()
        ),
        "ami_before_landmark_principal_n": int(
            (
                eligible["coded_ami"].eq(1)
                & pd.to_numeric(eligible["ami_min_seq"], errors="coerce").eq(1)
            ).sum()
        ),
        "stroke_before_landmark_principal_n": int(
            (
                eligible["coded_stroke"].eq(1)
                & pd.to_numeric(eligible["stroke_min_seq"], errors="coerce").eq(1)
            ).sum()
        ),
        "endpoint_time_quality": {
            "death": "exact timestamp",
            "revascularization": "procedure date, day-level",
            "ami": "coded diagnosis only; onset not recoverable without note NLP",
            "stroke": "coded diagnosis only; onset not recoverable without note NLP",
        },
    }


def mimic3_audit(con: duckdb.DuckDBPyConnection, horizon_hours: int) -> dict:
    view(con, "dx3", MIMIC3 / "DIAGNOSES_ICD.csv.gz")
    view(con, "adm3", MIMIC3 / "ADMISSIONS.csv.gz")
    view(con, "pat3", MIMIC3 / "PATIENTS.csv.gz")
    view(con, "icu3", MIMIC3 / "ICUSTAYS.csv.gz")
    view(con, "proc3", MIMIC3 / "PROCEDURES_ICD.csv.gz")

    cad = "replace(icd9_code, '.', '') LIKE '414%'"
    acute_mi = "replace(icd9_code, '.', '') LIKE '410%'"
    stroke = """
      replace(icd9_code, '.', '') LIKE '430%'
      OR replace(icd9_code, '.', '') LIKE '431%'
      OR replace(icd9_code, '.', '') LIKE '432%'
      OR replace(icd9_code, '.', '') LIKE '433%'
      OR replace(icd9_code, '.', '') LIKE '434%'
      OR replace(icd9_code, '.', '') LIKE '436%'
    """
    revasc = """
      replace(icd9_code, '.', '') LIKE '0066%'
      OR replace(icd9_code, '.', '') LIKE '361%'
    """
    base = q(
        con,
        f"""
        WITH cad_hadm AS (
          SELECT DISTINCT hadm_id FROM dx3
          WHERE {cad} AND try_cast(seq_num AS INTEGER) = 1
        ),
        ranked AS (
          SELECT i.*,
                 ROW_NUMBER() OVER (
                   PARTITION BY i.subject_id ORDER BY i.intime, i.icustay_id
                 ) AS patient_stay_rank,
                 ROW_NUMBER() OVER (
                   PARTITION BY i.hadm_id ORDER BY i.intime, i.icustay_id
                 ) AS admission_stay_rank
          FROM icu3 i JOIN cad_hadm USING (hadm_id)
        )
        SELECT r.icustay_id AS stay_id, r.hadm_id,
               r.subject_id AS patient_id,
               r.intime, r.outtime, a.admittime, a.dischtime,
               a.deathtime, a.hospital_expire_flag,
               p.dob
        FROM ranked r
        JOIN adm3 a USING (hadm_id)
        JOIN pat3 p ON p.subject_id = r.subject_id
        WHERE r.patient_stay_rank = 1 AND r.admission_stay_rank = 1
        """
    )
    if base.empty:
        return {"database": "mimic3", "eligible_n": 0}
    base["intime"] = pd.to_datetime(base["intime"], errors="coerce")
    base["dischtime"] = pd.to_datetime(base["dischtime"], errors="coerce")
    base["deathtime"] = pd.to_datetime(base["deathtime"], errors="coerce")
    base["dob"] = pd.to_datetime(base["dob"], errors="coerce")
    base["age"] = (base["intime"] - base["dob"]).dt.total_seconds() / (
        365.25 * 24 * 3600
    )
    base.loc[base["age"].gt(89), "age"] = 90
    base["death_hours"] = (
        base["deathtime"] - base["intime"]
    ).dt.total_seconds() / 3600
    base["eligible_24h"] = (
        base["age"].ge(18)
        & (base["death_hours"].isna() | base["death_hours"].gt(24))
    )
    ids = base.loc[base["eligible_24h"], ["stay_id", "hadm_id"]].copy()
    con.register("audit_ids3", ids)
    components = q(
        con,
        f"""
        SELECT c.stay_id,
          MAX(CASE WHEN {acute_mi} THEN 1 ELSE 0 END) AS coded_ami,
          MAX(CASE WHEN {stroke} THEN 1 ELSE 0 END) AS coded_stroke
        FROM dx3 d JOIN audit_ids3 c USING (hadm_id)
        GROUP BY c.stay_id
        """
    )
    # MIMIC-III PROC_ICD has no procedure date. Revascularization remains
    # ascertainable as an in-hospital event but not with a 7-day time stamp.
    revasc_events = pd.DataFrame(
        columns=["stay_id", "first_revasc_date"]
    )
    base = base.merge(components, on="stay_id", how="left")
    base = base.merge(revasc_events, on="stay_id", how="left")
    for column in ("coded_ami", "coded_stroke"):
        base[column] = base[column].fillna(0).astype(int)
    base["first_revasc_date"] = pd.to_datetime(
        base["first_revasc_date"], errors="coerce"
    )
    base["revasc_hours"] = (
        base["first_revasc_date"].dt.tz_localize(None)
        - base["intime"].dt.tz_localize(None).dt.normalize()
    ).dt.total_seconds() / 3600
    base["death_7d"] = (base["death_hours"] > 24) & (
        base["death_hours"] <= horizon_hours
    )
    base["revasc_7d"] = (base["revasc_hours"] > 24) & (
        base["revasc_hours"] <= horizon_hours
    )
    base["macce_observed_primary"] = base["death_7d"] | base["revasc_7d"]
    base["macce_broad_sensitivity"] = (
        base["macce_observed_primary"]
        | base[["coded_ami", "coded_stroke"]].max(axis=1).eq(1)
    )
    eligible = base[base["eligible_24h"]].copy()
    eligible.to_csv(
        RESULTS / "endpoint_audit_mimic3.csv",
        index=False,
    )
    return {
        "database": "mimic3",
        "principal_cad_admissions": int(
            con.execute(
                f"SELECT COUNT(DISTINCT hadm_id) FROM dx3 WHERE {cad} "
                "AND try_cast(seq_num AS INTEGER) = 1"
            ).fetchone()[0]
        ),
        "first_stay_patients": int(len(base)),
        "eligible_24h_n": int(len(eligible)),
        "death_24h_to_7d_n": int(eligible["death_7d"].sum()),
        "timed_revascularization_24h_to_7d_n": 0,
        "coded_ami_n": int(eligible["coded_ami"].sum()),
        "coded_stroke_n": int(eligible["coded_stroke"].sum()),
        "timed_primary_macce_n": int(eligible["macce_observed_primary"].sum()),
        "timed_primary_macce_rate": float(
            eligible["macce_observed_primary"].mean()
        ),
        "broad_coded_macce_n": int(eligible["macce_broad_sensitivity"].sum()),
        "broad_coded_macce_rate": float(
            eligible["macce_broad_sensitivity"].mean()
        ),
    }


def eicu_audit(
    con: duckdb.DuckDBPyConnection,
    horizon_hours: int,
    scope: str = "principal",
) -> dict:
    view(con, "epat", EICU / "patient.csv.gz")
    view(con, "edx", EICU / "diagnosis.csv.gz")
    cad = """
      coalesce(icd9code, '') LIKE '%414.%'
      OR coalesce(icd9code, '') LIKE '%I25.%'
    """
    acute_mi = """
      coalesce(icd9code, '') LIKE '%410.%'
      OR coalesce(icd9code, '') LIKE '%I21.%'
      OR coalesce(icd9code, '') LIKE '%I22.%'
    """
    stroke = """
      coalesce(icd9code, '') LIKE '%430.%'
      OR coalesce(icd9code, '') LIKE '%431.%'
      OR coalesce(icd9code, '') LIKE '%432.%'
      OR coalesce(icd9code, '') LIKE '%433.%'
      OR coalesce(icd9code, '') LIKE '%434.%'
      OR coalesce(icd9code, '') LIKE '%436.%'
      OR coalesce(icd9code, '') LIKE '%I60.%'
      OR coalesce(icd9code, '') LIKE '%I61.%'
      OR coalesce(icd9code, '') LIKE '%I62.%'
      OR coalesce(icd9code, '') LIKE '%I63.%'
      OR coalesce(icd9code, '') LIKE '%I64.%'
    """
    cad_text = """
      diagnosisstring ILIKE '%coronary artery disease%'
      OR diagnosisstring ILIKE '%coronary atherosclerosis%'
    """
    priority_clause = (
        "AND lower(trim(diagnosispriority)) = 'primary'"
        if scope == "principal"
        else ""
    )
    base = q(
        con,
        f"""
        WITH cad_stays AS (
          SELECT DISTINCT patientunitstayid FROM edx
          WHERE ({cad} OR {cad_text}) {priority_clause}
        ),
        ranked AS (
          SELECT p.*,
                 ROW_NUMBER() OVER (
                   PARTITION BY p.uniquepid
                   ORDER BY p.hospitaladmitoffset, p.patientunitstayid
                 ) AS patient_stay_rank
          FROM epat p JOIN cad_stays USING (patientunitstayid)
        )
        SELECT patientunitstayid AS stay_id, uniquepid AS subject_id,
               hospitalid, age, unitdischargeoffset,
               hospitaldischargeoffset, hospitaldischargestatus,
               unitdischargestatus, unitstaytype
        FROM ranked WHERE patient_stay_rank = 1
        """
    )
    if base.empty:
        return {"database": "eicu", "eligible_n": 0}
    age_text = base["age"].fillna("").astype(str).str.strip().str.lower()
    base["age_numeric"] = pd.to_numeric(base["age"], errors="coerce")
    base.loc[age_text.str.match(r"^\\d+$"), "age_numeric"] = pd.to_numeric(
        age_text[age_text.str.match(r"^\\d+$")], errors="coerce"
    )
    base.loc[age_text.str.startswith("> 89"), "age_numeric"] = 90
    base["eligible_24h"] = (
        base["age_numeric"].ge(18)
        & pd.to_numeric(base["unitdischargeoffset"], errors="coerce").ge(1440)
        & ~base["unitdischargestatus"].fillna("").astype(str).str.lower().eq("expired")
    )
    ids = base.loc[base["eligible_24h"], ["stay_id"]].copy()
    con.register("audit_eids", ids)
    dxcomp = q(
        con,
        f"""
        SELECT c.stay_id,
          MAX(CASE WHEN ({acute_mi}) AND diagnosisoffset > 1440
                    AND diagnosisoffset <= {horizon_hours * 60}
               THEN 1 ELSE 0 END) AS timed_ami,
          MAX(CASE WHEN ({stroke}) AND diagnosisoffset > 1440
                    AND diagnosisoffset <= {horizon_hours * 60}
               THEN 1 ELSE 0 END) AS timed_stroke
        FROM edx d JOIN audit_eids c ON d.patientunitstayid = c.stay_id
        GROUP BY c.stay_id
        """
    )
    base = base.merge(dxcomp, on="stay_id", how="left")
    base["timed_revasc"] = 0
    base[["timed_ami", "timed_stroke"]] = (
        base[["timed_ami", "timed_stroke"]].fillna(0).astype(int)
    )
    discharge_hours = pd.to_numeric(
        base["hospitaldischargeoffset"], errors="coerce"
    ) / 60.0
    base["death_7d"] = (
        base["hospitaldischargestatus"].fillna("").astype(str).str.lower().eq("expired")
        & discharge_hours.le(horizon_hours)
        & discharge_hours.gt(24)
    )
    base["macce"] = base[
        ["death_7d", "timed_ami", "timed_stroke", "timed_revasc"]
    ].max(axis=1)
    eligible = base[base["eligible_24h"]].copy()
    output_name = "eicu" if scope == "principal" else "eicu_anycad"
    eligible.to_csv(
        RESULTS / f"endpoint_audit_{output_name}.csv",
        index=False,
    )
    site = (
        eligible.groupby("hospitalid", as_index=False)
        .agg(n=("stay_id", "size"), events=("macce", "sum"))
        .sort_values("n", ascending=False)
    )
    site.to_csv(
        RESULTS / f"endpoint_audit_{output_name}_sites.csv", index=False
    )
    return {
        "database": output_name,
        "cad_scope": scope,
        "principal_cad_unit_stays": int(
            con.execute(
                f"SELECT COUNT(DISTINCT patientunitstayid) FROM edx "
                f"WHERE ({cad} OR {cad_text}) {priority_clause}"
            ).fetchone()[0]
        ),
        "first_stay_patients": int(len(base)),
        "eligible_24h_n": int(len(eligible)),
        "death_24h_to_7d_n": int(eligible["death_7d"].sum()),
        "timed_ami_n": int(eligible["timed_ami"].sum()),
        "timed_stroke_n": int(eligible["timed_stroke"].sum()),
        "timed_revascularization_n": int(eligible["timed_revasc"].sum()),
        "timed_macce_n": int(eligible["macce"].sum()),
        "timed_macce_rate": float(eligible["macce"].mean()),
        "sites_n": int(eligible["hospitalid"].nunique()),
        "sites_ge_30_n": int(site["n"].ge(30).sum()),
        "sites_ge_30_with_event_n": int(
            site.loc[site["n"].ge(30), "events"].gt(0).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", choices=["mimic4", "mimic3", "eicu", "all"], default="all")
    parser.add_argument(
        "--eicu-scope",
        choices=["principal", "any"],
        default="principal",
    )
    args = parser.parse_args()
    ensure_dirs()
    config = load_config()
    horizon = int(config["outcome_horizon_hours"])
    con = duckdb.connect()
    con.execute("SET threads=6")
    con.execute("SET memory_limit='10GB'")
    funcs = {
        "mimic4": mimic4_audit,
        "mimic3": mimic3_audit,
        "eicu": eicu_audit,
    }
    names = list(funcs) if args.database == "all" else [args.database]
    audits = [
        funcs[name](con, horizon, args.eicu_scope)
        if name == "eicu"
        else funcs[name](con, horizon)
        for name in names
    ]
    eicu_database = "eicu" if args.eicu_scope == "principal" else "eicu_anycad"
    output = {
        "config": config,
        "audits": audits,
        "go_no_go": {
            "mimic4_event_threshold_150": bool(
                next(a for a in audits if a["database"] == "mimic4")[
                    "timed_primary_macce_n"
                ]
                >= 150
            )
            if any(a["database"] == "mimic4" for a in audits)
            else None,
            "eicu_event_threshold_75": bool(
                next(a for a in audits if a["database"] == eicu_database)[
                    "timed_macce_n"
                ]
                >= 75
            )
            if any(a["database"] == "eicu" for a in audits)
            else None,
        },
    }
    path = (
        RESULTS / "endpoint_feasibility.json"
        if args.eicu_scope == "principal"
        else RESULTS / "endpoint_feasibility_eicu_anycad.json"
    )
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

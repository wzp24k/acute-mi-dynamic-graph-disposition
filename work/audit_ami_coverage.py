# -*- coding: utf-8 -*-
"""Read-only coverage audit for the AMI 24 h development cohort."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb


SOURCE = Path(r"D:\TT paper\0811Temporal Pathway")
MIMIC4 = SOURCE / "mimic-iv-3.1"
OUT = Path(__file__).resolve().parent / "audit_ami_coverage.json"


def q(path: Path) -> str:
    return str(path).replace("\\", "/")


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW adm AS
        SELECT * FROM read_csv_auto('{q(MIMIC4 / "hosp/admissions.csv.gz")}', header=true)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW dx AS
        SELECT * FROM read_csv_auto('{q(MIMIC4 / "hosp/diagnoses_icd.csv.gz")}', header=true)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW icu AS
        SELECT * FROM read_csv_auto('{q(MIMIC4 / "icu/icustays.csv.gz")}', header=true)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW pat AS
        SELECT * FROM read_csv_auto('{q(MIMIC4 / "hosp/patients.csv.gz")}', header=true)
        """
    )
    ami = """
        SELECT DISTINCT hadm_id
        FROM dx
        WHERE (icd_version = '10' AND icd_code LIKE 'I21%')
           OR (icd_version = '9' AND icd_code LIKE '410%')
    """
    cohort = f"""
        WITH ranked AS (
            SELECT i.*,
                   ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime, i.stay_id) AS rn
            FROM icu i
            JOIN ({ami}) a ON i.hadm_id = a.hadm_id
        )
        SELECT r.stay_id, r.hadm_id, r.subject_id, r.intime, r.outtime
        FROM ranked r
        JOIN adm a ON r.hadm_id = a.hadm_id
        JOIN pat p ON r.subject_id = p.subject_id
        WHERE r.rn = 1
          AND r.outtime >= r.intime + INTERVAL 24 HOUR
          AND (a.deathtime IS NULL OR a.deathtime > r.intime + INTERVAL 24 HOUR)
          AND p.anchor_age >= 18
    """
    con.execute(f"CREATE OR REPLACE TEMP TABLE cohort AS {cohort}")
    cohort_n = con.execute("SELECT COUNT(*) FROM cohort").fetchone()[0]
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE chart24 AS
        SELECT c.stay_id, c.charttime, c.itemid, c.valuenum
        FROM read_csv_auto('{q(MIMIC4 / "icu/chartevents.csv.gz")}', header=true) c
        JOIN cohort h ON c.stay_id = h.stay_id
        WHERE c.itemid IN (
            220045, 220210, 224690, 220277,
            220050, 220051, 220052,
            220179, 220180, 220181,
            223761, 223762,
            220739, 223900, 223901
        )
          AND c.charttime >= h.intime
          AND c.charttime < h.intime + INTERVAL 24 HOUR
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE lab24 AS
        SELECT h.hadm_id, l.charttime, l.itemid, l.valuenum
        FROM read_csv_auto('{q(MIMIC4 / "hosp/labevents.csv.gz")}', header=true) l
        JOIN cohort h ON l.hadm_id = h.hadm_id
        WHERE l.itemid IN (51002, 51003, 50911, 50910, 50813)
          AND l.charttime >= h.intime
          AND l.charttime < h.intime + INTERVAL 24 HOUR
        """
    )

    channel_expr = """
        CASE
            WHEN itemid = 220045 THEN 'hr'
            WHEN itemid IN (220210, 224690) THEN 'rr'
            WHEN itemid = 220277 THEN 'spo2'
            WHEN itemid IN (220050, 220179) THEN 'sbp'
            WHEN itemid IN (220051, 220180) THEN 'dbp'
            WHEN itemid IN (220052, 220181) THEN 'mbp'
            WHEN itemid IN (223761, 223762) THEN 'temp'
            WHEN itemid IN (220739, 223900, 223901) THEN 'gcs'
        END
    """
    coverage = con.execute(
        f"""
        WITH c AS (
            SELECT c.stay_id,
                   {channel_expr} AS channel
            FROM chart24 c
            WHERE c.valuenum IS NOT NULL
        ),
        per_stay AS (
            SELECT stay_id, COUNT(DISTINCT channel) AS n_channels
            FROM c
            WHERE channel IS NOT NULL
            GROUP BY stay_id
        )
        SELECT
            COALESCE(n_channels, 0) AS n_channels,
            COUNT(*) AS n_stays
        FROM cohort h
        LEFT JOIN per_stay p USING (stay_id)
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    dist = {int(r[0]): int(r[1]) for r in coverage}
    at_least_4 = sum(n for k, n in dist.items() if k >= 4)

    per_channel = []
    for channel in ["hr", "rr", "spo2", "sbp", "dbp", "mbp", "temp", "gcs"]:
        n = con.execute(
            f"""
            SELECT COUNT(DISTINCT c.stay_id)
            FROM chart24 c
            WHERE c.valuenum IS NOT NULL
              AND {channel_expr} = '{channel}'
            """
        ).fetchone()[0]
        per_channel.append(
            {"channel": channel, "n_stays": int(n), "fraction": round(n / cohort_n, 4)}
        )

    gcs_complete = con.execute(
        """
        WITH per_stay AS (
            SELECT c.stay_id,
                   COUNT(DISTINCT c.itemid) AS n_gcs_items
            FROM chart24 c
            WHERE c.valuenum IS NOT NULL
              AND c.itemid IN (220739, 223900, 223901)
            GROUP BY c.stay_id
        )
        SELECT COUNT(*) FROM per_stay WHERE n_gcs_items = 3
        """
    ).fetchone()[0]

    lab_counts = {}
    for name, itemids in {
        "troponin_any": "51002, 51003",
        "troponin_i": "51002",
        "troponin_t": "51003",
        "ckmb": "50911",
        "ck": "50910",
        "lactate": "50813",
    }.items():
        n_pre = con.execute(
            f"""
            SELECT COUNT(DISTINCT l.hadm_id)
            FROM lab24 l
            WHERE l.itemid IN ({itemids})
            """
        ).fetchone()[0]
        lab_counts[name] = {
            "n_hadm_pre_24h": int(n_pre),
            "fraction_pre_24h": round(n_pre / cohort_n, 4),
        }

    result = {
        "database": "MIMIC-IV v3.1",
        "cohort": "first ICU stay, AMI any diagnosis position, adult, alive and in ICU at 24 h",
        "n": int(cohort_n),
        "channel_count_distribution": dist,
        "at_least_4_of_8": {
            "n": int(at_least_4),
            "fraction": round(at_least_4 / cohort_n, 4),
        },
        "per_channel": per_channel,
        "gcs_complete_3_components": {
            "n": int(gcs_complete),
            "fraction": round(gcs_complete / cohort_n, 4),
        },
        "labs": lab_counts,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

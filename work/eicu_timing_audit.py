"""Audit whether eICU provides usable event timing for MI and stroke."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb


EICU = Path(r"D:\TT paper\0811Temporal Pathway\eicu")
OUT = Path(r"D:\TT paper\1107Invariant Continuous-Time\results")


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET memory_limit='8GB'")
    con.execute(
        f"CREATE VIEW patient AS SELECT * FROM read_csv_auto('{(EICU / 'patient.csv.gz').as_posix()}', header=true)"
    )
    con.execute(
        f"CREATE VIEW diagnosis AS SELECT * FROM read_csv_auto('{(EICU / 'diagnosis.csv.gz').as_posix()}', header=true)"
    )

    admission_reason = con.execute(
        """
        SELECT apacheadmissiondx, COUNT(*) AS n
        FROM patient
        WHERE lower(coalesce(apacheadmissiondx, '')) LIKE '%myocardial%'
           OR lower(coalesce(apacheadmissiondx, '')) LIKE '%coronary%'
           OR lower(coalesce(apacheadmissiondx, '')) LIKE '%angina%'
           OR lower(coalesce(apacheadmissiondx, '')) LIKE '%chest pain%'
           OR lower(coalesce(apacheadmissiondx, '')) LIKE '%stroke%'
           OR lower(coalesce(apacheadmissiondx, '')) LIKE '%cva%'
           OR lower(coalesce(apacheadmissiondx, '')) LIKE '%infarction%'
        GROUP BY 1
        ORDER BY n DESC
        """
    ).fetchdf()

    diagnoses = con.execute(
        """
        SELECT lower(diagnosisstring) AS diagnosis,
               COUNT(*) AS n,
               MIN(diagnosisoffset) AS min_offset,
               percentile_cont(0.25) WITHIN GROUP (ORDER BY diagnosisoffset) AS q1_offset,
               median(diagnosisoffset) AS median_offset,
               percentile_cont(0.75) WITHIN GROUP (ORDER BY diagnosisoffset) AS q3_offset,
               MAX(diagnosisoffset) AS max_offset
        FROM diagnosis
        WHERE lower(coalesce(diagnosisstring, ''))
                LIKE 'cardiovascular|chest pain / ashd|acute coronary syndrome|acute myocardial infarction%'
           OR lower(coalesce(diagnosisstring, ''))
                LIKE 'surgery|cardiac surgery|myocardial infarction%'
           OR lower(coalesce(diagnosisstring, ''))
                LIKE 'neurologic|disorders of vasculature|stroke%'
        GROUP BY 1
        ORDER BY n DESC
        """
    ).fetchdf()

    timing = con.execute(
        """
        WITH event_dx AS (
          SELECT patientunitstayid, diagnosisoffset,
                 lower(diagnosisstring) AS diagnosis,
                 activeupondischarge, diagnosispriority
          FROM diagnosis
          WHERE lower(coalesce(diagnosisstring, ''))
                  LIKE 'cardiovascular|chest pain / ashd|acute coronary syndrome|acute myocardial infarction%'
             OR lower(coalesce(diagnosisstring, ''))
                  LIKE 'surgery|cardiac surgery|myocardial infarction%'
             OR lower(coalesce(diagnosisstring, ''))
                  LIKE 'neurologic|disorders of vasculature|stroke%'
        ),
        timing_class AS (
          SELECT *,
            CASE
              WHEN diagnosisoffset < 0 THEN 'pre_icu'
              WHEN diagnosisoffset < 1440 THEN 'first_24h'
              WHEN diagnosisoffset < 10080 THEN '24h_to_day7'
              ELSE 'after_day7'
            END AS timing_group
          FROM event_dx
        )
        SELECT timing_group, COUNT(*) AS rows_n,
               COUNT(DISTINCT patientunitstayid) AS stays_n,
               MIN(diagnosisoffset) AS min_offset,
               median(diagnosisoffset) AS median_offset,
               MAX(diagnosisoffset) AS max_offset
        FROM timing_class
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()

    first_timing = con.execute(
        """
        WITH event_dx AS (
          SELECT patientunitstayid, diagnosisoffset,
                 lower(diagnosisstring) AS diagnosis
          FROM diagnosis
          WHERE lower(coalesce(diagnosisstring, ''))
                  LIKE 'cardiovascular|chest pain / ashd|acute coronary syndrome|acute myocardial infarction%'
             OR lower(coalesce(diagnosisstring, ''))
                  LIKE 'surgery|cardiac surgery|myocardial infarction%'
             OR lower(coalesce(diagnosisstring, ''))
                  LIKE 'neurologic|disorders of vasculature|stroke%'
        ),
        typed AS (
          SELECT *,
            CASE
              WHEN diagnosis LIKE 'cardiovascular|chest pain / ashd|acute coronary syndrome|acute myocardial infarction%'
                OR diagnosis LIKE 'surgery|cardiac surgery|myocardial infarction%'
              THEN 'acute_myocardial_infarction'
              ELSE 'stroke'
            END AS event_type
          FROM event_dx
        ),
        ranked AS (
          SELECT *,
            ROW_NUMBER() OVER (
              PARTITION BY patientunitstayid, event_type
              ORDER BY diagnosisoffset, diagnosis
            ) AS event_rank
          FROM typed
        ),
        classified AS (
          SELECT *,
            CASE
              WHEN diagnosisoffset < 0 THEN 'pre_icu'
              WHEN diagnosisoffset < 1440 THEN 'first_24h'
              WHEN diagnosisoffset < 10080 THEN '24h_to_day7'
              ELSE 'after_day7'
            END AS timing_group
          FROM ranked
          WHERE event_rank = 1
        )
        SELECT event_type, timing_group, COUNT(*) AS stays_n,
               MIN(diagnosisoffset) AS min_offset,
               median(diagnosisoffset) AS median_offset,
               MAX(diagnosisoffset) AS max_offset
        FROM classified
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).fetchdf()

    priority = con.execute(
        """
        SELECT diagnosispriority, COUNT(*) AS n
        FROM diagnosis
        WHERE lower(coalesce(diagnosisstring, ''))
                LIKE 'cardiovascular|chest pain / ashd|acute coronary syndrome|acute myocardial infarction%'
           OR lower(coalesce(diagnosisstring, ''))
                LIKE 'surgery|cardiac surgery|myocardial infarction%'
           OR lower(coalesce(diagnosisstring, ''))
                LIKE 'neurologic|disorders of vasculature|stroke%'
        GROUP BY 1
        ORDER BY n DESC
        """
    ).fetchdf()

    output = {
        "available_eicu_files": sorted(path.name for path in EICU.iterdir()),
        "admission_reason_matches": admission_reason.to_dict(orient="records"),
        "event_diagnosis_strings": diagnoses.to_dict(orient="records"),
        "event_offset_distribution": timing.to_dict(orient="records"),
        "first_event_per_stay_timing": first_timing.to_dict(orient="records"),
        "event_diagnosis_priority": priority.to_dict(orient="records"),
        "interpretation": {
            "apacheadmissiondx": (
                "Admission reason text is available but has no event timestamp. "
                "It can define the reason for ICU admission, not interval "
                "occurrence."
            ),
            "diagnosisoffset": (
                "Available for diagnosis rows and can provide a structured "
                "relative time, but it is a diagnosis-entry offset and may not "
                "equal the clinical event onset."
            ),
            "notes": "No eICU note table is present in this extract.",
        },
    }
    (OUT / "eicu_timing_audit.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

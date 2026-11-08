"""Audit indirect MIMIC event-time anchors for MI and stroke."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\TT paper\0811Temporal Pathway")
MIMIC4 = ROOT / "mimic-iv-3.1"
MIMIC3 = ROOT / "mimic-iii-clinical-database-1.4_1" / "mimic-iii-clinical-database-1.4"
PROJECT = Path(r"D:\TT paper\1107Invariant Continuous-Time")


def view(con, name, path, all_varchar=False):
    suffix = ", all_varchar=true" if all_varchar else ""
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM "
        f"read_csv_auto('{path.as_posix()}', header=true{suffix})"
    )


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads=12")
    con.execute("SET memory_limit='12GB'")
    ids4 = pd.read_csv(PROJECT / "results" / "endpoint_audit_mimic4.csv")
    ids3 = pd.read_csv(PROJECT / "results" / "endpoint_audit_mimic3.csv")
    con.register("ids4", ids4[["stay_id", "hadm_id", "intime"]])
    con.register("ids3", ids3[["stay_id", "hadm_id", "intime"]])

    view(con, "dx4", MIMIC4 / "hosp/diagnoses_icd.csv.gz")
    view(con, "proc4", MIMIC4 / "hosp/procedures_icd.csv.gz")
    view(con, "discharge4", MIMIC4 / "mimic iv note 2.2/note/discharge.csv.gz")
    view(con, "radiology4", MIMIC4 / "mimic iv note 2.2/note/radiology.csv.gz")
    view(con, "dx3", MIMIC3 / "DIAGNOSES_ICD.csv.gz")
    view(con, "proc3", MIMIC3 / "PROCEDURES_ICD.csv.gz")
    view(con, "notes3", MIMIC3 / "NOTEEVENTS.csv.gz")

    m4_proc = con.execute(
        """
        SELECT COUNT(*) AS rows_n,
               COUNT(DISTINCT p.hadm_id) AS hadm_n,
               COUNT(p.chartdate) AS dated_rows,
               COUNT(DISTINCT CASE WHEN p.chartdate IS NOT NULL THEN p.hadm_id END)
                 AS dated_hadm
        FROM proc4 p JOIN ids4 c USING (hadm_id)
        """
    ).fetchdf().to_dict(orient="records")[0]
    m4_discharge = con.execute(
        """
        SELECT COUNT(*) AS note_rows, COUNT(DISTINCT n.hadm_id) AS hadm_n
        FROM discharge4 n JOIN ids4 c USING (hadm_id)
        """
    ).fetchdf().to_dict(orient="records")[0]
    m4_radiology_stroke = con.execute(
        """
        SELECT COUNT(*) AS note_rows, COUNT(DISTINCT n.hadm_id) AS hadm_n
        FROM radiology4 n JOIN ids4 c USING (hadm_id)
        WHERE lower(coalesce(n.text, '')) LIKE '%acute infarct%'
           OR lower(coalesce(n.text, '')) LIKE '%intracranial hemorrhage%'
           OR lower(coalesce(n.text, '')) LIKE '%intraparenchymal hemorrhage%'
           OR lower(coalesce(n.text, '')) LIKE '%subarachnoid hemorrhage%'
        """
    ).fetchdf().to_dict(orient="records")[0]
    m3_proc_columns = con.execute("DESCRIBE proc3").fetchdf()
    m3_notes = con.execute(
        """
        SELECT category, COUNT(*) AS note_rows,
               COUNT(DISTINCT n.hadm_id) AS hadm_n,
               COUNT(try_cast(n.charttime AS TIMESTAMP)) AS timed_rows
        FROM notes3 n JOIN ids3 c USING (hadm_id)
        GROUP BY 1
        ORDER BY note_rows DESC
        """
    ).fetchdf()

    output = {
        "mimic4": {
            "diagnosis_table_columns": con.execute("DESCRIBE dx4").fetchdf()[
                ["column_name", "column_type"]
            ].to_dict(orient="records"),
            "procedure_date_coverage": m4_proc,
            "discharge_note_coverage": m4_discharge,
            "radiology_stroke_keyword_coverage": m4_radiology_stroke,
            "limitations": [
                "Diagnosis rows have no charttime or offset.",
                "Procedure dates are day-level.",
                "Discharge-note charttime is usually note finalization or discharge, not event onset.",
                "Only discharge summaries and radiology reports are present; no general progress-note table.",
            ],
        },
        "mimic3": {
            "procedure_table_columns": m3_proc_columns[
                ["column_name", "column_type"]
            ].to_dict(orient="records"),
            "note_coverage_by_category": m3_notes.to_dict(orient="records"),
            "limitations": [
                "Diagnosis rows have no charttime or offset.",
                "Procedure rows have no date column.",
                "NOTEEVENTS has charttime, but it is note time rather than verified clinical onset.",
            ],
        },
        "conclusion": (
            "MIMIC cannot reproduce eICU diagnosisoffset. Approximate event times "
            "can only be inferred from dated procedures, timed laboratory or order "
            "events, or note text with explicit dates; this requires a separate "
            "validated interval-extraction analysis."
        ),
    }
    destination = PROJECT / "results" / "mimic_event_timing_audit.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

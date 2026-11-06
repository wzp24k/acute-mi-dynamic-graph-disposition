# -*- coding: utf-8 -*-
"""Check eICU keys used by the first-unit rule."""
from __future__ import annotations

import duckdb


def main() -> None:
    con = duckdb.connect()
    path = "D:/TT paper/0811Temporal Pathway/eicu/patient.csv.gz"
    print(
        con.execute(
            f"""
            SELECT
                COUNT(*) AS n,
                COUNT(*) FILTER (WHERE patienthealthsystemstayid IS NULL) AS null_system_stay,
                COUNT(DISTINCT patienthealthsystemstayid) AS n_system_stays,
                COUNT(DISTINCT uniquepid) AS n_patients,
                COUNT(DISTINCT patientunitstayid) AS n_unit_stays
            FROM read_csv_auto('{path}', header=true)
            """
        ).fetchdf().to_string(index=False)
    )


if __name__ == "__main__":
    main()

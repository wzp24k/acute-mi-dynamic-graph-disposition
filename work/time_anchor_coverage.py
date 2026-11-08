"""Quantify MIMIC timing anchors for interval endpoint construction."""
from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(r"D:\TT paper\0811Temporal Pathway")
MIMIC4 = ROOT / "mimic-iv-3.1"
MIMIC3 = ROOT / "mimic-iii-clinical-database-1.4_1" / "mimic-iii-clinical-database-1.4"


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads=12")
    con.execute("SET memory_limit='12GB'")
    for name, path in (
        ("d_lab4", MIMIC4 / "hosp/d_labitems.csv.gz"),
        ("d_lab3", MIMIC3 / "D_LABITEMS.csv.gz"),
    ):
        print(f"\n[{name}: troponin]")
        print(
            con.execute(
                f"""
                SELECT itemid, label, fluid, category
                FROM read_csv_auto('{path.as_posix()}', header=true)
                WHERE lower(label) LIKE '%troponin%'
                ORDER BY itemid
                """
            )
            .fetchdf()
            .to_string(index=False)
        )

    discharge = MIMIC4 / "mimic iv note 2.2/note/discharge.csv.gz"
    con.execute(
        f"CREATE VIEW discharge AS SELECT * FROM read_csv_auto('{discharge.as_posix()}', header=true)"
    )
    print("\n[discharge note timing anchor text]")
    print(
        con.execute(
            """
            SELECT COUNT(*) notes, COUNT(DISTINCT hadm_id) hadm,
                   SUM(
                     CASE WHEN regexp_matches(
                       lower(text),
                       'hospital day|hd ?#? ?[0-9]|postoperative day|pod ?#? ?[0-9]'
                     ) THEN 1 ELSE 0 END
                   ) notes_with_day,
                   SUM(
                     CASE WHEN regexp_matches(
                       lower(text),
                       'acute myocardial infarction|nstemi|stemi'
                     ) AND regexp_matches(
                       lower(text),
                       'hospital day|hd ?#? ?[0-9]|postoperative day|pod ?#? ?[0-9]'
                     ) THEN 1 ELSE 0 END
                   ) mi_with_day
            FROM discharge
            """
        )
        .fetchdf()
        .to_string(index=False)
    )

    radiology = MIMIC4 / "mimic iv note 2.2/note/radiology.csv.gz"
    con.execute(
        f"CREATE VIEW radiology AS SELECT * FROM read_csv_auto('{radiology.as_posix()}', header=true)"
    )
    print("\n[radiology stroke anchors]")
    print(
        con.execute(
            """
            SELECT COUNT(*) notes, COUNT(DISTINCT hadm_id) hadm,
                   SUM(CASE WHEN lower(text) LIKE '%acute infarct%'
                            THEN 1 ELSE 0 END) acute_infarct,
                   SUM(CASE WHEN lower(text) LIKE '%intracranial hemorrhage%'
                             OR lower(text) LIKE '%intraparenchymal hemorrhage%'
                             OR lower(text) LIKE '%subarachnoid hemorrhage%'
                            THEN 1 ELSE 0 END) hemorrhage
            FROM radiology
            """
        )
        .fetchdf()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

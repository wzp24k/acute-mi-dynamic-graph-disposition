"""Inspect source fields used to harmonize organ measurements."""
from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(r"D:\TT paper\0811Temporal Pathway")


def show(title: str, path: Path, sql: str) -> None:
    con = duckdb.connect()
    query = sql.format(path=path.as_posix())
    print(f"\n[{title}]")
    print(con.execute(query).fetchdf().to_string(index=False))
    con.close()


show(
    "MIMIC-IV output items",
    ROOT / "mimic-iv-3.1/icu/d_items.csv.gz",
    """
    SELECT itemid, label, linksto, category, unitname
    FROM read_csv_auto('{path}', header=true)
    WHERE lower(label) LIKE '%urine%'
       OR lower(label) LIKE '%foley%'
       OR lower(label) LIKE '%void%'
    ORDER BY label
    """,
)
show(
    "MIMIC-III output items",
    ROOT
    / "mimic-iii-clinical-database-1.4_1/"
    "mimic-iii-clinical-database-1.4/D_ITEMS.csv.gz",
    """
    SELECT itemid, label, linksto, category, unitname
    FROM read_csv_auto('{path}', header=true)
    WHERE lower(label) LIKE '%urine%'
       OR lower(label) LIKE '%foley%'
       OR lower(label) LIKE '%void%'
    ORDER BY label
    """,
)
show(
    "eICU laboratory names",
    ROOT / "eicu/lab.csv.gz",
    """
    SELECT lower(labname) AS lab, COUNT(*) AS n
    FROM read_csv_auto('{path}', header=true)
    WHERE lower(labname) LIKE '%lactate%'
       OR lower(labname) LIKE '%creatinine%'
       OR lower(labname) LIKE '%platelet%'
       OR lower(labname) LIKE '%inr%'
       OR lower(labname) LIKE '%ptt%'
       OR lower(labname) LIKE '%wbc%'
       OR lower(labname) LIKE '%glucose%'
       OR lower(labname) LIKE '%bicarbonate%'
       OR lower(labname) = 'ph'
    GROUP BY 1
    ORDER BY n DESC
    """,
)
show(
    "eICU nurse fields",
    ROOT / "eicu/nurseCharting.csv.gz",
    """
    SELECT lower(nursingchartcelltypevallabel) AS label,
    lower(nursingchartcelltypevalname) AS item,
    COUNT(*) AS n
    FROM read_csv_auto('{path}', header=true)
    WHERE lower(nursingchartcelltypevallabel) LIKE '%urine%'
       OR lower(nursingchartcelltypevalname) LIKE '%urine%'
       OR lower(nursingchartcelltypevalname) LIKE '%gcs%'
    GROUP BY 1, 2
    ORDER BY n DESC
    """,
)

"""Build harmonized 24-hour organ trajectories and 7-day MACCE cohorts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .common import (
    EICU,
    MIMIC3,
    MIMIC4,
    ORGAN_CHANNELS,
    PROCESSED,
    RESULTS,
    build_sequence_tensors,
    ensure_dirs,
    hourly_medians,
    load_config,
    normalize_hourly,
    sequence_summary,
)


MIMIC_LAB_ITEMS = {
    "lactate": (50813,),
    "creatinine": (50912,),
    "platelets": (51265,),
    "inr": (51237,),
    "aptt": (51275,),
    "wbc": (51301,),
    "glucose": (50809, 50931),
    "ph": (50820,),
    "bicarbonate": (50882,),
}

EICU_LAB_NAMES = {
    "lactate": ("lactate",),
    "creatinine": ("creatinine",),
    "platelets": ("platelets x 1000",),
    "inr": ("pt - inr",),
    "aptt": ("ptt",),
    "wbc": ("wbc x 1000",),
    "glucose": ("glucose", "bedside glucose"),
    "ph": ("ph",),
    "bicarbonate": ("bicarbonate",),
}

MIMIC4_VITALS = {
    220045: "hr",
    220210: "rr",
    224690: "rr",
    220277: "spo2",
    220050: "sbp",
    220179: "sbp",
    220051: "dbp",
    220180: "dbp",
    220052: "mbp",
    220181: "mbp",
    223762: "temp",
    223761: "temp_f",
}
MIMIC3_VITALS = {
    211: "hr",
    220045: "hr",
    615: "rr",
    618: "rr",
    220210: "rr",
    224690: "rr",
    646: "spo2",
    220277: "spo2",
    51: "sbp",
    455: "sbp",
    220050: "sbp",
    220179: "sbp",
    8368: "dbp",
    8441: "dbp",
    220051: "dbp",
    220180: "dbp",
    52: "mbp",
    456: "mbp",
    220052: "mbp",
    220181: "mbp",
    676: "temp",
    678: "temp_f",
    223762: "temp",
    223761: "temp_f",
}
GCS_COMPONENTS = {220739: "eye", 223900: "verbal", 223901: "motor"}
GCS_TOTAL_ITEMS = (198, 226755)

MIMIC4_URINE_ITEMS = (
    226559,
    226560,
    226561,
    226563,
    226564,
    226565,
    226566,
    226567,
    226584,
    226627,
    226631,
    227489,
)
MIMIC3_URINE_ITEMS = (
    40055,
    40056,
    40057,
    40069,
    40085,
    40094,
    40096,
    40405,
    40428,
    40473,
    40651,
    40715,
    226559,
    226560,
    226561,
    226563,
    226564,
    226565,
    226566,
    226567,
    226584,
    226627,
    226631,
    227489,
)

CHANNEL_RANGES = {
    "gcs_total": (3, 15),
    "pupil_reactivity": (0, 2),
    "hr": (20, 260),
    "sbp": (40, 300),
    "dbp": (20, 180),
    "mbp": (30, 220),
    "lactate": (0.1, 30),
    "rr": (2, 80),
    "spo2": (50, 100),
    "fio2": (0.21, 1.0),
    "creatinine": (0.1, 30),
    "urine_output": (0, 5000),
    "platelets": (1, 1500),
    "inr": (0.5, 20),
    "aptt": (5, 300),
    "wbc": (0.05, 300),
    "temp": (30, 44),
    "glucose": (10, 1500),
    "ph": (6.5, 7.9),
    "bicarbonate": (2, 60),
}


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        f"SET threads={max(1, int(os.environ.get('CTOGT_DUCKDB_THREADS', '8')))}"
    )
    con.execute(
        f"SET memory_limit='{os.environ.get('CTOGT_DUCKDB_MEMORY', '12GB')}'"
    )
    return con


def view(
    con: duckdb.DuckDBPyConnection,
    name: str,
    path: Path,
    *,
    all_varchar: bool = False,
) -> None:
    suffix = ", all_varchar=true" if all_varchar else ""
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_csv_auto('{path.as_posix()}', header=true{suffix})"
    )


def frame(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    out = con.execute(sql).fetchdf()
    out.columns = [str(column).lower() for column in out.columns]
    return out


def audit(name: str) -> pd.DataFrame:
    path = RESULTS / f"endpoint_audit_{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run python -m ctogt.audit_endpoint --database all"
        )
    out = pd.read_csv(path)
    out["eligible_24h"] = out["eligible_24h"].astype(str).str.lower().isin(
        ["true", "1"]
    )
    for column in (
        "intime",
        "outtime",
        "admittime",
        "dischtime",
        "deathtime",
    ):
        if column in out:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    return out[out["eligible_24h"]].copy()


def code_series(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(".", "", regex=False)
    )


def any_prefix(codes: pd.Series, prefixes: tuple[str, ...]) -> pd.Series:
    result = pd.Series(False, index=codes.index)
    for prefix in prefixes:
        result |= codes.str.startswith(prefix)
    return result


def diagnosis_flags(
    diagnoses: pd.DataFrame,
    ids: pd.DataFrame,
) -> pd.DataFrame:
    if diagnoses.empty:
        return ids[["stay_id"]].copy()
    work = diagnoses.copy()
    work["code"] = code_series(work["icd_code"])
    format_version = (
        pd.to_numeric(work.get("icd_version"), errors="coerce")
        .fillna(9)
        .astype(int)
    )
    work["v10"] = format_version.eq(10)
    flags = {
        "prior_mi": (("I252", "412"),),
        "prior_stroke": (("I69", "438", "Z8673", "V1254"),),
        "heart_failure": (("I50", "428"),),
        "atrial_fib": (("I48", "42731"),),
        "diabetes": (
            tuple(f"E{number}" for number in range(10, 15))
            + ("250",),
        ),
        "hypertension": (
            ("I10", "I11", "I12", "I13", "I15", "401", "402", "403", "404", "405"),
        ),
        "ckd": (("N18", "585"),),
        "copd": (("J44", "496"),),
        "pad": (("I70", "440"),),
        "smoking": (("F17", "3051", "Z720", "V1582"),),
        "prior_pci": (("Z955", "V4582"),),
        "prior_cabg": (("Z951", "V4581"),),
        "acs_history": (("I20", "I21", "I22", "410", "4111"),),
    }
    rows = []
    for flag, prefix_groups in flags.items():
        matched = pd.Series(False, index=work.index)
        for prefixes in prefix_groups:
            v9_only = format_version.ne(10) & any_prefix(work["code"], prefixes)
            v10_only = format_version.eq(10) & any_prefix(work["code"], prefixes)
            matched |= v9_only | v10_only
        work[flag] = matched.astype(int)
        rows.append(flag)
    result = (
        work.groupby("stay_id", as_index=False)[rows]
        .max()
        .reset_index(drop=True)
    )
    result = ids[["stay_id"]].merge(result, on="stay_id", how="left")
    for column in rows:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
    return result


def mimic4_base(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    selected = audit("mimic4")
    con.register("selected4", selected)
    view(con, "m4_icu", MIMIC4 / "icu/icustays.csv.gz")
    view(con, "m4_adm", MIMIC4 / "hosp/admissions.csv.gz")
    view(con, "m4_pat", MIMIC4 / "hosp/patients.csv.gz")
    context = frame(
        con,
        """
        SELECT c.stay_id, i.first_careunit, i.last_careunit,
               a.admission_type, a.admission_location, a.insurance, a.race,
               a.admittime AS admittime_context,
               a.dischtime AS dischtime_context, p.gender
        FROM selected4 c
        JOIN m4_icu i ON i.stay_id = c.stay_id
        JOIN m4_adm a ON a.hadm_id = c.hadm_id
        JOIN m4_pat p ON p.subject_id = i.subject_id
        """,
    )
    base = selected.merge(context, on="stay_id", how="left", validate="one_to_one")
    base["sex_male"] = (
        base["gender"].fillna("").astype(str).str.upper().eq("M").astype(float)
    )
    race = base["race"].fillna("").astype(str).str.upper()
    base["race_white"] = race.str.contains("WHITE").astype(float)
    base["race_black"] = race.str.contains("BLACK").astype(float)
    base["emergency_admission"] = (
        base["admission_type"].fillna("").astype(str).str.upper().str.contains("EMER")
    ).astype(float)
    base["admission_year"] = pd.to_datetime(
        base["admittime_context"], errors="coerce"
    ).dt.year
    base["environment"] = (
        base["first_careunit"].fillna("unknown").astype(str)
        + "_"
        + pd.cut(
            base["admission_year"],
            bins=[2007, 2012, 2016, 2019, 2022],
            labels=["early", "middle", "late", "latest"],
            include_lowest=True,
        ).astype(str)
    )
    view(con, "m4_dx", MIMIC4 / "hosp/diagnoses_icd.csv.gz")
    diagnoses = frame(
        con,
        """
        SELECT c.stay_id, d.icd_code, d.icd_version
        FROM m4_dx d JOIN selected4 c USING (hadm_id)
        """,
    )
    flags = diagnosis_flags(diagnoses, selected[["stay_id"]])
    return base.merge(flags, on="stay_id", how="left", validate="one_to_one")


def mimic3_base(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    selected = audit("mimic3")
    con.register("selected3", selected)
    view(con, "m3_icu", MIMIC3 / "ICUSTAYS.csv.gz")
    view(con, "m3_adm", MIMIC3 / "ADMISSIONS.csv.gz")
    view(con, "m3_pat", MIMIC3 / "PATIENTS.csv.gz")
    context = frame(
        con,
        """
        SELECT c.stay_id, i.first_careunit, i.last_careunit,
               a.admission_type, a.admission_location, a.insurance,
               a.ethnicity AS race, a.admittime AS admittime_context,
               a.dischtime AS dischtime_context, p.gender
        FROM selected3 c
        JOIN m3_icu i ON i.icustay_id = c.stay_id
        JOIN m3_adm a ON a.hadm_id = c.hadm_id
        JOIN m3_pat p ON p.subject_id = i.subject_id
        """,
    )
    base = selected.merge(context, on="stay_id", how="left", validate="one_to_one")
    base["sex_male"] = (
        base["gender"].fillna("").astype(str).str.upper().eq("M").astype(float)
    )
    race = base["race"].fillna("").astype(str).str.upper()
    base["race_white"] = race.str.contains("WHITE").astype(float)
    base["race_black"] = race.str.contains("BLACK").astype(float)
    base["emergency_admission"] = (
        base["admission_type"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.contains("EMERGENCY")
    ).astype(float)
    base["admission_year"] = pd.to_datetime(
        base["admittime_context"], errors="coerce"
    ).dt.year
    base["environment"] = (
        base["first_careunit"].fillna("unknown").astype(str)
        + "_"
        + pd.cut(
            base["admission_year"],
            bins=[2000, 2003, 2006, 2009, 2013],
            labels=["early", "middle", "late", "latest"],
            include_lowest=True,
        ).astype(str)
    )
    view(con, "m3_dx", MIMIC3 / "DIAGNOSES_ICD.csv.gz")
    diagnoses = frame(
        con,
        """
        SELECT c.stay_id, d.icd9_code AS icd_code, 9 AS icd_version
        FROM m3_dx d JOIN selected3 c USING (hadm_id)
        """,
    )
    flags = diagnosis_flags(diagnoses, selected[["stay_id"]])
    return base.merge(flags, on="stay_id", how="left", validate="one_to_one")


def eicu_base(con: duckdb.DuckDBPyConnection, audit_name: str = "eicu") -> pd.DataFrame:
    selected = audit(audit_name)
    con.register("selected_e", selected)
    view(con, "e_pat", EICU / "patient.csv.gz")
    context = frame(
        con,
        """
        SELECT c.stay_id, p.hospitalid AS hospitalid_context,
               p.unittype, p.unitadmitsource,
               p.ethnicity AS race, p.gender, p.age AS age_context,
               p.hospitaladmitoffset,
               p.hospitaldischargeoffset
        FROM selected_e c
        JOIN e_pat p ON p.patientunitstayid = c.stay_id
        """,
    )
    base = selected.merge(context, on="stay_id", how="left", validate="one_to_one")
    age_text = (
        base["age_context"].fillna("").astype(str).str.strip().str.lower()
    )
    base["age"] = pd.to_numeric(base["age_context"], errors="coerce")
    numeric_text = age_text.str.match(r"^\d+$")
    base.loc[numeric_text, "age"] = pd.to_numeric(
        age_text[numeric_text], errors="coerce"
    )
    base.loc[age_text.str.startswith("> 89"), "age"] = 90
    base["sex_male"] = (
        base["gender"].fillna("").astype(str).str.lower().eq("male").astype(float)
    )
    race = base["race"].fillna("").astype(str).str.lower()
    base["race_white"] = race.str.contains("white").astype(float)
    base["race_black"] = race.str.contains("black").astype(float)
    base["emergency_admission"] = (
        base["unitadmitsource"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains("emergency|ed|transfer")
    ).astype(float)
    base["environment"] = (
        "hospital_"
        + pd.to_numeric(base["hospitalid_context"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .astype(str)
    )
    view(con, "e_dx", EICU / "diagnosis.csv.gz")
    diagnoses = frame(
        con,
        """
        SELECT c.stay_id, d.icd9code AS icd_code, 9 AS icd_version,
               d.diagnosisstring
        FROM e_dx d JOIN selected_e c
          ON d.patientunitstayid = c.stay_id
        """,
    )
    flags = diagnosis_flags(diagnoses, selected[["stay_id"]])
    flag_columns = [column for column in flags if column != "stay_id"]
    text = (
        diagnoses.assign(text=diagnoses["diagnosisstring"].fillna("").str.lower())
        .groupby("stay_id")
        .agg(
            prior_mi_text=("text", lambda x: int(x.str.contains("prior mi").any())),
            prior_stroke_text=(
                "text",
                lambda x: int(
                    x.str.contains("prior stroke|history of stroke|cva history").any()
                ),
            ),
            heart_failure_text=(
                "text",
                lambda x: int(x.str.contains("heart failure|chf").any()),
            ),
            diabetes_text=("text", lambda x: int(x.str.contains("diabetes").any())),
            hypertension_text=(
                "text",
                lambda x: int(x.str.contains("hypertension|htn").any()),
            ),
            ckd_text=(
                "text",
                lambda x: int(x.str.contains("chronic kidney|renal failure").any()),
            ),
            copd_text=("text", lambda x: int(x.str.contains("copd").any())),
        )
        .reset_index()
    )
    flags = flags.merge(text, on="stay_id", how="left")
    for column in flag_columns:
        text_column = f"{column}_text"
        if text_column in flags:
            flags[column] = flags[[column, text_column]].max(axis=1)
    return base.merge(flags, on="stay_id", how="left", validate="one_to_one")


def mimic_labs(
    con: duckdb.DuckDBPyConnection,
    table: str,
    selected: pd.DataFrame,
    max_hour: int,
) -> pd.DataFrame:
    items = sorted({value for values in MIMIC_LAB_ITEMS.values() for value in values})
    data = frame(
        con,
        f"""
        SELECT c.stay_id, lab.itemid::INTEGER AS itemid,
               date_diff('second', c.intime, lab.charttime) / 3600.0
                 AS time_hour,
               try_cast(lab.valuenum AS DOUBLE) AS value
        FROM {table} lab
        JOIN cohort_selected c USING (hadm_id)
        WHERE lab.itemid::INTEGER IN ({','.join(str(v) for v in items)})
          AND lab.charttime >= c.intime
          AND lab.charttime < c.intime + INTERVAL {int(max_hour)} HOUR
          AND try_cast(lab.valuenum AS DOUBLE) IS NOT NULL
        """,
    )
    item_map = {
        item: channel
        for channel, values in MIMIC_LAB_ITEMS.items()
        for item in values
    }
    data["channel"] = data["itemid"].map(item_map)
    data["hour"] = np.floor(
        pd.to_numeric(data["time_hour"], errors="coerce")
    )
    data = data[data["hour"].between(0, int(max_hour) - 1)].copy()
    data["hour"] = data["hour"].astype(int)
    return (
        data.dropna(subset=["channel", "value"])
        .groupby(["stay_id", "hour", "channel"], as_index=False)
        .agg(value=("value", "median"), time_hour=("time_hour", "median"))
    )


def eicu_labs(
    con: duckdb.DuckDBPyConnection,
    selected: pd.DataFrame,
    max_hour: int,
) -> pd.DataFrame:
    names = sorted({value for values in EICU_LAB_NAMES.values() for value in values})
    escaped = ",".join("'" + value.replace("'", "''") + "'" for value in names)
    view(con, "e_lab", EICU / "lab.csv.gz")
    data = frame(
        con,
        f"""
        SELECT c.stay_id, lower(l.labname) AS labname,
               l.labresultoffset / 60.0 AS time_hour,
               try_cast(l.labresult AS DOUBLE) AS value
        FROM e_lab l
        JOIN cohort_selected c ON l.patientunitstayid = c.stay_id
        WHERE lower(l.labname) IN ({escaped})
          AND l.labresultoffset >= 0
          AND l.labresultoffset < {int(max_hour) * 60}
          AND try_cast(l.labresult AS DOUBLE) IS NOT NULL
        """,
    )
    name_map = {
        name: channel
        for channel, values in EICU_LAB_NAMES.items()
        for name in values
    }
    data["channel"] = data["labname"].map(name_map)
    data["hour"] = np.floor(
        pd.to_numeric(data["time_hour"], errors="coerce")
    )
    data = data[data["hour"].between(0, int(max_hour) - 1)].copy()
    data["hour"] = data["hour"].astype(int)
    return (
        data.dropna(subset=["channel", "value"])
        .groupby(["stay_id", "hour", "channel"], as_index=False)
        .agg(value=("value", "median"), time_hour=("time_hour", "median"))
    )


def reconstruct_gcs(events: pd.DataFrame) -> pd.DataFrame:
    totals = events[events["itemid"].isin(GCS_TOTAL_ITEMS)].copy()
    totals["value"] = pd.to_numeric(totals["value"], errors="coerce")
    totals = totals[totals["value"].between(3, 15)]
    components = events[events["itemid"].isin(GCS_COMPONENTS)].copy()
    if components.empty:
        return totals[["stay_id", "time_hour", "value"]]
    components["component"] = components["itemid"].map(GCS_COMPONENTS)
    components["time_key"] = (
        pd.to_numeric(components["time_hour"], errors="coerce").round(6)
    )
    pivot = components.pivot_table(
        index=["stay_id", "time_key"],
        columns="component",
        values="value",
        aggfunc="first",
    ).reindex(columns=["eye", "verbal", "motor"])
    complete = pivot.dropna().sum(axis=1).rename("value").reset_index()
    if not complete.empty:
        complete = complete.rename(columns={"time_key": "time_hour"})
    total_rows = totals[["stay_id", "time_hour", "value"]]
    if complete.empty:
        return total_rows
    return pd.concat(
        [total_rows, complete[["stay_id", "time_hour", "value"]]],
        ignore_index=True,
    )


def mimic_hourly(
    con: duckdb.DuckDBPyConnection,
    database: str,
    selected: pd.DataFrame,
    max_hour: int,
) -> pd.DataFrame:
    is_m4 = database == "mimic4"
    chart = "m4_chartevents" if is_m4 else "m3_chartevents"
    view(
        con,
        chart,
        MIMIC4 / "icu/chartevents.csv.gz"
        if is_m4
        else MIMIC3 / "CHARTEVENTS.csv.gz",
        all_varchar=not is_m4,
    )
    vitals = MIMIC4_VITALS if is_m4 else MIMIC3_VITALS
    items = sorted({*vitals, *GCS_COMPONENTS, *GCS_TOTAL_ITEMS, 223835})
    join = (
        "ce.stay_id = c.stay_id"
        if is_m4
        else "try_cast(ce.icustay_id AS BIGINT) = c.stay_id"
    )
    charttime = "ce.charttime" if is_m4 else "try_cast(ce.charttime AS TIMESTAMP)"
    events = frame(
        con,
        f"""
        SELECT c.stay_id, try_cast(ce.itemid AS INTEGER) AS itemid,
               date_diff('second', c.intime, {charttime}) / 3600.0
                 AS time_hour,
               try_cast(ce.valuenum AS DOUBLE) AS value
        FROM {chart} ce
        JOIN cohort_selected c ON {join}
        WHERE try_cast(ce.itemid AS INTEGER)
                IN ({','.join(str(v) for v in items)})
          AND {charttime} >= c.intime
          AND {charttime} < c.intime + INTERVAL {int(max_hour)} HOUR
        """,
    )
    vital = events[events["itemid"].isin(vitals)].copy()
    vital["channel"] = vital["itemid"].map(vitals)
    fahrenheit = vital["channel"].eq("temp_f")
    vital.loc[fahrenheit, "value"] = (
        vital.loc[fahrenheit, "value"] - 32.0
    ) * 5.0 / 9.0
    vital.loc[fahrenheit, "channel"] = "temp"
    gcs = reconstruct_gcs(events)
    gcs["channel"] = "gcs_total"
    fio2 = events[events["itemid"].eq(223835)][
        ["stay_id", "time_hour", "value"]
    ].copy()
    fio2["channel"] = "fio2"
    fio2.loc[fio2["value"].gt(1.0), "value"] /= 100.0
    events_long = pd.concat(
        [
            vital[["stay_id", "time_hour", "channel", "value"]],
            gcs[["stay_id", "time_hour", "channel", "value"]],
            fio2[["stay_id", "time_hour", "channel", "value"]],
        ],
        ignore_index=True,
    )
    output_table = "m4_output" if is_m4 else "m3_output"
    view(
        con,
        output_table,
        MIMIC4 / "icu/outputevents.csv.gz"
        if is_m4
        else MIMIC3 / "OUTPUTEVENTS.csv.gz",
        all_varchar=not is_m4,
    )
    urine_items = MIMIC4_URINE_ITEMS if is_m4 else MIMIC3_URINE_ITEMS
    output_join = (
        "o.stay_id = c.stay_id"
        if is_m4
        else "try_cast(o.icustay_id AS BIGINT) = c.stay_id"
    )
    output_time = "o.charttime" if is_m4 else "try_cast(o.charttime AS TIMESTAMP)"
    urine = frame(
        con,
        f"""
        SELECT c.stay_id,
               date_diff('second', c.intime, {output_time}) / 3600.0
                 AS time_hour,
               try_cast(o.value AS DOUBLE) AS value
        FROM {output_table} o
        JOIN cohort_selected c ON {output_join}
        WHERE try_cast(o.itemid AS INTEGER)
                IN ({','.join(str(v) for v in urine_items)})
          AND {output_time} >= c.intime
          AND {output_time} < c.intime + INTERVAL {int(max_hour)} HOUR
        """,
    )
    urine["channel"] = "urine_output"
    events_long = pd.concat(
        [
            events_long,
            urine[["stay_id", "time_hour", "channel", "value"]],
        ],
        ignore_index=True,
    )
    labs = mimic_labs(con, "labevents", selected, max_hour)
    events_long = pd.concat(
        [
            events_long,
            labs[["stay_id", "time_hour", "channel", "value"]],
        ],
        ignore_index=True,
    )
    events_long["hour"] = np.floor(
        pd.to_numeric(events_long["time_hour"], errors="coerce")
    )
    return events_long[
        ["stay_id", "hour", "time_hour", "channel", "value"]
    ].dropna()


def eicu_hourly(
    con: duckdb.DuckDBPyConnection,
    selected: pd.DataFrame,
    max_hour: int,
) -> pd.DataFrame:
    view(con, "e_vital", EICU / "vitalPeriodic.csv.gz")
    vital = frame(
        con,
        f"""
        SELECT c.stay_id, v.observationoffset / 60.0 AS time_hour,
               try_cast(v.heartrate AS DOUBLE) AS hr,
               try_cast(v.respiration AS DOUBLE) AS rr,
               try_cast(v.sao2 AS DOUBLE) AS spo2,
               try_cast(v.systemicsystolic AS DOUBLE) AS sbp,
               try_cast(v.systemicdiastolic AS DOUBLE) AS dbp,
               try_cast(v.systemicmean AS DOUBLE) AS mbp,
               try_cast(v.temperature AS DOUBLE) AS temp
        FROM e_vital v
        JOIN cohort_selected c ON v.patientunitstayid = c.stay_id
        WHERE v.observationoffset >= 0
          AND v.observationoffset < {int(max_hour) * 60}
        """,
    )
    vital_long = vital.melt(
        id_vars=["stay_id", "time_hour"],
        value_vars=["hr", "rr", "spo2", "sbp", "dbp", "mbp", "temp"],
        var_name="channel",
        value_name="value",
    )
    view(con, "e_nurse", EICU / "nurseCharting.csv.gz")
    nurse = frame(
        con,
        f"""
        SELECT c.stay_id, n.nursingchartoffset / 60.0 AS time_hour,
               lower(n.nursingchartcelltypevalname) AS item,
               try_cast(n.nursingchartvalue AS DOUBLE) AS value
        FROM e_nurse n
        JOIN cohort_selected c ON n.patientunitstayid = c.stay_id
        WHERE n.nursingchartoffset >= 0
          AND n.nursingchartoffset < {int(max_hour) * 60}
          AND lower(n.nursingchartcelltypevalname) = 'gcs total'
        """,
    )
    nurse["channel"] = "gcs_total"
    labs = eicu_labs(con, selected, max_hour)
    hourly = pd.concat(
        [
            vital_long,
            nurse[["stay_id", "time_hour", "channel", "value"]],
            labs[["stay_id", "time_hour", "channel", "value"]],
        ],
        ignore_index=True,
    )
    hourly["hour"] = np.floor(
        pd.to_numeric(hourly["time_hour"], errors="coerce")
    )
    return hourly[
        ["stay_id", "hour", "time_hour", "channel", "value"]
    ].dropna()


def finalize(
    database: str,
    base: pd.DataFrame,
    hourly: pd.DataFrame,
    min_channels: int,
    max_hour: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    ids = base["stay_id"].astype(int).to_numpy()
    hourly = normalize_hourly(
        hourly,
        ids,
        ORGAN_CHANNELS,
        max_hour,
        CHANNEL_RANGES,
    )
    values, masks, observed, sequence_ids = build_sequence_tensors(
        hourly, ids, ORGAN_CHANNELS, max_hour
    )
    channel_count = (masks.sum(axis=1) > 0).sum(axis=1)
    base = base.copy()
    base["n_channels_24h"] = channel_count
    base["eligible_channels"] = channel_count >= int(min_channels)
    base = base[base["eligible_channels"]].copy()
    keep = set(base["stay_id"].astype(int))
    keep_mask = np.isin(sequence_ids, np.fromiter(keep, dtype=np.int64))
    values = values[keep_mask]
    masks = masks[keep_mask]
    observed = observed[keep_mask]
    sequence_ids = sequence_ids[keep_mask]
    summaries = sequence_summary(values, masks, sequence_ids, ORGAN_CHANNELS)
    base = base.merge(summaries, on="stay_id", how="left", validate="one_to_one")

    for channel in ORGAN_CHANNELS:
        base[f"{channel}_missing"] = base[f"{channel}_n"].eq(0).astype(float)
    base["total_missing_channels"] = base[
        [f"{channel}_missing" for channel in ORGAN_CHANNELS]
    ].sum(axis=1)
    if "age" not in base:
        base["age"] = np.nan
    base["age"] = pd.to_numeric(base["age"], errors="coerce")
    base = base[base["age"].ge(18)].copy()
    keep = set(base["stay_id"].astype(int))
    keep_mask = np.isin(sequence_ids, np.fromiter(keep, dtype=np.int64))
    values = values[keep_mask]
    masks = masks[keep_mask]
    observed = observed[keep_mask]
    sequence_ids = sequence_ids[keep_mask]

    for column in (
        "prior_mi",
        "prior_stroke",
        "heart_failure",
        "atrial_fib",
        "diabetes",
        "hypertension",
        "ckd",
        "copd",
        "pad",
        "smoking",
        "prior_pci",
        "prior_cabg",
        "acs_history",
    ):
        if column not in base:
            base[column] = 0.0
        base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0)
    base["comorbidity_burden"] = base[
        [
            "prior_mi",
            "prior_stroke",
            "heart_failure",
            "atrial_fib",
            "diabetes",
            "hypertension",
            "ckd",
            "copd",
            "pad",
        ]
    ].sum(axis=1)
    for component in ("death_7d", "revasc_7d", "coded_ami", "coded_stroke"):
        if component not in base:
            base[component] = 0
        base[component] = pd.to_numeric(base[component], errors="coerce").fillna(0)
    if database.startswith("eicu"):
        base["ami_7d"] = pd.to_numeric(
            base.get("timed_ami", pd.Series(0, index=base.index)),
            errors="coerce",
        ).fillna(0)
        base["stroke_7d"] = pd.to_numeric(
            base.get("timed_stroke", pd.Series(0, index=base.index)),
            errors="coerce",
        ).fillna(0)
    else:
        base["ami_7d"] = base["coded_ami"]
        base["stroke_7d"] = base["coded_stroke"]
    base["macce7"] = base[
        ["death_7d", "revasc_7d", "ami_7d", "stroke_7d"]
    ].max(axis=1).astype(int)
    base["macce7_timestamped"] = base[
        ["death_7d", "revasc_7d"]
    ].max(axis=1).astype(int)
    if database.startswith("eicu"):
        base["macce7_timestamped"] = base[
            ["death_7d", "ami_7d", "stroke_7d"]
        ].max(axis=1)
    base["database"] = database
    base = base.sort_values("stay_id").reset_index(drop=True)
    order = {int(value): index for index, value in enumerate(base["stay_id"])}
    reorder = np.asarray([order[int(value)] for value in sequence_ids], dtype=int)
    values = values[reorder]
    masks = masks[reorder]
    observed = observed[reorder]
    sequence_ids = sequence_ids[reorder]
    flow = {
        "database": database,
        "audited_n": int(len(audit(database))),
        "minimum_channels": int(min_channels),
        "analytic_n": int(len(base)),
        "macce7_events": int(base["macce7"].sum()),
        "macce7_rate": float(base["macce7"].mean()),
        "timestamped_events": int(base["macce7_timestamped"].sum()),
        "death7_events": int(base["death_7d"].sum()),
        "ami7_events": int(base["ami_7d"].sum()),
        "stroke7_events": int(base["stroke_7d"].sum()),
        "revasc7_events": int(base["revasc_7d"].sum()),
        "channels": list(ORGAN_CHANNELS),
        "sequence_shape": list(values.shape),
    }
    return base, hourly[hourly["stay_id"].isin(keep)], flow


def build(database: str, config: dict) -> None:
    con = connect()
    max_hour = int(config["sequence_hours"])
    if database == "mimic4":
        base = mimic4_base(con)
        selected = base[["stay_id", "hadm_id", "intime"]].copy()
        con.register("cohort_selected", selected)
        view(con, "labevents", MIMIC4 / "hosp/labevents.csv.gz")
        hourly = mimic_hourly(con, database, selected, max_hour)
    elif database == "mimic3":
        base = mimic3_base(con)
        selected = base[["stay_id", "hadm_id", "intime"]].copy()
        con.register("cohort_selected", selected)
        view(con, "labevents", MIMIC3 / "LABEVENTS.csv.gz")
        hourly = mimic_hourly(con, database, selected, max_hour)
    else:
        base = eicu_base(con, database)
        selected = base[["stay_id"]].copy()
        con.register("cohort_selected", selected)
        hourly = eicu_hourly(con, selected, max_hour)
    base, hourly, flow = finalize(
        database,
        base,
        hourly,
        int(config["minimum_channels"]),
        max_hour,
    )
    out = PROCESSED / database
    out.mkdir(parents=True, exist_ok=True)
    base.to_csv(out / "cohort.csv", index=False)
    hourly.to_csv(out / "hourly_24h.csv", index=False)
    values, masks, observed, ids = build_sequence_tensors(
        hourly,
        base["stay_id"].astype(int),
        ORGAN_CHANNELS,
        max_hour,
    )
    np.savez_compressed(
        out / "sequences_24h.npz",
        values=values,
        masks=masks,
        observed=observed,
        stay_id=ids,
        channels=np.asarray(ORGAN_CHANNELS),
    )
    pd.DataFrame([flow]).to_csv(out / "cohort_flow.csv", index=False)
    print(
        f"[{database}] n={flow['analytic_n']:,}, "
        f"events={flow['macce7_events']:,}, sequence={values.shape}",
        flush=True,
    )
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        choices=["mimic4", "mimic3", "eicu", "eicu_anycad", "all"],
        default="all",
    )
    args = parser.parse_args()
    ensure_dirs()
    config = load_config()
    requested = (
        ["mimic4", "mimic3", "eicu"]
        if args.database == "all"
        else [args.database]
    )
    for database in requested:
        build(database, config)


if __name__ == "__main__":
    main()

"""Construct a conservative event-time interval sensitivity endpoint."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .common import EICU, MIMIC3, MIMIC4, PROCESSED, RESULTS, ensure_dirs, load_config


TARGET_START = 24.0
TARGET_END = 168.0

MI_TERMS = (
    "acute myocardial infarction",
    "acute mi",
    "nstemi",
    "stemi",
)
STROKE_TERMS = (
    "acute ischemic stroke",
    "ischemic stroke",
    "hemorrhagic stroke",
    "intracerebral hemorrhage",
    "intraparenchymal hemorrhage",
    "subarachnoid hemorrhage",
    "acute infarct",
)
NEGATION_TERMS = (
    "history of",
    "h/o",
    "prior",
    "old ",
    "s/p",
    "rule out",
    "ruled out",
    "negative for",
    "no acute",
    "without",
    "not ",
)
UNCERTAINTY_TERMS = (
    "cannot be excluded",
    "can still represent",
    "likely represent",
    "concern for",
    "possible",
    "suggestive",
    "may represent",
    "subacute",
)
DAY_PATTERN = re.compile(
    r"(?:hospital day|hd|postoperative day|post-op day|pod)\s*#?\s*(\d+)",
    flags=re.IGNORECASE,
)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET threads=12")
    con.execute("SET memory_limit='12GB'")
    return con


def view(con: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_csv_auto('{path.as_posix()}', header=true)"
    )


def sentence_has_event(sentence: str, terms: tuple[str, ...]) -> bool:
    if not any(term in sentence for term in terms):
        return False
    if any(negation in sentence for negation in NEGATION_TERMS):
        return False
    return True


def evidence_confidence(sentence: str) -> str:
    lowered = sentence.lower()
    if any(term in lowered for term in UNCERTAINTY_TERMS):
        return "probable"
    return "definite"


def extract_hospital_day(
    text: str,
    terms: tuple[str, ...],
) -> tuple[int, str] | None:
    lowered = str(text).lower().replace("\n", " ")
    course_start = lowered.find("brief hospital course")
    if course_start < 0:
        course_start = lowered.find("hospital course")
    if course_start >= 0:
        lowered = lowered[course_start:]
    sentences = re.split(r"(?<=[.!?])\s+", lowered)
    for sentence in sentences:
        if not sentence_has_event(sentence, terms):
            continue
        match = DAY_PATTERN.search(sentence)
        if match:
            return int(match.group(1)), evidence_confidence(sentence)
    return None


def interval_status(lower: float, upper: float) -> tuple[int, int]:
    if not np.isfinite(lower) or not np.isfinite(upper):
        return 0, 0
    lower, upper = sorted([float(lower), float(upper)])
    confirmed = int(lower >= TARGET_START and upper <= TARGET_END)
    overlap = int(upper > TARGET_START and lower <= TARGET_END)
    return confirmed, overlap


def finalize_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in (
        "death_interval",
        "revasc_interval",
        "ami_interval",
        "stroke_interval",
    ):
        if column not in frame:
            frame[column] = False
    interval_columns = (
        "interval_lower_hours",
        "interval_upper_hours",
    )
    if "definite_interval_lower_hours" not in frame:
        frame["definite_interval_lower_hours"] = np.nan
        frame["definite_interval_upper_hours"] = np.nan
    if "specific_interval_lower_hours" not in frame:
        frame["specific_interval_lower_hours"] = frame["interval_lower_hours"]
        frame["specific_interval_upper_hours"] = frame["interval_upper_hours"]
    event_columns = (
        "death_interval",
        "revasc_interval",
        "ami_interval",
        "stroke_interval",
    )
    frame["macce_interval_specific"] = (
        frame[list(event_columns)].any(axis=1).astype(int)
    )
    frame["macce_interval_present"] = (
        frame["interval_lower_hours"].notna()
        & frame["interval_upper_hours"].notna()
    ).astype(int)
    overlap_status = [
        interval_status(lower, upper)[1]
        if present
        else 0
        for lower, upper, present in zip(
            frame["specific_interval_lower_hours"].fillna(np.nan),
            frame["specific_interval_upper_hours"].fillna(np.nan),
            frame["macce_interval_specific"].astype(bool),
        )
    ]
    all_overlap = [
        interval_status(lower, upper)[1]
        for lower, upper in zip(
            frame["interval_lower_hours"],
            frame["interval_upper_hours"],
        )
    ]
    confirmed_status = [
        interval_status(lower, upper)[0]
        if np.isfinite(lower) and np.isfinite(upper)
        else 0
        for lower, upper in zip(
            frame["definite_interval_lower_hours"],
            frame["definite_interval_upper_hours"],
        )
    ]
    frame["macce_interval_confirmed"] = confirmed_status
    frame["macce_interval_overlap"] = overlap_status
    frame["macce_interval_overlap_all"] = all_overlap
    frame["macce_interval_possible"] = (
        frame["macce_interval_overlap"].astype(int)
        - frame["macce_interval_confirmed"].astype(int)
    ).clip(lower=0)
    return frame


def m4_notes(
    con: duckdb.DuckDBPyConnection,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    mi_ids = cohort.loc[cohort["coded_ami"].eq(1), ["stay_id", "hadm_id", "intime", "admittime"]]
    stroke_ids = cohort.loc[
        cohort["coded_stroke"].eq(1), ["stay_id", "hadm_id", "intime"]
    ]
    if not mi_ids.empty:
        con.register("mi_ids4", mi_ids)
        view(con, "discharge4", MIMIC4 / "mimic iv note 2.2/note/discharge.csv.gz")
        notes = con.execute(
            """
            SELECT c.stay_id, c.intime, c.admittime, n.text
            FROM discharge4 n JOIN mi_ids4 c USING (hadm_id)
            """
        ).fetchdf()
        for row in notes.itertuples(index=False):
            extracted = extract_hospital_day(row.text, MI_TERMS)
            if extracted is None:
                continue
            day, confidence = extracted
            admit = pd.Timestamp(row.admittime)
            icu = pd.Timestamp(row.intime)
            lower = (admit + pd.Timedelta(days=day - 1) - icu).total_seconds() / 3600
            upper = (admit + pd.Timedelta(days=day) - icu).total_seconds() / 3600
            rows.append(
                {
                    "stay_id": int(row.stay_id),
                    "component": "ami",
                    "support": "discharge_note_hospital_day",
                    "confidence": confidence,
                    "lower": lower,
                    "upper": upper,
                }
            )
    if not stroke_ids.empty:
        con.register("stroke_ids4", stroke_ids)
        view(con, "radiology4", MIMIC4 / "mimic iv note 2.2/note/radiology.csv.gz")
        notes = con.execute(
            """
            SELECT c.stay_id, c.intime, n.charttime, n.text
            FROM radiology4 n JOIN stroke_ids4 c USING (hadm_id)
            """
        ).fetchdf()
        for row in notes.itertuples(index=False):
            text = str(row.text).lower().replace("\n", " ")
            matched = next(
                (
                    sentence
                    for sentence in re.split(r"(?<=[.!?])\s+", text)
                    if sentence_has_event(sentence, STROKE_TERMS)
                ),
                "",
            )
            if not matched:
                continue
            chart = pd.Timestamp(row.charttime)
            icu = pd.Timestamp(row.intime)
            upper = (chart - icu).total_seconds() / 3600
            lower = upper - 24.0
            rows.append(
                {
                    "stay_id": int(row.stay_id),
                    "component": "stroke",
                    "support": "radiology_acute_infarct_or_hemorrhage",
                    "confidence": evidence_confidence(matched),
                    "lower": lower,
                    "upper": upper,
                }
            )
    return pd.DataFrame(rows)


def m3_notes(
    con: duckdb.DuckDBPyConnection,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    target = cohort.loc[
        cohort["coded_ami"].eq(1) | cohort["coded_stroke"].eq(1),
        ["stay_id", "hadm_id", "intime"],
    ]
    if target.empty:
        return pd.DataFrame()
    con.register("target_ids3", target)
    view(con, "notes3", MIMIC3 / "NOTEEVENTS.csv.gz")
    notes = con.execute(
        """
        SELECT c.stay_id, c.intime, n.category, n.chartdate, n.charttime, n.text
        FROM notes3 n JOIN target_ids3 c USING (hadm_id)
        WHERE lower(n.category) LIKE '%radiology%'
           OR lower(n.category) LIKE '%physician%'
           OR lower(n.category) LIKE '%nursing%'
        """
    ).fetchdf()
    notes.columns = [str(column).lower() for column in notes.columns]
    rows = []
    for row in notes.itertuples(index=False):
        text = str(row.text).lower().replace("\n", " ")
        sentences = re.split(r"(?<=[.!?])\s+", text)
        timestamp = pd.to_datetime(row.charttime, errors="coerce")
        if pd.isna(timestamp):
            timestamp = pd.to_datetime(row.chartdate, errors="coerce")
        if pd.isna(timestamp):
            continue
        icu = pd.Timestamp(row.intime)
        for component, terms, support in (
            ("ami", MI_TERMS, "timed_note_mi"),
            ("stroke", STROKE_TERMS, "timed_note_stroke"),
        ):
            matched = next(
                (
                    sentence
                    for sentence in sentences
                    if sentence_has_event(sentence, terms)
                ),
                "",
            )
            if not matched:
                continue
            upper = (timestamp - icu).total_seconds() / 3600
            lower = upper - 24.0
            rows.append(
                {
                    "stay_id": int(row.stay_id),
                    "component": component,
                    "support": support,
                    "confidence": evidence_confidence(matched),
                    "lower": lower,
                    "upper": upper,
                }
            )
    return pd.DataFrame(rows)


def troponin_events(
    con: duckdb.DuckDBPyConnection,
    table: str,
    ids: pd.DataFrame,
    *,
    database: str,
    threshold_multiplier: float = 1.0,
) -> pd.DataFrame:
    if ids.empty:
        return pd.DataFrame()
    con.register(f"trop_ids_{database}", ids)
    items = (51002, 51003, 52642) if database == "mimic4" else (51002, 51003)
    labs = con.execute(
        f"""
        SELECT c.stay_id, c.intime, l.itemid, l.charttime,
               try_cast(l.valuenum AS DOUBLE) AS value
        FROM {table} l JOIN trop_ids_{database} c USING (hadm_id)
        WHERE l.itemid IN ({','.join(str(v) for v in items)})
          AND l.charttime >= c.intime
          AND l.charttime <= c.intime + INTERVAL 168 HOUR
          AND try_cast(l.valuenum AS DOUBLE) IS NOT NULL
        ORDER BY c.stay_id, l.charttime
        """
    ).fetchdf()
    labs.columns = [str(column).lower() for column in labs.columns]
    if labs.empty:
        return labs
    base_threshold = np.where(
        labs["itemid"].eq(51003),
        0.01,
        0.04,
    )
    labs["threshold"] = base_threshold * float(threshold_multiplier)
    elevated = labs[labs["value"] > labs["threshold"]].copy()
    elevated["hours"] = (
        pd.to_datetime(elevated["charttime"]) - pd.to_datetime(elevated["intime"])
    ).dt.total_seconds() / 3600
    elevated = elevated[elevated["hours"].between(24, 168, inclusive="both")]
    elevated = elevated.sort_values(["stay_id", "hours"]).drop_duplicates("stay_id")
    elevated["component"] = "ami"
    elevated["support"] = "troponin_elevation"
    elevated["confidence"] = "probable"
    elevated["lower"] = elevated["hours"] - 6.0
    elevated["upper"] = elevated["hours"]
    return elevated[["stay_id", "component", "support", "lower", "upper"]]


def eicu_intervals(
    con: duckdb.DuckDBPyConnection,
    cohort: pd.DataFrame,
    *,
    documentation_lag_hours: float = 6.0,
) -> pd.DataFrame:
    if cohort.empty:
        return pd.DataFrame()
    con.register("ids_e_interval", cohort[["stay_id"]])
    view(con, "e_dx_interval", EICU / "diagnosis.csv.gz")
    diagnosis = con.execute(
        """
        WITH e AS (
          SELECT patientunitstayid AS stay_id, diagnosisoffset,
                 lower(diagnosisstring) AS diagnosis,
                 CASE
                   WHEN lower(diagnosisstring) LIKE 'cardiovascular|chest pain / ashd|acute coronary syndrome|acute myocardial infarction%'
                     OR lower(diagnosisstring) LIKE 'surgery|cardiac surgery|myocardial infarction%'
                   THEN 'ami'
                   WHEN lower(diagnosisstring) LIKE 'neurologic|disorders of vasculature|stroke%'
                   THEN 'stroke'
                 END AS component
          FROM e_dx_interval
          WHERE lower(coalesce(diagnosisstring, ''))
                  LIKE 'cardiovascular|chest pain / ashd|acute coronary syndrome|acute myocardial infarction%'
             OR lower(coalesce(diagnosisstring, ''))
                  LIKE 'surgery|cardiac surgery|myocardial infarction%'
             OR lower(coalesce(diagnosisstring, ''))
                  LIKE 'neurologic|disorders of vasculature|stroke%'
        )
        SELECT c.stay_id, e.component,
               MIN(e.diagnosisoffset) / 60.0 AS upper
        FROM e JOIN ids_e_interval c USING (stay_id)
        WHERE e.component IS NOT NULL
        GROUP BY c.stay_id, e.component
        """
    ).fetchdf()
    if diagnosis.empty:
        return diagnosis
    # eICU offsets are documentation-entry times. The lag is a sensitivity
    # parameter, not a validated clinical onset interval.
    diagnosis["lower"] = diagnosis["upper"] - float(documentation_lag_hours)
    diagnosis["support"] = "eicu_diagnosis_offset"
    diagnosis["confidence"] = "probable"
    return diagnosis[
        ["stay_id", "component", "support", "confidence", "lower", "upper"]
    ]


def build_mimic4(
    con: duckdb.DuckDBPyConnection,
    *,
    troponin_threshold_multiplier: float = 1.0,
) -> pd.DataFrame:
    cohort = pd.read_csv(PROCESSED / "mimic4/cohort.csv")
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    cohort["admittime"] = pd.to_datetime(cohort["admittime"], errors="coerce")
    cohort["dischtime"] = pd.to_datetime(cohort["dischtime"], errors="coerce")
    cohort["discharge_hours"] = (
        cohort["dischtime"] - cohort["intime"]
    ).dt.total_seconds() / 3600
    events = []
    death = cohort.loc[cohort["death_hours"].notna(), ["stay_id", "death_hours"]]
    for row in death.itertuples(index=False):
        events.append(
            {
                "stay_id": int(row.stay_id),
                "component": "death",
                "support": "exact_death",
                "confidence": "definite",
                "lower": float(row.death_hours),
                "upper": float(row.death_hours),
            }
        )
    revasc = cohort.loc[
        cohort["first_revasc_date"].notna(), ["stay_id", "intime", "first_revasc_date"]
    ]
    for row in revasc.itertuples(index=False):
        date = pd.Timestamp(row.first_revasc_date).normalize()
        icu = pd.Timestamp(row.intime)
        lower = (date - icu).total_seconds() / 3600
        events.append(
            {
                "stay_id": int(row.stay_id),
                "component": "revasc",
                "support": "procedure_day",
                "confidence": "definite",
                "lower": lower,
                "upper": lower + 24.0,
            }
        )
    events.append(m4_notes(con, cohort))
    view(con, "labevents4_interval", MIMIC4 / "hosp/labevents.csv.gz")
    events.append(
        troponin_events(
            con,
            "labevents4_interval",
            cohort.loc[
                cohort["coded_ami"].eq(1), ["stay_id", "hadm_id", "intime"]
            ],
            database="mimic4",
            threshold_multiplier=troponin_threshold_multiplier,
        )
    )
    return finalize_mimic(
        cohort,
        pd.concat([value for value in events if isinstance(value, pd.DataFrame)], ignore_index=True),
        database="mimic4",
    )


def build_mimic3(
    con: duckdb.DuckDBPyConnection,
    *,
    troponin_threshold_multiplier: float = 1.0,
) -> pd.DataFrame:
    cohort = pd.read_csv(PROCESSED / "mimic3/cohort.csv")
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    cohort["dischtime"] = pd.to_datetime(cohort["dischtime"], errors="coerce")
    cohort["discharge_hours"] = (
        cohort["dischtime"] - cohort["intime"]
    ).dt.total_seconds() / 3600
    events = []
    death = cohort.loc[cohort["death_hours"].notna(), ["stay_id", "death_hours"]]
    for row in death.itertuples(index=False):
        events.append(
            {
                "stay_id": int(row.stay_id),
                "component": "death",
                "support": "exact_death",
                "confidence": "definite",
                "lower": float(row.death_hours),
                "upper": float(row.death_hours),
            }
        )
    events.append(m3_notes(con, cohort))
    view(con, "labevents3_interval", MIMIC3 / "LABEVENTS.csv.gz")
    events.append(
        troponin_events(
            con,
            "labevents3_interval",
            cohort.loc[
                cohort["coded_ami"].eq(1), ["stay_id", "hadm_id", "intime"]
            ],
            database="mimic3",
            threshold_multiplier=troponin_threshold_multiplier,
        )
    )
    return finalize_mimic(
        cohort,
        pd.concat([value for value in events if isinstance(value, pd.DataFrame)], ignore_index=True),
        database="mimic3",
    )


def build_eicu(
    con: duckdb.DuckDBPyConnection,
    database: str = "eicu",
    *,
    documentation_lag_hours: float = 6.0,
) -> pd.DataFrame:
    cohort = pd.read_csv(PROCESSED / database / "cohort.csv")
    events = []
    death = cohort.loc[
        cohort["death_7d"].eq(1), ["stay_id", "hospitaldischargeoffset_x"]
    ]
    for row in death.itertuples(index=False):
        hours = float(row.hospitaldischargeoffset_x) / 60.0
        events.append(
            {
                "stay_id": int(row.stay_id),
                "component": "death",
                "support": "hospital_discharge_offset",
                "confidence": "definite",
                "lower": hours,
                "upper": hours,
            }
        )
    events.append(
        eicu_intervals(
            con,
            cohort,
            documentation_lag_hours=documentation_lag_hours,
        )
    )
    frame = cohort[["stay_id", "macce7"]].copy()
    event_frame = pd.concat(
        [value for value in events if isinstance(value, pd.DataFrame)],
        ignore_index=True,
    )
    if not event_frame.empty:
        lower = event_frame.groupby("stay_id")["lower"].min()
        upper = event_frame.groupby("stay_id")["upper"].max()
        supports = (
            event_frame.groupby("stay_id")["support"]
            .agg(lambda values: "|".join(sorted(set(values))))
        )
        frame["interval_lower_hours"] = frame["stay_id"].map(lower)
        frame["interval_upper_hours"] = frame["stay_id"].map(upper)
        frame["specific_interval_lower_hours"] = frame["interval_lower_hours"]
        frame["specific_interval_upper_hours"] = frame["interval_upper_hours"]
        frame["event_interval_support"] = frame["stay_id"].map(supports)
        definite = event_frame[event_frame["confidence"].eq("definite")]
        frame["definite_interval_lower_hours"] = frame["stay_id"].map(
            definite.groupby("stay_id")["lower"].min()
        )
        frame["definite_interval_upper_hours"] = frame["stay_id"].map(
            definite.groupby("stay_id")["upper"].max()
        )
        confidence_summary = (
            event_frame.groupby("stay_id")["confidence"]
            .agg(lambda values: "|".join(sorted(set(values))))
        )
        frame["event_interval_confidence"] = frame["stay_id"].map(
            confidence_summary
        )
    else:
        frame["interval_lower_hours"] = np.nan
        frame["interval_upper_hours"] = np.nan
        frame["specific_interval_lower_hours"] = np.nan
        frame["specific_interval_upper_hours"] = np.nan
        frame["event_interval_support"] = ""
        frame["definite_interval_lower_hours"] = np.nan
        frame["definite_interval_upper_hours"] = np.nan
        frame["event_interval_confidence"] = ""
    for component in ("death", "revasc", "ami", "stroke"):
        ids = set(
            event_frame.loc[event_frame["component"].eq(component), "stay_id"].astype(int)
        ) if not event_frame.empty else set()
        frame[f"{component}_interval"] = frame["stay_id"].isin(ids)
        definite_ids = set(
            event_frame.loc[
                event_frame["component"].eq(component)
                & event_frame["confidence"].eq("definite"),
                "stay_id",
            ].astype(int)
        ) if not event_frame.empty else set()
        frame[f"{component}_interval_definite"] = frame["stay_id"].isin(
            definite_ids
        )
    frame["database"] = database
    return finalize_intervals(frame)


def finalize_mimic(
    cohort: pd.DataFrame,
    events: pd.DataFrame,
    *,
    database: str,
) -> pd.DataFrame:
    frame = cohort[["stay_id", "macce7", "discharge_hours"]].copy()
    if not events.empty:
        events = events[np.isfinite(events["lower"]) & np.isfinite(events["upper"])]
        lower = events.groupby("stay_id")["lower"].min()
        upper = events.groupby("stay_id")["upper"].max()
        supports = (
            events.groupby("stay_id")["support"]
            .agg(lambda values: "|".join(sorted(set(values))))
        )
        frame["interval_lower_hours"] = frame["stay_id"].map(lower)
        frame["interval_upper_hours"] = frame["stay_id"].map(upper)
        frame["specific_interval_lower_hours"] = frame["interval_lower_hours"]
        frame["specific_interval_upper_hours"] = frame["interval_upper_hours"]
        frame["event_interval_support"] = frame["stay_id"].map(supports)
        if "confidence" not in events:
            events["confidence"] = "probable"
        events["confidence"] = events["confidence"].fillna("probable")
        definite = events[events["confidence"].eq("definite")]
        frame["definite_interval_lower_hours"] = frame["stay_id"].map(
            definite.groupby("stay_id")["lower"].min()
        )
        frame["definite_interval_upper_hours"] = frame["stay_id"].map(
            definite.groupby("stay_id")["upper"].max()
        )
        confidence_summary = (
            events.groupby("stay_id")["confidence"]
            .agg(lambda values: "|".join(sorted(set(values))))
        )
        frame["event_interval_confidence"] = frame["stay_id"].map(
            confidence_summary
        )
    else:
        frame["interval_lower_hours"] = np.nan
        frame["interval_upper_hours"] = np.nan
        frame["specific_interval_lower_hours"] = np.nan
        frame["specific_interval_upper_hours"] = np.nan
        frame["event_interval_support"] = ""
        frame["definite_interval_lower_hours"] = np.nan
        frame["definite_interval_upper_hours"] = np.nan
        frame["event_interval_confidence"] = ""
    for component in ("death", "revasc", "ami", "stroke"):
        ids = set(
            events.loc[events["component"].eq(component), "stay_id"].astype(int)
        ) if not events.empty else set()
        frame[f"{component}_interval"] = frame["stay_id"].isin(ids)
        definite_ids = set(
            events.loc[
                events["component"].eq(component)
                & events["confidence"].eq("definite"),
                "stay_id",
            ].astype(int)
        ) if not events.empty else set()
        frame[f"{component}_interval_definite"] = frame["stay_id"].isin(
            definite_ids
        )

    # Diagnosis-coded events without a time anchor receive the broad
    # admission-to-discharge interval and enter only the possible-event layer.
    missing_timing = frame["macce7"].eq(1) & frame["interval_lower_hours"].isna()
    frame.loc[missing_timing, "interval_lower_hours"] = 0.0
    frame.loc[missing_timing, "interval_upper_hours"] = frame.loc[
        missing_timing, "discharge_hours"
    ].clip(lower=0)
    frame.loc[missing_timing, "event_interval_support"] = "coded_only_broad_admission_to_discharge"
    frame.loc[missing_timing, "event_interval_confidence"] = "probable"
    if "definite_interval_lower_hours" not in frame:
        frame["definite_interval_lower_hours"] = np.nan
        frame["definite_interval_upper_hours"] = np.nan
    frame = finalize_intervals(frame)
    frame["database"] = database
    return frame


def summarize(frame: pd.DataFrame, database: str) -> dict:
    return {
        "database": database,
        "n": int(len(frame)),
        "coded_macce_n": int(frame["macce7"].sum()),
        "event_interval_present_n": int(frame["macce_interval_present"].sum()),
        "specific_anchor_n": int(frame["macce_interval_specific"].sum()),
        "confirmed_24h_to_day7_n": int(frame["macce_interval_confirmed"].sum()),
        "possible_overlap_n": int(frame["macce_interval_possible"].sum()),
        "overlap_total_n": int(frame["macce_interval_overlap"].sum()),
        "overlap_all_intervals_n": int(
            frame["macce_interval_overlap_all"].sum()
        ),
        "confirmed_rate": float(frame["macce_interval_confirmed"].mean()),
        "possible_rate": float(frame["macce_interval_possible"].mean()),
        "support_counts": frame["event_interval_support"]
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(20)
        .to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        choices=["mimic4", "mimic3", "eicu", "eicu_anycad", "all"],
        default="all",
    )
    args = parser.parse_args()
    ensure_dirs()
    con = connect()
    frames = {}
    summaries = []
    requested = (
        ["mimic4", "mimic3", "eicu"]
        if args.database == "all"
        else [args.database]
    )
    existing_summary_path = RESULTS / "interval_endpoint_summary.json"
    if existing_summary_path.is_file() and args.database != "all":
        existing = pd.read_json(existing_summary_path)
        summaries = existing.to_dict(orient="records")
    for database in requested:
        if database == "mimic4":
            frame = build_mimic4(con)
        elif database == "mimic3":
            frame = build_mimic3(con)
        else:
            frame = build_eicu(con, database)
        frame.to_csv(PROCESSED / database / "interval_endpoint.csv", index=False)
        frames[database] = frame
        summaries = [
            row for row in summaries if row.get("database") != database
        ]
        summaries.append(summarize(frame, database))
    (RESULTS / "interval_endpoint_summary.json").write_text(
        pd.Series(summaries).to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    print(pd.Series(summaries).to_string(index=False))


if __name__ == "__main__":
    main()

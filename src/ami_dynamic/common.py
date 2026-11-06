"""Shared contracts, paths, and preprocessing helpers for the AMI study."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(
    os.environ.get("AMI_SOURCE_ROOT", r"D:\TT paper\0811Temporal Pathway")
)
CONFIG_PATH = PROJECT_ROOT / "configs" / "study.json"
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
OUTPUTS = PROJECT_ROOT / "outputs"

DATASETS = ("mimic4", "mimic3", "eicu")
CHANNELS = ("hr", "rr", "spo2", "sbp", "dbp", "mbp", "temp", "gcs")
SEQUENCE_HOURS = 24
CARRY_FORWARD_HOURS = 2

LAB_RANGES = {
    "troponin": (0.0, 500000.0),
    "ck_mb": (0.0, 100000.0),
    "creatinine": (0.1, 30.0),
    "blood_urea_nitrogen": (1.0, 300.0),
    "lactate": (0.1, 50.0),
    "white_blood_cell_count": (0.0, 300.0),
    "hemoglobin": (1.0, 30.0),
    "platelets": (0.0, 2000.0),
    "sodium": (80.0, 200.0),
    "potassium": (1.0, 15.0),
    "glucose": (10.0, 1000.0),
    "inr": (0.1, 30.0),
    "total_bilirubin": (0.1, 80.0),
    "alt": (1.0, 20000.0),
    "ast": (1.0, 20000.0),
    "albumin": (0.5, 8.0),
    "bicarbonate": (1.0, 80.0),
    "chloride": (40.0, 200.0),
}
LAB_NAMES = tuple(LAB_RANGES)
LOG_LABS = {
    "troponin",
    "ck_mb",
    "creatinine",
    "blood_urea_nitrogen",
    "lactate",
    "total_bilirubin",
    "alt",
    "ast",
}
LAB_STATS = ("first", "min", "max", "mean")

MIMIC4_LABS = {
    "troponin_i": 51002,
    "troponin_t": 51003,
    "ck_mb": 50911,
    "creatinine": 50912,
    "blood_urea_nitrogen": 51006,
    "lactate": 50813,
    "white_blood_cell_count": 51301,
    "hemoglobin": 51222,
    "platelets": 51265,
    "sodium": 50983,
    "potassium": 50971,
    "glucose": 50931,
    "inr": 51237,
    "total_bilirubin": 50885,
    "alt": 50861,
    "ast": 50878,
    "albumin": 50862,
    "bicarbonate": 50882,
    "chloride": 50902,
}

EICU_LABS = {
    "troponin_i": ("troponin - I",),
    "troponin_t": ("troponin - T",),
    "ck_mb": ("CK-MB", "CK MB", "creatine kinase mb"),
    "creatinine": ("creatinine",),
    "blood_urea_nitrogen": ("BUN",),
    "lactate": ("lactate",),
    "white_blood_cell_count": ("WBC x 1000",),
    "hemoglobin": ("Hgb", "hemoglobin"),
    "platelets": ("platelets x 1000", "platelets"),
    "sodium": ("sodium",),
    "potassium": ("potassium",),
    "glucose": ("glucose", "bedside glucose"),
    "inr": ("PT - INR",),
    "total_bilirubin": ("total bilirubin",),
    "alt": ("ALT (SGPT)", "ALT"),
    "ast": ("AST (SGOT)", "AST"),
    "albumin": ("albumin",),
    "bicarbonate": ("bicarbonate",),
    "chloride": ("chloride",),
}

PHYSIOLOGIC_RANGES = {
    "hr": (20.0, 250.0),
    "rr": (0.0, 80.0),
    "spo2": (50.0, 100.0),
    "sbp": (40.0, 300.0),
    "dbp": (20.0, 200.0),
    "mbp": (20.0, 250.0),
    "temp": (25.0, 45.0),
    "gcs": (3.0, 15.0),
}

DEMOGRAPHIC_FEATURES = (
    "age",
    "age_gt89",
    "sex_male",
    "race_white",
    "race_black",
    "emergency_admission",
    "care_unit_ccu",
    "care_unit_cvicu",
    "care_unit_micu",
    "care_unit_sicu",
    "care_unit_tsicu",
    "care_unit_neuro",
    "admission_source_er",
    "admission_source_transfer",
)

OUTCOME_5_NAMES = (
    "Home or against advice",
    "Home health care",
    "Inpatient rehabilitation or assisted living",
    "Skilled nursing or other facility",
    "Hospice or in-hospital death",
)
OUTCOME_6_NAMES = (
    "Home or against advice",
    "Home health care",
    "Inpatient rehabilitation or assisted living",
    "Skilled nursing or other facility",
    "Hospice",
    "In-hospital death",
)

MIMIC4_DISPOSITION_5 = {
    "HOME": 0,
    "AGAINST ADVICE": 0,
    "HOME HEALTH CARE": 1,
    "REHAB": 2,
    "ASSISTED LIVING": 2,
    "SKILLED NURSING FACILITY": 3,
    "CHRONIC/LONG TERM ACUTE CARE": 3,
    "ACUTE HOSPITAL": 3,
    "PSYCH FACILITY": 3,
    "OTHER FACILITY": 3,
    "HEALTHCARE FACILITY": 3,
    "HOSPICE": 4,
    "DIED": 4,
}
MIMIC3_DISPOSITION_5 = {
    "HOME": 0,
    "LEFT AGAINST MEDICAL ADVI": 0,
    "HOME HEALTH CARE": 1,
    "HOME WITH HOME IV PROVIDR": 1,
    "REHAB/DISTINCT PART HOSP": 2,
    "SNF": 3,
    "LONG TERM CARE HOSPITAL": 3,
    "DISC-TRAN CANCER/CHLDRN H": 3,
    "SHORT TERM HOSPITAL": 3,
    "DISCH-TRAN TO PSYCH HOSP": 3,
    "OTHER FACILITY": 3,
    "HOSPICE-HOME": 4,
    "HOSPICE-MEDICAL FACILITY": 4,
    "DEAD/EXPIRED": 4,
}
EICU_DISPOSITION_4 = {
    "HOME": 0,
    "REHABILITATION": 1,
    "SKILLED NURSING FACILITY": 2,
    "NURSING HOME": 2,
    "OTHER": 2,
    "OTHER HOSPITAL": 2,
    "OTHER EXTERNAL": 2,
    "DEATH": 3,
    "EXPIRED": 3,
}


def ensure_dirs() -> None:
    for path in (
        PROCESSED,
        RESULTS,
        OUTPUTS,
        RESULTS / "predictions",
        RESULTS / "tables",
        RESULTS / "figures",
        RESULTS / "audits",
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def map_five_level(disposition: object, dataset: str) -> float:
    text = str(disposition or "").strip().upper()
    if dataset == "mimic4":
        return float(MIMIC4_DISPOSITION_5.get(text, np.nan))
    if dataset == "mimic3":
        return float(MIMIC3_DISPOSITION_5.get(text, np.nan))
    raise ValueError("Five-level mapping is only defined for MIMIC cohorts")


def map_six_level(disposition: object, dataset: str) -> float:
    text = str(disposition or "").strip().upper()
    five = map_five_level(text, dataset)
    if not np.isfinite(five):
        return np.nan
    if five < 4:
        return five
    if text in {"HOSPICE", "HOSPICE-HOME", "HOSPICE-MEDICAL FACILITY"}:
        return 4.0
    return 5.0


def five_to_four(level: object) -> float:
    value = pd.to_numeric(pd.Series([level]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        return np.nan
    value = int(value)
    if value == 0:
        return 0.0
    if value in {1, 2}:
        return 1.0
    if value == 3:
        return 2.0
    return 3.0


def five_to_three(level: object) -> float:
    """Map five levels to community, post-acute care, and hospice/death."""
    value = pd.to_numeric(pd.Series([level]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        return np.nan
    value = int(value)
    if value == 0:
        return 0.0
    if value in {1, 2, 3}:
        return 1.0
    return 2.0


def four_to_three(level: object) -> float:
    """Map harmonized four-class transport labels to the common scale."""
    value = pd.to_numeric(pd.Series([level]), errors="coerce").iloc[0]
    if not np.isfinite(value):
        return np.nan
    value = int(value)
    if value == 0:
        return 0.0
    if value in {1, 2}:
        return 1.0
    return 2.0


def parse_eicu_age(value: object) -> float:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.nan
    if text.startswith(">"):
        return 90.0
    try:
        return float(text)
    except ValueError:
        return np.nan


def _flag_from_text(series: pd.Series, pattern: str) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.contains(pattern, regex=True)
        .astype(float)
    )


def add_demographic_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    race = out.get("race", pd.Series("", index=out.index))
    care_unit = out.get("first_careunit", pd.Series("", index=out.index))
    admission_type = out.get("admission_type", pd.Series("", index=out.index))
    source = out.get("admission_source", pd.Series("", index=out.index))
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["age_gt89"] = out["age"].ge(90).astype(float)
    out["sex_male"] = (
        out.get("sex", out.get("gender", pd.Series("", index=out.index)))
        .fillna("")
        .astype(str)
        .str.upper()
        .isin({"M", "MALE"})
        .astype(float)
    )
    out["race_white"] = _flag_from_text(race, r"WHITE|CAUCASIAN")
    out["race_black"] = _flag_from_text(race, r"BLACK|AFRICAN")
    out["emergency_admission"] = (
        _flag_from_text(admission_type, r"EMER|URGENT")
        .combine(_flag_from_text(source, r"EMERGENCY|ER"), max)
    )
    care_upper = care_unit.fillna("").astype(str).str.upper()
    out["care_unit_ccu"] = care_upper.str.contains(r"(?:^| )CCU(?:$| )").astype(float)
    out["care_unit_cvicu"] = care_upper.str.contains("CVICU|CICU").astype(float)
    out["care_unit_micu"] = care_upper.str.contains("MICU").astype(float)
    out["care_unit_sicu"] = care_upper.str.contains("SICU").astype(float)
    out["care_unit_tsicu"] = care_upper.str.contains("TSICU|TRAUMA").astype(float)
    out["care_unit_neuro"] = care_upper.str.contains("NEURO").astype(float)
    out["admission_source_er"] = _flag_from_text(source, r"EMERGENCY|ER")
    out["admission_source_transfer"] = _flag_from_text(
        source, r"TRANSFER|SNF|OTHER HOSPITAL"
    )
    return out


def clean_events(
    frame: pd.DataFrame,
    *,
    stay_col: str = "stay_id",
    time_col: str = "time_hour",
    channel_col: str = "channel",
    value_col: str = "value",
    max_hour: int = SEQUENCE_HOURS,
) -> pd.DataFrame:
    work = frame.copy()
    keep = pd.Series(False, index=work.index)
    for channel, (low, high) in PHYSIOLOGIC_RANGES.items():
        mask = work[channel_col].eq(channel)
        work.loc[mask, value_col] = pd.to_numeric(
            work.loc[mask, value_col], errors="coerce"
        )
        work.loc[mask, value_col] = work.loc[mask, value_col].where(
            work.loc[mask, value_col].between(low, high)
        )
        keep |= mask
    work = work[keep].dropna(subset=[stay_col, time_col, value_col])
    work[stay_col] = pd.to_numeric(work[stay_col], errors="raise").astype(np.int64)
    work["hour"] = np.floor(
        pd.to_numeric(work[time_col], errors="coerce")
    ).astype("Int64")
    work = work[work["hour"].between(0, max_hour - 1)].copy()
    return work


def hourly_last_by_priority(events: pd.DataFrame) -> pd.DataFrame:
    """Keep the last valid value per stay-hour-channel and source priority."""
    if events.empty:
        return pd.DataFrame(columns=["stay_id", "hour", "channel", "value"])
    work = events.copy()
    if "priority" not in work:
        work["priority"] = 0
    if "event_time" not in work:
        work["event_time"] = work["hour"]
    work["priority"] = pd.to_numeric(work["priority"], errors="coerce").fillna(0)
    work = work.sort_values(
        ["stay_id", "hour", "channel", "priority", "event_time"]
    )
    # Lower numeric priority is preferred. Use the highest-priority source that
    # exists in an hour, then retain its last observation.
    min_priority = work.groupby(["stay_id", "hour", "channel"])[
        "priority"
    ].transform("min")
    work = work[work["priority"].eq(min_priority)]
    work = work.drop_duplicates(
        ["stay_id", "hour", "channel"], keep="last"
    )
    return work[["stay_id", "hour", "channel", "value"]].reset_index(drop=True)


def build_sequence_tensor(
    hourly: pd.DataFrame,
    stay_ids: Iterable[int],
    *,
    channels: Iterable[str] = CHANNELS,
    max_hour: int = SEQUENCE_HOURS,
    carry_hours: int = CARRY_FORWARD_HOURS,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build values, observed masks, and IDs with past-only carry.

    Carried values enter the value tensor for imputation continuity but their
    mask remains zero. Never-observed cells remain NaN for training-fold mean
    imputation.
    """
    channel_list = list(channels)
    ids = [int(value) for value in stay_ids]
    values = np.full((len(ids), max_hour, len(channel_list)), np.nan, dtype=np.float32)
    masks = np.zeros_like(values, dtype=np.float32)
    positions = {stay_id: index for index, stay_id in enumerate(ids)}
    work = hourly[
        hourly["channel"].isin(channel_list)
        & hourly["hour"].between(0, max_hour - 1)
    ].copy()
    for row in work.itertuples(index=False):
        row_index = positions.get(int(row.stay_id))
        if row_index is None:
            continue
        channel_index = channel_list.index(row.channel)
        hour = int(row.hour)
        value = float(row.value)
        values[row_index, hour, channel_index] = value
        masks[row_index, hour, channel_index] = 1.0

    if carry_hours > 0:
        for row_index in range(len(ids)):
            for channel_index in range(len(channel_list)):
                series = values[row_index, :, channel_index]
                observed = masks[row_index, :, channel_index] > 0
                last_value = np.nan
                last_hour = -10_000
                for hour in range(max_hour):
                    if observed[hour] and np.isfinite(series[hour]):
                        last_value = series[hour]
                        last_hour = hour
                    elif (
                        np.isfinite(last_value)
                        and hour - last_hour <= carry_hours
                        and not np.isfinite(series[hour])
                    ):
                        series[hour] = last_value
    return values, masks, ids


def summarize_timeseries(
    values: np.ndarray,
    masks: np.ndarray,
    stay_ids: Iterable[int],
    channels: Iterable[str] = CHANNELS,
) -> pd.DataFrame:
    channel_list = list(channels)
    records: list[dict[str, float | int]] = []
    for row_index, stay_id in enumerate(stay_ids):
        record: dict[str, float | int] = {"stay_id": int(stay_id)}
        for channel_index, channel in enumerate(channel_list):
            sequence = values[row_index, :, channel_index]
            observed = masks[row_index, :, channel_index] > 0
            observed_values = sequence[observed & np.isfinite(sequence)]
            record[f"{channel}_count"] = int(len(observed_values))
            for stat in ("first", "last", "min", "max", "mean"):
                record[f"{channel}_{stat}"] = np.nan
            if len(observed_values):
                record[f"{channel}_first"] = float(observed_values[0])
                record[f"{channel}_last"] = float(observed_values[-1])
                record[f"{channel}_min"] = float(np.min(observed_values))
                record[f"{channel}_max"] = float(np.max(observed_values))
                record[f"{channel}_mean"] = float(np.mean(observed_values))
        record["gcs_change"] = record["gcs_last"] - record["gcs_first"]
        records.append(record)
    return pd.DataFrame(records)


def summarize_labs(
    long: pd.DataFrame,
    all_stay_ids: pd.Series,
) -> pd.DataFrame:
    """Summarize pre-landmark labs and retain explicit missingness at a row level."""
    rows = []
    for stay_id, group in long.groupby("stay_id", sort=False):
        rows.append((int(stay_id), group))
    grouped = {stay_id: group for stay_id, group in rows}
    records: list[dict[str, float | int]] = []
    for stay_id in pd.to_numeric(all_stay_ids, errors="raise").astype(np.int64):
        row: dict[str, float | int] = {"stay_id": int(stay_id)}
        group = grouped.get(int(stay_id), long.iloc[0:0])
        for lab in LAB_NAMES:
            values = pd.to_numeric(
                group.loc[group["lab"].eq(lab), "value"], errors="coerce"
            ).dropna()
            row[f"lab_{lab}_missing"] = int(len(values) == 0)
            for stat in LAB_STATS:
                row[f"lab_{lab}_{stat}"] = np.nan
            if len(values):
                row[f"lab_{lab}_first"] = float(values.iloc[0])
                row[f"lab_{lab}_min"] = float(values.min())
                row[f"lab_{lab}_max"] = float(values.max())
                row[f"lab_{lab}_mean"] = float(values.mean())
        records.append(row)
    out = pd.DataFrame(records)
    out["troponin_assay_t"] = (
        long.groupby("stay_id")["assay_t"]
        .max()
        .reindex(pd.to_numeric(all_stay_ids, errors="raise").astype(np.int64))
        .fillna(0)
        .to_numpy(dtype=float)
    )
    return out


def static_feature_columns(frame: pd.DataFrame) -> list[str]:
    lab_columns = [
        column
        for column in frame.columns
        if column.startswith("lab_") and column != "lab_rows"
    ]
    summary_columns = [
        column
        for channel in CHANNELS
        for column in (
            f"{channel}_first",
            f"{channel}_last",
            f"{channel}_min",
            f"{channel}_max",
            f"{channel}_mean",
            f"{channel}_count",
        )
        if column in frame
    ]
    return list(DEMOGRAPHIC_FEATURES) + lab_columns + ["troponin_assay_t"] + summary_columns + ["gcs_change"]


def transform_lab_for_model(name: str, value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    if name in LOG_LABS:
        return float(np.log1p(max(value, 0.0)))
    return float(value)


def dataset_dir(name: str) -> Path:
    path = PROCESSED / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_processed(
    name: str,
    cohort: pd.DataFrame,
    hourly: pd.DataFrame,
    labs_long: pd.DataFrame,
    flow: pd.DataFrame,
    *,
    max_hour: int = SEQUENCE_HOURS,
) -> None:
    outdir = dataset_dir(name)
    cohort = cohort.sort_values("stay_id").reset_index(drop=True)
    values, masks, ids = build_sequence_tensor(
        hourly, cohort["stay_id"], max_hour=max_hour
    )
    summaries = summarize_timeseries(values, masks, ids)
    labs = summarize_labs(labs_long, cohort["stay_id"])
    cohort = cohort.merge(summaries, on="stay_id", how="left", validate="one_to_one")
    cohort = cohort.merge(labs, on="stay_id", how="left", validate="one_to_one")
    cohort.to_csv(outdir / "cohort.csv", index=False)
    hourly.to_csv(outdir / "hourly.csv", index=False)
    labs_long.to_csv(outdir / "labs_long.csv", index=False)
    flow.to_csv(outdir / "cohort_flow.csv", index=False)
    np.savez_compressed(
        outdir / "sequences.npz",
        values=values,
        masks=masks,
        stay_id=np.asarray(ids, dtype=np.int64),
        channels=np.asarray(CHANNELS),
    )
    audit = {
        "dataset": name,
        "n": int(len(cohort)),
        "outcome_counts": {
            str(int(key)): int(value)
            for key, value in cohort["outcome"].value_counts().sort_index().items()
        },
        "sequence_shape": list(values.shape),
        "observed_fraction": float(masks.mean()),
        "mean_channels_observed": float(
            (masks.sum(axis=1) > 0).sum(axis=1).mean()
        ),
        "at_least_4_of_8": int(
            ((masks.sum(axis=1) > 0).sum(axis=1) >= 4).sum()
        ),
        "troponin_available": int(
            cohort["lab_troponin_missing"].eq(0).sum()
        ),
        "files": {
            filename: sha256_file(outdir / filename)
            for filename in (
                "cohort.csv",
                "hourly.csv",
                "labs_long.csv",
                "cohort_flow.csv",
                "sequences.npz",
            )
        },
    }
    (outdir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

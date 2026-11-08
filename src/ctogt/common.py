"""Shared paths, constants, and utility functions."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(
    r"D:\TT paper\0811Temporal Pathway"
)

MIMIC4 = SOURCE_ROOT / "mimic-iv-3.1"
MIMIC3 = (
    SOURCE_ROOT
    / "mimic-iii-clinical-database-1.4_1"
    / "mimic-iii-clinical-database-1.4"
)
EICU = SOURCE_ROOT / "eicu"

CONFIG_PATH = PROJECT_ROOT / "configs" / "study.json"
WORK = PROJECT_ROOT / "work"
RESULTS = PROJECT_ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
PROCESSED = PROJECT_ROOT / "data" / "processed"
PREDICTIONS = RESULTS / "predictions"
OUTPUTS = PROJECT_ROOT / "outputs"

DATABASES = ("mimic4", "mimic3", "eicu")

ORGAN_CHANNELS = (
    "gcs_total",
    "pupil_reactivity",
    "hr",
    "sbp",
    "dbp",
    "mbp",
    "lactate",
    "rr",
    "spo2",
    "fio2",
    "creatinine",
    "urine_output",
    "platelets",
    "inr",
    "aptt",
    "wbc",
    "temp",
    "glucose",
    "ph",
    "bicarbonate",
)

ORGAN_GROUPS = {
    "neurologic": ("gcs_total", "pupil_reactivity"),
    "circulatory": ("hr", "sbp", "dbp", "mbp", "lactate"),
    "respiratory": ("rr", "spo2", "fio2"),
    "renal": ("creatinine", "urine_output"),
    "coagulation": ("platelets", "inr", "aptt"),
    "inflammation": ("wbc", "temp"),
    "metabolic": ("glucose", "ph", "bicarbonate"),
}

CHANNEL_TO_ORGAN = {
    channel: organ for organ, channels in ORGAN_GROUPS.items() for channel in channels
}

CLINICAL_PRIOR_EDGES = (
    ("circulatory", "neurologic"),
    ("circulatory", "renal"),
    ("circulatory", "respiratory"),
    ("respiratory", "neurologic"),
    ("respiratory", "metabolic"),
    ("renal", "metabolic"),
    ("renal", "coagulation"),
    ("coagulation", "inflammation"),
    ("inflammation", "circulatory"),
    ("metabolic", "neurologic"),
)


def ensure_dirs() -> None:
    for path in (
        WORK,
        RESULTS,
        TABLES,
        FIGURES,
        PROCESSED,
        PREDICTIONS,
        OUTPUTS,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def to_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce")


def clean_numeric(
    values: pd.Series | np.ndarray,
    low: float | None = None,
    high: float | None = None,
) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if low is not None:
        numeric[numeric < low] = np.nan
    if high is not None:
        numeric[numeric > high] = np.nan
    return numeric


def binary_metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    labels = np.asarray(labels, dtype=int)
    scores = np.clip(np.asarray(scores, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, scores)),
        "prevalence": float(labels.mean()),
    }


def hourly_medians(
    events: pd.DataFrame,
    *,
    value_column: str = "value",
    time_column: str = "time_hour",
    max_hour: int = 24,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["stay_id", "hour", value_column])
    work = events.dropna(subset=["stay_id", time_column, value_column]).copy()
    work["hour"] = np.floor(
        pd.to_numeric(work[time_column], errors="coerce")
    )
    work = work[work["hour"].between(0, max_hour - 1)]
    if work.empty:
        return pd.DataFrame(columns=["stay_id", "hour", value_column])
    work["hour"] = work["hour"].astype(int)
    return (
        work.groupby(["stay_id", "hour"], as_index=False)[value_column]
        .median()
        .sort_values(["stay_id", "hour"])
        .reset_index(drop=True)
    )


def build_sequence_tensors(
    hourly: pd.DataFrame,
    stay_ids: Iterable[int],
    channels: Iterable[str],
    max_hour: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channel_list = list(channels)
    ids = np.asarray([int(value) for value in stay_ids], dtype=np.int64)
    shape = (len(ids), int(max_hour), len(channel_list))
    values = np.full(shape, np.nan, dtype=np.float32)
    masks = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=np.float32)
    position = {int(stay_id): index for index, stay_id in enumerate(ids)}
    channel_position = {
        str(channel): index for index, channel in enumerate(channel_list)
    }
    for row in hourly.itertuples(index=False):
        patient_index = position.get(int(row.stay_id))
        if patient_index is None:
            continue
        channel_index = channel_position.get(str(row.channel))
        if channel_index is None:
            continue
        hour = int(row.hour)
        if not 0 <= hour < max_hour:
            continue
        value = float(row.value)
        if not math.isfinite(value):
            continue
        values[patient_index, hour, channel_index] = value
        masks[patient_index, hour, channel_index] = 1.0
        observed[patient_index, hour, channel_index] = float(
            getattr(row, "observed", 1.0)
        )
    return values, masks, observed, ids


def sequence_summary(
    values: np.ndarray,
    masks: np.ndarray,
    stay_ids: Iterable[int],
    channels: Iterable[str],
) -> pd.DataFrame:
    channel_list = list(channels)
    rows: list[dict[str, float | int]] = []
    for index, stay_id in enumerate(stay_ids):
        record: dict[str, float | int] = {"stay_id": int(stay_id)}
        for channel_index, channel in enumerate(channel_list):
            channel_values = values[index, :, channel_index]
            valid = (masks[index, :, channel_index] > 0) & np.isfinite(
                channel_values
            )
            observed_values = channel_values[valid]
            record[f"{channel}_n"] = int(valid.sum())
            record[f"{channel}_mean"] = (
                float(observed_values.mean()) if observed_values.size else np.nan
            )
            record[f"{channel}_min"] = (
                float(observed_values.min()) if observed_values.size else np.nan
            )
            record[f"{channel}_max"] = (
                float(observed_values.max()) if observed_values.size else np.nan
            )
            record[f"{channel}_sd"] = (
                float(observed_values.std(ddof=1))
                if observed_values.size > 1
                else np.nan
            )
            record[f"{channel}_slope"] = np.nan
            if observed_values.size > 1:
                hours = np.flatnonzero(valid).astype(float)
                record[f"{channel}_slope"] = float(
                    np.polyfit(hours, observed_values, 1)[0]
                )
        rows.append(record)
    return pd.DataFrame(rows)


def normalize_hourly(
    hourly: pd.DataFrame,
    stay_ids: Iterable[int],
    channels: Iterable[str],
    max_hour: int,
    ranges: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    channel_list = list(channels)
    work = hourly[
        hourly["channel"].isin(channel_list)
        & hourly["stay_id"].isin([int(value) for value in stay_ids])
    ].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "stay_id",
                "hour",
                "time_hour",
                "channel",
                "value",
                "observed",
                "carried",
            ]
        )
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    for channel, (low, high) in ranges.items():
        selected = work["channel"].eq(channel)
        work.loc[selected, "value"] = work.loc[selected, "value"].where(
            work.loc[selected, "value"].between(low, high)
        )
    work = work.dropna(subset=["value"]).copy()
    work = (
        work.groupby(["stay_id", "hour", "channel"], as_index=False)
        .agg(value=("value", "median"), time_hour=("time_hour", "median"))
    )
    if work.empty:
        return pd.DataFrame(
            columns=[
                "stay_id",
                "hour",
                "time_hour",
                "channel",
                "value",
                "observed",
                "carried",
            ]
        )
    work = work[work["hour"].between(0, max_hour - 1)].copy()
    work["observed"] = 1.0
    work["carried"] = 0.0
    return work[
        ["stay_id", "hour", "time_hour", "channel", "value", "observed", "carried"]
    ].reset_index(drop=True)


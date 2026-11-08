"""Retrain the tabular temporal benchmark on interval-censored endpoints."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .common import PROCESSED, RESULTS, ensure_dirs, load_config
from .train_evaluate import (
    ALL_FEATURES,
    fit_xgboost,
    load_dataset,
    metric_block,
)


def merge_endpoint(dataset, cohort: str):
    endpoint = pd.read_csv(PROCESSED / cohort / "interval_endpoint.csv")
    frame = dataset.cohort.merge(
        endpoint,
        on="stay_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_interval"),
    )
    return frame


def cv_predict(dataset, frame, outcome: str, config: dict):
    labels = frame[outcome].fillna(0).astype(int).to_numpy()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    probability = np.full(len(frame), np.nan)
    features = dataset.frame(ALL_FEATURES)
    for train, test in splitter.split(features, labels):
        probability[test] = fit_xgboost(
            features.iloc[train],
            features.iloc[test],
            labels[train],
            config,
            7,
        )
    return labels, probability


def external_predict(development, development_frame, external, external_frame, outcome, config):
    labels = external_frame[outcome].fillna(0).astype(int).to_numpy()
    probability = fit_xgboost(
        development.frame(ALL_FEATURES),
        external.frame(ALL_FEATURES),
        development_frame["macce_interval_confirmed"].astype(int).to_numpy(),
        config,
        7,
    )
    return labels, probability


def main() -> None:
    ensure_dirs()
    config = load_config()
    development = load_dataset("mimic4")
    development_frame = merge_endpoint(development, "mimic4")
    output = {"mimic4": {}, "external": {}}
    for outcome in ("macce_interval_confirmed", "macce_interval_overlap"):
        events = int(development_frame[outcome].fillna(0).sum())
        if events < 100:
            output["mimic4"][outcome] = {
                "status": "insufficient_events",
                "events": events,
            }
            continue
        labels, probability = cv_predict(
            development, development_frame, outcome, config
        )
        result = metric_block(labels, probability)
        result["events"] = int(labels.sum())
        output["mimic4"][outcome] = result

    for external_name in ("mimic3", "eicu", "eicu_anycad"):
        external = load_dataset(external_name)
        external_frame = merge_endpoint(external, external_name)
        output["external"][external_name] = {}
        for outcome in ("macce_interval_confirmed", "macce_interval_overlap"):
            labels = external_frame[outcome].fillna(0).astype(int).to_numpy()
            if labels.sum() < 20:
                output["external"][external_name][outcome] = {
                    "status": "insufficient_events",
                    "events": int(labels.sum()),
                }
                continue
            probability = fit_xgboost(
                development.frame(ALL_FEATURES),
                external.frame(ALL_FEATURES),
                development_frame[outcome].astype(int).to_numpy(),
                config,
                7,
            )
            result = metric_block(labels, probability)
            result["events"] = int(labels.sum())
            output["external"][external_name][outcome] = result
    (RESULTS / "interval_model_analysis.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

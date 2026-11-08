"""Develop a stronger temporal model on early years, then test late years."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from .common import ORGAN_CHANNELS, ORGAN_GROUPS, PREDICTIONS, RESULTS, ensure_dirs, load_config
from .temporal_holdout import load_years
from .train_evaluate import (
    ALL_FEATURES,
    bootstrap_paired_auc,
    fit_xgboost,
    load_dataset,
    metric_block,
)


ADVERSE_DIRECTION = {
    "gcs_total": -1,
    "pupil_reactivity": -1,
    "hr": 1,
    "sbp": -1,
    "dbp": -1,
    "mbp": -1,
    "lactate": 1,
    "rr": 1,
    "spo2": -1,
    "fio2": 1,
    "creatinine": 1,
    "urine_output": -1,
    "platelets": -1,
    "inr": 1,
    "aptt": 1,
    "wbc": 1,
    "temp": 1,
    "glucose": 1,
    "ph": -1,
    "bicarbonate": -1,
}


def enhanced_features(dataset) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = dataset.frame(ALL_FEATURES).copy()
    for channel in ORGAN_CHANNELS:
        minimum = pd.to_numeric(frame.get(f"{channel}_min"), errors="coerce")
        maximum = pd.to_numeric(frame.get(f"{channel}_max"), errors="coerce")
        mean = pd.to_numeric(frame.get(f"{channel}_mean"), errors="coerce")
        slope = pd.to_numeric(frame.get(f"{channel}_slope"), errors="coerce")
        frame[f"{channel}_range"] = maximum - minimum
        frame[f"{channel}_relative_slope"] = (
            slope / mean.abs().clip(lower=1e-6)
        ).clip(-5, 5)
        frame[f"{channel}_adverse_slope"] = (
            ADVERSE_DIRECTION[channel]
            * frame[f"{channel}_relative_slope"]
            > 0
        ).astype(float)
        frame.loc[frame[f"{channel}_slope"].isna(), f"{channel}_adverse_slope"] = np.nan

    frame["shock_index_mean"] = frame["hr_mean"] / frame["sbp_mean"].clip(lower=1)
    frame["shock_index_extreme"] = frame["hr_max"] / frame["sbp_min"].clip(lower=1)
    frame["pulse_pressure_mean"] = frame["sbp_mean"] - frame["dbp_mean"]
    frame["spo2_fio2_ratio"] = frame["spo2_mean"] / frame["fio2_mean"].clip(lower=0.21)
    frame["platelet_inr_ratio"] = frame["platelets_min"] / frame["inr_max"].clip(lower=0.5)
    frame["renal_metabolic_product"] = (
        frame["creatinine_max"].clip(lower=0.1)
        * frame["lactate_max"].clip(lower=0.1)
    )

    derived = []
    for organ, channels in ORGAN_GROUPS.items():
        observed = pd.concat(
            [
                frame[f"{channel}_n"].fillna(0).gt(0).astype(float)
                for channel in channels
            ],
            axis=1,
        )
        adverse = frame[[f"{channel}_adverse_slope" for channel in channels]]
        relative = frame[[f"{channel}_relative_slope" for channel in channels]]
        organ_observed = observed.mean(axis=1)
        organ_adverse = adverse.sum(axis=1, min_count=1)
        organ_mean_slope = relative.mean(axis=1)
        organ_worst_slope = relative.max(axis=1)
        frame[f"{organ}_observed_fraction"] = organ_observed
        frame[f"{organ}_adverse_slope_count"] = organ_adverse
        frame[f"{organ}_adverse_slope_fraction"] = (
            organ_adverse / observed.sum(axis=1).replace(0, np.nan)
        )
        frame[f"{organ}_mean_relative_slope"] = organ_mean_slope
        frame[f"{organ}_worst_relative_slope"] = organ_worst_slope
        derived.extend(
            [
                f"{organ}_observed_fraction",
                f"{organ}_adverse_slope_count",
                f"{organ}_adverse_slope_fraction",
                f"{organ}_mean_relative_slope",
                f"{organ}_worst_relative_slope",
            ]
        )

    ratio_features = (
        "shock_index_mean",
        "shock_index_extreme",
        "pulse_pressure_mean",
        "spo2_fio2_ratio",
        "platelet_inr_ratio",
        "renal_metabolic_product",
    )
    range_features = tuple(f"{channel}_range" for channel in ORGAN_CHANNELS)
    relative_features = tuple(
        f"{channel}_relative_slope" for channel in ORGAN_CHANNELS
    )
    adverse_features = tuple(
        f"{channel}_adverse_slope" for channel in ORGAN_CHANNELS
    )
    feature_names = (
        *ALL_FEATURES,
        *range_features,
        *relative_features,
        *adverse_features,
        *ratio_features,
        *derived,
    )
    return frame.loc[:, list(feature_names)], feature_names


def candidate_configs(config: dict) -> list[dict[str, object]]:
    base = copy.deepcopy(config["xgboost"])
    deeper = {
        **base,
        "n_estimators": 620,
        "max_depth": 4,
        "learning_rate": 0.02,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "reg_lambda": 4.0,
    }
    return [
        {"name": "base_features_base_params", "features": "base", "xgboost": base},
        {
            "name": "enhanced_features_base_params",
            "features": "enhanced",
            "xgboost": base,
        },
        {
            "name": "enhanced_features_deeper",
            "features": "enhanced",
            "xgboost": deeper,
        },
    ]


def main() -> None:
    ensure_dirs()
    config = load_config()
    dataset = load_dataset("mimic4")
    groups = load_years(dataset)
    base_frame = dataset.frame(ALL_FEATURES)
    enhanced_frame, enhanced_names = enhanced_features(dataset)
    frames = {"base": base_frame, "enhanced": enhanced_frame}

    development_rows = np.flatnonzero(
        np.isin(groups, ["2008 - 2010", "2011 - 2013", "2014 - 2016"])
    )
    validation_rows = np.flatnonzero(groups == "2017 - 2019")
    final_rows = np.flatnonzero(groups == "2020 - 2022")
    candidates = candidate_configs(config)
    validation_records = []
    validation_predictions = {}
    for candidate in candidates:
        frame = frames[str(candidate["features"])]
        candidate_config = copy.deepcopy(config)
        candidate_config["xgboost"] = candidate["xgboost"]
        probability = fit_xgboost(
            frame.iloc[development_rows],
            frame.iloc[validation_rows],
            dataset.labels[development_rows],
            candidate_config,
            7,
        )
        metrics = metric_block(dataset.labels[validation_rows], probability)
        validation_predictions[str(candidate["name"])] = probability
        validation_records.append(
            {
                "name": candidate["name"],
                "features": candidate["features"],
                "n_features": int(frame.shape[1]),
                "xgboost": candidate["xgboost"],
                **metrics,
            }
        )
        print(
            f"[internal validation] {candidate['name']} "
            f"AUROC={metrics['auroc']:.4f}",
            flush=True,
        )
    np.savez_compressed(
        PREDICTIONS / "mimic4_internal_era_validation_xgb.npz",
        labels=dataset.labels[validation_rows],
        stay_id=dataset.cohort["stay_id"].iloc[validation_rows].to_numpy(
            dtype=np.int64
        ),
        **{
            name: probability.astype(np.float32)
            for name, probability in validation_predictions.items()
        },
    )

    selected = max(validation_records, key=lambda record: record["auroc"])
    selected_candidate = next(
        candidate for candidate in candidates if candidate["name"] == selected["name"]
    )
    selected_frame = frames[str(selected_candidate["features"])]
    final_config = copy.deepcopy(config)
    final_config["xgboost"] = selected_candidate["xgboost"]
    train_rows = np.flatnonzero(groups != "2020 - 2022")
    final_probability = fit_xgboost(
        selected_frame.iloc[train_rows],
        selected_frame.iloc[final_rows],
        dataset.labels[train_rows],
        final_config,
        7,
    )
    with np.load(
        PREDICTIONS / "mimic4_temporal_holdout.npz",
        allow_pickle=True,
    ) as data:
        baseline_xgb = data["xgb_static_ts"].astype(float)
        transformer = data["transformer"].astype(float)
    labels = dataset.labels[final_rows]
    selected_metrics = metric_block(labels, final_probability)
    baseline_metrics = metric_block(labels, baseline_xgb)
    contrast = bootstrap_paired_auc(
        labels,
        final_probability,
        baseline_xgb,
        2000,
        2026,
    )
    output = {
        "selection_split": {
            "development_groups": ["2008 - 2010", "2011 - 2013", "2014 - 2016"],
            "validation_groups": ["2017 - 2019"],
            "development_n": int(len(development_rows)),
            "validation_n": int(len(validation_rows)),
            "validation_events": int(dataset.labels[validation_rows].sum()),
        },
        "candidates": validation_records,
        "locked_selected_candidate": selected["name"],
        "locked_enhanced_feature_count": int(len(enhanced_names)),
        "final_holdout": {
            "n": int(len(final_rows)),
            "events": int(labels.sum()),
            "baseline_xgb": baseline_metrics,
            "selected_xgb": selected_metrics,
            "selected_minus_baseline": contrast,
            "transformer": metric_block(labels, transformer),
        },
    }
    (RESULTS / "strong_temporal_holdout.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        PREDICTIONS / "mimic4_temporal_holdout_strong.npz",
        labels=labels,
        stay_id=dataset.cohort["stay_id"].iloc[final_rows].to_numpy(dtype=np.int64),
        baseline_xgb=baseline_xgb.astype(np.float32),
        selected_xgb=final_probability.astype(np.float32),
        transformer=transformer.astype(np.float32),
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

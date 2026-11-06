"""Run prespecified single-cohort sensitivity models with shared outer folds."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import PROCESSED, RESULTS  # noqa: E402
from src.ami_dynamic.evaluation import (  # noqa: E402
    bootstrap_metric_ci,
    calibration_slope_intercept,
    point_metrics,
)
from src.ami_dynamic.train_evaluate import (  # noqa: E402
    TRAIN_DEFAULTS,
    _assert_feature_contract,
    _feature_columns,
    _fit_xgb,
    _json_default,
    _static_matrix,
    _temporal_predictions,
    load_dataset,
)


OUTPUT = RESULTS / "sensitivity"


def _prepare(
    analysis: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data = load_dataset("mimic4")
    cohort = data["cohort"].copy()
    if analysis in {"primary_position", "exclude_type2"}:
        flags = pd.read_csv(PROCESSED / "mimic4" / "diagnosis_flags.csv")
        cohort = cohort.merge(flags, on=["stay_id", "hadm_id"], how="left")
    keep = np.ones(len(cohort), dtype=bool)
    outcome_column = "outcome"
    if analysis == "six_level":
        outcome_column = "outcome_six"
        keep &= cohort[outcome_column].notna().to_numpy()
    elif analysis == "primary_position":
        keep &= cohort["ami_primary_position"].fillna(0).eq(1).to_numpy()
    elif analysis == "exclude_type2":
        keep &= cohort["ami_type2"].fillna(0).eq(0).to_numpy()
    elif analysis == "exclude_against_advice":
        keep &= ~cohort["disposition_raw"].fillna("").astype(str).str.upper().isin(
            {"AGAINST ADVICE", "LEFT AGAINST MEDICAL ADVI"}
        ).to_numpy()
    elif analysis != "exclude_troponin":
        raise ValueError(analysis)
    cohort = cohort.loc[keep].reset_index(drop=True)
    values = data["values"][keep]
    masks = data["masks"][keep]
    static_columns, full_columns = _feature_columns(cohort)
    if analysis == "exclude_troponin":
        static_columns = [
            column
            for column in static_columns
            if not column.startswith("lab_troponin")
            and column != "troponin_assay_t"
        ]
        full_columns = [
            column
            for column in full_columns
            if not column.startswith("lab_troponin")
            and column != "troponin_assay_t"
        ]
    _assert_feature_contract(static_columns)
    _assert_feature_contract(full_columns)
    if outcome_column != "outcome":
        cohort["sensitivity_outcome"] = cohort[outcome_column]
    else:
        cohort["sensitivity_outcome"] = cohort["outcome"]
    return cohort, values, masks, keep, full_columns


def run(
    *,
    analysis: str,
    model_name: str,
    seeds: list[int],
    folds: int,
    epochs: int,
) -> dict:
    cohort, values, masks, original_rows, full_columns = _prepare(analysis)
    labels = cohort["sensitivity_outcome"].to_numpy(dtype=int)
    groups = cohort["subject_key"].astype(str).to_numpy()
    static = _static_matrix(cohort, full_columns)
    n_classes = int(labels.max()) + 1
    tag = f"mimic4_{analysis}_{model_name}"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prediction_dir = RESULTS / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    seed_probabilities = []
    seed_metrics = []
    for seed in seeds:
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=seed
        )
        probability = np.full((len(cohort), n_classes), np.nan, dtype=np.float32)
        fold_id = np.full(len(cohort), -1, dtype=int)
        for fold, (train, test) in enumerate(
            splitter.split(static, labels, groups)
        ):
            fold_id[test] = fold
            if model_name == "xgb_static_ts":
                model = _fit_xgb(
                    static[train],
                    labels[train],
                    n_classes=n_classes,
                    seed=seed + fold,
                    n_jobs=int(os.environ.get("AMI_XGB_THREADS", "4")),
                )
                fold_probability = model.predict_proba(static[test])
            elif model_name == "dggru_tr":
                fold_probability, _ = _temporal_predictions(
                    values[train],
                    masks[train],
                    static[train],
                    labels[train],
                    values[test],
                    masks[test],
                    static[test],
                    model_name="dggru_tr",
                    n_classes=n_classes,
                    seed=seed + fold,
                    epochs=epochs,
                    train_defaults=TRAIN_DEFAULTS,
                    lambda_smooth=0.2,
                    lambda_drift=0.05,
                )
            else:
                raise ValueError(model_name)
            probability[test] = fold_probability
        seed_probabilities.append(probability)
        metrics = point_metrics(labels, probability)
        metrics.pop("cumulative_probabilities", None)
        seed_metrics.append(metrics)
        np.savez_compressed(
            prediction_dir / f"{tag}_seed{seed}.npz",
            labels=labels.astype(int),
            probabilities=probability,
            fold=fold_id,
            stay_id=cohort["stay_id"].to_numpy(dtype=np.int64),
            subject_key=groups,
            original_row=original_rows,
        )
        (prediction_dir / f"{tag}_seed{seed}.json").write_text(
            json.dumps(
                {
                    "analysis": analysis,
                    "model": model_name,
                    "seed": seed,
                    "n": len(cohort),
                    "metrics": metrics,
                },
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        print(
            f"[{analysis}] model={model_name} seed={seed} "
            f"n={len(cohort)} macro_auc={metrics['macro_auc']:.4f}",
            flush=True,
        )
    average = np.mean(np.stack(seed_probabilities), axis=0)
    metrics = point_metrics(labels, average)
    metrics.pop("cumulative_probabilities", None)
    metrics["macro_auc_ci"] = bootstrap_metric_ci(
        labels,
        average,
        groups=groups,
        replicates=int(os.environ.get("AMI_BOOTSTRAP_REPLICATES", "2000")),
    )
    result = {
        "analysis": analysis,
        "model": model_name,
        "dataset": "mimic4",
        "n": int(len(cohort)),
        "seeds": seeds,
        "folds": folds,
        "outcome_counts": {
            str(int(key)): int(value)
            for key, value in pd.Series(labels).value_counts().sort_index().items()
        },
        "metrics": metrics,
        "calibration": {
            "p_y_ge_3": calibration_slope_intercept(
                (labels >= 3).astype(int), average[:, 3:].sum(axis=1)
            )
            if n_classes == 5
            else None,
            "highest_class": calibration_slope_intercept(
                (labels == n_classes - 1).astype(int), average[:, -1]
            ),
        },
    }
    path = OUTPUT / f"metrics_mimic4_{analysis}_{model_name}.json"
    path.write_text(
        json.dumps(result, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        required=True,
        choices=[
            "six_level",
            "primary_position",
            "exclude_type2",
            "exclude_against_advice",
            "exclude_troponin",
        ],
    )
    parser.add_argument(
        "--model",
        default="xgb_static_ts",
        choices=["xgb_static_ts", "dggru_tr"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 42, 123, 2024])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    result = run(
        analysis=args.analysis,
        model_name=args.model,
        seeds=args.seeds,
        folds=args.folds,
        epochs=args.epochs,
    )
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

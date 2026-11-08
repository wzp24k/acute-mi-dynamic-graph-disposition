"""Calibration, classification, and net-benefit uncertainty analysis."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .common import PREDICTIONS, RESULTS, ensure_dirs, load_config


MODELS = ("xgb_static_ts", "transformer", "temporal_ensemble")
DCA_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
OPERATING_THRESHOLDS = (0.10, 0.15)


def calibration(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    logits = np.log(
        np.clip(probability, 1e-7, 1 - 1e-7)
        / np.clip(1 - probability, 1e-7, 1 - 1e-7)
    ).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logits, labels)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def ece(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges) - 1, 0, bins - 1)
    value = 0.0
    for bin_index in range(bins):
        selected = index == bin_index
        if selected.any():
            value += selected.mean() * abs(
                probability[selected].mean() - labels[selected].mean()
            )
    return float(value)


def net_benefit(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> float:
    predicted = probability >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    return float(
        tp / len(labels)
        - fp / len(labels) * threshold / (1.0 - threshold)
    )


def operating_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predicted = probability >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    tn = int((~predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    return {
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "predicted_positive_fraction": float(predicted.mean()),
    }


def point_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration(labels, probability)
    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece": ece(labels, probability),
        "prevalence": float(labels.mean()),
    }


def bootstrap_uncertainty(
    labels: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray | None,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    brier_values = []
    calibrated_brier_values = []
    net_values = {f"{threshold:.2f}": {"raw": [], "calibrated": [], "difference": []}
                  for threshold in DCA_THRESHOLDS}
    operating_values = {
        f"{threshold:.2f}": {
            "raw": [],
            "calibrated": [],
        }
        for threshold in OPERATING_THRESHOLDS
    }
    for _ in range(int(replicates)):
        index = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[index]
        if len(np.unique(sampled_labels)) < 2:
            continue
        sampled_raw = raw[index]
        brier_values.append(brier_score_loss(sampled_labels, sampled_raw))
        sampled_calibrated = calibrated[index] if calibrated is not None else None
        if sampled_calibrated is not None:
            calibrated_brier_values.append(
                brier_score_loss(sampled_labels, sampled_calibrated)
            )
        for threshold in DCA_THRESHOLDS:
            key = f"{threshold:.2f}"
            raw_net = net_benefit(sampled_labels, sampled_raw, threshold)
            net_values[key]["raw"].append(raw_net)
            if sampled_calibrated is not None:
                calibrated_net = net_benefit(
                    sampled_labels, sampled_calibrated, threshold
                )
                net_values[key]["calibrated"].append(calibrated_net)
                net_values[key]["difference"].append(calibrated_net - raw_net)
        for threshold in OPERATING_THRESHOLDS:
            key = f"{threshold:.2f}"
            operating_values[key]["raw"].append(
                operating_metrics(sampled_labels, sampled_raw, threshold)
            )
            if sampled_calibrated is not None:
                operating_values[key]["calibrated"].append(
                    operating_metrics(
                        sampled_labels, sampled_calibrated, threshold
                    )
                )

    def interval(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return [np.nan, np.nan]
        return [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]

    output = {
        "brier_ci": interval(brier_values),
        "calibrated_brier_ci": (
            interval(calibrated_brier_values)
            if calibrated is not None
            else None
        ),
        "net_benefit_ci": {},
        "operating_thresholds_ci": {},
    }
    for key, values in net_values.items():
        output["net_benefit_ci"][key] = {
            "raw": interval(values["raw"]),
            "calibrated": (
                interval(values["calibrated"])
                if calibrated is not None
                else None
            ),
            "calibration_minus_raw": (
                interval(values["difference"])
                if calibrated is not None
                else None
            ),
        }
    for key, values in operating_values.items():
        output["operating_thresholds_ci"][key] = {}
        for source, records in values.items():
            if not records:
                output["operating_thresholds_ci"][key][source] = None
                continue
            output["operating_thresholds_ci"][key][source] = {
                metric: interval([record[metric] for record in records])
                for metric in records[0]
            }
    return output


def evaluate(
    labels: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray | None,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    result = {
        "point": point_metrics(labels, raw),
        "calibrated_point": (
            point_metrics(labels, calibrated)
            if calibrated is not None
            else None
        ),
        "operating_thresholds": {
            f"{threshold:.2f}": {
                "raw": operating_metrics(labels, raw, threshold),
                "calibrated": (
                    operating_metrics(labels, calibrated, threshold)
                    if calibrated is not None
                    else None
                ),
            }
            for threshold in OPERATING_THRESHOLDS
        },
        "bootstrap": bootstrap_uncertainty(
            labels,
            raw,
            calibrated,
            replicates=replicates,
            seed=seed,
        ),
    }
    return result


def main() -> None:
    ensure_dirs()
    config = load_config()
    replicates = int(config["bootstrap_replicates"])
    output: dict[str, object] = {
        "dca_thresholds": list(DCA_THRESHOLDS),
        "operating_thresholds": list(OPERATING_THRESHOLDS),
        "cohorts": {},
    }
    for cohort in ("mimic4", "mimic3", "eicu", "eicu_anycad"):
        output["cohorts"][cohort] = {"models": {}}
        for model in MODELS:
            path = (
                PREDICTIONS / f"mimic4_cv_{model}.npz"
                if cohort == "mimic4"
                else PREDICTIONS / f"{cohort}_external_{model}.npz"
            )
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=True) as data:
                labels = data["labels"].astype(int)
                raw_key = "probability" if cohort == "mimic4" else model
                raw = data[raw_key].astype(float)
                recalibrated_key = f"{model}__recalibrated"
                calibrated = (
                    data[recalibrated_key].astype(float)
                    if recalibrated_key in data
                    else None
                )
            output["cohorts"][cohort]["models"][model] = evaluate(
                labels,
                raw,
                calibrated,
                replicates=replicates,
                seed=2026,
            )
            print(f"[utility] {cohort} {model}", flush=True)
        strong_path = PREDICTIONS / f"{cohort}_external_strong_xgb.npz"
        if strong_path.is_file():
            with np.load(strong_path, allow_pickle=True) as data:
                labels = data["labels"].astype(int)
                raw = data["strong_xgb"].astype(float)
                calibrated = data["strong_xgb__recalibrated"].astype(float)
                fusion = data["locked_75_25_fusion"].astype(float)
            output["cohorts"][cohort]["models"]["strong_xgb"] = evaluate(
                labels,
                raw,
                calibrated,
                replicates=replicates,
                seed=2027,
            )
            output["cohorts"][cohort]["models"]["locked_75_25_fusion"] = evaluate(
                labels,
                fusion,
                None,
                replicates=replicates,
                seed=2028,
            )
            print(f"[utility] {cohort} strong models", flush=True)

    (RESULTS / "clinical_utility_analysis.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    rows = []
    for cohort, cohort_result in output["cohorts"].items():
        for model, record in cohort_result["models"].items():
            for calibration_state in ("raw", "calibrated"):
                metrics = (
                    record["point"]
                    if calibration_state == "raw"
                    else record["calibrated_point"]
                )
                if metrics is None:
                    continue
                for threshold in OPERATING_THRESHOLDS:
                    operating = record["operating_thresholds"][
                        f"{threshold:.2f}"
                    ][calibration_state]
                    ci = record["bootstrap"]["operating_thresholds_ci"][
                        f"{threshold:.2f}"
                    ][calibration_state]
                    rows.append(
                        {
                            "cohort": cohort,
                            "model": model,
                            "calibration": calibration_state,
                            "threshold": threshold,
                            "auroc": metrics["auroc"],
                            "brier": metrics["brier"],
                            "calibration_intercept": metrics[
                                "calibration_intercept"
                            ],
                            "calibration_slope": metrics["calibration_slope"],
                            "ece": metrics["ece"],
                            **operating,
                            "ppv_ci_low": ci["ppv"][0] if ci else np.nan,
                            "ppv_ci_high": ci["ppv"][1] if ci else np.nan,
                            "npv_ci_low": ci["npv"][0] if ci else np.nan,
                            "npv_ci_high": ci["npv"][1] if ci else np.nan,
                        }
                    )
    pd.DataFrame(rows).to_csv(
        RESULTS / "clinical_utility_analysis.csv",
        index=False,
    )
    print(
        f"Wrote {RESULTS / 'clinical_utility_analysis.json'} "
        f"({len(rows)} rows)"
    )


if __name__ == "__main__":
    main()

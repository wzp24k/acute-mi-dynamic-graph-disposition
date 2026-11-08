"""Export aggregate plot inputs from saved NPZ and JSON artifacts for R."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
RESULTS = ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
OUTPUT = RESULTS / "figures_r" / "data"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    development = json.loads(
        (RESULTS / "metrics_development.json").read_text(encoding="utf-8")
    )
    rows = []
    for model, metrics in development["metrics"].items():
        rows.append(
            {
                "model": model,
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT / "development_performance.csv", index=False)

    validation = json.loads(
        (RESULTS / "temporal_holdout_validation.json").read_text(
            encoding="utf-8"
        )
    )
    pd.DataFrame(
        [
            {
                "model": model,
                "auroc": record["metrics"]["auroc"],
                "auprc": record["metrics"]["auprc"],
                "brier": record["metrics"]["brier"],
            }
            for model, record in validation["models"].items()
        ]
    ).to_csv(OUTPUT / "temporal_validation.csv", index=False)

    strong = json.loads(
        (RESULTS / "strong_temporal_holdout_fusion.json").read_text(
            encoding="utf-8"
        )
    )
    pd.DataFrame(
        [
            {
                "model": "Original XGB+TS",
                "auroc": strong["final_holdout"]["baseline_xgb"]["auroc"],
            },
            {
                "model": "Enhanced XGB",
                "auroc": strong["final_holdout"]["strong_xgb"]["auroc"],
            },
            {
                "model": "Transformer",
                "auroc": strong["final_holdout"]["transformer"]["auroc"],
            },
            {
                "model": "Locked 75/25 fusion",
                "auroc": strong["final_holdout"]["locked_fusion"]["auroc"],
            },
        ]
    ).to_csv(OUTPUT / "strong_holdout.csv", index=False)

    utility = json.loads(
        (RESULTS / "clinical_utility_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for cohort in ("mimic3", "eicu_anycad"):
        path = PREDICTIONS / f"{cohort}_external_temporal_ensemble.npz"
        with np.load(path, allow_pickle=True) as data:
            labels = data["labels"].astype(int)
            raw = data["temporal_ensemble"].astype(float)
            calibrated = data["temporal_ensemble__recalibrated"].astype(float)
        for source, probability in (
            ("Raw", raw),
            ("Recalibrated", calibrated),
        ):
            observed, predicted = calibration_curve(
                labels,
                probability,
                n_bins=10,
                strategy="quantile",
            )
            for index, (x, y) in enumerate(zip(predicted, observed), start=1):
                rows.append(
                    {
                        "cohort": cohort,
                        "source": source,
                        "bin": index,
                        "predicted": x,
                        "observed": y,
                    }
                )
    pd.DataFrame(rows).to_csv(OUTPUT / "calibration_curves.csv", index=False)

    rows = []
    thresholds = utility["dca_thresholds"]
    for cohort in ("mimic3", "eicu_anycad"):
        record = utility["cohorts"][cohort]["models"]["temporal_ensemble"]
        path = PREDICTIONS / f"{cohort}_external_temporal_ensemble.npz"
        with np.load(path, allow_pickle=True) as data:
            labels = data["labels"].astype(int)
            raw = data["temporal_ensemble"].astype(float)
            calibrated = data["temporal_ensemble__recalibrated"].astype(float)
        for source, probability in (
            ("Raw", raw),
            ("Recalibrated", calibrated),
        ):
            for threshold in thresholds:
                key = f"{threshold:.2f}"
                selected = probability >= threshold
                tp = np.sum(selected & (labels == 1))
                fp = np.sum(selected & (labels == 0))
                net = (
                    tp / len(labels)
                    - fp / len(labels) * threshold / (1 - threshold)
                )
                ci = record["bootstrap"]["net_benefit_ci"][key][
                    "raw" if source == "Raw" else "calibrated"
                ]
                rows.append(
                    {
                        "cohort": cohort,
                        "source": source,
                        "threshold": threshold,
                        "net_benefit": net,
                        "low": ci[0],
                        "high": ci[1],
                    }
                )
    pd.DataFrame(rows).to_csv(OUTPUT / "decision_curves.csv", index=False)

    rows = []
    for cohort in ("mimic3", "eicu_anycad"):
        record = utility["cohorts"][cohort]["models"]["temporal_ensemble"]
        for source in ("raw", "calibrated"):
            operating = record["operating_thresholds"]["0.10"][source]
            for metric in ("sensitivity", "specificity", "ppv", "npv"):
                rows.append(
                    {
                        "cohort": cohort,
                        "source": source.title(),
                        "metric": metric.upper(),
                        "value": operating[metric],
                    }
                )
    pd.DataFrame(rows).to_csv(OUTPUT / "threshold_metrics.csv", index=False)

    attention_path = PREDICTIONS / "mimic4_attention_ctogt.npz"
    with np.load(attention_path, allow_pickle=True) as data:
        organ_keys = [key for key in data.files if "organ" in key]
        temporal_keys = [key for key in data.files if "temporal" in key]
        organ = np.mean([data[key] for key in organ_keys], axis=0)
        temporal = np.mean([data[key] for key in temporal_keys], axis=0)
        if organ.ndim == 3:
            organ = organ.mean(axis=1)
        if temporal.ndim == 3:
            temporal = temporal.mean(axis=1)
    cohort = pd.read_csv(ROOT / "data" / "processed" / "mimic4" / "cohort.csv")
    labels = cohort["macce7"].astype(int).to_numpy()
    channels = [
        "GCS",
        "Pupil",
        "HR",
        "SBP",
        "DBP",
        "MBP",
        "Lactate",
        "RR",
        "SpO2",
        "FiO2",
        "Creatinine",
        "Urine",
        "Platelets",
        "INR",
        "aPTT",
        "WBC",
        "Temperature",
        "Glucose",
        "pH",
        "Bicarbonate",
    ]
    organ_rows = []
    for index, channel in enumerate(channels):
        organ_rows.append(
            {
                "channel": channel,
                "MACCE": float(organ[labels == 1, index].mean()),
                "Event-free": float(organ[labels == 0, index].mean()),
            }
        )
    pd.DataFrame(organ_rows).to_csv(
        OUTPUT / "organ_attention.csv", index=False
    )
    temporal_rows = []
    for hour in range(24):
        temporal_rows.append(
            {
                "hour": hour + 1,
                "MACCE": float(temporal[labels == 1, hour].mean()),
                "Event-free": float(temporal[labels == 0, hour].mean()),
            }
        )
    pd.DataFrame(temporal_rows).to_csv(
        OUTPUT / "temporal_attention.csv", index=False
    )

    pd.read_csv(RESULTS / "tables" / "table8_interval_threshold_sensitivity.csv").to_csv(
        OUTPUT / "interval_sensitivity.csv",
        index=False,
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()

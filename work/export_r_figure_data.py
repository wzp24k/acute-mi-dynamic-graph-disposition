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

    external = json.loads(
        (RESULTS / "metrics_external.json").read_text(encoding="utf-8")
    )
    transport_rows = []
    for cohort in ("mimic3", "eicu_anycad"):
        for model in (
            "xgb_static_ts",
            "transformer",
            "temporal_ensemble",
            "ctogt",
        ):
            metrics = (
                external.get("external", {})
                .get(cohort, {})
                .get("models", {})
                .get(model)
            )
            if not metrics:
                continue
            transport_rows.append(
                {
                    "cohort": cohort,
                    "model": model,
                    "auroc_raw": metrics["unrecalibrated"]["auroc"],
                    "auroc_recalibrated": metrics["recalibrated"]["auroc"],
                }
            )
    strong_transport = json.loads(
        (RESULTS / "strong_external_transport.json").read_text(encoding="utf-8")
    )
    for cohort in ("mimic3", "eicu_anycad"):
        record = strong_transport["models"].get(cohort, {})
        if record:
            transport_rows.append(
                {
                    "cohort": cohort,
                    "model": "enhanced_fusion",
                    "auroc_raw": record["locked_75_25_fusion"]["auroc"],
                    "auroc_recalibrated": np.nan,
                }
            )
    pd.DataFrame(transport_rows).to_csv(
        OUTPUT / "transport_summary.csv",
        index=False,
    )

    interval_summary = json.loads(
        (RESULTS / "interval_endpoint_summary.json").read_text(encoding="utf-8")
    )
    pd.DataFrame(interval_summary).to_csv(
        OUTPUT / "endpoint_composition.csv",
        index=False,
    )

    nested = json.loads(
        (RESULTS / "nested_selection_and_utility.json").read_text(encoding="utf-8")
    )
    nested_transformer = json.loads(
        (RESULTS / "nested_transformer_selection.json").read_text(encoding="utf-8")
    )
    selection_rows = [
        {
            "analysis": "Nested selection",
            "model": "XGB+TS",
            "auroc": nested["development"]["standard_xgb_ts"]["metrics"]["auroc"],
        },
        {
            "analysis": "Nested selection",
            "model": "Nested XGB",
            "auroc": nested["development"]["nested_xgb_ts"]["metrics"]["auroc"],
        },
        {
            "analysis": "Nested selection",
            "model": "Nested XGB + Transformer",
            "auroc": nested["development"]["nested_temporal_ensemble"]["metrics"]["auroc"],
        },
        {
            "analysis": "Nested selection",
            "model": "Joint nested ensemble",
            "auroc": nested_transformer["development"]["joint_nested_temporal_ensemble"]["metrics"]["auroc"],
        },
    ]
    pd.DataFrame(selection_rows).to_csv(
        OUTPUT / "model_selection.csv",
        index=False,
    )

    organ_rows = []
    organ_labels = {
        "ct_fixed": "Fixed graph",
        "ct_graph": "Dynamic graph",
        "organ_graph": "Seven-organ token graph",
        "ctogt": "Full CT-OGT",
        "ct_random": "Random graph",
        "ct_reverse": "Reversed time",
        "ct_permute": "Permuted time",
        "ct_no_time": "No continuous time",
    }
    for model, label in organ_labels.items():
        metrics = development["metrics"].get(model)
        if metrics:
            organ_rows.append({"model": label, "auroc": metrics["auroc"]})
    pd.DataFrame(organ_rows).to_csv(
        OUTPUT / "organ_graph_controls.csv",
        index=False,
    )

    table1 = pd.read_csv(RESULTS / "tables" / "table1_baseline.csv")
    risk_rows = []
    risk_variables = {
        "Prior myocardial infarction": "Prior MI",
        "Prior stroke": "Prior stroke",
        "Heart failure": "Heart failure",
        "Atrial fibrillation": "Atrial fibrillation",
        "Diabetes": "Diabetes",
        "Hypertension": "Hypertension",
        "Chronic kidney disease": "CKD",
        "COPD": "COPD",
        "Peripheral artery disease": "PAD",
    }
    cohort_columns = {
        "MIMIC-IV": "MIMIC-IV development",
        "MIMIC-III": "MIMIC-III temporal transport",
        "eICU expanded": "eICU expanded CAD transport",
    }
    for source_label, label in risk_variables.items():
        row = table1.loc[table1["Characteristic"] == source_label]
        if row.empty:
            continue
        for cohort, column in cohort_columns.items():
            raw = str(row.iloc[0][column])
            try:
                value = float(raw)
            except ValueError:
                value = np.nan
            risk_rows.append(
                {
                    "cohort": cohort,
                    "risk_factor": label,
                    "percent": value,
                }
            )
    pd.DataFrame(risk_rows).to_csv(
        OUTPUT / "cohort_risk_profiles.csv",
        index=False,
    )

    channel_labels = {
        "gcs_total": "GCS",
        "pupil_reactivity": "Pupil",
        "hr": "Heart rate",
        "sbp": "SBP",
        "dbp": "DBP",
        "mbp": "MBP",
        "lactate": "Lactate",
        "rr": "Respiratory rate",
        "spo2": "SpO2",
        "fio2": "FiO2",
        "creatinine": "Creatinine",
        "urine_output": "Urine output",
        "platelets": "Platelets",
        "inr": "INR",
        "aptt": "aPTT",
        "wbc": "WBC",
        "temp": "Temperature",
        "glucose": "Glucose",
        "ph": "pH",
        "bicarbonate": "Bicarbonate",
    }
    coverage_rows = []
    channel_organs = {
        "gcs_total": "Neurologic",
        "pupil_reactivity": "Neurologic",
        "hr": "Circulatory",
        "sbp": "Circulatory",
        "dbp": "Circulatory",
        "mbp": "Circulatory",
        "lactate": "Circulatory",
        "rr": "Respiratory",
        "spo2": "Respiratory",
        "fio2": "Respiratory",
        "creatinine": "Renal",
        "urine_output": "Renal",
        "platelets": "Coagulation",
        "inr": "Coagulation",
        "aptt": "Coagulation",
        "wbc": "Inflammation",
        "temp": "Inflammation",
        "glucose": "Metabolic",
        "ph": "Metabolic",
        "bicarbonate": "Metabolic",
    }
    for cohort, label in (
        ("mimic4", "MIMIC-IV"),
        ("mimic3", "MIMIC-III"),
        ("eicu_anycad", "eICU expanded"),
    ):
        frame = pd.read_csv(ROOT / "data" / "processed" / cohort / "cohort.csv")
        for channel, channel_label in channel_labels.items():
            column = f"{channel}_n"
            if column not in frame:
                coverage = np.nan
            else:
                coverage = float(pd.to_numeric(frame[column], errors="coerce").gt(0).mean())
            coverage_rows.append(
                {
                    "cohort": label,
                    "channel": channel_label,
                    "organ": channel_organs[channel],
                    "coverage": coverage,
                }
            )
    pd.DataFrame(coverage_rows).to_csv(
        OUTPUT / "observation_coverage.csv",
        index=False,
    )

    distribution_rows = []
    decision_rows = []
    for cohort, label in (
        ("mimic3", "MIMIC-III"),
        ("eicu_anycad", "eICU expanded"),
    ):
        path = PREDICTIONS / f"{cohort}_external_temporal_ensemble.npz"
        with np.load(path, allow_pickle=True) as data:
            labels = data["labels"].astype(int)
            raw = data["temporal_ensemble"].astype(float)
            calibrated = data["temporal_ensemble__recalibrated"].astype(float)
        bins = np.linspace(0, 1, 31)
        for source, probability in (("Raw", raw), ("Recalibrated", calibrated)):
            counts, edges = np.histogram(probability, bins=bins)
            for index, count in enumerate(counts):
                distribution_rows.append(
                    {
                        "cohort": label,
                        "source": source,
                        "left": edges[index],
                        "right": edges[index + 1],
                        "proportion": count / len(probability),
                    }
                )
        for source, probability in (("Raw", raw), ("Recalibrated", calibrated)):
            predicted = probability >= 0.10
            tp = int(np.sum(predicted & (labels == 1)))
            fp = int(np.sum(predicted & (labels == 0)))
            tn = int(np.sum(~predicted & (labels == 0)))
            fn = int(np.sum(~predicted & (labels == 1)))
            for outcome, count in (
                ("True positive", tp),
                ("False positive", fp),
                ("True negative", tn),
                ("False negative", fn),
            ):
                decision_rows.append(
                    {
                        "cohort": label,
                        "source": source,
                        "outcome": outcome,
                        "count": count,
                        "proportion": count / len(labels),
                    }
                )
    pd.DataFrame(distribution_rows).to_csv(
        OUTPUT / "risk_distribution.csv",
        index=False,
    )
    pd.DataFrame(decision_rows).to_csv(
        OUTPUT / "decision_matrix.csv",
        index=False,
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()

"""Create publication tables and figures from saved study artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .common import (
    FIGURES,
    ORGAN_CHANNELS,
    PREDICTIONS,
    PROCESSED,
    RESULTS,
    TABLES,
    ensure_dirs,
)
from .train_evaluate import MODEL_LABELS


COLORS = {
    "blue": "#2C6E9B",
    "teal": "#2A9D8F",
    "orange": "#E76F51",
    "gold": "#E9C46A",
    "ink": "#24323D",
    "gray": "#7D8B96",
    "light": "#EEF3F6",
}


def percent(value: float) -> float:
    return float(value) * 100.0


def table_one() -> pd.DataFrame:
    rows = []
    specifications = {
        "mimic4": "MIMIC-IV development",
        "mimic3": "MIMIC-III temporal transport",
        "eicu": "eICU principal-CAD sensitivity",
        "eicu_anycad": "eICU expanded CAD transport",
    }
    variables = [
        ("Age, years", "age", "continuous"),
        ("Male", "sex_male", "binary"),
        ("White race", "race_white", "binary"),
        ("Black race", "race_black", "binary"),
        ("Emergency admission", "emergency_admission", "binary"),
        ("Prior myocardial infarction", "prior_mi", "binary"),
        ("Prior stroke", "prior_stroke", "binary"),
        ("Heart failure", "heart_failure", "binary"),
        ("Atrial fibrillation", "atrial_fib", "binary"),
        ("Diabetes", "diabetes", "binary"),
        ("Hypertension", "hypertension", "binary"),
        ("Chronic kidney disease", "ckd", "binary"),
        ("COPD", "copd", "binary"),
        ("Peripheral artery disease", "pad", "binary"),
        ("Prior PCI", "prior_pci", "binary"),
        ("Prior CABG", "prior_cabg", "binary"),
        ("Observed organ channels, median", "n_channels_24h", "median"),
    ]
    cohorts = {name: pd.read_csv(PROCESSED / name / "cohort.csv") for name in specifications}
    for label, column, kind in variables:
        row: dict[str, object] = {"Characteristic": label}
        for name in specifications:
            frame = cohorts[name]
            values = pd.to_numeric(frame.get(column), errors="coerce")
            if kind == "continuous":
                row[specifications[name]] = (
                    f"{values.mean():.1f} ({values.std():.1f})"
                )
            elif kind == "binary":
                row[specifications[name]] = (
                    "NR"
                    if values.dropna().nunique() <= 1
                    else f"{percent(values.mean()):.1f}"
                )
            else:
                row[specifications[name]] = f"{values.median():.0f}"
        rows.append(row)
    event_row: dict[str, object] = {
        "Characteristic": "Diagnosis-coded MACCE phenotype, n (%)"
    }
    death_row: dict[str, object] = {"Characteristic": "7-day death, n (%)"}
    ami_row: dict[str, object] = {"Characteristic": "Acute MI, n (%)"}
    stroke_row: dict[str, object] = {"Characteristic": "Stroke, n (%)"}
    revasc_row: dict[str, object] = {"Characteristic": "Revascularization, n (%)"}
    n_row: dict[str, object] = {"Characteristic": "Analytic cohort, n"}
    site_row: dict[str, object] = {"Characteristic": "Sites with >=30 patients, n"}
    for name in specifications:
        frame = cohorts[name]
        n = len(frame)
        events = int(frame["macce7"].sum())
        n_row[specifications[name]] = f"{n:,}"
        event_row[specifications[name]] = f"{events:,} ({percent(events / n):.1f})"
        death_row[specifications[name]] = (
            f"{int(frame['death_7d'].sum()):,} "
            f"({percent(frame['death_7d'].mean()):.1f})"
        )
        ami_row[specifications[name]] = (
            f"{int(frame['ami_7d'].sum()):,} "
            f"({percent(frame['ami_7d'].mean()):.1f})"
        )
        stroke_row[specifications[name]] = (
            f"{int(frame['stroke_7d'].sum()):,} "
            f"({percent(frame['stroke_7d'].mean()):.1f})"
        )
        revasc_row[specifications[name]] = (
            f"{int(frame['revasc_7d'].sum()):,} "
            f"({percent(frame['revasc_7d'].mean()):.1f})"
        )
        if name in ("eicu", "eicu_anycad") and "hospitalid" in frame:
            counts = frame.groupby("hospitalid").size()
            eligible_sites = counts[counts >= 30]
            site_row[specifications[name]] = f"{len(eligible_sites)}"
        else:
            site_row[specifications[name]] = "-"
    for extra in (n_row, event_row, death_row, ami_row, stroke_row, revasc_row, site_row):
        rows.append(extra)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "table1_baseline.csv", index=False)
    (TABLES / "table1_baseline.md").write_text(
        table.to_markdown(index=False), encoding="utf-8"
    )
    return table


def load_development_metrics() -> tuple[pd.DataFrame, dict]:
    records = []
    for path in sorted(RESULTS.glob("metrics_development_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data["metrics"]
        bootstrap = data["bootstrap"]
        records.append(
            {
                "model": data["model"],
                "label": MODEL_LABELS.get(data["model"], data["model"]),
                "auroc": metrics["auroc"],
                "auroc_low": bootstrap["auroc_ci"][0],
                "auroc_high": bootstrap["auroc_ci"][1],
                "auprc": metrics["auprc"],
                "brier": metrics["brier"],
                "calibration_slope": metrics["calibration_slope"],
                "ece": metrics["ece"],
            }
        )
    order = {
        "logit_static": 1,
        "xgb_static": 2,
        "xgb_static_ts": 3,
        "gru": 4,
        "transformer": 5,
        "tft": 6,
        "ct_fixed": 7,
        "ct_graph": 8,
        "organ_graph": 9,
        "temporal_ensemble": 10,
        "ctogt": 11,
        "ct_random": 12,
        "ct_reverse": 13,
        "ct_permute": 14,
        "ct_no_time": 15,
    }
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["order"] = frame["model"].map(order).fillna(99)
        frame = frame.sort_values("order").drop(columns="order")
    aggregate = {}
    path = RESULTS / "metrics_development.json"
    if path.is_file():
        aggregate = json.loads(path.read_text(encoding="utf-8"))
    return frame, aggregate


def table_two(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    table = metrics.copy()
    table["AUROC (95% CI)"] = table.apply(
        lambda row: (
            f"{row['auroc']:.3f} "
            f"({row['auroc_low']:.3f}-{row['auroc_high']:.3f})"
        ),
        axis=1,
    )
    output = table[
        [
            "label",
            "AUROC (95% CI)",
            "auprc",
            "brier",
            "calibration_slope",
            "ece",
        ]
    ].rename(
        columns={
            "label": "Model",
            "auprc": "AUPRC",
            "brier": "Brier",
            "calibration_slope": "Calibration slope",
            "ece": "ECE",
        }
    )
    for column in ("AUPRC", "Brier", "Calibration slope", "ECE"):
        output[column] = output[column].map(lambda value: f"{value:.3f}")
    output.to_csv(TABLES / "table2_development_performance.csv", index=False)
    (TABLES / "table2_development_performance.md").write_text(
        output.to_markdown(index=False), encoding="utf-8"
    )
    return output


def load_external_metrics() -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for name in ("mimic3", "eicu", "eicu_anycad"):
        rows = []
        paths = sorted(RESULTS.glob(f"metrics_external_{name}_*.json"))
        if name == "eicu":
            paths = [
                path
                for path in paths
                if not path.name.startswith("metrics_external_eicu_anycad_")
            ]
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            model = data["model"]
            raw = data["metrics"]["unrecalibrated"]
            recal = data["metrics"]["recalibrated"]
            rows.append(
                {
                    "model": model,
                    "label": MODEL_LABELS.get(model, model),
                    "auroc_raw": raw["auroc"],
                    "auroc_recal": recal["auroc"],
                    "auprc_raw": raw["auprc"],
                    "brier_raw": raw["brier"],
                    "brier_recal": recal["brier"],
                    "intercept_raw": raw["calibration_intercept"],
                    "slope_raw": raw["calibration_slope"],
                    "ece_raw": raw["ece"],
                    "ece_recal": recal["ece"],
                }
            )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            order = {
                "logit_static": 1,
                "xgb_static": 2,
                "xgb_static_ts": 3,
                "gru": 4,
                "transformer": 5,
                "tft": 6,
                "ct_fixed": 7,
                "ct_graph": 8,
                "temporal_ensemble": 9,
                "ctogt": 10,
            }
            frame["order"] = frame["model"].map(order).fillna(99)
            frame = frame.sort_values("order").drop(columns="order")
        result[name] = frame
    return result


def table_three(external: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    cohort_labels = {
        "mimic3": "MIMIC-III",
        "eicu": "eICU principal-CAD sensitivity",
        "eicu_anycad": "eICU expanded CAD",
    }
    for cohort, frame in external.items():
        for record in frame.to_dict(orient="records"):
            rows.append(
                {
                    "Cohort": cohort_labels.get(cohort, cohort),
                    "Model": record["label"],
                    "AUROC raw": f"{record['auroc_raw']:.3f}",
                    "AUROC recalibrated": f"{record['auroc_recal']:.3f}",
                    "AUPRC raw": f"{record['auprc_raw']:.3f}",
                    "Brier raw": f"{record['brier_raw']:.3f}",
                    "Brier recalibrated": f"{record['brier_recal']:.3f}",
                    "Calibration intercept": f"{record['intercept_raw']:.3f}",
                    "Calibration slope": f"{record['slope_raw']:.3f}",
                    "ECE raw": f"{record['ece_raw']:.3f}",
                    "ECE recalibrated": f"{record['ece_recal']:.3f}",
                }
            )
    strong_path = RESULTS / "strong_external_transport.json"
    if strong_path.is_file():
        strong = json.loads(strong_path.read_text(encoding="utf-8"))
        for cohort, label in (
            ("mimic3", "MIMIC-III"),
            ("eicu_anycad", "eICU expanded CAD"),
        ):
            record = strong.get("models", {}).get(cohort, {})
            xgb = record.get("strong_xgb", {})
            if xgb:
                raw = xgb["unrecalibrated"]
                recal = xgb["recalibrated"]
                rows.append(
                    {
                        "Cohort": label,
                        "Model": "Enhanced organ-dynamics XGBoost",
                        "AUROC raw": f"{raw['auroc']:.3f}",
                        "AUROC recalibrated": f"{recal['auroc']:.3f}",
                        "AUPRC raw": f"{raw['auprc']:.3f}",
                        "Brier raw": f"{raw['brier']:.3f}",
                        "Brier recalibrated": f"{recal['brier']:.3f}",
                        "Calibration intercept": f"{raw['calibration_intercept']:.3f}",
                        "Calibration slope": f"{raw['calibration_slope']:.3f}",
                        "ECE raw": f"{raw['ece']:.3f}",
                        "ECE recalibrated": f"{recal['ece']:.3f}",
                    }
                )
            fusion = record.get("locked_75_25_fusion", {})
            if fusion:
                rows.append(
                    {
                        "Cohort": label,
                        "Model": "Locked 75/25 enhanced fusion",
                        "AUROC raw": f"{fusion['auroc']:.3f}",
                        "AUROC recalibrated": "NA",
                        "AUPRC raw": f"{fusion['auprc']:.3f}",
                        "Brier raw": f"{fusion['brier']:.3f}",
                        "Brier recalibrated": "NA",
                        "Calibration intercept": f"{fusion['calibration_intercept']:.3f}",
                        "Calibration slope": f"{fusion['calibration_slope']:.3f}",
                        "ECE raw": f"{fusion['ece']:.3f}",
                        "ECE recalibrated": "NA",
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "table3_transport.csv", index=False)
    if not table.empty:
        (TABLES / "table3_transport.md").write_text(
            table.to_markdown(index=False), encoding="utf-8"
        )
    return table


def table_four_interval() -> pd.DataFrame:
    model_path = RESULTS / "interval_model_analysis.json"
    prediction_path = RESULTS / "interval_endpoint_analysis.json"
    if not model_path.is_file() or not prediction_path.is_file():
        return pd.DataFrame()
    model_data = json.loads(model_path.read_text(encoding="utf-8"))
    prediction_data = json.loads(prediction_path.read_text(encoding="utf-8"))
    rows = []
    for cohort, label in (
        ("mimic4", "MIMIC-IV"),
        ("mimic3", "MIMIC-III"),
        ("eicu", "eICU"),
        ("eicu_anycad", "eICU expanded CAD"),
    ):
        if cohort == "mimic4":
            source = model_data["mimic4"]
        else:
            source = model_data["external"][cohort]
        fixed = prediction_data["cohorts"][cohort]["models"].get(
            "temporal_ensemble", {}
        )
        for outcome, outcome_label in (
            ("macce_interval_confirmed", "Fully confirmed interval in target"),
            (
                "macce_interval_overlap",
                "Specific anchor interval overlaps target",
            ),
        ):
            retrained = source.get(outcome, {})
            fixed_result = fixed.get(outcome, {})
            rows.append(
                {
                    "Cohort": label,
                    "Interval endpoint": outcome_label,
                    "Events": retrained.get("events", ""),
                    "Retrained XGB+TS AUROC": (
                        f"{retrained['auroc']:.3f}"
                        if "auroc" in retrained
                        else "NA"
                    ),
                    "Existing temporal ensemble AUROC": (
                        f"{fixed_result['auroc']:.3f}"
                        if "auroc" in fixed_result
                        else "NA"
                    ),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "table4_interval_endpoint.csv", index=False)
    (TABLES / "table4_interval_endpoint.md").write_text(
        table.to_markdown(index=False), encoding="utf-8"
    )
    return table


def table_five_validation() -> pd.DataFrame:
    rows = []
    holdout_path = RESULTS / "temporal_holdout_validation.json"
    if holdout_path.is_file():
        holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "Validation": "Locked temporal holdout",
                "Test set": (
                    f"2020-2022; n={holdout['split']['test_n']:,}; "
                    f"events={holdout['split']['test_events']:,}"
                ),
                "XGB+TS AUROC": f"{holdout['models']['xgb_static_ts']['metrics']['auroc']:.3f}",
                "Temporal ensemble AUROC": f"{holdout['models']['temporal_ensemble']['metrics']['auroc']:.3f}",
                "Difference (95% CI)": (
                    f"{holdout['ensemble_minus_xgb_static_ts']['estimate']:.3f} "
                    f"({holdout['ensemble_minus_xgb_static_ts']['ci'][0]:.3f} to "
                    f"{holdout['ensemble_minus_xgb_static_ts']['ci'][1]:.3f})"
                ),
                "0.01 threshold met": (
                    "Yes"
                    if holdout["ensemble_minus_xgb_static_ts"]["estimate"] >= 0.01
                    and holdout["ensemble_minus_xgb_static_ts"]["ci"][0] > 0
                    else "No"
                ),
            }
        )
    nested_path = RESULTS / "nested_selection_and_utility.json"
    if nested_path.is_file():
        nested = json.loads(nested_path.read_text(encoding="utf-8"))
        contrast = nested["contrasts"]["nested_ensemble_minus_standard_xgb"]
        rows.append(
            {
                "Validation": "Nested model selection",
                "Test set": "Five-fold outer CV with inner 3-fold XGB tuning",
                "XGB+TS AUROC": f"{nested['development']['standard_xgb_ts']['metrics']['auroc']:.3f}",
                "Temporal ensemble AUROC": f"{nested['development']['nested_temporal_ensemble']['metrics']['auroc']:.3f}",
                "Difference (95% CI)": (
                    f"{contrast['estimate']:.4f} "
                    f"({contrast['ci'][0]:.4f} to {contrast['ci'][1]:.4f})"
                ),
                "0.01 threshold met": (
                    "Yes"
                    if contrast["estimate"] >= 0.01 and contrast["ci"][0] > 0
                    else "No"
                ),
            }
        )
    strong_path = RESULTS / "strong_temporal_holdout_fusion.json"
    if strong_path.is_file():
        strong = json.loads(strong_path.read_text(encoding="utf-8"))
        final = strong["final_holdout"]
        contrast = final["locked_fusion_minus_baseline_xgb"]
        rows.append(
            {
                "Validation": "Internal-era locked strong fusion",
                "Test set": (
                    "Select on 2017-2019; test 2020-2022; "
                    f"n={final['n']:,}; events={final['events']:,}"
                ),
                "XGB+TS AUROC": f"{final['baseline_xgb']['auroc']:.3f}",
                "Temporal ensemble AUROC": f"{final['locked_fusion']['auroc']:.3f}",
                "Difference (95% CI)": (
                    f"{contrast['difference']:.4f} "
                    f"({contrast['ci'][0]:.4f} to {contrast['ci'][1]:.4f})"
                ),
                "0.01 threshold met": (
                    "Yes"
                    if contrast["difference"] >= 0.01
                    and contrast["ci"][0] > 0
                    else "No"
                ),
            }
        )
    nested_transformer_path = RESULTS / "nested_transformer_selection.json"
    if nested_transformer_path.is_file():
        nested_transformer = json.loads(
            nested_transformer_path.read_text(encoding="utf-8")
        )
        contrast = nested_transformer["contrasts"][
            "joint_nested_ensemble_minus_standard_xgb"
        ]
        rows.append(
            {
                "Validation": "Joint nested XGB and Transformer",
                "Test set": (
                    "Five outer folds; inner 20% hyperparameter selection; "
                    "2 Transformer candidates"
                ),
                "XGB+TS AUROC": (
                    f"{nested['development']['standard_xgb_ts']['metrics']['auroc']:.3f}"
                ),
                "Temporal ensemble AUROC": (
                    f"{nested_transformer['development']['joint_nested_temporal_ensemble']['metrics']['auroc']:.3f}"
                ),
                "Difference (95% CI)": (
                    f"{contrast['estimate']:.4f} "
                    f"({contrast['ci'][0]:.4f} to {contrast['ci'][1]:.4f})"
                ),
                "0.01 threshold met": (
                    "Yes"
                    if contrast["estimate"] >= 0.01
                    and contrast["ci"][0] > 0
                    else "No"
                ),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(TABLES / "table5_validation.csv", index=False)
        (TABLES / "table5_validation.md").write_text(
            table.to_markdown(index=False), encoding="utf-8"
        )
    return table


def table_six_manual_review() -> pd.DataFrame:
    path = RESULTS / "manual_endpoint_review_pilot.json"
    if not path.is_file():
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    table = pd.DataFrame(
        [
            {"Review result": key, "Records": value}
            for key, value in data["label_counts"].items()
        ]
    )
    table.loc[len(table)] = {
        "Review result": "Total reviewed",
        "Records": data["reviewed_n"],
    }
    scored_path = RESULTS / "manual_endpoint_review_scored.json"
    if scored_path.is_file():
        scored = json.loads(scored_path.read_text(encoding="utf-8"))
        table.loc[len(table)] = {
            "Review result": (
                "Two-reader human review pending"
                if scored.get("status") == "pending"
                else "Two-reader human review completed"
            ),
            "Records": int(scored.get("pending_rows", 0)),
        }
    table.to_csv(TABLES / "table6_manual_review.csv", index=False)
    (TABLES / "table6_manual_review.md").write_text(
        table.to_markdown(index=False), encoding="utf-8"
    )
    return table


def table_seven_clinical_utility() -> pd.DataFrame:
    path = RESULTS / "clinical_utility_analysis.json"
    if not path.is_file():
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    cohort_labels = {
        "mimic3": "MIMIC-III",
        "eicu": "eICU principal-CAD",
        "eicu_anycad": "eICU expanded CAD",
    }
    for cohort in ("mimic3", "eicu", "eicu_anycad"):
        model_records = data.get("cohorts", {}).get(cohort, {}).get(
            "models", {}
        )
        for model_key, model_label in (
            ("temporal_ensemble", "Original temporal ensemble"),
            ("strong_xgb", "Enhanced organ-dynamics XGBoost"),
            ("locked_75_25_fusion", "Locked 75/25 enhanced fusion"),
        ):
            record = model_records.get(model_key)
            if not record:
                continue
            for state, point_key, ci_key in (
                ("Raw", "point", "raw"),
                ("Locally recalibrated", "calibrated_point", "calibrated"),
            ):
                point = record.get(point_key)
                if not point:
                    continue
                operating = record["operating_thresholds"]["0.10"][ci_key]
                brier_ci = (
                    record["bootstrap"]["brier_ci"]
                    if ci_key == "raw"
                    else record["bootstrap"]["calibrated_brier_ci"]
                )
                rows.append(
                    {
                        "Cohort": cohort_labels[cohort],
                        "Model": model_label,
                        "Probability": state,
                        "AUROC": f"{point['auroc']:.3f}",
                        "Brier (95% CI)": (
                            f"{point['brier']:.3f} "
                            f"({brier_ci[0]:.3f}-{brier_ci[1]:.3f})"
                        ),
                        "Calibration slope": f"{point['calibration_slope']:.3f}",
                        "Calibration intercept": f"{point['calibration_intercept']:.3f}",
                        "ECE": f"{point['ece']:.3f}",
                        "Sensitivity @10%": f"{operating['sensitivity']:.3f}",
                        "Specificity @10%": f"{operating['specificity']:.3f}",
                        "PPV @10%": f"{operating['ppv']:.3f}",
                        "NPV @10%": f"{operating['npv']:.3f}",
                    }
                )
    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(TABLES / "table7_clinical_utility.csv", index=False)
        (TABLES / "table7_clinical_utility.md").write_text(
            table.to_markdown(index=False),
            encoding="utf-8",
        )
    return table


def table_eight_interval_sensitivity() -> pd.DataFrame:
    path = RESULTS / "interval_threshold_sensitivity.json"
    if not path.is_file():
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for family, label in (
        ("troponin_threshold_multiplier", "Troponin threshold multiplier"),
        ("eicu_documentation_lag_hours", "eICU documentation lag, hours"),
    ):
        for parameter, cohorts in data.get(family, {}).items():
            for cohort, record in cohorts.items():
                xgb = record["models"].get("xgb_static_ts", {})
                ensemble = record["models"].get("temporal_ensemble", {})
                rows.append(
                    {
                        "Assumption": label,
                        "Value": parameter,
                        "Cohort": cohort,
                        "Coded events": record["coded_events"],
                        "Fully confirmed": record["confirmed_events"],
                        "Anchor-overlap": record["anchor_overlap_events"],
                        "XGB+TS AUROC": (
                            f"{xgb['auroc']:.3f}" if "auroc" in xgb else "NA"
                        ),
                        "Temporal ensemble AUROC": (
                            f"{ensemble['auroc']:.3f}"
                            if "auroc" in ensemble
                            else "NA"
                        ),
                    }
                )
    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(
            TABLES / "table8_interval_threshold_sensitivity.csv",
            index=False,
        )
        (TABLES / "table8_interval_threshold_sensitivity.md").write_text(
            table.to_markdown(index=False),
            encoding="utf-8",
        )
    return table


def figure_one() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def box(x, y, width, height, title, body, color):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height - 0.35,
            title,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=COLORS["ink"],
        )
        ax.text(
            x + width / 2,
            y + height - 0.85,
            body,
            ha="center",
            va="top",
            fontsize=8.5,
            color=COLORS["ink"],
            linespacing=1.4,
        )

    box(
        0.3,
        4.5,
        2.3,
        1.45,
        "MIMIC-IV",
        "Principal CAD diagnosis\nFirst ICU stay\nn = 13,131",
        COLORS["blue"],
    )
    box(
        0.3,
        2.55,
        2.3,
        1.45,
        "MIMIC-III",
        "Same-institution\ntemporal transport\nn = 3,470",
        COLORS["teal"],
    )
    box(
        0.3,
        0.6,
        2.3,
        1.45,
        "eICU-CRD",
        "Expanded CAD transport\nn = 5,913",
        COLORS["orange"],
    )
    box(
        3.45,
        2.7,
        2.5,
        2.0,
        "24-hour landmark",
        "Age >=18 years\nAlive and event-free at 24 h\n>=6 organ channels observed",
        COLORS["gold"],
    )
    box(
        6.75,
        2.7,
        2.4,
        2.0,
        "CT-OGT",
        "Irregular observation time\nDynamic organ graph\nEnvironment invariance",
        COLORS["blue"],
    )
    box(
        9.85,
        4.4,
        1.85,
        1.45,
        "Development",
        "5-fold CV\nRepeated model seeds",
        COLORS["teal"],
    )
    box(
        9.85,
        2.65,
        1.85,
        1.35,
        "Transport",
        "Direct prediction\nLocal recalibration",
        COLORS["orange"],
    )
    box(
        9.85,
        0.9,
        1.85,
        1.35,
        "Evaluation",
        "AUROC / AUPRC\nCalibration / DCA",
        COLORS["gold"],
    )
    arrows = [
        ((2.6, 5.2), (3.45, 3.8)),
        ((2.6, 3.28), (3.45, 3.6)),
        ((2.6, 1.33), (3.45, 3.3)),
        ((5.95, 3.7), (6.75, 3.7)),
        ((9.15, 3.95), (9.85, 5.0)),
        ((9.15, 3.7), (9.85, 3.3)),
        ((9.15, 3.4), (9.85, 1.6)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color=COLORS["gray"],
                connectionstyle="arc3,rad=0.03",
            )
        )
    ax.text(
        6.0,
        0.2,
        "Outcome window: ICU hour 24 to day 7",
        ha="center",
        fontsize=9,
        color=COLORS["gray"],
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_study_design.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / "figure1_study_design.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_two() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    def module(x, y, width, height, title, subtitle, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="white",
                edgecolor=color,
                linewidth=1.3,
            )
        )
        ax.text(
            x + width / 2,
            y + height * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=COLORS["ink"],
        )
        ax.text(
            x + width / 2,
            y + height * 0.3,
            subtitle,
            ha="center",
            va="center",
            fontsize=8,
            color=COLORS["gray"],
        )

    module(
        0.25,
        2.4,
        1.8,
        1.55,
        "Organ observations",
        "7 organ systems\n20 channels x 24 h",
        COLORS["blue"],
    )
    module(
        2.45,
        4.35,
        2.05,
        1.35,
        "Continuous-time cues",
        "Elapsed time since last\nobservation per channel",
        COLORS["teal"],
    )
    module(
        2.45,
        0.65,
        2.05,
        1.35,
        "Clinical prior",
        "Sparse organ edges\nand self connections",
        COLORS["gold"],
    )
    module(
        4.95,
        2.55,
        2.35,
        1.55,
        "Dynamic organ graph",
        "Observation-aware\nattention over organ nodes",
        COLORS["orange"],
    )
    module(
        7.7,
        2.55,
        2.0,
        1.55,
        "Temporal encoder",
        "Time2Vec +\nTransformer encoder",
        COLORS["blue"],
    )
    module(
        10.05,
        2.55,
        1.65,
        1.55,
        "7-day MACCE",
        "Risk trajectory\nand final risk",
        COLORS["teal"],
    )
    module(
        7.7,
        0.65,
        2.0,
        1.2,
        "Static context",
        "Demographics and\ncomorbidity burden",
        COLORS["gray"],
    )
    module(
        4.95,
        0.65,
        2.35,
        1.2,
        "Invariance loss",
        "Environment-balanced risk\nand graph sparsity",
        COLORS["orange"],
    )
    for start, end in [
        ((2.05, 3.15), (2.45, 4.9)),
        ((2.05, 3.15), (2.45, 1.35)),
        ((4.5, 5.0), (4.95, 3.5)),
        ((4.5, 1.35), (4.95, 3.0)),
        ((7.3, 3.33), (7.7, 3.33)),
        ((9.7, 3.33), (10.05, 3.33)),
        ((8.7, 1.85), (8.7, 2.55)),
        ((7.3, 1.25), (7.7, 1.25)),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color=COLORS["gray"],
            )
        )
    ax.text(
        6.0,
        6.25,
        "CT-OGT: Continuous-Time Organ Graph Transformer",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_model_architecture.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / "figure2_model_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_three(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return
    plot = metrics.iloc[::-1].copy()
    colors = [
        COLORS["orange"] if model.startswith("ct_") or model == "ctogt" else COLORS["blue"]
        for model in plot["model"]
    ]
    fig, ax = plt.subplots(figsize=(9.3, 6.2))
    y = np.arange(len(plot))
    ax.hlines(
        y,
        plot["auroc_low"],
        plot["auroc_high"],
        color=colors,
        linewidth=2.2,
        alpha=0.75,
    )
    ax.scatter(plot["auroc"], y, color=colors, s=48, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"], fontsize=9)
    ax.set_xlabel("Cross-validated AUROC")
    ax.grid(axis="x", color="#D9E1E7", linewidth=0.7)
    ax.set_axisbelow(True)
    best = plot.loc[plot["auroc"].idxmax()]
    ax.axvline(best["auroc"], color=COLORS["gold"], linestyle="--", linewidth=1.2)
    ax.text(
        best["auroc"],
        len(plot) - 0.2,
        f"Best: {best['label']}",
        ha="right",
        va="bottom",
        fontsize=8,
        color=COLORS["ink"],
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_performance_ablation.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / "figure3_performance_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def calibration_curve(labels: np.ndarray, probability: np.ndarray, bins: int = 10):
    edges = np.quantile(probability, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    observed = []
    predicted = []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (probability >= left) & (probability <= right)
        if selected.sum() >= 10:
            observed.append(labels[selected].mean())
            predicted.append(probability[selected].mean())
    return np.asarray(predicted), np.asarray(observed)


def figure_four() -> None:
    selected = ("xgb_static_ts", "transformer", "temporal_ensemble")
    labels_map = {
        "xgb_static_ts": "XGBoost + temporal summaries",
        "transformer": "Transformer",
        "temporal_ensemble": "Temporal ensemble",
    }
    colors = [COLORS["blue"], COLORS["teal"], COLORS["orange"]]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    for column, cohort in enumerate(("mimic3", "eicu_anycad")):
        ax = axes[0, column]
        ax.plot([0, 1], [0, 1], color=COLORS["gray"], linestyle="--", linewidth=1)
        for model, color in zip(selected, colors):
            path = PREDICTIONS / f"{cohort}_external_{model}.npz"
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=True) as data:
                labels = data["labels"].astype(int)
                probability = data[f"{model}__recalibrated"].astype(float)
            predicted, observed = calibration_curve(labels, probability)
            ax.plot(
                predicted,
                observed,
                marker="o",
                markersize=3.5,
                linewidth=1.7,
                color=color,
                label=labels_map[model],
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted risk")
        ax.set_ylabel("Observed risk")
        ax.set_title(
            "MIMIC-III" if cohort == "mimic3" else "eICU expanded CAD",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(color="#E4EAEE", linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)

        dca_ax = axes[1, column]
        reference = PREDICTIONS / f"{cohort}_external_xgb_static_ts.npz"
        if reference.is_file():
            with np.load(reference, allow_pickle=True) as data:
                referral_labels = data["labels"].astype(int)
        else:
            referral_labels = None
        for model, color in zip(selected, colors):
            path = PREDICTIONS / f"{cohort}_external_{model}.npz"
            if not path.is_file() or referral_labels is None:
                continue
            with np.load(path, allow_pickle=True) as data:
                probability = data[model].astype(float)
            thresholds = np.linspace(0.02, 0.35, 34)
            net = []
            for threshold in thresholds:
                predicted = probability >= threshold
                tp = np.sum(predicted & (referral_labels == 1))
                fp = np.sum(predicted & (referral_labels == 0))
                net.append(
                    tp / len(referral_labels)
                    - fp
                    / len(referral_labels)
                    * threshold
                    / (1 - threshold)
                )
            dca_ax.plot(
                thresholds,
                net,
                linewidth=1.7,
                color=color,
                label=labels_map[model],
            )
        prevalence = referral_labels.mean() if referral_labels is not None else 0
        thresholds = np.linspace(0.02, 0.35, 34)
        treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
        dca_ax.plot(
            thresholds,
            treat_all,
            color=COLORS["gray"],
            linestyle="--",
            linewidth=1,
            label="Treat all",
        )
        dca_ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
        dca_ax.set_xlabel("Threshold probability")
        dca_ax.set_ylabel("Net benefit")
        dca_ax.grid(color="#E4EAEE", linewidth=0.7)
        dca_ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4_calibration_dca.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / "figure4_calibration_dca.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_five() -> None:
    path = PREDICTIONS / "mimic4_attention_ctogt.npz"
    if not path.is_file():
        return
    with np.load(path, allow_pickle=True) as data:
        organ_keys = [key for key in data.files if "organ" in key]
        temporal_keys = [key for key in data.files if "temporal" in key]
        if not organ_keys or not temporal_keys:
            return
        organ = np.mean([data[key] for key in organ_keys], axis=0)
        temporal = np.mean([data[key] for key in temporal_keys], axis=0)
    cohort = pd.read_csv(PROCESSED / "mimic4" / "cohort.csv")
    labels = cohort["macce7"].astype(int).to_numpy()
    organ_event = organ[labels == 1].mean(axis=0)
    organ_free = organ[labels == 0].mean(axis=0)
    difference = organ_event - organ_free
    order = np.argsort(difference)
    top = list(order[-10:])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    y = np.arange(len(top))
    ax.barh(
        y,
        difference[top],
        color=[COLORS["orange"] if value > 0 else COLORS["blue"] for value in difference[top]],
    )
    ax.set_yticks(y)
    ax.set_yticklabels([ORGAN_CHANNELS[index] for index in top])
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xlabel("MACCE minus event-free attention")
    ax.set_title("Organ attention", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#E4EAEE", linewidth=0.7)
    ax = axes[1]
    event_temporal = temporal[labels == 1].mean(axis=0)
    free_temporal = temporal[labels == 0].mean(axis=0)
    ax.plot(range(1, 25), free_temporal, color=COLORS["blue"], linewidth=1.8, label="Event-free")
    ax.plot(range(1, 25), event_temporal, color=COLORS["orange"], linewidth=1.8, label="MACCE")
    ax.set_xlabel("ICU hour")
    ax.set_ylabel("Mean temporal attention")
    ax.set_title("Temporal attention", fontsize=11, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(color="#E4EAEE", linewidth=0.7)
    for a in axes:
        for spine in ("top", "right"):
            a.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure5_attention.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / "figure5_attention.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_six_utility_ci() -> None:
    path = RESULTS / "clinical_utility_analysis.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    cohorts = (("mimic3", "MIMIC-III"), ("eicu_anycad", "eICU expanded CAD"))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharey=True)
    for ax, (cohort, label) in zip(axes, cohorts):
        settings = (
            ("temporal_ensemble", "raw", "Raw", COLORS["blue"]),
            (
                "temporal_ensemble",
                "calibrated",
                "Locally recalibrated",
                COLORS["orange"],
            ),
        )
        for model, source, title, color in settings:
            prediction_path = PREDICTIONS / f"{cohort}_external_{model}.npz"
            if not prediction_path.is_file():
                continue
            with np.load(prediction_path, allow_pickle=True) as prediction:
                labels = prediction["labels"].astype(int)
                key = (
                    model
                    if source == "raw"
                    else f"{model}__recalibrated"
                )
                probability = prediction[key].astype(float)
            thresholds = np.asarray(
                data.get("dca_thresholds", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
            )
            net = []
            low = []
            high = []
            for threshold in thresholds:
                selected = probability >= threshold
                tp = np.sum(selected & (labels == 1))
                fp = np.sum(selected & (labels == 0))
                net.append(
                    tp / len(labels)
                    - fp / len(labels) * threshold / (1.0 - threshold)
                )
                ci = data["cohorts"][cohort]["models"][model]["bootstrap"][
                    "net_benefit_ci"
                ][f"{threshold:.2f}"][source]
                low.append(ci[0])
                high.append(ci[1])
            ax.plot(
                thresholds,
                net,
                marker="o",
                markersize=4,
                linewidth=1.8,
                color=color,
                label=title,
            )
            ax.fill_between(
                thresholds,
                low,
                high,
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        with np.load(
            PREDICTIONS / f"{cohort}_external_xgb_static_ts.npz",
            allow_pickle=True,
        ) as prediction:
            prevalence = prediction["labels"].astype(int).mean()
        thresholds = np.asarray(
            data.get("dca_thresholds", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        )
        treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
        ax.plot(
            thresholds,
            treat_all,
            color=COLORS["gray"],
            linestyle="--",
            linewidth=1,
            label="Treat all",
        )
        ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Threshold probability")
        ax.grid(color="#E4EAEE", linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Net benefit")
    fig.tight_layout()
    fig.savefig(
        FIGURES / "figure6_net_benefit_ci.png",
        dpi=320,
        bbox_inches="tight",
    )
    fig.savefig(FIGURES / "figure6_net_benefit_ci.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_seven_interval_sensitivity() -> None:
    path = RESULTS / "interval_threshold_sensitivity.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    panels = (
        ("eicu_documentation_lag_hours", "eicu", "eICU principal CAD"),
        (
            "eicu_documentation_lag_hours",
            "eicu_anycad",
            "eICU expanded CAD",
        ),
        ("troponin_threshold_multiplier", "mimic4", "MIMIC-IV"),
        ("troponin_threshold_multiplier", "mimic3", "MIMIC-III"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    for ax, (family, cohort, label) in zip(axes.flat, panels):
        variants = data.get(family, {})
        parameters = sorted(float(value) for value in variants)
        events = []
        xgb = []
        ensemble = []
        for parameter in parameters:
            record = variants[f"{parameter:.1f}"][cohort]
            events.append(record["anchor_overlap_events"])
            xgb.append(
                record["models"].get("xgb_static_ts", {}).get("auroc", np.nan)
            )
            ensemble.append(
                record["models"].get("temporal_ensemble", {}).get(
                    "auroc", np.nan
                )
            )
        positions = np.arange(len(parameters))
        ax.bar(
            positions,
            events,
            color=COLORS["light"],
            edgecolor=COLORS["gray"],
            linewidth=0.7,
            label="Anchor-overlap events",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{value:g}" for value in parameters])
        ax.set_xlabel(
            "Documentation lag, hours"
            if family == "eicu_documentation_lag_hours"
            else "Troponin threshold multiplier"
        )
        ax.set_ylabel("Anchor-overlap events")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#E4EAEE", linewidth=0.7)
        ax2 = ax.twinx()
        ax2.plot(
            positions,
            xgb,
            color=COLORS["blue"],
            marker="o",
            linewidth=1.8,
            label="XGB+TS AUROC",
        )
        ax2.plot(
            positions,
            ensemble,
            color=COLORS["orange"],
            marker="o",
            linewidth=1.8,
            label="Temporal ensemble AUROC",
        )
        ax2.set_ylabel("AUROC")
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            handles1 + handles2,
            labels1 + labels2,
            frameon=False,
            fontsize=8,
            loc="best",
        )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "figure7_interval_threshold_sensitivity.png",
        dpi=320,
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURES / "figure7_interval_threshold_sensitivity.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    table1 = table_one()
    metrics, _ = load_development_metrics()
    table2 = table_two(metrics)
    external = load_external_metrics()
    table3 = table_three(external)
    table4 = table_four_interval()
    table5 = table_five_validation()
    table6 = table_six_manual_review()
    table7 = table_seven_clinical_utility()
    table8 = table_eight_interval_sensitivity()
    figure_one()
    figure_two()
    figure_three(metrics)
    figure_four()
    figure_five()
    figure_six_utility_ci()
    figure_seven_interval_sensitivity()
    print(table1.to_string(index=False))
    print("\nDevelopment performance\n", table2.to_string(index=False))
    print("\nTransport performance\n", table3.to_string(index=False))
    print("\nInterval endpoint sensitivity\n", table4.to_string(index=False))
    print("\nLocked temporal and nested validation\n", table5.to_string(index=False))
    print("\nManual review pilot\n", table6.to_string(index=False))
    print("\nClinical utility\n", table7.to_string(index=False))
    print("\nInterval threshold sensitivity\n", table8.to_string(index=False))


if __name__ == "__main__":
    main()

"""Create manuscript tables from the locked processed cohorts and results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import (  # noqa: E402
    OUTCOME_5_NAMES,
    OUTCOME_6_NAMES,
    PROCESSED,
    RESULTS,
)


TABLE_DIR = RESULTS / "tables"
CALIBRATION_DIR = RESULTS / "calibration"
ORDINAL_DIR = RESULTS / "ordinal_evaluation"
OUTCOME_LABELS = {
    0: "Home/against advice",
    1: "Home health care",
    2: "Rehabilitation/assisted living",
    3: "Skilled nursing/facility",
    4: "Hospice/in-hospital death",
}


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _median_iqr(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "Missing"
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return f"{median:.1f} ({q1:.1f}-{q3:.1f})"


def _n_percent(mask: pd.Series) -> str:
    valid = mask.notna()
    count = int(mask.fillna(False).sum())
    denominator = int(valid.sum())
    if denominator == 0:
        return "Missing"
    return f"{count} ({100 * count / denominator:.1f}%)"


def baseline_table() -> pd.DataFrame:
    cohort = pd.read_csv(PROCESSED / "mimic4" / "cohort.csv")
    flags = pd.read_csv(PROCESSED / "mimic4" / "diagnosis_flags.csv")
    cohort = cohort.merge(flags, on=["stay_id", "hadm_id"], how="left")
    continuous = [
        ("Age, years", "age"),
        ("Heart rate, mean", "hr_mean"),
        ("Respiratory rate, mean", "rr_mean"),
        ("SpO2, mean (%)", "spo2_mean"),
        ("Systolic pressure, mean", "sbp_mean"),
        ("Diastolic pressure, mean", "dbp_mean"),
        ("Mean arterial pressure, mean", "mbp_mean"),
        ("Temperature, mean (C)", "temp_mean"),
        ("GCS, mean", "gcs_mean"),
        ("Troponin, first", "lab_troponin_first"),
        ("Creatinine, first", "lab_creatinine_first"),
        ("Lactate, first", "lab_lactate_first"),
        ("White blood cell count, first", "lab_white_blood_cell_count_first"),
        ("Haemoglobin, first", "lab_hemoglobin_first"),
        ("Platelets, first", "lab_platelets_first"),
    ]
    categorical = [
        ("Male sex", "sex_male"),
        ("White race", "race_white"),
        ("Black race", "race_black"),
        ("Emergency admission", "emergency_admission"),
        ("CCU", "care_unit_ccu"),
        ("CVICU", "care_unit_cvicu"),
        ("MICU", "care_unit_micu"),
        ("SICU", "care_unit_sicu"),
        ("TSICU", "care_unit_tsicu"),
        ("Neurological ICU", "care_unit_neuro"),
    ]
    subtype = [
        ("STEMI", "STEMI"),
        ("NSTEMI", "NSTEMI"),
        ("Type 2 MI", "Type 2 MI"),
        ("Unspecified or other MI", "Other/Unspecified"),
        ("Primary-position AMI", "primary"),
    ]
    rows = []
    groups = {
        int(outcome): frame
        for outcome, frame in cohort.groupby("outcome", sort=True)
    }
    for label, column in continuous:
        values = pd.to_numeric(cohort[column], errors="coerce")
        grouped = [
            pd.to_numeric(groups[outcome][column], errors="coerce").dropna()
            for outcome in sorted(groups)
        ]
        p_value = (
            float(stats.kruskal(*grouped).pvalue)
            if all(len(values) > 1 for values in grouped)
            else float("nan")
        )
        rows.append(
            {
                "characteristic": label,
                "overall": _median_iqr(values),
                **{
                    OUTCOME_LABELS[outcome]: _median_iqr(groups[outcome][column])
                    for outcome in sorted(groups)
                },
                "p_value": p_value,
                "summary": "median (IQR)",
            }
        )
    for label, column in categorical:
        mask = cohort[column].astype(float)
        table = pd.crosstab(cohort["outcome"], cohort[column].fillna(0))
        p_value = (
            float(stats.chi2_contingency(table)[1])
            if table.shape == (len(groups), 2)
            else float("nan")
        )
        rows.append(
            {
                "characteristic": label,
                "overall": _n_percent(mask),
                **{
                    OUTCOME_LABELS[outcome]: _n_percent(
                        groups[outcome][column].astype(float)
                    )
                    for outcome in sorted(groups)
                },
                "p_value": p_value,
                "summary": "n (%)",
            }
        )
    for label, subtype_value in subtype:
        if subtype_value == "primary":
            mask = cohort["ami_primary_position"].fillna(0).eq(1)
            p_value = float(
                stats.chi2_contingency(
                    pd.crosstab(cohort["outcome"], mask)
                )[1]
            )
        else:
            mask = (
                cohort["ami_subtype"].isin(["Other", "Unspecified"])
                if subtype_value == "Other/Unspecified"
                else cohort["ami_subtype"].eq(subtype_value)
            )
            p_value = float(
                stats.chi2_contingency(
                    pd.crosstab(
                        cohort["outcome"],
                        cohort["ami_subtype"].isin(
                            ["Other", "Unspecified"]
                        )
                        if subtype_value == "Other/Unspecified"
                        else cohort["ami_subtype"].eq(subtype_value),
                    )
                )[1]
            )
        rows.append(
            {
                "characteristic": label,
                "overall": _n_percent(mask),
                **{
                    OUTCOME_LABELS[outcome]: _n_percent(
                        (
                            groups[outcome]["ami_subtype"].isin(
                                ["Other", "Unspecified"]
                            )
                            if subtype_value == "Other/Unspecified"
                            else (
                                groups[outcome]["ami_primary_position"]
                                .fillna(0)
                                .eq(1)
                                if subtype_value == "primary"
                                else groups[outcome]["ami_subtype"].eq(
                                    subtype_value
                                )
                            )
                        )
                    )
                    for outcome in sorted(groups)
                },
                "p_value": p_value,
                "summary": "n (%)",
            }
        )
    table = pd.DataFrame(rows)
    table.insert(1, "overall_n", len(cohort))
    return table


def outcome_table() -> pd.DataFrame:
    rows = []
    for dataset, outcome_column, labels in (
        ("mimic4", "outcome", OUTCOME_5_NAMES),
        ("mimic3", "outcome", OUTCOME_5_NAMES),
    ):
        cohort = pd.read_csv(PROCESSED / dataset / "cohort.csv")
        for level, label in enumerate(labels):
            rows.append(
                {
                    "dataset": dataset,
                    "scale": "five-level",
                    "level": level,
                    "label": label,
                    "n": int(cohort[outcome_column].eq(level).sum()),
                    "fraction": float(cohort[outcome_column].eq(level).mean()),
                }
            )
    eicu = pd.read_csv(PROCESSED / "eicu" / "cohort.csv")
    eicu_labels = (
        "Home",
        "Rehabilitation (eICU only)",
        "Skilled nursing or other facility",
        "Death",
    )
    for level, label in enumerate(eicu_labels):
        rows.append(
            {
                "dataset": "eicu",
                "scale": "four-class harmonized",
                "level": level,
                "label": label,
                "n": int(eicu["outcome4"].eq(level).sum()),
                "fraction": float(eicu["outcome4"].eq(level).mean()),
            }
        )
    return pd.DataFrame(rows)


def model_table() -> pd.DataFrame | None:
    payload = _read_json(RESULTS / "metrics_mimic4.json")
    if payload is None:
        return None
    rows = []
    for model, metrics in payload["metrics"].items():
        ci = metrics["macro_auc_ci"]
        rows.append(
            {
                "model": model,
                "n": payload["n"],
                "seeds": len(payload["seeds"]),
                "macro_auc": metrics["macro_auc"],
                "macro_auc_ci_low": ci["ci_low"],
                "macro_auc_ci_high": ci["ci_high"],
                "accuracy": metrics["accuracy"],
                "quadratic_weighted_kappa": metrics[
                    "quadratic_weighted_kappa"
                ],
                "ordinal_mae": metrics["ordinal_mae"],
                "log_loss": metrics["log_loss"],
                "multiclass_brier": metrics["multiclass_brier"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        "macro_auc", ascending=False
    )


def contrast_table() -> pd.DataFrame | None:
    payload = _read_json(RESULTS / "metrics_mimic4.json")
    if payload is None:
        return None
    rows = []
    for key, label in (
        ("primary_contrast", "DG-GRU+TR minus XGB-static+TS"),
        ("regularization_contrast", "DG-GRU+TR minus DG-GRU"),
        ("time_series_increment", "XGB-static+TS minus XGB-static"),
        (
            "time_series_increment_mlp",
            "MLP-static+TS minus MLP-static",
        ),
        ("regularization_increment", "DG-GRU+TR minus DG-GRU"),
        ("deep_vs_static_ts", "DG-GRU+TR minus XGB-static+TS"),
        (
            "same_static_input_mlp_vs_xgb",
            "MLP-static minus XGB-static",
        ),
        (
            "same_full_input_xgb_vs_mlp",
            "XGB-static+TS minus MLP-static+TS",
        ),
        (
            "same_full_input_mlp_vs_dggru",
            "MLP-static+TS minus DG-GRU (representation plus structure)",
        ),
        (
            "same_full_input_gru_vs_dggru",
            "SeqOrdinal GRU minus DG-GRU",
        ),
        (
            "static_input_dggru_increment",
            "DG-GRU minus DG-GRU-static (raw sequence-plus-availability "
            "input toggle; fixed network structure)",
        ),
        (
            "cumulative_curvature_increment",
            "DG-GRU+CTR minus DG-GRU",
        ),
    ):
        if key not in payload:
            continue
        value = payload[key]
        if "delta" in value:
            rows.append(
                {
                    "contrast": label,
                    "estimate": value["delta"],
                    "ci_low": value.get("ci_low"),
                    "ci_high": value.get("ci_high"),
                    "metric": "macro-AUC",
                }
            )
        elif "delta_macro_auc" in value:
            result = value["delta_macro_auc"]
            rows.append(
                {
                    "contrast": label,
                    "estimate": result["delta"],
                    "ci_low": result.get("ci_low"),
                    "ci_high": result.get("ci_high"),
                    "metric": "macro-AUC",
                }
            )
    frame = pd.DataFrame(rows).drop_duplicates()
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["contrast"], keep="first")
    return frame


def _interval_text(
    estimate: float,
    low: float,
    high: float,
    *,
    digits: int = 3,
    signed: bool = False,
) -> str:
    if pd.isna(estimate):
        return "NA"
    template = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    estimate_text = template.format(float(estimate))
    if pd.isna(low) or pd.isna(high):
        return estimate_text
    low_text = template.format(float(low))
    high_text = template.format(float(high))
    return f"{estimate_text} ({low_text} to {high_text})"


def external_calibration_table() -> pd.DataFrame | None:
    path = CALIBRATION_DIR / "table7_external_calibration.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    rows = []
    for row in frame.itertuples(index=False):
        if row.target_key == "high_need":
            target_label = r"High-need, P(y>=2)"
        elif row.target_key == "death" and row.dataset == "MIMIC-III":
            target_label = "Hospice-or-death, P(y=3)"
        elif row.target_key == "death":
            target_label = "In-hospital death, P(y=3)"
        else:
            target_label = row.target
        rows.append(
            {
                "Dataset": row.dataset,
                "Target": target_label,
                "Model": row.model,
                "n/events": f"{int(row.n):,}/{int(row.events):,}",
                "Raw calibration slope": _interval_text(
                    row.raw_slope,
                    row.raw_slope_ci_low,
                    row.raw_slope_ci_high,
                ),
                "Raw calibration intercept": _interval_text(
                    row.raw_intercept,
                    row.raw_intercept_ci_low,
                    row.raw_intercept_ci_high,
                    signed=True,
                ),
                "OOF-recalibrated calibration slope": _interval_text(
                    row.oof_recal_slope,
                    row.oof_recal_slope_ci_low,
                    row.oof_recal_slope_ci_high,
                ),
                "OOF-recalibrated calibration intercept": _interval_text(
                    row.oof_recal_intercept,
                    row.oof_recal_intercept_ci_low,
                    row.oof_recal_intercept_ci_high,
                    signed=True,
                ),
                "Raw Brier": _interval_text(
                    row.raw_brier,
                    row.raw_brier_ci_low,
                    row.raw_brier_ci_high,
                ),
                "OOF-recalibrated Brier": _interval_text(
                    row.oof_recal_brier,
                    row.oof_recal_brier_ci_low,
                    row.oof_recal_brier_ci_high,
                ),
            }
        )
    return pd.DataFrame(rows)


def common_three_class_table() -> pd.DataFrame | None:
    path = CALIBRATION_DIR / "table8_common_three_class.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "Dataset": row.dataset,
                "Model": row.model,
                "n (class 0/1/2)": (
                    f"{int(row.n):,} "
                    f"({int(row.class_0_n):,}/"
                    f"{int(row.class_1_n):,}/"
                    f"{int(row.class_2_n):,})"
                ),
                "Macro-AUC": _interval_text(
                    row.macro_auc,
                    row.macro_auc_ci_low,
                    row.macro_auc_ci_high,
                ),
                "Ordinal MAE": _interval_text(
                    row.ordinal_mae,
                    row.ordinal_mae_ci_low,
                    row.ordinal_mae_ci_high,
                ),
                "QWK": _interval_text(
                    row.quadratic_weighted_kappa,
                    row.quadratic_weighted_kappa_ci_low,
                    row.quadratic_weighted_kappa_ci_high,
                ),
            }
        )
    return pd.DataFrame(rows)


def ordinal_metrics_table() -> pd.DataFrame | None:
    path = ORDINAL_DIR / "table9_ordinal_metrics.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "Model": row.model_label,
                "Ordinal C-index": _interval_text(
                    row.ordinal_c_index,
                    row.ordinal_c_index_ci_low,
                    row.ordinal_c_index_ci_high,
                ),
                "QWK": _interval_text(
                    row.quadratic_weighted_kappa,
                    row.qwk_ci_low,
                    row.qwk_ci_high,
                ),
                "Ordinal MAE": _interval_text(
                    row.ordinal_mae,
                    row.ordinal_mae_ci_low,
                    row.ordinal_mae_ci_high,
                ),
                "Brier score": _interval_text(
                    row.multiclass_brier,
                    row.multiclass_brier_ci_low,
                    row.multiclass_brier_ci_high,
                ),
            }
        )
    return pd.DataFrame(rows)


def ordinal_contrasts_table() -> pd.DataFrame | None:
    path = ORDINAL_DIR / "table9b_ordinal_contrasts.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "Contrast": row.contrast,
                "Delta ordinal C-index": _interval_text(
                    row.ordinal_c_index_delta,
                    row.ordinal_c_index_ci_low,
                    row.ordinal_c_index_ci_high,
                    signed=True,
                ),
                "Delta QWK": _interval_text(
                    row.qwk_delta,
                    row.qwk_ci_low,
                    row.qwk_ci_high,
                    signed=True,
                ),
                "Delta ordinal MAE": _interval_text(
                    row.ordinal_mae_delta,
                    row.ordinal_mae_ci_low,
                    row.ordinal_mae_ci_high,
                    signed=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def threshold_calibration_table() -> pd.DataFrame | None:
    path = ORDINAL_DIR / "table10_threshold_calibration.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    frame = frame.loc[frame["record_type"].eq("threshold_performance")]
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "Model": row.model_label,
                "Threshold": row.threshold_label,
                "Events/N": f"{int(row.events):,}/{int(row.n):,}",
                "AUROC": _interval_text(
                    row.auroc,
                    row.auroc_ci_low,
                    row.auroc_ci_high,
                ),
                "Calibration slope": _interval_text(
                    row.calibration_slope,
                    row.calibration_slope_ci_low,
                    row.calibration_slope_ci_high,
                ),
                "Intercept": _interval_text(
                    row.calibration_intercept,
                    row.calibration_intercept_ci_low,
                    row.calibration_intercept_ci_high,
                    signed=True,
                ),
                "Brier": _interval_text(
                    row.brier,
                    row.brier_ci_low,
                    row.brier_ci_high,
                ),
            }
        )
    return pd.DataFrame(rows)


def threshold_risk_strata_table() -> pd.DataFrame | None:
    path = ORDINAL_DIR / "table10_threshold_calibration.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    frame = frame.loc[frame["record_type"].eq("risk_stratum")]
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "Model": row.model_label,
                "Threshold": row.threshold_label,
                "Risk stratum": row.risk_stratum,
                "N": f"{int(row.n):,}",
                "Events": f"{int(row.events):,}",
                "Mean predicted": f"{float(row.mean_predicted):.3f}",
                "Observed": f"{float(row.observed_rate):.3f}",
                "Gap": f"{float(row.calibration_gap):+.3f}",
            }
        )
    return pd.DataFrame(rows)


def multiseed_temporal_table() -> pd.DataFrame | None:
    path = TABLE_DIR / "table4b_multiseed_temporal_anchors.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if "delta_macro_auc_vs_no_tr" not in frame:
        no_tr = frame.loc[frame["configuration"].eq("No TR"), "macro_auc"]
        if len(no_tr) != 1:
            raise ValueError("Expected exactly one No TR temporal anchor")
        frame["delta_macro_auc_vs_no_tr"] = (
            frame["macro_auc"] - float(no_tr.iloc[0])
        )
    reference = frame.loc[frame["configuration"].eq("No TR")]
    if len(reference) != 1:
        raise ValueError("Expected exactly one No TR temporal anchor")
    output = frame[
        [
            "configuration",
            "seeds",
            "macro_auc",
            "macro_auc_ci_low",
            "macro_auc_ci_high",
            "seed_macro_auc_mean",
            "seed_macro_auc_sd",
            "delta_macro_auc_vs_no_tr",
        ]
    ].copy()
    output["macro_auc"] = output.apply(
        lambda row: _interval_text(
            row["macro_auc"],
            row["macro_auc_ci_low"],
            row["macro_auc_ci_high"],
            digits=4,
        ),
        axis=1,
    )
    output["delta_macro_auc_vs_no_tr"] = output[
        "delta_macro_auc_vs_no_tr"
    ].map(lambda value: f"{float(value):+.6f}")
    output["seed_macro_auc"] = output.apply(
        lambda row: (
            f"{float(row['seed_macro_auc_mean']):.4f} "
            f"(SD {float(row['seed_macro_auc_sd']):.4f})"
        ),
        axis=1,
    )
    result = output[
        [
            "configuration",
            "seeds",
            "macro_auc",
            "delta_macro_auc_vs_no_tr",
            "seed_macro_auc",
        ]
    ].rename(
        columns={
            "configuration": "Configuration",
            "seeds": "Seeds",
            "macro_auc": "Seed-averaged macro-AUC (95% CI)",
            "delta_macro_auc_vs_no_tr": (
                "Delta vs No TR (current minus No TR)"
            ),
            "seed_macro_auc": "Single-seed macro-AUC, mean (SD)",
        }
    )
    result.attrs["disable_numparse"] = True
    return result


def cumulative_curvature_table() -> pd.DataFrame | None:
    path = (
        RESULTS
        / "sensitivity"
        / "cumulative_curvature"
        / "summary_seed42.json"
    )
    if not path.is_file():
        return None
    payload = _read_json(path)
    if not payload:
        return None
    rows = []
    for record in payload:
        rows.append(
            {
                "Lambda curvature": float(record["lambda_curvature"]),
                "Macro-AUC": float(record["metrics"]["macro_auc"]),
                "QWK": float(
                    record["metrics"]["quadratic_weighted_kappa"]
                ),
                "Ordinal MAE": float(record["metrics"]["ordinal_mae"]),
                "Brier": float(record["metrics"]["multiclass_brier"]),
            }
        )
    return pd.DataFrame(rows).sort_values("Lambda curvature")


def external_table() -> pd.DataFrame | None:
    payload = _read_json(RESULTS / "metrics_transport.json")
    if payload is None:
        return None
    rows = []
    for dataset, dataset_result in payload.items():
        for model, model_result in dataset_result["models"].items():
            metrics = model_result["metrics"]
            highest = model_result["highest_class_auc_ci"]
            strict = model_result.get("strict_mortality_auc_ci")
            rows.append(
                {
                    "dataset": dataset,
                    "validation_type": dataset_result["status"],
                    "model": model,
                    "n": dataset_result["n"],
                    "macro_auc": metrics["macro_auc"],
                    "macro_auc_ci_low": model_result["macro_auc_ci"][
                        "ci_low"
                    ],
                    "macro_auc_ci_high": model_result["macro_auc_ci"][
                        "ci_high"
                    ],
                    "highest_class_auc": highest["estimate"],
                    "highest_class_ci_low": highest["ci_low"],
                    "highest_class_ci_high": highest["ci_high"],
                    "strict_death_auc": (
                        float("nan") if strict is None else strict["estimate"]
                    ),
                    "strict_death_ci_low": (
                        float("nan") if strict is None else strict["ci_low"]
                    ),
                    "strict_death_ci_high": (
                        float("nan") if strict is None else strict["ci_high"]
                    ),
                    "calibration_slope_high_need": model_result[
                        "calibration"
                    ]["high_need_y_ge_2"]["slope"],
                    "calibration_intercept_high_need": model_result[
                        "calibration"
                    ]["high_need_y_ge_2"]["intercept"],
                }
            )
    return pd.DataFrame(rows)


def cohort_flow_table() -> pd.DataFrame:
    labels = {
        "source_ami_admissions_or_units": "AMI admissions or unit stays",
        "first_icu_or_unit": "First ICU stay or unit",
        "adult": "Age >=18 years",
        "alive_and_in_icu_at_24h": "Alive and in ICU at 24 h",
        "mapped_disposition": "Mapped discharge disposition",
        "deduplicated_analysis_unit": "Analytic unit after deduplication",
    }
    rows = []
    for dataset in ("mimic4", "mimic3", "eicu"):
        flow = pd.read_csv(PROCESSED / dataset / "cohort_flow.csv")
        for record in flow.itertuples(index=False):
            rows.append(
                {
                    "dataset": dataset,
                    "step": getattr(record, "reason_code", ""),
                    "description": labels.get(
                        getattr(record, "reason_code", ""),
                        getattr(record, "reason_code", ""),
                    ),
                    "n": int(record.n),
                }
            )
    return pd.DataFrame(rows)


def coverage_table() -> pd.DataFrame:
    audit = _read_json(RESULTS / "audits" / "cohort_audit.json")
    if audit is None:
        raise FileNotFoundError("results/audits/cohort_audit.json")
    rows = []
    for dataset in ("mimic4", "mimic3", "eicu"):
        payload = audit[dataset]
        n = int(payload["n"])
        rows.append(
            {
                "dataset": dataset,
                "feature": "valid physiology in >=4/8 channels",
                "available_n": int(payload["at_least_4_of_8"]),
                "available_fraction": float(
                    payload["at_least_4_of_8"] / n
                ),
            }
        )
        for channel, count in payload["per_channel"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "feature": channel,
                    "available_n": int(count),
                    "available_fraction": float(count / n),
                }
            )
        for lab, values in payload["lab_coverage"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "feature": f"lab_{lab}",
                    "available_n": int(values["n"]),
                    "available_fraction": float(values["fraction"]),
                }
            )
    return pd.DataFrame(rows)


def temporal_controls_table() -> pd.DataFrame | None:
    path = (
        RESULTS
        / "sensitivity"
        / "temporal_controls"
        / "summary_seed42.json"
    )
    if not path.is_file():
        return None
    rows = []
    for result in json.loads(path.read_text(encoding="utf-8")):
        rows.append(
            {
                "configuration": result["name"],
                "mode": result["mode"],
                "lambda_smooth": result["lambda_smooth"],
                "lambda_drift": result["lambda_drift"],
                "macro_auc": result["metrics"]["macro_auc"],
                "quadratic_weighted_kappa": result["metrics"][
                    "quadratic_weighted_kappa"
                ],
                "ordinal_mae": result["metrics"]["ordinal_mae"],
                "mean_adjacent_squared_change": result["trajectory"][
                    "mean_adjacent_squared_change"
                ],
                "mean_absolute_net_drift": result["trajectory"][
                    "mean_absolute_net_drift"
                ],
            }
        )
    cluster = _read_json(
        RESULTS / "audits" / "eicu_hospital_cluster_cv_audit.json"
    )
    if cluster:
        metric = cluster["hospital_cluster_grouped"]
        interval = metric["macro_auc_ci"]
        rows.append(
            {
                "dataset": "eicu",
                "validation_type": (
                    "eICU internal CV, hospital-cluster grouped"
                ),
                "model": "xgb_static_ts",
                "n": cluster["n"],
                "macro_auc": interval["estimate"],
                "macro_auc_ci_low": interval["ci_low"],
                "macro_auc_ci_high": interval["ci_high"],
                "highest_class_auc": float("nan"),
                "highest_class_ci_low": float("nan"),
                "highest_class_ci_high": float("nan"),
                "highest_class_auc": float("nan"),
                "highest_class_ci_low": float("nan"),
                "highest_class_ci_high": float("nan"),
                "strict_death_auc": float("nan"),
                "strict_death_ci_low": float("nan"),
                "strict_death_ci_high": float("nan"),
                "calibration_slope_high_need": float("nan"),
                "calibration_intercept_high_need": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_table() -> pd.DataFrame | None:
    paths = sorted(
        (RESULTS / "sensitivity").glob(
            "metrics_mimic4_*_xgb_static_ts.json"
        )
    )
    paths = [
        path
        for path in paths
        if "severity" not in path.name
    ]
    if not paths:
        return None
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        high_need_calibration = (
            payload.get("calibration", {}).get("p_y_ge_3")
        )
        rows.append(
            {
                "analysis": payload["analysis"],
                "model": payload["model"],
                "n": payload["n"],
                "outcome_counts": json.dumps(
                    payload["outcome_counts"], sort_keys=True
                ),
                "macro_auc": metrics["macro_auc"],
                "macro_auc_ci_low": metrics["macro_auc_ci"]["ci_low"],
                "macro_auc_ci_high": metrics["macro_auc_ci"]["ci_high"],
                "calibration_slope_high_need": (
                    high_need_calibration.get("slope")
                    if high_need_calibration
                    else None
                ),
                "calibration_intercept_high_need": (
                    high_need_calibration.get("intercept")
                    if high_need_calibration
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def landmark_performance_table() -> pd.DataFrame | None:
    sources = (
        (6, RESULTS / "metrics_mimic4_6h.json"),
        (24, RESULTS / "metrics_mimic4.json"),
        (72, RESULTS / "metrics_mimic4_72h.json"),
    )
    models = ("xgb_static", "xgb_static_ts", "dggru", "dggru_tr")
    rows = []
    seen = False
    for landmark, path in sources:
        payload = _read_json(path)
        if payload is None:
            continue
        seen = True
        for model in models:
            metric = payload.get("metrics", {}).get(model)
            if metric is None:
                continue
            interval = metric["macro_auc_ci"]
            rows.append(
                {
                    "landmark_hours": landmark,
                    "model": model,
                    "n": int(payload["n"]),
                    "macro_auc": float(interval["estimate"]),
                    "macro_auc_ci_low": float(interval["ci_low"]),
                    "macro_auc_ci_high": float(interval["ci_high"]),
                    "accuracy": float(metric["accuracy"]),
                    "quadratic_weighted_kappa": float(
                        metric["quadratic_weighted_kappa"]
                    ),
                    "ordinal_mae": float(metric["ordinal_mae"]),
                    "multiclass_brier": float(metric["multiclass_brier"]),
                }
            )
    return pd.DataFrame(rows) if seen else None


def full_sensitivity_table() -> pd.DataFrame | None:
    rows: list[dict] = []

    def add_metric(
        scenario_id: int,
        scenario: str,
        database: str,
        outcome_scale: str,
        model: str,
        payload: dict,
        *,
        reference: float | None,
        note: str,
    ) -> None:
        metric = payload.get("metrics", payload)
        interval = metric.get("macro_auc_ci")
        if interval is None:
            interval = {
                "estimate": metric.get("macro_auc"),
                "ci_low": np.nan,
                "ci_high": np.nan,
            }
        estimate = pd.to_numeric(
            pd.Series([interval.get("estimate")]), errors="coerce"
        ).iloc[0]
        rows.append(
            {
                "scenario_id": scenario_id,
                "scenario": scenario,
                "database": database,
                "outcome_scale": outcome_scale,
                "model": model,
                "n": int(payload.get("n", np.nan)),
                "macro_auc": float(estimate),
                "macro_auc_ci_low": float(interval.get("ci_low", np.nan)),
                "macro_auc_ci_high": float(interval.get("ci_high", np.nan)),
                "reference_macro_auc": reference,
                "delta_from_reference": (
                    float(estimate - reference)
                    if reference is not None
                    and np.isfinite(float(estimate))
                    else np.nan
                ),
                "note": note,
            }
        )

    primary = _read_json(RESULTS / "metrics_mimic4.json")
    primary_reference = (
        float(primary["metrics"]["xgb_static_ts"]["macro_auc"])
        if primary
        else None
    )
    mimic_specs = (
        (
            1,
            "6 h landmark",
            "landmark_6h",
            RESULTS / "metrics_mimic4_6h.json",
        ),
        (
            2,
            "72 h landmark",
            "landmark_72h",
            RESULTS / "metrics_mimic4_72h.json",
        ),
        (
            3,
            "Six-level outcome",
            "six_level",
            RESULTS / "sensitivity" / "metrics_mimic4_six_level_xgb_static_ts.json",
        ),
        (
            4,
            "Primary-position AMI",
            "primary_position",
            RESULTS
            / "sensitivity"
            / "metrics_mimic4_primary_position_xgb_static_ts.json",
        ),
        (
            5,
            "Exclude type 2 MI",
            "exclude_type2",
            RESULTS
            / "sensitivity"
            / "metrics_mimic4_exclude_type2_xgb_static_ts.json",
        ),
        (
            6,
            "Exclude discharge against advice",
            "exclude_against_advice",
            RESULTS
            / "sensitivity"
            / "metrics_mimic4_exclude_against_advice_xgb_static_ts.json",
        ),
        (
            7,
            "Exclude troponin",
            "exclude_troponin",
            RESULTS
            / "sensitivity"
            / "metrics_mimic4_exclude_troponin_xgb_static_ts.json",
        ),
        (
            8,
            "Multiple imputation",
            "multiple_imputation",
            RESULTS
            / "sensitivity"
            / "metrics_mimic4_multiple_imputation_xgb_static_ts.json",
        ),
        (
            9,
            "Pre-24 h revascularization",
            "include_pre24h_revascularization",
            RESULTS
            / "sensitivity"
            / "metrics_mimic4_include_pre24h_revascularization_xgb_static_ts.json",
        ),
        (
            10,
            "Pre-24 h vasopressor or MCS",
            "include_pre24h_vasopressor_or_mcs",
            RESULTS
            / "sensitivity"
            / "metrics_mimic4_include_pre24h_vasopressor_or_mcs_xgb_static_ts.json",
        ),
        (
            15,
            "Merge hospice/death and rehabilitation/facility",
            "label_merge_hospice_death_and_rehab_facility",
            RESULTS
            / "sensitivity"
            / "metrics_mimic4_label_merge_hospice_death_and_rehab_facility_xgb_static_ts.json",
        ),
    )
    for scenario_id, scenario, key, path in mimic_specs:
        report = _read_json(path)
        if report is None:
            continue
        if key.startswith("landmark_"):
            selected = report.get("metrics", {}).get("xgb_static_ts")
            if selected is None:
                continue
            payload = {"n": int(report["n"]), **selected}
        else:
            payload = report
        payload.setdefault("n", payload.get("n"))
        if key == "six_level":
            outcome_scale = "Six-level"
            reference = None
            note = "Prespecified outcome-scale sensitivity"
        elif key == "label_merge_hospice_death_and_rehab_facility":
            outcome_scale = "Four-class merged labels"
            reference = None
            note = "Prespecified merged-label sensitivity"
        else:
            outcome_scale = "Five-level"
            reference = primary_reference
            note = (
                "Separate landmark cohort"
                if key.startswith("landmark_")
                else "Prespecified cohort or feature sensitivity"
            )
        add_metric(
            scenario_id,
            scenario,
            "MIMIC-IV",
            outcome_scale,
            "XGB-static+TS",
            payload,
            reference=reference,
            note=note,
        )

    temporal_path = (
        RESULTS
        / "sensitivity"
        / "temporal_controls"
        / "summary_seed42.json"
    )
    temporal_payload = _read_json(temporal_path)
    if temporal_payload:
        temporal = {record["name"]: record for record in temporal_payload}
        no_tr = temporal.get("no_tr", {})
        temporal_reference = (
            float(no_tr["metrics"]["macro_auc"]) if no_tr else None
        )
        temporal_specs = (
            (
                11,
                "Reversed-direction monotone trajectory",
                "monotone_decrease",
            ),
            (
                12,
                "Time-permuted trajectory",
                "permuted",
            ),
            (
                13,
                "Matched generic smoothness",
                "smooth_only",
            ),
        )
        for scenario_id, scenario, name in temporal_specs:
            record = temporal.get(name)
            if record is None:
                continue
            add_metric(
                scenario_id,
                scenario,
                "MIMIC-IV",
                "Five-level",
                "DG-GRU+TR",
                {"n": int(primary["n"]) if primary else np.nan, **record},
                reference=temporal_reference,
                note="Seed-42 matched temporal-control analysis",
            )

    site_report = _read_json(
        RESULTS / "audits" / "eicu_site_heterogeneity_report.json"
    )
    transport = _read_json(RESULTS / "metrics_transport.json")
    eicu_reference = (
        float(
            transport["eicu"]["models"]["dggru_tr"]["macro_auc_ci"][
                "estimate"
            ]
        )
        if transport
        and "eicu" in transport
        and "dggru_tr" in transport["eicu"].get("models", {})
        else None
    )
    if site_report:
        site_primary = site_report["models"]["dggru_tr"]
        rows.append(
            {
                "scenario_id": 14,
                "scenario": "eICU patient-safe hospital-cluster bootstrap",
                "database": "eICU-CRD",
                "outcome_scale": "Four-class",
                "model": "DG-GRU+TR",
                "n": int(site_report["cohort"]["n_admissions"]),
                "macro_auc": float(
                    site_primary["cluster_holdout_full_macro_auc"]
                ),
                "macro_auc_ci_low": float(
                    site_primary["cluster_bootstrap_macro_auc_ci_low"]
                ),
                "macro_auc_ci_high": float(
                    site_primary["cluster_bootstrap_macro_auc_ci_high"]
                ),
                "reference_macro_auc": eicu_reference,
                "delta_from_reference": (
                    float(
                        site_primary["cluster_holdout_full_macro_auc"]
                        - eicu_reference
                    )
                    if eicu_reference is not None
                    else np.nan
                ),
                "note": (
                    "Patient-safe hospital-cluster bootstrap; no model "
                    "refitting"
                ),
            }
        )

    transport_sensitivity = _read_json(
        RESULTS / "metrics_transport_sensitivity.json"
    )
    if transport_sensitivity:
        exclude_other = transport_sensitivity.get(
            "exclude_other_disposition", {}
        ).get("models", {}).get("dggru_tr")
        if exclude_other:
            add_metric(
                16,
                "eICU Other disposition excluded",
                "eICU-CRD",
                "Four-class",
                "DG-GRU+TR",
                exclude_other,
                reference=eicu_reference,
                note="Evaluation-only restriction of frozen transport predictions",
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return None
    frame = frame.sort_values("scenario_id").reset_index(drop=True)
    missing = sorted(set(range(1, 17)).difference(frame["scenario_id"]))
    frame.attrs["missing_scenarios"] = missing
    return frame


def primary_per_class_table() -> pd.DataFrame | None:
    payload = _read_json(RESULTS / "metrics_mimic4.json")
    if payload is None or "per_class_delong" not in payload:
        return None
    rows = []
    for record in payload["per_class_delong"]:
        class_index = int(record["class"])
        rows.append(
            {
                "class": class_index,
                "label": OUTCOME_LABELS[class_index],
                "auc_dggru_tr": record["auc_a"],
                "auc_xgb_static_ts": record["auc_b"],
                "delong_p": record["p_value"],
                "holm_p": record["holm_p"],
                "bonferroni_p": record["bonferroni_p"],
            }
        )
    return pd.DataFrame(rows)


def calibration_table() -> pd.DataFrame | None:
    payload = _read_json(RESULTS / "metrics_mimic4.json")
    if payload is None or "calibration" not in payload:
        return None
    rows = []
    for model in ("xgb_static_ts", "dggru", "dggru_tr"):
        if model not in payload["calibration"]:
            continue
        calibration = payload["calibration"][model]
        rows.append(
            {
                "model": model,
                "target": "P(y>=3)",
                "slope": calibration["p_y_ge_3"]["slope"],
                "intercept": calibration["p_y_ge_3"]["intercept"],
            }
        )
        rows.append(
            {
                "model": model,
                "target": "P(y=4)",
                "slope": calibration["p_y_4"]["slope"],
                "intercept": calibration["p_y_4"]["intercept"],
            }
        )
    return pd.DataFrame(rows)


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    options = {"disable_numparse": True} if frame.attrs.get(
        "disable_numparse"
    ) else {}
    path.write_text(
        frame.to_markdown(index=False, **options),
        encoding="utf-8",
    )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "table1_baseline_mimic4": baseline_table(),
        "table2_outcome_distribution": outcome_table(),
        "table4_temporal_controls": temporal_controls_table(),
        "table4c_cumulative_curvature_path": cumulative_curvature_table(),
        "table5_sensitivity_analyses": sensitivity_table(),
        "table3c_landmark_performance": landmark_performance_table(),
        "tableS15_full_16_scenario_sensitivity": full_sensitivity_table(),
        "tableS1_cohort_flow": cohort_flow_table(),
        "tableS2_feature_coverage": coverage_table(),
        "tableS3_primary_per_class": primary_per_class_table(),
        "tableS4_calibration": calibration_table(),
        "table7_external_calibration_main": external_calibration_table(),
        "table8_common_three_class_main": common_three_class_table(),
        "table9_ordinal_metrics_main": ordinal_metrics_table(),
        "table9b_ordinal_contrasts_main": ordinal_contrasts_table(),
        "table10_threshold_calibration_main": threshold_calibration_table(),
        "table10b_threshold_risk_strata": threshold_risk_strata_table(),
        "table4b_multiseed_temporal_anchors_main": multiseed_temporal_table(),
    }
    for name, getter in (
        ("table3_model_performance", model_table),
        ("table3b_paired_contrasts", contrast_table),
        ("table6_external_validation", external_table),
    ):
        table = getter()
        if table is not None:
            tables[name] = table
    written = {}
    for name, table in tables.items():
        if table is None:
            continue
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False)
        write_markdown(table, TABLE_DIR / f"{name}.md")
        written[name] = len(table)
    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()

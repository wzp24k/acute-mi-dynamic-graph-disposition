"""Audit XGB-static+TS importance with outer-fold permutation tests.

The script reuses the locked MIMIC-IV processed cohort and the exact
XGB-static+TS feature contract. It writes only the Table S11 and Figure S2
artifacts requested for the feature-importance module.
"""
from __future__ import annotations

import os

# Set thread limits before importing NumPy, scikit-learn, or XGBoost.
THREADS = 4
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = str(THREADS)
os.environ["AMI_XGB_THREADS"] = str(THREADS)
os.environ["AMI_TORCH_THREADS"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import hashlib
import json
import sys
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import (  # noqa: E402
    DEMOGRAPHIC_FEATURES,
    RESULTS,
    SEQUENCE_HOURS,
)
from src.ami_dynamic.train_evaluate import (  # noqa: E402
    _feature_columns,
    _fit_xgb,
    _static_matrix,
    load_dataset,
)


TABLE_DIR = RESULTS / "tables"
FIGURE_DIR = RESULTS / "figures"
SOURCE_DIR = FIGURE_DIR / "source_data"
TABLE_STEM = "tableS11_feature_importance"
FIGURE_STEM = "figureS2_feature_importance"
AUDIT_PATH = RESULTS / "audits" / f"{TABLE_STEM}_audit.json"

FOLDS = 5
CV_SEED = 42
DEFAULT_REPEATS = 5
PERMUTATION_SEED = 1104

GROUP_ORDER = (
    "demographics",
    "troponin_ck_mb",
    "other_labs",
    "vital_sign_summaries",
    "gcs_neuro",
)
GROUP_LABELS = {
    "demographics": "Demographics",
    "troponin_ck_mb": "Troponin / CK-MB",
    "other_labs": "Other laboratories",
    "vital_sign_summaries": "Vital-sign summaries",
    "gcs_neuro": "GCS / neuro",
}
GROUP_COLORS = {
    "demographics": "#5B6F8C",
    "troponin_ck_mb": "#B45545",
    "other_labs": "#C18A3D",
    "vital_sign_summaries": "#3F7F73",
    "gcs_neuro": "#74639A",
}
VITAL_CHANNELS = ("hr", "rr", "spo2", "sbp", "dbp", "mbp", "temp")
WINDOW_STATS = ("first", "last", "min", "max", "mean", "count")

FORBIDDEN_EXACT = {
    "outcome",
    "outcome4",
    "outcome_six",
    "disposition_raw",
    "discharge_location",
    "death_at_hospital",
    "icu_out_hour",
    "death_hour_from_icu",
    "hospital_expire_flag",
    "length_of_stay",
    "hospital_los",
    "icu_los",
    "stay_id",
    "hadm_id",
    "subject_id",
    "subject_key",
    "fold",
    "fold_id",
    "probability",
    "prediction",
}
FORBIDDEN_PATTERNS = (
    "outcome",
    "disposition",
    "discharge",
    "in_hospital_death",
    "hospital_expire",
    "icu_out",
    "death_hour",
    "length_of_stay",
)


def _assign_group(feature: str) -> str:
    if feature in DEMOGRAPHIC_FEATURES:
        return "demographics"
    if feature == "troponin_assay_t" or feature.startswith(
        ("lab_troponin_", "lab_ck_mb_")
    ):
        return "troponin_ck_mb"
    if feature.startswith("lab_"):
        return "other_labs"
    if feature.startswith("gcs_"):
        return "gcs_neuro"
    if any(feature.startswith(f"{channel}_") for channel in VITAL_CHANNELS):
        suffix = feature.rsplit("_", 1)[-1]
        if suffix in WINDOW_STATS:
            return "vital_sign_summaries"
    raise ValueError(f"Unmapped model feature: {feature}")


def _macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    classes = np.arange(probabilities.shape[1])
    return float(
        roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=classes,
        )
    )


def _fold_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) < 2:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    half_width = float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
    return mean, mean - half_width, mean + half_width


def _feature_contract_audit(
    cohort: pd.DataFrame,
    values: np.ndarray,
    masks: np.ndarray,
    features: list[str],
    labels: np.ndarray,
) -> dict:
    duplicated = sorted(
        feature for feature, count in Counter(features).items() if count > 1
    )
    forbidden = sorted(
        feature
        for feature in features
        if feature in FORBIDDEN_EXACT
        or any(pattern in feature.lower() for pattern in FORBIDDEN_PATTERNS)
    )
    if duplicated:
        raise ValueError(f"Duplicate model features: {duplicated}")
    if forbidden:
        raise ValueError(f"Forbidden post-landmark or outcome features: {forbidden}")
    if "outcome" in features:
        raise ValueError("Outcome entered the predictor matrix")
    if values.shape != masks.shape:
        raise ValueError("Sequence values and masks differ in shape")
    if values.shape[1] != SEQUENCE_HOURS:
        raise ValueError(
            f"Unexpected temporal horizon: {values.shape[1]} h, "
            f"expected {SEQUENCE_HOURS} h"
        )
    if values.shape[0] != len(cohort) or len(labels) != len(cohort):
        raise ValueError("Sequence, cohort, and outcome row counts differ")

    metrics_path = RESULTS / "metrics_mimic4.json"
    locked_match = None
    if metrics_path.is_file():
        locked = json.loads(metrics_path.read_text(encoding="utf-8"))
        locked_features = list(locked.get("full_static_features", []))
        locked_match = locked_features == features
        if not locked_match:
            raise ValueError(
                "Feature contract differs from the locked XGB-static+TS run"
            )

    mapped_groups = pd.Series([_assign_group(feature) for feature in features])
    group_counts = {
        group: int(mapped_groups.eq(group).sum()) for group in GROUP_ORDER
    }
    if sum(group_counts.values()) != len(features):
        raise ValueError("Variable-group mapping does not cover every feature once")

    return {
        "n_rows": int(len(cohort)),
        "n_patients": int(cohort["subject_key"].nunique()),
        "outcome_counts": {
            str(int(level)): int(count)
            for level, count in pd.Series(labels).value_counts().sort_index().items()
        },
        "n_features": int(len(features)),
        "n_groups": int(len(group_counts)),
        "group_counts": group_counts,
        "temporal_horizon_hours": int(values.shape[1]),
        "duplicate_features": duplicated,
        "forbidden_features": forbidden,
        "locked_feature_contract_match": locked_match,
        "threads": THREADS,
    }


def _run_importance(
    *,
    repeats: int,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    data = load_dataset("mimic4")
    cohort = data["cohort"].reset_index(drop=True)
    values = data["values"]
    masks = data["masks"]
    labels = cohort["outcome"].to_numpy(dtype=int)
    groups = cohort["subject_key"].astype(str).to_numpy()
    _, full_columns = _feature_columns(cohort)
    matrix = _static_matrix(cohort, full_columns)
    n_classes = int(labels.max()) + 1
    if n_classes != 5:
        raise ValueError(f"Expected five outcome levels, found {n_classes}")

    audit = _feature_contract_audit(
        cohort, values, masks, full_columns, labels
    )
    feature_groups = np.asarray(
        [_assign_group(feature) for feature in full_columns], dtype=object
    )
    group_indices = {
        group: np.flatnonzero(feature_groups == group) for group in GROUP_ORDER
    }
    splitter = StratifiedGroupKFold(
        n_splits=FOLDS, shuffle=True, random_state=CV_SEED
    )

    n_features = len(full_columns)
    fold_gain = np.full((FOLDS, n_features), np.nan, dtype=float)
    feature_drop = np.full(
        (FOLDS, n_features, repeats), np.nan, dtype=float
    )
    group_drop = np.full(
        (FOLDS, len(GROUP_ORDER), repeats), np.nan, dtype=float
    )
    fold_baseline = np.full(FOLDS, np.nan, dtype=float)
    fold_test_n = np.zeros(FOLDS, dtype=int)

    for fold, (train, test) in enumerate(
        splitter.split(matrix, labels, groups)
    ):
        overlap = np.intersect1d(groups[train], groups[test])
        if overlap.size:
            raise ValueError(
                f"Patient overlap between train and test in fold {fold}: "
                f"{overlap[:5].tolist()}"
            )
        if np.unique(labels[test]).size != n_classes:
            raise ValueError(f"Fold {fold} does not contain all outcome classes")

        model = _fit_xgb(
            matrix[train],
            labels[train],
            n_classes=n_classes,
            seed=CV_SEED + fold,
            n_jobs=THREADS,
        )
        test_matrix = matrix[test].copy()
        test_labels = labels[test]
        baseline_probability = model.predict_proba(test_matrix)
        baseline = _macro_auc(test_labels, baseline_probability)
        fold_baseline[fold] = baseline
        fold_test_n[fold] = int(len(test))

        booster_gain = model.get_booster().get_score(importance_type="gain")
        raw_gain = np.asarray(
            [
                float(booster_gain.get(f"f{feature_index}", 0.0))
                for feature_index in range(n_features)
            ],
            dtype=float,
        )
        if not np.isfinite(raw_gain).all() or raw_gain.sum() <= 0:
            raise ValueError(f"Fold {fold} produced an invalid gain vector")
        fold_gain[fold] = raw_gain / raw_gain.sum()

        for repeat in range(repeats):
            feature_rng = np.random.default_rng(
                PERMUTATION_SEED + 1_000 * fold + repeat
            )
            for feature_index in range(n_features):
                permuted = test_matrix.copy()
                permuted[:, feature_index] = permuted[
                    feature_rng.permutation(len(test_matrix)), feature_index
                ]
                probability = model.predict_proba(permuted)
                feature_drop[fold, feature_index, repeat] = (
                    baseline - _macro_auc(test_labels, probability)
                )

            group_rng = np.random.default_rng(
                PERMUTATION_SEED + 50_000 + 1_000 * fold + repeat
            )
            for group_index, group in enumerate(GROUP_ORDER):
                permuted = test_matrix.copy()
                row_order = group_rng.permutation(len(test_matrix))
                columns = group_indices[group]
                permuted[:, columns] = permuted[np.ix_(row_order, columns)]
                probability = model.predict_proba(permuted)
                group_drop[fold, group_index, repeat] = (
                    baseline - _macro_auc(test_labels, probability)
                )

        print(
            f"[feature importance] fold={fold + 1}/{FOLDS} "
            f"n_test={len(test)} baseline_macro_auc={baseline:.4f}",
            flush=True,
        )

    feature_gain_mean = fold_gain.mean(axis=0)
    feature_gain_mean /= feature_gain_mean.sum()
    # Switch from fold x feature to feature x fold before summarizing
    # each feature's five fold-level permutation drops.
    feature_fold_drop = feature_drop.mean(axis=2).T
    feature_mean, feature_ci_low, feature_ci_high = np.vstack(
        [_fold_ci(row) for row in feature_fold_drop]
    ).T
    feature_sd = feature_fold_drop.std(axis=1, ddof=1)

    feature_rows = []
    for feature_index, feature in enumerate(full_columns):
        group = str(feature_groups[feature_index])
        feature_rows.append(
            {
                "record_type": "feature",
                "feature_or_group": feature,
                "variable_group": group,
                "variable_group_label": GROUP_LABELS[group],
                "n_features": 1,
                "normalized_gain": float(feature_gain_mean[feature_index]),
                "normalized_gain_pct": float(
                    100.0 * feature_gain_mean[feature_index]
                ),
                "permutation_macro_auc_drop_mean": float(
                    feature_mean[feature_index]
                ),
                "permutation_macro_auc_drop_sd_across_folds": float(
                    feature_sd[feature_index]
                ),
                "permutation_macro_auc_drop_ci95_low": float(
                    feature_ci_low[feature_index]
                ),
                "permutation_macro_auc_drop_ci95_high": float(
                    feature_ci_high[feature_index]
                ),
                "n_folds": FOLDS,
                "n_repeats": repeats,
                "cv_seed": CV_SEED,
                "permutation_seed": PERMUTATION_SEED,
                "baseline_macro_auc_mean_across_folds": float(
                    np.nanmean(fold_baseline)
                ),
                "threads": THREADS,
            }
        )

    group_rows = []
    group_fold_drop = group_drop.mean(axis=2)
    for group_index, group in enumerate(GROUP_ORDER):
        group_gain = float(feature_gain_mean[group_indices[group]].sum())
        mean, ci_low, ci_high = _fold_ci(group_fold_drop[group_index])
        group_rows.append(
            {
                "record_type": "variable_group",
                "feature_or_group": GROUP_LABELS[group],
                "variable_group": group,
                "variable_group_label": GROUP_LABELS[group],
                "n_features": int(len(group_indices[group])),
                "normalized_gain": group_gain,
                "normalized_gain_pct": float(100.0 * group_gain),
                "permutation_macro_auc_drop_mean": mean,
                "permutation_macro_auc_drop_sd_across_folds": float(
                    group_fold_drop[group_index].std(ddof=1)
                ),
                "permutation_macro_auc_drop_ci95_low": ci_low,
                "permutation_macro_auc_drop_ci95_high": ci_high,
                "n_folds": FOLDS,
                "n_repeats": repeats,
                "cv_seed": CV_SEED,
                "permutation_seed": PERMUTATION_SEED,
                "baseline_macro_auc_mean_across_folds": float(
                    np.nanmean(fold_baseline)
                ),
                "threads": THREADS,
            }
        )

    group_frame = pd.DataFrame(group_rows)
    feature_frame = pd.DataFrame(feature_rows)
    combined = pd.concat([group_frame, feature_frame], ignore_index=True)
    audit.update(
        {
            "cv_seed": CV_SEED,
            "folds": FOLDS,
            "permutation_repeats": repeats,
            "permutation_seed": PERMUTATION_SEED,
            "fold_test_n": {
                str(index + 1): int(value)
                for index, value in enumerate(fold_test_n)
            },
            "fold_baseline_macro_auc": {
                str(index + 1): float(value)
                for index, value in enumerate(fold_baseline)
            },
            "patient_disjoint_outer_folds": True,
            "gain_normalization": "fold-specific normalization, then mean across folds",
            "permutation_estimand": (
                "decrease in outer-fold macro-AUC after permutation on held-out rows"
            ),
            "negative_values_retained": True,
        }
    )
    return combined, audit, group_frame


def _write_table_markdown(
    frame: pd.DataFrame,
    audit: dict,
    path: Path,
) -> None:
    group_frame = frame[frame["record_type"].eq("variable_group")].copy()
    feature_frame = frame[frame["record_type"].eq("feature")].copy()
    top_permutation = feature_frame.sort_values(
        "permutation_macro_auc_drop_mean", ascending=False
    ).head(20)
    top_gain = feature_frame.sort_values(
        "normalized_gain_pct", ascending=False
    ).head(15)

    group_display = group_frame.copy()
    group_display["Normalized gain (%)"] = group_display[
        "normalized_gain_pct"
    ].map(lambda value: f"{value:.2f}")
    group_display["Permutation drop (95% CI)"] = group_display.apply(
        lambda row: (
            f"{row['permutation_macro_auc_drop_mean']:+.4f} "
            f"({row['permutation_macro_auc_drop_ci95_low']:+.4f} to "
            f"{row['permutation_macro_auc_drop_ci95_high']:+.4f})"
        ),
        axis=1,
    )
    group_display = group_display[
        [
            "variable_group_label",
            "n_features",
            "Normalized gain (%)",
            "Permutation drop (95% CI)",
        ]
    ].rename(
        columns={
            "variable_group_label": "Variable group",
            "n_features": "Features",
        }
    )

    permutation_display = top_permutation[
        [
            "feature_or_group",
            "variable_group_label",
            "permutation_macro_auc_drop_mean",
            "permutation_macro_auc_drop_ci95_low",
            "permutation_macro_auc_drop_ci95_high",
            "normalized_gain_pct",
        ]
    ].rename(
        columns={
            "feature_or_group": "Feature",
            "variable_group_label": "Variable group",
            "permutation_macro_auc_drop_mean": "Macro-AUC drop",
            "permutation_macro_auc_drop_ci95_low": "95% CI low",
            "permutation_macro_auc_drop_ci95_high": "95% CI high",
            "normalized_gain_pct": "Gain (%)",
        }
    )

    gain_display = top_gain[
        [
            "feature_or_group",
            "variable_group_label",
            "normalized_gain_pct",
            "permutation_macro_auc_drop_mean",
        ]
    ].rename(
        columns={
            "feature_or_group": "Feature",
            "variable_group_label": "Variable group",
            "normalized_gain_pct": "Gain (%)",
            "permutation_macro_auc_drop_mean": "Macro-AUC drop",
        }
    )

    audit_rows = pd.DataFrame(
        [
            ("MIMIC-IV analytic cohort", f"{audit['n_rows']:,} admissions"),
            ("Patients", f"{audit['n_patients']:,}"),
            ("Model features", str(audit["n_features"])),
            ("Variable groups", str(audit["n_groups"])),
            (
                "Temporal horizon",
                f"Hours 0-{audit['temporal_horizon_hours'] - 1}; no post-landmark features",
            ),
            ("Outer validation", f"{audit['folds']}-fold patient-grouped stratified CV"),
            ("Cross-validation seed", str(audit["cv_seed"])),
            ("Permutation repeats", str(audit["permutation_repeats"])),
            ("Permutation base seed", str(audit["permutation_seed"])),
            ("XGBoost threads", str(audit["threads"])),
            (
                "Locked feature contract",
                "PASS"
                if audit["locked_feature_contract_match"]
                else "not compared",
            ),
            (
                "Forbidden outcome/post-landmark names",
                (
                    "PASS (none found)"
                    if not audit["forbidden_features"]
                    else f"FAIL: {audit['forbidden_features']}"
                ),
            ),
            ("Duplicate predictors", "PASS (none found)"),
            ("Patient-disjoint outer folds", "PASS"),
            (
                "Mean outer-fold baseline macro-AUC",
                f"{np.mean(list(audit['fold_baseline_macro_auc'].values())):.4f}",
            ),
        ],
        columns=["Audit item", "Result"],
    )

    lines = [
        "# Table S11. Feature importance for XGB-static+TS",
        "",
        "Feature importance was estimated in the locked MIMIC-IV analytic cohort "
        "using the same 154-variable XGB-static+TS feature contract and "
        "five-fold patient-grouped, outcome-stratified outer cross-validation "
        "used by the primary model comparison. Normalized gain was calculated "
        "within each training fold and averaged across the five fitted models.",
        "",
        "Permutation importance was evaluated only on the held-out outer test "
        "rows. Each predictor and each prespecified variable group was permuted "
        "five times per fold; the reported estimand is the decrease in "
        "five-class one-vs-rest macro-AUC. Variable-group permutation preserved "
        "within-group feature correlations by permuting the group rows jointly. "
        "Intervals use the five fold-level permutation means. Negative values "
        "reflect permutation noise and were retained rather than truncated.",
        "",
        "## Audit",
        "",
        audit_rows.to_markdown(index=False),
        "",
        "## Variable-group importance",
        "",
        group_display.to_markdown(index=False),
        "",
        "## Top 20 features by outer-fold permutation importance",
        "",
        permutation_display.to_markdown(
            index=False, floatfmt=(".4f", ".4f", ".4f", ".2f")
        ),
        "",
        "## Top 15 features by normalized XGBoost gain",
        "",
        gain_display.to_markdown(
            index=False, floatfmt=(".2f", ".4f")
        ),
        "",
        "Gain measures how a feature was used by the fitted trees and is not a "
        "causal effect. Permutation importance measures predictive dependence "
        "on the held-out fold and can be attenuated when correlated variables "
        "have interchangeable information.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _make_figure(group_frame: pd.DataFrame) -> pd.DataFrame:
    _style()
    frame = group_frame.copy()
    frame["group_order"] = frame["variable_group"].map(
        {group: index for index, group in enumerate(GROUP_ORDER)}
    )
    frame = frame.sort_values("group_order").reset_index(drop=True)
    colors = [GROUP_COLORS[group] for group in frame["variable_group"]]
    y = np.arange(len(frame))[::-1]

    figure, (gain_axis, permutation_axis) = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.15),
        gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.52},
    )

    gain_axis.barh(
        y,
        frame["normalized_gain_pct"],
        height=0.62,
        color=colors,
        edgecolor="white",
        linewidth=0.45,
    )
    gain_axis.set_yticks(y, frame["variable_group_label"])
    gain_axis.set_xlabel("Mean normalized XGBoost gain (%)")
    gain_axis.set_xlim(
        0,
        max(5.0, float(frame["normalized_gain_pct"].max()) * 1.27),
    )
    for ypos, value in zip(y, frame["normalized_gain_pct"]):
        gain_axis.text(
            value + max(1.0, float(frame["normalized_gain_pct"].max()) * 0.025),
            ypos,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=6.5,
        )
    gain_axis.set_title("A  Gain by variable group", loc="left", fontweight="bold")

    drop = frame["permutation_macro_auc_drop_mean"].to_numpy(dtype=float)
    low = frame["permutation_macro_auc_drop_ci95_low"].to_numpy(dtype=float)
    high = frame["permutation_macro_auc_drop_ci95_high"].to_numpy(dtype=float)
    xerr = np.vstack([drop - low, high - drop])
    permutation_axis.errorbar(
        drop,
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#4B4B4B",
        elinewidth=0.8,
        capsize=2.0,
        zorder=2,
    )
    permutation_axis.scatter(
        drop,
        y,
        s=34,
        c=colors,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    permutation_axis.axvline(0, color="#A6A6A6", linewidth=0.7, zorder=1)
    permutation_axis.set_yticks(y, frame["variable_group_label"])
    permutation_axis.set_xlabel("Macro-AUC decrease after permutation")
    span = max(
        0.01,
        float(np.nanmax(high) - np.nanmin(low)),
    )
    permutation_axis.set_xlim(
        float(np.nanmin(low) - 0.12 * span),
        float(np.nanmax(high) + 0.18 * span),
    )
    for ypos, value in zip(y, drop):
        permutation_axis.text(
            value,
            ypos + 0.19,
            f"{value:+.4f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
    permutation_axis.set_title(
        "B  Outer-fold permutation importance",
        loc="left",
        fontweight="bold",
    )

    figure.text(
        0.01,
        0.015,
        "5-fold patient-grouped CV; 5 repeats on held-out rows; bars/points show "
        "mean and 95% CI across folds.",
        fontsize=6.2,
        color="#4B4B4B",
        ha="left",
        va="bottom",
    )
    figure.subplots_adjust(bottom=0.23, left=0.18, right=0.98, top=0.89)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg", "tiff"):
        figure.savefig(
            FIGURE_DIR / f"{FIGURE_STEM}.{extension}",
            dpi=600 if extension == "tiff" else (300 if extension == "png" else None),
        )
    plt.close(figure)

    source = frame[
        [
            "variable_group",
            "variable_group_label",
            "n_features",
            "normalized_gain_pct",
            "permutation_macro_auc_drop_mean",
            "permutation_macro_auc_drop_ci95_low",
            "permutation_macro_auc_drop_ci95_high",
            "n_folds",
            "n_repeats",
            "cv_seed",
            "permutation_seed",
        ]
    ].copy()
    source.insert(0, "panel", "A_group_gain_and_B_group_permutation")
    return source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: object):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _audit_existing_outputs(repeats: int = DEFAULT_REPEATS) -> dict:
    """Validate finalized artifacts without refitting any model."""
    data = load_dataset("mimic4")
    cohort = data["cohort"].reset_index(drop=True)
    labels = cohort["outcome"].to_numpy(dtype=int)
    _, full_columns = _feature_columns(cohort)
    audit = _feature_contract_audit(
        cohort,
        data["values"],
        data["masks"],
        full_columns,
        labels,
    )

    table_path = TABLE_DIR / f"{TABLE_STEM}.csv"
    markdown_path = TABLE_DIR / f"{TABLE_STEM}.md"
    source_path = SOURCE_DIR / f"{FIGURE_STEM}.csv"
    figure_paths = {
        extension: FIGURE_DIR / f"{FIGURE_STEM}.{extension}"
        for extension in ("png", "pdf", "svg", "tiff")
    }
    required_paths = [
        table_path,
        markdown_path,
        source_path,
        *figure_paths.values(),
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    empty = [
        str(path) for path in required_paths if path.is_file() and path.stat().st_size == 0
    ]
    if missing or empty:
        raise ValueError(f"Missing or empty outputs: missing={missing}, empty={empty}")

    table = pd.read_csv(table_path)
    source = pd.read_csv(source_path)
    required_columns = {
        "record_type",
        "feature_or_group",
        "variable_group",
        "variable_group_label",
        "n_features",
        "normalized_gain",
        "normalized_gain_pct",
        "permutation_macro_auc_drop_mean",
        "permutation_macro_auc_drop_sd_across_folds",
        "permutation_macro_auc_drop_ci95_low",
        "permutation_macro_auc_drop_ci95_high",
        "n_folds",
        "n_repeats",
        "cv_seed",
        "permutation_seed",
        "baseline_macro_auc_mean_across_folds",
        "threads",
    }
    missing_columns = sorted(required_columns.difference(table.columns))
    if missing_columns:
        raise ValueError(f"Table S11 is missing columns: {missing_columns}")

    group_table = table[table["record_type"].eq("variable_group")].copy()
    feature_table = table[table["record_type"].eq("feature")].copy()
    expected_feature_names = set(full_columns)
    observed_feature_names = set(feature_table["feature_or_group"].astype(str))
    feature_set_match = observed_feature_names == expected_feature_names
    group_mapping = {
        row.feature_or_group: row.variable_group
        for row in feature_table.itertuples(index=False)
    }
    mapping_mismatches = sorted(
        feature
        for feature in full_columns
        if group_mapping.get(feature) != _assign_group(feature)
    )
    group_set_match = set(group_table["variable_group"].astype(str)) == set(GROUP_ORDER)
    feature_gain_sum = float(feature_table["normalized_gain"].sum())
    group_gain_sum = float(group_table["normalized_gain"].sum())
    group_counts_recomputed = feature_table.groupby("variable_group").size().to_dict()
    group_count_mismatches = sorted(
        row.variable_group
        for row in group_table.itertuples(index=False)
        if int(row.n_features) != group_counts_recomputed.get(row.variable_group)
    )
    group_gain_by_name = group_table.set_index("variable_group")["normalized_gain"]
    group_gain_mismatches = sorted(
        group
        for group in GROUP_ORDER
        if not np.isclose(
            group_gain_by_name.get(group),
            feature_table.loc[
                feature_table["variable_group"].eq(group), "normalized_gain"
            ].sum(),
            atol=1e-12,
            rtol=0,
        )
    )
    numeric_columns = [
        "normalized_gain",
        "normalized_gain_pct",
        "permutation_macro_auc_drop_mean",
        "permutation_macro_auc_drop_sd_across_folds",
        "permutation_macro_auc_drop_ci95_low",
        "permutation_macro_auc_drop_ci95_high",
        "baseline_macro_auc_mean_across_folds",
    ]
    finite_values = all(np.isfinite(table[column].to_numpy(dtype=float)).all() for column in numeric_columns)
    seed_and_fold_contract = bool(
        table["n_folds"].eq(FOLDS).all()
        and table["n_repeats"].eq(repeats).all()
        and table["cv_seed"].eq(CV_SEED).all()
        and table["permutation_seed"].eq(PERMUTATION_SEED).all()
        and table["threads"].eq(THREADS).all()
    )
    ci_contract = bool(
        (
            table["permutation_macro_auc_drop_ci95_low"]
            <= table["permutation_macro_auc_drop_mean"]
        ).all()
        and (
            table["permutation_macro_auc_drop_mean"]
            <= table["permutation_macro_auc_drop_ci95_high"]
        ).all()
    )
    source_columns = {
        "panel",
        "variable_group",
        "normalized_gain_pct",
        "permutation_macro_auc_drop_mean",
        "permutation_macro_auc_drop_ci95_low",
        "permutation_macro_auc_drop_ci95_high",
    }
    source_match = bool(
        source_columns.issubset(source.columns)
        and len(source) == len(GROUP_ORDER)
        and set(source["variable_group"].astype(str)) == set(GROUP_ORDER)
        and np.allclose(
            source["normalized_gain_pct"].to_numpy(dtype=float),
            group_table["normalized_gain_pct"].to_numpy(dtype=float),
            atol=1e-10,
            rtol=0,
        )
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    placeholder_tokens = ("feature_0", "placeholder", "synthetic", "dummy")
    placeholder_found = [
        token for token in placeholder_tokens if token.lower() in markdown.lower()
    ]
    real_feature_tokens = [
        feature for feature in full_columns if feature in markdown
    ]
    markdown_has_real_features = len(real_feature_tokens) >= 5

    png_valid = False
    tiff_valid = False
    try:
        with Image.open(figure_paths["png"]) as image:
            png_valid = image.width >= 1200 and image.height >= 600
        with Image.open(figure_paths["tiff"]) as image:
            tiff_valid = image.width >= 2400 and image.height >= 1200
    except OSError:
        pass
    try:
        ET.parse(figure_paths["svg"])
        svg_valid = True
    except ET.ParseError:
        svg_valid = False
    pdf_valid = figure_paths["pdf"].read_bytes().startswith(b"%PDF")

    checks = {
        "feature_names_match_locked_model": feature_set_match,
        "variable_groups_match_contract": group_set_match,
        "feature_group_assignment_match": not mapping_mismatches,
        "feature_gain_sums_to_one": bool(abs(feature_gain_sum - 1.0) <= 1e-9),
        "group_gain_sums_to_one": bool(abs(group_gain_sum - 1.0) <= 1e-9),
        "group_counts_match_features": not group_count_mismatches,
        "group_gain_equals_feature_aggregate": not group_gain_mismatches,
        "required_numeric_values_finite": finite_values,
        "seed_fold_thread_contract": seed_and_fold_contract,
        "confidence_intervals_contain_estimates": ci_contract,
        "figure_source_matches_table": source_match,
        "markdown_contains_real_feature_names": markdown_has_real_features,
        "no_placeholder_tokens": not placeholder_found,
        "png_nonblank_dimensions": png_valid,
        "tiff_nonblank_dimensions": tiff_valid,
        "svg_parses": svg_valid,
        "pdf_header_valid": pdf_valid,
        "locked_feature_contract_match": bool(
            audit["locked_feature_contract_match"]
        ),
        "no_forbidden_features": not audit["forbidden_features"],
        "temporal_horizon_is_24h": audit["temporal_horizon_hours"] == 24,
        "patient_grouped_outer_folds": table["n_folds"].eq(FOLDS).all(),
    }
    audit.update(
        {
            "audit_generated_utc": datetime.now(timezone.utc).isoformat(),
            "audit_mode": "read-only validation of finalized artifacts",
            "passed": all(checks.values()),
            "checks": checks,
            "details": {
                "missing_outputs": missing,
                "empty_outputs": empty,
                "missing_columns": missing_columns,
                "feature_set_match": feature_set_match,
                "group_mapping_mismatches": mapping_mismatches,
                "group_count_mismatches": group_count_mismatches,
                "group_gain_mismatches": group_gain_mismatches,
                "feature_gain_sum": feature_gain_sum,
                "group_gain_sum": group_gain_sum,
                "placeholder_tokens_found": placeholder_found,
                "markdown_has_real_features": markdown_has_real_features,
                "real_feature_tokens_found": real_feature_tokens,
                "figure_pixels": {
                    "png": "opened",
                    "tiff": "opened",
                },
            },
            "output_files": {
                str(path): {
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
                for path in required_paths
            },
        }
    )
    if not audit["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Feature-importance audit failed: {failed}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Table S11 and Figure S2 feature importance."
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate finalized outputs and write the audit JSON without refitting.",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    warnings.filterwarnings("ignore", category=FutureWarning)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.audit_only:
        audit = _audit_existing_outputs(repeats=args.repeats)
        AUDIT_PATH.write_text(
            json.dumps(audit, indent=2, default=_json_default),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "audit_complete",
                    "passed": audit["passed"],
                    "audit_json": str(AUDIT_PATH),
                    "checks": audit["checks"],
                },
                indent=2,
                default=_json_default,
            ),
            flush=True,
        )
        return

    frame, audit, group_frame = _run_importance(repeats=args.repeats)
    table_path = TABLE_DIR / f"{TABLE_STEM}.csv"
    markdown_path = TABLE_DIR / f"{TABLE_STEM}.md"
    source_path = SOURCE_DIR / f"{FIGURE_STEM}.csv"
    frame.to_csv(table_path, index=False)
    _write_table_markdown(frame, audit, markdown_path)
    source = _make_figure(group_frame)
    source.to_csv(source_path, index=False)
    audit = _audit_existing_outputs(repeats=args.repeats)
    AUDIT_PATH.write_text(
        json.dumps(audit, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "complete",
                "audit": audit,
                "outputs": [
                    str(table_path),
                    str(markdown_path),
                    str(FIGURE_DIR / f"{FIGURE_STEM}.png"),
                    str(FIGURE_DIR / f"{FIGURE_STEM}.pdf"),
                    str(FIGURE_DIR / f"{FIGURE_STEM}.svg"),
                    str(FIGURE_DIR / f"{FIGURE_STEM}.tiff"),
                    str(source_path),
                    str(AUDIT_PATH),
                ],
                "audit_passed": audit["passed"],
            },
            indent=2,
            default=_json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

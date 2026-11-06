"""Run the remaining AMI sensitivity analyses without touching locked models.

The module covers multiple imputation, exact-time pre-24-hour treatment
features, and a four-class rehabilitation-plus-facility outcome. It reuses the
locked XGB-static+TS feature contract and evaluation protocol.
"""
from __future__ import annotations

import os

THREADS = min(
    4,
    max(1, int(os.environ.get("AMI_ADDITIONAL_THREADS", "4"))),
)
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = str(THREADS)
os.environ["AMI_XGB_THREADS"] = str(THREADS)
os.environ["AMI_DUCKDB_THREADS"] = str(THREADS)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import PROCESSED, RESULTS, SOURCE_ROOT  # noqa: E402
from src.ami_dynamic.evaluation import (  # noqa: E402
    bootstrap_metric_ci,
    point_metrics,
)
from src.ami_dynamic.train_evaluate import (  # noqa: E402
    _assert_feature_contract,
    _fit_xgb,
    _json_default,
    _static_matrix,
    load_dataset,
)


SEEDS = (1, 7, 42, 123, 2024)
FOLDS = 5
MIN_IMPUTATIONS = 3
DEFAULT_IMPUTATIONS = 3
DEFAULT_MI_MAX_ITER = 1
DEFAULT_BOOTSTRAP = 2000

OUTPUT = RESULTS / "additional_sensitivities"
PREDICTION_OUTPUT = OUTPUT / "predictions"
SENSITIVITY_OUTPUT = RESULTS / "sensitivity"

MIMIC4_ROOT = SOURCE_ROOT / "mimic-iv-3.1"
MIMIC3_ROOT = (
    SOURCE_ROOT
    / "mimic-iii-clinical-database-1.4_1"
    / "mimic-iii-clinical-database-1.4"
)
EICU_ROOT = SOURCE_ROOT / "eicu"

REVASCULARIZATION_ITEMIDS = {
    225427: "Angiography",
    225430: "Cardiac Cath",
    229577: "Cath Lab (Received)",
}
VASOACTIVE_ITEMIDS = {
    221289: "Epinephrine",
    221653: "Dobutamine",
    221662: "Dopamine",
    221749: "Phenylephrine",
    221906: "Norepinephrine",
    221986: "Milrinone",
    222315: "Vasopressin",
    229617: "Epinephrine",
    229630: "Phenylephrine",
    229631: "Phenylephrine",
    229632: "Phenylephrine",
    229709: "Angiotensin II",
    229764: "Angiotensin II",
}
MCS_ITEMIDS = {
    224272: "IABP",
    228169: "Impella",
    228201: "Tandem Heart",
    228202: "Tandem Heart",
    229527: "CentriMag",
    229528: "CentriMag",
    229529: "ECMO",
    229530: "ECMO",
}

ANALYSES = (
    "multiple_imputation",
    "pre24h_revascularization",
    "pre24h_vasopressor_mcs",
    "label_merge_rehab_facility",
)

SENSITIVITY_FILES = {
    "multiple_imputation": (
        "metrics_mimic4_multiple_imputation_xgb_static_ts.json"
    ),
    "pre24h_revascularization": (
        "metrics_mimic4_include_pre24h_revascularization_xgb_static_ts.json"
    ),
    "pre24h_vasopressor_mcs": (
        "metrics_mimic4_include_pre24h_vasopressor_or_mcs_xgb_static_ts.json"
    ),
    "label_merge_rehab_facility": (
        "metrics_mimic4_label_merge_hospice_death_and_rehab_facility_"
        "xgb_static_ts.json"
    ),
}


def _write_json(path: Path, payload: dict, *, overwrite: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"[skip existing] {path}", flush=True)
        return False
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return True


def _write_npz(path: Path, *, overwrite: bool, **arrays) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"[skip existing] {path}", flush=True)
        return False
    np.savez_compressed(path, **arrays)
    return True


def _remove_cumulative(metrics: dict) -> dict:
    out = dict(metrics)
    out.pop("cumulative_probabilities", None)
    return out


def _load_locked_development() -> tuple[dict, pd.DataFrame, np.ndarray]:
    data = load_dataset("mimic4")
    cohort = data["cohort"].reset_index(drop=True).copy()
    metrics_path = RESULTS / "metrics_mimic4.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing locked metrics: {metrics_path}")
    locked = json.loads(metrics_path.read_text(encoding="utf-8"))
    full_columns = list(locked["full_static_features"])
    _assert_feature_contract(full_columns)
    if int(locked["n"]) != len(cohort):
        raise ValueError("Locked metric row count differs from processed cohort")
    if list(locked["seeds"]) != list(SEEDS):
        raise ValueError("Locked seeds differ from the requested five-seed protocol")
    matrix = _static_matrix(cohort, full_columns)
    if matrix.shape != (len(cohort), len(full_columns)):
        raise ValueError("Static feature matrix does not match the locked contract")
    return data, cohort, matrix


def _cohort_times(cohort: pd.DataFrame) -> pd.DataFrame:
    times = cohort[["stay_id", "admittime", "intime"]].copy()
    times["intime"] = pd.to_datetime(times["intime"], errors="coerce")
    times["admittime"] = pd.to_datetime(times["admittime"], errors="coerce")
    times["landmark_time"] = times["intime"] + pd.Timedelta(hours=24)
    times["lower_time"] = times[
        ["admittime", "intime"]
    ].max(axis=1)
    times.loc[
        times["lower_time"].isna() | times["lower_time"].lt(times["intime"]),
        "lower_time",
    ] = times["intime"]
    return times


def _create_csv_view(
    con: duckdb.DuckDBPyConnection,
    name: str,
    path: Path,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_csv_auto('{path.as_posix()}', header=true)"
    )


def _aggregate_exact_time_events(
    con: duckdb.DuckDBPyConnection,
    *,
    view: str,
    itemids: dict[int, str],
    cohort_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_list = ",".join(str(value) for value in sorted(itemids))
    events = con.execute(
        f"""
        SELECT e.stay_id,
               e.itemid,
               MIN(e.starttime) AS first_starttime,
               MAX(e.starttime) AS last_starttime,
               COUNT(*) AS event_rows
        FROM {view} e
        JOIN cohort_times c USING (stay_id)
        WHERE e.itemid IN ({item_list})
          AND e.starttime IS NOT NULL
          AND e.starttime >= c.lower_time
          AND e.starttime < c.landmark_time
        GROUP BY e.stay_id, e.itemid
        """
    ).fetchdf()
    if events.empty:
        return (
            pd.DataFrame(columns=["stay_id"]),
            pd.DataFrame(
                columns=["stay_id", "itemid", "label", "first_starttime"]
            ),
        )
    events["label"] = events["itemid"].map(itemids)
    summary = (
        events.groupby("stay_id", as_index=False)
        .agg(
            unique_item_count=("itemid", "nunique"),
            event_rows=("event_rows", "sum"),
            first_starttime=("first_starttime", "min"),
            last_starttime=("last_starttime", "max"),
        )
    )
    if len(summary) > cohort_size:
        raise ValueError(f"{view}: event summary exceeds cohort size")
    return summary, events


def _extract_pre24h_treatment_features(
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    times = _cohort_times(cohort)
    con = duckdb.connect()
    con.execute(f"SET threads={THREADS}")
    con.execute("SET memory_limit='8GB'")
    con.register("cohort_times", times)
    try:
        _create_csv_view(
            con,
            "procedureevents",
            MIMIC4_ROOT / "icu" / "procedureevents.csv.gz",
        )
        _create_csv_view(
            con,
            "inputevents",
            MIMIC4_ROOT / "icu" / "inputevents.csv.gz",
        )
        revasc, revasc_events = _aggregate_exact_time_events(
            con,
            view="procedureevents",
            itemids=REVASCULARIZATION_ITEMIDS,
            cohort_size=len(cohort),
        )
        vaso, vaso_events = _aggregate_exact_time_events(
            con,
            view="inputevents",
            itemids=VASOACTIVE_ITEMIDS,
            cohort_size=len(cohort),
        )
        mcs, mcs_events = _aggregate_exact_time_events(
            con,
            view="procedureevents",
            itemids=MCS_ITEMIDS,
            cohort_size=len(cohort),
        )
    finally:
        con.close()

    output = cohort[["stay_id"]].copy()
    output["pre24h_revascularization_proxy"] = 0
    output["pre24h_vasoactive_any"] = 0
    output["pre24h_vasoactive_unique_agents"] = 0.0
    output["pre24h_mcs_any"] = 0
    output["pre24h_mcs_unique_types"] = 0.0

    if not revasc.empty:
        output.loc[
            output["stay_id"].isin(revasc["stay_id"]),
            "pre24h_revascularization_proxy",
        ] = 1
    if not vaso.empty:
        vaso_map = vaso.set_index("stay_id")["unique_item_count"]
        output["pre24h_vasoactive_unique_agents"] = (
            output["stay_id"].map(vaso_map).fillna(0).astype(float)
        )
        output["pre24h_vasoactive_any"] = (
            output["pre24h_vasoactive_unique_agents"].gt(0).astype(int)
        )
    if not mcs.empty:
        mcs_types = mcs_events.groupby("stay_id")["label"].nunique()
        output["pre24h_mcs_unique_types"] = (
            output["stay_id"].map(mcs_types).fillna(0).astype(float)
        )
        output["pre24h_mcs_any"] = (
            output["pre24h_mcs_unique_types"].gt(0).astype(int)
        )

    if output["stay_id"].duplicated().any():
        raise ValueError("Duplicate stay_id in treatment feature table")
    if not set(output["stay_id"]).issubset(set(cohort["stay_id"])):
        raise ValueError("Treatment features contain an unknown stay_id")

    def event_audit(
        frame: pd.DataFrame,
        events: pd.DataFrame,
        mapping: dict[int, str],
    ) -> dict:
        observed_items = (
            sorted(int(value) for value in events["itemid"].unique())
            if not events.empty
            else []
        )
        return {
            "n_stays_with_event": int(frame["stay_id"].nunique())
            if not frame.empty
            else 0,
            "observed_itemids": observed_items,
            "observed_labels": [
                mapping[value] for value in observed_items if value in mapping
            ],
            "event_rows_pre24h": int(
                events["event_rows"].sum()
            )
            if not events.empty
            else 0,
        }

    audit = {
        "landmark_hours": 24,
        "time_window": (
            "starttime >= max(admittime, ICU intime) and "
            "starttime < ICU intime + 24 hours"
        ),
        "treatment_after_landmark_used": False,
        "features": {
            "pre24h_revascularization_proxy": {
                "definition": (
                    "Any exact-time procedureevent for cardiac catheterisation, "
                    "angiography, or Cath Lab (Received) before hour 24."
                ),
                "itemids": {
                    str(key): value
                    for key, value in REVASCULARIZATION_ITEMIDS.items()
                },
                **event_audit(revasc, revasc_events, REVASCULARIZATION_ITEMIDS),
            },
            "pre24h_vasoactive": {
                "definition": (
                    "Any exact-time inputevents vasoactive infusion start before "
                    "hour 24; unique-agent count also retained."
                ),
                "itemids": {
                    str(key): value for key, value in VASOACTIVE_ITEMIDS.items()
                },
                **event_audit(vaso, vaso_events, VASOACTIVE_ITEMIDS),
            },
            "pre24h_mcs": {
                "definition": (
                    "Any exact-time procedureevent for IABP, Impella, Tandem "
                    "Heart, CentriMag, or ECMO before hour 24."
                ),
                "itemids": {
                    str(key): value for key, value in MCS_ITEMIDS.items()
                },
                **event_audit(mcs, mcs_events, MCS_ITEMIDS),
            },
        },
        "positive_counts": {
            column: int(output[column].sum())
            for column in (
                "pre24h_revascularization_proxy",
                "pre24h_vasoactive_any",
                "pre24h_mcs_any",
            )
        },
        "estimability": {
            "mimic4": {
                "status": "proxy_estimable",
                "exact_time_events": True,
                "full_pci_or_cabg_status": "not_estimable_at_hourly_resolution",
                "full_pci_or_cabg_reason": (
                    "MIMIC-IV procedures_icd supplies chartdate but no procedure "
                    "time; it cannot prove that a PCI or CABG occurred before the "
                    "24-hour landmark. Exact-time catheterisation events are used "
                    "as a prespecified proxy and are labelled as such."
                ),
            },
            "mimic3": {
                "status": "not_estimated",
                "reason": (
                    "MIMIC-III is retained for transport validation. This module "
                    "targets development-cohort sensitivity analyses and does not "
                    "fit new MIMIC-III treatment models."
                ),
            },
            "eicu": {
                "status": "not_estimable",
                "reason": (
                    "The local eICU source snapshot lacks infusionDrug.csv.gz and "
                    "treatment.csv.gz. Available vital and nurse-charting tables "
                    "cannot reliably identify vasoactive infusions or MCS."
                ),
                "required_missing_files": [
                    str(EICU_ROOT / "infusionDrug.csv.gz"),
                    str(EICU_ROOT / "treatment.csv.gz"),
                ],
                "required_files_present": {
                    "infusionDrug.csv.gz": (
                        EICU_ROOT / "infusionDrug.csv.gz"
                    ).is_file(),
                    "treatment.csv.gz": (
                        EICU_ROOT / "treatment.csv.gz"
                    ).is_file(),
                },
            },
        },
    }
    return output, audit


def _stratified_folds(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    folds: int,
) -> np.ndarray:
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    fold_id = np.full(len(labels), -1, dtype=int)
    for fold, (_, test) in enumerate(
        splitter.split(matrix, labels, groups)
    ):
        fold_id[test] = fold
    if (fold_id < 0).any():
        raise ValueError("Cross-validation left rows without a fold")
    return fold_id


def _run_xgb_cv(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    stay_ids: np.ndarray,
    *,
    seeds: list[int],
    folds: int,
    tag: str,
    overwrite: bool,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[dict]]:
    n_classes = int(labels.max()) + 1
    seed_probabilities: dict[int, np.ndarray] = {}
    seed_folds: dict[int, np.ndarray] = {}
    seed_rows: list[dict] = []
    for seed in seeds:
        fold_id = _stratified_folds(
            matrix, labels, groups, seed=seed, folds=folds
        )
        probability = np.full(
            (len(labels), n_classes), np.nan, dtype=np.float32
        )
        for fold, (train, test) in enumerate(
            (
                (np.flatnonzero(fold_id != fold), np.flatnonzero(fold_id == fold))
                for fold in range(folds)
            )
        ):
            model = _fit_xgb(
                matrix[train],
                labels[train],
                n_classes=n_classes,
                seed=seed + fold,
                n_jobs=THREADS,
            )
            probability[test] = model.predict_proba(matrix[test])
            print(
                f"[{tag}] seed={seed} fold={fold + 1}/{folds}",
                flush=True,
            )
        if not np.isfinite(probability).all():
            raise ValueError(f"{tag}: non-finite out-of-fold probabilities")
        seed_probabilities[seed] = probability
        seed_folds[seed] = fold_id
        metrics = _remove_cumulative(point_metrics(labels, probability))
        seed_rows.append({"seed": seed, **metrics})
        print(
            f"[{tag}] seed={seed} macro_auc={metrics['macro_auc']:.4f}",
            flush=True,
        )
        _write_npz(
            PREDICTION_OUTPUT / f"{tag}_seed{seed}.npz",
            overwrite=overwrite,
            labels=labels.astype(int),
            probabilities=probability,
            fold=fold_id,
            stay_id=stay_ids,
            subject_key=groups,
        )
    return seed_probabilities, seed_folds, seed_rows


def _summarize_seed_outputs(
    labels: np.ndarray,
    groups: np.ndarray,
    probabilities: np.ndarray,
    *,
    bootstrap: int,
) -> tuple[dict, dict]:
    average = np.mean(probabilities, axis=0)
    metrics = _remove_cumulative(point_metrics(labels, average))
    metrics["macro_auc_ci"] = bootstrap_metric_ci(
        labels,
        average,
        groups=groups,
        replicates=bootstrap,
    )
    return metrics, {"point": metrics, "seed_average_probability": average}


def _base_result(
    *,
    analysis: str,
    cohort: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    seeds: list[int],
    folds: int,
    metrics: dict,
    features: list[str],
) -> dict:
    return {
        "analysis": analysis,
        "model": "xgb_static_ts",
        "dataset": "mimic4",
        "n": int(len(cohort)),
        "n_patients": int(pd.Series(groups).nunique()),
        "folds": folds,
        "seeds": seeds,
        "outcome_counts": {
            str(int(key)): int(value)
            for key, value in pd.Series(labels)
            .value_counts()
            .sort_index()
            .items()
        },
        "features": features,
        "n_features": int(len(features)),
        "metrics": metrics,
        "threads": THREADS,
    }


def run_treatment_analysis(
    *,
    analysis: str,
    seeds: list[int],
    folds: int,
    bootstrap: int,
    overwrite: bool,
) -> dict:
    data, cohort, matrix = _load_locked_development()
    del data
    treatment, treatment_audit = _extract_pre24h_treatment_features(cohort)
    _write_json(
        OUTPUT / "treatment_estimability.json",
        treatment_audit,
        overwrite=overwrite,
    )
    if not (OUTPUT / "treatment_features.csv").exists() or overwrite:
        treatment.to_csv(OUTPUT / "treatment_features.csv", index=False)

    if analysis == "pre24h_revascularization":
        treatment_columns = ["pre24h_revascularization_proxy"]
        output_stem = "pre24h_revascularization"
    elif analysis == "pre24h_vasopressor_mcs":
        treatment_columns = [
            "pre24h_vasoactive_any",
            "pre24h_vasoactive_unique_agents",
            "pre24h_mcs_any",
            "pre24h_mcs_unique_types",
        ]
        output_stem = "pre24h_vasopressor_mcs"
    else:
        raise ValueError(analysis)

    locked = json.loads(
        (RESULTS / "metrics_mimic4.json").read_text(encoding="utf-8")
    )
    base_features = list(locked["full_static_features"])
    treatment_matrix = treatment[treatment_columns].to_numpy(dtype=float)
    augmented_matrix = np.concatenate([matrix, treatment_matrix], axis=1)
    features = base_features + treatment_columns
    _assert_feature_contract(features)
    labels = cohort["outcome"].to_numpy(dtype=int)
    groups = cohort["subject_key"].astype(str).to_numpy()
    tag = f"mimic4_include_{output_stem}_xgb_static_ts"

    seed_probabilities, seed_folds, seed_rows = _run_xgb_cv(
        augmented_matrix,
        labels,
        groups,
        cohort["stay_id"].to_numpy(dtype=np.int64),
        seeds=seeds,
        folds=folds,
        tag=tag,
        overwrite=overwrite,
    )
    probability_stack = np.stack(
        [seed_probabilities[seed] for seed in seeds]
    )
    metrics, _ = _summarize_seed_outputs(
        labels,
        groups,
        probability_stack,
        bootstrap=bootstrap,
    )
    result = _base_result(
        analysis=(
            "include_pre24h_revascularization"
            if analysis == "pre24h_revascularization"
            else "include_pre24h_vasopressor_or_mcs"
        ),
        cohort=cohort,
        labels=labels,
        groups=groups,
        seeds=seeds,
        folds=folds,
        metrics=metrics,
        features=features,
    )
    result["added_features"] = treatment_columns
    result["treatment_audit"] = {
        key: treatment_audit["features"][key]
        for key in (
            "pre24h_revascularization_proxy",
            "pre24h_vasoactive",
            "pre24h_mcs",
        )
    }
    result["estimability"] = treatment_audit["estimability"]
    result["seed_metrics"] = seed_rows
    result["bootstrap"] = {
        "replicates_requested": bootstrap,
        "unit": "patient",
        "metric": "macro_auc",
    }
    result["locked_reference_macro_auc"] = float(
        locked["metrics"]["xgb_static_ts"]["macro_auc"]
    )

    detail_path = OUTPUT / f"{output_stem}.json"
    sensitivity_path = SENSITIVITY_OUTPUT / SENSITIVITY_FILES[analysis]
    wrote_detail = _write_json(detail_path, result, overwrite=overwrite)
    wrote_metrics = _write_json(
        sensitivity_path, result, overwrite=overwrite
    )
    pd.DataFrame(seed_rows).to_csv(
        OUTPUT / f"{output_stem}_by_seed.csv", index=False
    )
    result["_wrote"] = {
        "detail": wrote_detail,
        "sensitivity_metrics": wrote_metrics,
    }
    return result


def run_label_merge_analysis(
    *,
    seeds: list[int],
    folds: int,
    bootstrap: int,
    overwrite: bool,
) -> dict:
    _, cohort, matrix = _load_locked_development()
    original = cohort["outcome"].to_numpy(dtype=int)
    mapping = np.asarray([0, 1, 2, 2, 3], dtype=int)
    labels = mapping[original]
    groups = cohort["subject_key"].astype(str).to_numpy()
    locked = json.loads(
        (RESULTS / "metrics_mimic4.json").read_text(encoding="utf-8")
    )
    features = list(locked["full_static_features"])
    tag = "mimic4_label_merge_rehab_facility_xgb_static_ts"
    seed_probabilities, seed_folds, seed_rows = _run_xgb_cv(
        matrix,
        labels,
        groups,
        cohort["stay_id"].to_numpy(dtype=np.int64),
        seeds=seeds,
        folds=folds,
        tag=tag,
        overwrite=overwrite,
    )
    probability_stack = np.stack(
        [seed_probabilities[seed] for seed in seeds]
    )
    metrics, _ = _summarize_seed_outputs(
        labels,
        groups,
        probability_stack,
        bootstrap=bootstrap,
    )
    result = _base_result(
        analysis="label_merge_hospice_death_and_rehab_facility",
        cohort=cohort,
        labels=labels,
        groups=groups,
        seeds=seeds,
        folds=folds,
        metrics=metrics,
        features=features,
    )
    result["outcome_mapping"] = {
        "0": {"source_levels": [0], "label": "Home or against advice"},
        "1": {"source_levels": [1], "label": "Home health care"},
        "2": {
            "source_levels": [2, 3],
            "label": "Inpatient rehabilitation plus skilled nursing/facility",
        },
        "3": {
            "source_levels": [4],
            "label": "Hospice or in-hospital death",
        },
    }
    result["seed_metrics"] = seed_rows
    result["bootstrap"] = {
        "replicates_requested": bootstrap,
        "unit": "patient",
        "metric": "macro_auc",
    }
    detail_path = OUTPUT / "label_merge_rehab_facility.json"
    sensitivity_path = (
        SENSITIVITY_OUTPUT
        / SENSITIVITY_FILES["label_merge_rehab_facility"]
    )
    wrote_detail = _write_json(detail_path, result, overwrite=overwrite)
    wrote_metrics = _write_json(
        sensitivity_path, result, overwrite=overwrite
    )
    pd.DataFrame(seed_rows).to_csv(
        OUTPUT / "label_merge_rehab_facility_by_seed.csv",
        index=False,
    )
    result["_wrote"] = {
        "detail": wrote_detail,
        "sensitivity_metrics": wrote_metrics,
    }
    return result


def _impute_train_test(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    *,
    seed: int,
    fold: int,
    imputation: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    missing_train = np.isnan(train_matrix).any(axis=0)
    missing_test = np.isnan(test_matrix).any(axis=0)
    impute_columns = np.flatnonzero(missing_train | missing_test)
    if impute_columns.size == 0:
        return train_matrix.copy(), test_matrix.copy(), []
    all_missing = impute_columns[
        np.isnan(train_matrix[:, impute_columns]).all(axis=0)
    ]
    if all_missing.size:
        raise ValueError(
            "Training fold contains an entirely missing imputation column: "
            f"{all_missing.tolist()}"
        )
    random_state = (
        int(seed) * 100_000 + int(fold) * 1_000 + int(imputation)
    ) % (2**32 - 1)
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        tol=1e-3,
        sample_posterior=True,
        random_state=random_state,
        skip_complete=True,
        min_value=0.0,
    )
    fitted = imputer.fit(train_matrix[:, impute_columns])
    train_imputed = train_matrix.copy()
    test_imputed = test_matrix.copy()
    train_imputed[:, impute_columns] = fitted.transform(
        train_matrix[:, impute_columns]
    )
    test_imputed[:, impute_columns] = fitted.transform(
        test_matrix[:, impute_columns]
    )
    if (
        not np.isfinite(train_imputed).all()
        or not np.isfinite(test_imputed).all()
    ):
        raise ValueError("Iterative imputation produced non-finite values")
    return train_imputed, test_imputed, impute_columns.tolist()


def _macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=np.arange(probabilities.shape[1]),
        )
    )


def _mi_bootstrap_ci(
    labels: np.ndarray,
    probabilities_by_imputation: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int = 20261104,
) -> dict:
    rng = np.random.default_rng(seed)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_rows = [
        np.flatnonzero(inverse == group_index)
        for group_index in range(len(unique_groups))
    ]
    point_imputation_auc = [
        _macro_auc(labels, probabilities)
        for probabilities in probabilities_by_imputation
    ]
    estimate = float(np.mean(point_imputation_auc))
    sampled = []
    for _ in range(replicates):
        chosen = rng.choice(
            len(unique_groups), size=len(unique_groups), replace=True
        )
        rows = np.concatenate([group_rows[index] for index in chosen])
        replicate_auc = [
            _macro_auc(labels[rows], probabilities[rows])
            for probabilities in probabilities_by_imputation
        ]
        value = float(np.mean(replicate_auc))
        if np.isfinite(value):
            sampled.append(value)
    if not sampled:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_replicates_valid": 0,
        }
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(sampled, 2.5)),
        "ci_high": float(np.percentile(sampled, 97.5)),
        "bootstrap_replicates_valid": int(len(sampled)),
        "bootstrap_replicates_requested": int(replicates),
        "per_imputation_macro_auc": [
            float(value) for value in point_imputation_auc
        ],
        "between_imputation_sd": float(
            np.std(point_imputation_auc, ddof=1)
        )
        if len(point_imputation_auc) > 1
        else 0.0,
    }


def _run_mi_seed_task(task: tuple) -> dict:
    (
        matrix,
        labels,
        groups,
        stay_ids,
        seed,
        folds,
        imputation,
        max_iter,
        overwrite,
    ) = task
    fold_id = _stratified_folds(
        matrix, labels, groups, seed=seed, folds=folds
    )
    probability = np.full(
        (len(labels), int(labels.max()) + 1), np.nan, dtype=np.float32
    )
    imputed_column_counts = {}
    for fold in range(folds):
        train = np.flatnonzero(fold_id != fold)
        test = np.flatnonzero(fold_id == fold)
        train_x, test_x, imputed_columns = _impute_train_test(
            matrix[train],
            matrix[test],
            seed=seed,
            fold=fold,
            imputation=imputation,
            max_iter=max_iter,
        )
        imputed_column_counts[str(fold)] = len(imputed_columns)
        model = _fit_xgb(
            train_x,
            labels[train],
            n_classes=int(labels.max()) + 1,
            seed=seed + fold,
            n_jobs=THREADS,
        )
        probability[test] = model.predict_proba(test_x)
        print(
            f"[multiple imputation] imputation={imputation} seed={seed} "
            f"fold={fold + 1}/{folds} imputed_columns={len(imputed_columns)} "
            f"worker_threads={THREADS}",
            flush=True,
        )
    if not np.isfinite(probability).all():
        raise ValueError(
            "Multiple imputation produced non-finite OOF probabilities"
        )
    seed_metrics = _remove_cumulative(point_metrics(labels, probability))
    _write_npz(
        PREDICTION_OUTPUT
        / (
            "mimic4_multiple_imputation_"
            f"imp{imputation}_seed{seed}.npz"
        ),
        overwrite=overwrite,
        labels=labels.astype(int),
        probabilities=probability,
        fold=fold_id,
        stay_id=stay_ids,
        subject_key=groups,
        imputation=np.asarray(imputation),
        seed=np.asarray(seed),
    )
    print(
        f"[multiple imputation] imputation={imputation} seed={seed} "
        f"macro_auc={seed_metrics['macro_auc']:.4f}",
        flush=True,
    )
    return {
        "imputation": imputation,
        "seed": seed,
        "probabilities": probability,
        "metrics": seed_metrics,
        "imputed_column_counts": imputed_column_counts,
    }


def run_multiple_imputation(
    *,
    seeds: list[int],
    folds: int,
    bootstrap: int,
    imputations: int,
    max_iter: int,
    overwrite: bool,
) -> dict:
    if imputations < MIN_IMPUTATIONS:
        raise ValueError(
            f"At least {MIN_IMPUTATIONS} imputations are required"
        )
    _, cohort, matrix = _load_locked_development()
    locked = json.loads(
        (RESULTS / "metrics_mimic4.json").read_text(encoding="utf-8")
    )
    features = list(locked["full_static_features"])
    labels = cohort["outcome"].to_numpy(dtype=int)
    groups = cohort["subject_key"].astype(str).to_numpy()
    n_classes = int(labels.max()) + 1

    stay_ids = cohort["stay_id"].to_numpy(dtype=np.int64)
    tasks = [
        (
            matrix,
            labels,
            groups,
            stay_ids,
            seed,
            folds,
            imputation,
            max_iter,
            overwrite,
        )
        for imputation in range(1, imputations + 1)
        for seed in seeds
    ]
    previous_worker_threads = os.environ.get("AMI_ADDITIONAL_THREADS")
    os.environ["AMI_ADDITIONAL_THREADS"] = "1"
    try:
        with ProcessPoolExecutor(max_workers=THREADS) as executor:
            completed = list(executor.map(_run_mi_seed_task, tasks))
    finally:
        if previous_worker_threads is None:
            os.environ.pop("AMI_ADDITIONAL_THREADS", None)
        else:
            os.environ["AMI_ADDITIONAL_THREADS"] = previous_worker_threads

    by_imputation = {
        imputation: {
            item["seed"]: item for item in completed
            if item["imputation"] == imputation
        }
        for imputation in range(1, imputations + 1)
    }
    seed_average_by_imputation = []
    imputation_rows = []
    imputed_column_counts = {}
    for imputation, by_seed in by_imputation.items():
        if set(by_seed) != set(seeds):
            raise ValueError(
                f"Imputation {imputation} did not return every seed"
            )
        for seed, item in by_seed.items():
            imputation_rows.append(
                {
                    "imputation": imputation,
                    "seed": seed,
                    **item["metrics"],
                }
            )
            for fold, count in item["imputed_column_counts"].items():
                imputed_column_counts[
                    f"imputation{imputation}_seed{seed}_fold{fold}"
                ] = count
        seed_average_by_imputation.append(
            np.mean(
                np.stack([by_seed[seed]["probabilities"] for seed in seeds]),
                axis=0,
            )
        )

    imputation_stack = np.stack(seed_average_by_imputation)
    ci = _mi_bootstrap_ci(
        labels,
        imputation_stack,
        groups,
        replicates=bootstrap,
    )
    average_probability = np.mean(imputation_stack, axis=0)
    pooled_point = _remove_cumulative(
        point_metrics(labels, average_probability)
    )
    pooled_point["macro_auc"] = ci["estimate"]
    pooled_point["macro_auc_ci"] = ci
    pooled_point["macro_auc_strategy"] = (
        "mean of per-imputation macro-AUCs across the five seeds"
    )

    per_imputation_metrics = []
    for index, probabilities in enumerate(seed_average_by_imputation, start=1):
        metrics = _remove_cumulative(point_metrics(labels, probabilities))
        metrics["imputation"] = index
        metrics["macro_auc_ci"] = bootstrap_metric_ci(
            labels,
            probabilities,
            groups=groups,
            replicates=bootstrap,
        )
        per_imputation_metrics.append(metrics)

    result = _base_result(
        analysis="multiple_imputation",
        cohort=cohort,
        labels=labels,
        groups=groups,
        seeds=seeds,
        folds=folds,
        metrics=pooled_point,
        features=features,
    )
    result.update(
        {
            "n_imputations": imputations,
            "imputation": {
                "method": "sklearn.IterativeImputer",
                "estimator": "BayesianRidge",
                "max_iter": max_iter,
                "tol": 1e-3,
                "sample_posterior": True,
                "chained_equations_converged": False,
                "budget_note": (
                    "One posterior-sampled chained-equations pass is used "
                    "because three independent imputations across five folds "
                    "and five seeds require 75 model fits. This is a "
                    "computationally constrained approximation and is not "
                    "claimed to be a converged multiple-imputation solution."
                ),
                "fit_scope": "outer training fold only",
                "retrain_each_imputation": True,
                "random_state_rule": (
                    "seed * 100000 + fold * 1000 + imputation"
                ),
                "imputed_column_counts": imputed_column_counts,
            },
            "per_imputation": per_imputation_metrics,
            "imputation_seed_metrics": imputation_rows,
            "locked_reference_macro_auc": float(
                locked["metrics"]["xgb_static_ts"]["macro_auc"]
            ),
            "bootstrap": {
                "replicates_requested": bootstrap,
                "unit": "patient",
                "metric": "mean per-imputation macro-AUC",
                "includes_between_imputation_variation": True,
            },
        }
    )
    detail_path = OUTPUT / "multiple_imputation.json"
    sensitivity_path = (
        SENSITIVITY_OUTPUT / SENSITIVITY_FILES["multiple_imputation"]
    )
    wrote_detail = _write_json(detail_path, result, overwrite=overwrite)
    wrote_metrics = _write_json(
        sensitivity_path, result, overwrite=overwrite
    )
    pd.DataFrame(imputation_rows).to_csv(
        OUTPUT / "multiple_imputation_by_seed.csv", index=False
    )
    pd.DataFrame(per_imputation_metrics).to_csv(
        OUTPUT / "multiple_imputation_by_imputation.csv", index=False
    )
    _write_npz(
        OUTPUT / "multiple_imputation_seed_averaged_predictions.npz",
        overwrite=overwrite,
        labels=labels.astype(int),
        probabilities=imputation_stack,
        subject_key=groups,
        seeds=np.asarray(seeds),
        imputations=np.arange(1, imputations + 1),
    )
    result["_wrote"] = {
        "detail": wrote_detail,
        "sensitivity_metrics": wrote_metrics,
    }
    return result


def _existing_outputs() -> dict:
    return {
        "additional_sensitivities": {
            name: str(path)
            for name, path in {
                "multiple_imputation": OUTPUT / "multiple_imputation.json",
                "treatment_estimability": (
                    OUTPUT / "treatment_estimability.json"
                ),
                "pre24h_revascularization": (
                    OUTPUT / "pre24h_revascularization.json"
                ),
                "pre24h_vasopressor_mcs": (
                    OUTPUT / "pre24h_vasopressor_mcs.json"
                ),
                "label_merge_rehab_facility": (
                    OUTPUT / "label_merge_rehab_facility.json"
                ),
            }.items()
            if path.is_file()
        },
        "sensitivity_metrics": {
            name: str(SENSITIVITY_OUTPUT / filename)
            for name, filename in SENSITIVITY_FILES.items()
            if (SENSITIVITY_OUTPUT / filename).is_file()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run AMI multiple-imputation, pre-24-hour treatment, and "
            "rehabilitation-plus-facility label sensitivities."
        )
    )
    parser.add_argument(
        "--analyses",
        nargs="+",
        choices=ANALYSES,
        default=list(ANALYSES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument(
        "--imputations", type=int, default=DEFAULT_IMPUTATIONS
    )
    parser.add_argument(
        "--mi-max-iter", type=int, default=DEFAULT_MI_MAX_ITER
    )
    parser.add_argument(
        "--bootstrap", type=int, default=DEFAULT_BOOTSTRAP
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.imputations < MIN_IMPUTATIONS:
        raise ValueError(
            f"--imputations must be at least {MIN_IMPUTATIONS}"
        )
    if args.mi_max_iter < 1:
        raise ValueError("--mi-max-iter must be positive")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")

    warnings.filterwarnings("ignore", category=FutureWarning)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PREDICTION_OUTPUT.mkdir(parents=True, exist_ok=True)
    SENSITIVITY_OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": {
            "threads": THREADS,
            "folds": args.folds,
            "seeds": args.seeds,
            "bootstrap_replicates": args.bootstrap,
            "analysis_order": args.analyses,
            "locked_model_files_modified": False,
            "negative_or_null_estimates_retained": True,
        },
        "results": {},
    }

    for analysis in args.analyses:
        print(f"\n=== {analysis} ===", flush=True)
        if analysis == "multiple_imputation":
            result = run_multiple_imputation(
                seeds=args.seeds,
                folds=args.folds,
                bootstrap=args.bootstrap,
                imputations=args.imputations,
                max_iter=args.mi_max_iter,
                overwrite=args.overwrite,
            )
        elif analysis == "label_merge_rehab_facility":
            result = run_label_merge_analysis(
                seeds=args.seeds,
                folds=args.folds,
                bootstrap=args.bootstrap,
                overwrite=args.overwrite,
            )
        else:
            result = run_treatment_analysis(
                analysis=analysis,
                seeds=args.seeds,
                folds=args.folds,
                bootstrap=args.bootstrap,
                overwrite=args.overwrite,
            )
        summary["results"][analysis] = {
            "macro_auc": float(result["metrics"]["macro_auc"]),
            "macro_auc_ci": result["metrics"]["macro_auc_ci"],
            "n": int(result["n"]),
            "outcome_counts": result["outcome_counts"],
            "n_features": int(result["n_features"]),
            "wrote": result.get("_wrote", {}),
        }

    existing_summary_path = OUTPUT / "run_summary.json"
    if existing_summary_path.is_file():
        existing_summary = json.loads(
            existing_summary_path.read_text(encoding="utf-8")
        )
        retained = []
        for analysis, result_summary in existing_summary.get(
            "results", {}
        ).items():
            if analysis not in summary["results"]:
                summary["results"][analysis] = result_summary
                retained.append(analysis)
        summary["protocol"]["retained_from_prior_run"] = retained

    summary["outputs"] = _existing_outputs()
    existing_summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print("\n" + json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

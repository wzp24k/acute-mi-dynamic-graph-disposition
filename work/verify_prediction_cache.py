"""Verify that all final model and temporal-control caches are complete."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
PROCESSED = PROJECT_ROOT / "data" / "processed"
SEEDS = (1, 7, 42, 123, 2024)
MODELS = {
    "mimic4": (
        "xgb_static",
        "xgb_static_ts",
        "mlp_static",
        "mlp_static_ts",
        "gru",
        "grud",
        "transformer",
        "stgcn_lite",
        "dggru_static",
        "dggru",
        "dggru_tr",
        "dggru_ctr",
    ),
    "eicu": ("xgb_static", "xgb_static_ts", "dggru_tr"),
    "mimic4_6h": ("xgb_static", "xgb_static_ts", "dggru", "dggru_tr"),
    "mimic4_72h": ("xgb_static", "xgb_static_ts", "dggru", "dggru_tr"),
}


def _verify_cache(
    dataset: str,
    model: str,
    seed: int,
) -> dict:
    stem = f"{dataset}_{model}_seed{seed}"
    npz_path = RESULTS / "predictions" / f"{stem}.npz"
    json_path = RESULTS / "predictions" / f"{stem}.json"
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Missing cache pair: {stem}")
    cohort = pd.read_csv(PROCESSED / dataset / "cohort.csv")
    with np.load(npz_path, allow_pickle=True) as data:
        labels = data["labels"].astype(int)
        probabilities = data["probabilities"].astype(float)
        subject_key = data["subject_key"].astype(str)
        fold = data["fold"].astype(int)
    expected_labels = cohort["outcome"].to_numpy(dtype=int)
    expected_groups = cohort["subject_key"].astype(str).to_numpy()
    if labels.shape != expected_labels.shape or not np.array_equal(
        labels, expected_labels
    ):
        raise ValueError(f"{stem}: labels do not match the locked cohort")
    if subject_key.shape != expected_groups.shape or not np.array_equal(
        subject_key, expected_groups
    ):
        raise ValueError(f"{stem}: subject keys do not match the locked cohort")
    if probabilities.shape != (len(cohort), int(labels.max()) + 1):
        raise ValueError(f"{stem}: probability shape is incorrect")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{stem}: probabilities contain non-finite values")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-5):
        raise ValueError(f"{stem}: probabilities do not sum to one")
    if set(np.unique(fold)) != set(range(5)):
        raise ValueError(f"{stem}: outer-fold assignment is incomplete")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "n": int(len(cohort)),
        "n_classes": int(probabilities.shape[1]),
        "macro_auc": float(metadata["metrics"]["macro_auc"]),
    }


def _verify_temporal_controls() -> dict:
    directory = RESULTS / "sensitivity" / "temporal_controls"
    expected = [
        *(f"smooth_drift_lambda{value.replace('.', 'p')}_seed42" for value in (
            "0",
            "0p0005",
            "0p001",
            "0p002",
            "0p005",
            "0p01",
            "0p02",
            "0p03",
            "0p05",
            "0p075",
            "0p1",
            "0p15",
            "0p2",
            "0p3",
            "0p5",
            "0p75",
            "1",
            "1p5",
        )),
        "no_tr_seed42",
        "intact_seed42",
        "monotone_decrease_seed42",
        "monotone_increase_seed42",
        "permuted_seed42",
        "smooth_only_seed42",
    ]
    missing = [
        name
        for name in expected
        if not (directory / f"{name}.json").is_file()
        or not (directory / f"{name}.npz").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing temporal-control caches: " + ", ".join(missing)
        )
    curvature_directory = RESULTS / "sensitivity" / "cumulative_curvature"
    curvature_expected = [
        f"dggru_ctr_lambda{value.replace('.', 'p')}_seed42"
        for value in ("0", "0p001", "0p005", "0p01", "0p05", "0p1")
    ]
    curvature_missing = [
        name
        for name in curvature_expected
        if not (curvature_directory / f"{name}.json").is_file()
        or not (curvature_directory / f"{name}.npz").is_file()
    ]
    if curvature_missing:
        raise FileNotFoundError(
            "Missing cumulative-curvature caches: "
            + ", ".join(curvature_missing)
        )
    return {
        "expected_scalar_configurations": len(expected),
        "missing_scalar": missing,
        "expected_cumulative_curvature_configurations": len(
            curvature_expected
        ),
        "missing_cumulative_curvature": curvature_missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-controls", action="store_true")
    args = parser.parse_args()
    verified = [
        _verify_cache(dataset, model, seed)
        for dataset, models in MODELS.items()
        for model in models
        for seed in SEEDS
    ]
    output = {
        "status": "passed",
        "n_cache_pairs": len(verified),
        "cache_pairs": verified,
    }
    if args.temporal_controls:
        output["temporal_controls"] = _verify_temporal_controls()
    path = RESULTS / "audits" / "prediction_cache_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": output["status"],
                "n_cache_pairs": output["n_cache_pairs"],
                "temporal_controls": output.get("temporal_controls"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Evaluate cumulative-outcome curvature regularization."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import RESULTS  # noqa: E402
from src.ami_dynamic.evaluation import point_metrics  # noqa: E402
from src.ami_dynamic.train_evaluate import (  # noqa: E402
    TRAIN_DEFAULTS,
    _feature_columns,
    _json_default,
    _static_matrix,
    _temporal_predictions,
    load_dataset,
)


OUTPUT = RESULTS / "sensitivity" / "cumulative_curvature"
DEFAULT_LAMBDAS = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1)


def _run_config(
    lambda_curvature: float,
    *,
    seed: int,
    folds: int,
    epochs: int,
) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = f"dggru_ctr_lambda{lambda_curvature:g}".replace(".", "p")
    json_path = OUTPUT / f"{stem}_seed{seed}.json"
    npz_path = OUTPUT / f"{stem}_seed{seed}.npz"
    data = load_dataset("mimic4")
    cohort = data["cohort"]
    labels = cohort["outcome"].to_numpy(dtype=int)
    groups = cohort["subject_key"].astype(str).to_numpy()
    if json_path.is_file() and npz_path.is_file():
        with np.load(npz_path, allow_pickle=True) as cached:
            cache_matches = (
                np.array_equal(cached["labels"].astype(int), labels)
                and np.array_equal(
                    cached["subject_key"].astype(str), groups
                )
            )
        if cache_matches:
            return json.loads(json_path.read_text(encoding="utf-8"))
    _, full_columns = _feature_columns(cohort)
    static = _static_matrix(cohort, full_columns)
    n_classes = int(labels.max()) + 1
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    probability = np.full((len(cohort), n_classes), np.nan, dtype=np.float32)
    fold_id = np.full(len(cohort), -1, dtype=int)
    model_name = "dggru" if lambda_curvature == 0.0 else "dggru_ctr"
    for fold, (train, test) in enumerate(
        splitter.split(static, labels, groups)
    ):
        fold_id[test] = fold
        probability[test], _ = _temporal_predictions(
            data["values"][train],
            data["masks"][train],
            static[train],
            labels[train],
            data["values"][test],
            data["masks"][test],
            static[test],
            model_name=model_name,
            n_classes=n_classes,
            seed=seed + fold,
            epochs=epochs,
            train_defaults=TRAIN_DEFAULTS,
            lambda_smooth=0.0,
            lambda_drift=0.0,
            lambda_curvature=lambda_curvature,
        )
        print(
            f"[cumulative_curvature lambda={lambda_curvature:g}] "
            f"seed={seed} fold={fold + 1}/{folds}",
            flush=True,
        )
    metrics = point_metrics(labels, probability)
    metrics.pop("cumulative_probabilities", None)
    result = {
        "model": model_name,
        "regularization": "cumulative_curvature",
        "lambda_curvature": lambda_curvature,
        "seed": seed,
        "folds": folds,
        "epochs": epochs,
        "metrics": metrics,
    }
    np.savez_compressed(
        npz_path,
        labels=labels,
        probabilities=probability,
        fold=fold_id,
        stay_id=cohort["stay_id"].to_numpy(dtype=np.int64),
        subject_key=groups,
    )
    json_path.write_text(
        json.dumps(result, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lambdas", nargs="*", type=float, default=list(DEFAULT_LAMBDAS))
    args = parser.parse_args()
    summary = [
        _run_config(
            float(value),
            seed=args.seed,
            folds=args.folds,
            epochs=args.epochs,
        )
        for value in args.lambdas
    ]
    summary_path = OUTPUT / f"summary_seed{args.seed}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()

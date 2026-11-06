"""Run the prespecified lambda path and temporal-regularization controls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import RESULTS, SEQUENCE_HOURS, load_config  # noqa: E402
from src.ami_dynamic.evaluation import point_metrics  # noqa: E402
from src.ami_dynamic.train_evaluate import (  # noqa: E402
    TRAIN_DEFAULTS,
    _feature_columns,
    _json_default,
    _static_matrix,
    _temporal_predictions,
    load_dataset,
)


OUTPUT = RESULTS / "sensitivity" / "temporal_controls"


def _config_name(lambda_smooth: float, mode: str, no_tr: bool = False) -> str:
    if no_tr:
        return "no_tr"
    value = f"{lambda_smooth:g}".replace(".", "p")
    return f"{mode}_lambda{value}"


def _trajectory_summary(
    trajectory: np.ndarray,
    masks: np.ndarray,
) -> dict:
    valid = masks.sum(axis=2) > 0
    difference = trajectory[:, 1:] - trajectory[:, :-1]
    pair_valid = valid[:, 1:] & valid[:, :-1]
    smooth_values = np.square(difference)[pair_valid]
    first = []
    last = []
    for row in range(len(trajectory)):
        index = np.flatnonzero(valid[row])
        if len(index):
            first.append(trajectory[row, index[0]])
            last.append(trajectory[row, index[-1]])
    drift = np.asarray(last) - np.asarray(first)
    return {
        "mean_adjacent_squared_change": float(smooth_values.mean()),
        "mean_absolute_net_drift": float(np.abs(drift).mean()),
        "mean_squared_net_drift": float(np.square(drift).mean()),
        "n_patients_with_trajectory": int(len(first)),
    }


def run_config(
    *,
    name: str,
    no_tr: bool,
    mode: str,
    lambda_smooth: float,
    lambda_drift: float,
    seed: int,
    folds: int,
    epochs: int,
) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT / f"{name}_seed{seed}.npz"
    json_path = OUTPUT / f"{name}_seed{seed}.json"
    data = load_dataset("mimic4")
    cohort = data["cohort"]
    labels = cohort["outcome"].to_numpy(dtype=int)
    groups = cohort["subject_key"].astype(str).to_numpy()
    if npz_path.is_file() and json_path.is_file():
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
    trajectory_all = np.full(
        (len(cohort), SEQUENCE_HOURS), np.nan, dtype=np.float32
    )
    fold_id = np.full(len(cohort), -1, dtype=int)
    for fold, (train, test) in enumerate(
        splitter.split(static, labels, groups)
    ):
        fold_id[test] = fold
        probability[test], trajectory_all[test] = _temporal_predictions(
            data["values"][train],
            data["masks"][train],
            static[train],
            labels[train],
            data["values"][test],
            data["masks"][test],
            static[test],
            model_name="dggru" if no_tr else "dggru_tr",
            n_classes=n_classes,
            seed=seed + fold,
            epochs=epochs,
            train_defaults=TRAIN_DEFAULTS,
            lambda_smooth=lambda_smooth,
            lambda_drift=lambda_drift,
            regularization_mode=mode,
        )
        print(
            f"[{name}] seed={seed} fold={fold + 1}/{folds}",
            flush=True,
        )
    metrics = point_metrics(labels, probability)
    metrics.pop("cumulative_probabilities", None)
    result = {
        "name": name,
        "seed": seed,
        "folds": folds,
        "epochs": epochs,
        "no_tr": no_tr,
        "mode": mode,
        "lambda_smooth": lambda_smooth,
        "lambda_drift": lambda_drift,
        "metrics": metrics,
        "trajectory": _trajectory_summary(
            trajectory_all,
            data["masks"],
        ),
    }
    np.savez_compressed(
        npz_path,
        labels=labels,
        probabilities=probability,
        trajectories=trajectory_all,
        fold=fold_id,
        stay_id=cohort["stay_id"].to_numpy(dtype=np.int64),
        subject_key=groups,
    )
    json_path.write_text(
        json.dumps(result, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(
        f"[{name}] seed={seed} macro_auc={metrics['macro_auc']:.4f}",
        flush=True,
    )
    return result


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--lambdas",
        nargs="*",
        type=float,
        default=config["temporal_regularization"]["lambda_s_grid"],
    )
    parser.add_argument(
        "--controls",
        nargs="*",
        choices=[
            "no_tr",
            "intact",
            "monotone_decrease",
            "monotone_increase",
            "permuted",
            "smooth_only",
        ],
        default=[
            "no_tr",
            "intact",
            "monotone_decrease",
            "monotone_increase",
            "permuted",
            "smooth_only",
        ],
    )
    parser.add_argument("--skip-lambdas", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    args = parser.parse_args()
    lambda_drift_ratio = float(
        config["temporal_regularization"]["lambda_d_ratio_to_lambda_s"]
    )
    default_lambda = float(
        config["temporal_regularization"]["default_lambda_s"]
    )
    for control in [] if args.skip_controls else args.controls:
        if control == "no_tr":
            run_config(
                name="no_tr",
                no_tr=True,
                mode="smooth_drift",
                lambda_smooth=0.0,
                lambda_drift=0.0,
                seed=args.seed,
                folds=args.folds,
                epochs=args.epochs,
            )
            continue
        mode = "smooth_drift" if control == "intact" else control
        lambda_smooth = default_lambda
        lambda_drift = lambda_smooth * lambda_drift_ratio
        run_config(
            name=control,
            no_tr=False,
            mode=mode,
            lambda_smooth=lambda_smooth,
            lambda_drift=lambda_drift,
            seed=args.seed,
            folds=args.folds,
            epochs=args.epochs,
        )
    if not args.skip_lambdas:
        for lambda_smooth in args.lambdas:
            name = _config_name(lambda_smooth, "smooth_drift")
            run_config(
                name=name,
                no_tr=False,
                mode="smooth_drift",
                lambda_smooth=float(lambda_smooth),
                lambda_drift=float(lambda_smooth) * lambda_drift_ratio,
                seed=args.seed,
                folds=args.folds,
                epochs=args.epochs,
            )
    summary = []
    for path in sorted(OUTPUT.glob(f"*_seed{args.seed}.json")):
        if path.name.startswith("summary_"):
            continue
        summary.append(json.loads(path.read_text(encoding="utf-8")))
    summary_path = OUTPUT / f"summary_seed{args.seed}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()

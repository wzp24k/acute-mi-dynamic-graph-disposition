"""Pre-specified endpoint and population sensitivity analyses."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .common import RESULTS, ensure_dirs, load_config
from .train_evaluate import (
    ALL_FEATURES,
    fit_xgboost,
    load_dataset,
    metric_block,
)


def cv_predictions(
    dataset,
    labels: np.ndarray,
    rows: np.ndarray,
    config: dict,
    seed: int,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=int)
    subset_labels = labels[rows]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prediction = np.full(len(rows), np.nan)
    features = dataset.frame(ALL_FEATURES)
    for train_position, test_position in splitter.split(
        np.zeros((len(rows), 1)), subset_labels
    ):
        train_rows = rows[train_position]
        test_rows = rows[test_position]
        prediction[test_position] = fit_xgboost(
            features.iloc[train_rows],
            features.iloc[test_rows],
            labels[train_rows],
            config,
            seed,
        )
    if np.isnan(prediction).any():
        raise RuntimeError("Incomplete sensitivity predictions")
    return prediction


def main() -> None:
    ensure_dirs()
    config = load_config()
    dataset = load_dataset("mimic4")
    labels_coded = dataset.labels.copy()
    labels_timed = (
        pd.to_numeric(
            dataset.cohort["macce7_timestamped"], errors="raise"
        )
        .astype(int)
        .to_numpy()
    )
    all_rows = np.arange(len(dataset.cohort))
    analyses = {
        "primary_coded_macce": (labels_coded, all_rows),
        "timestamped_macce": (labels_timed, all_rows),
        "no_prior_mi_or_stroke": (
            labels_coded,
            np.flatnonzero(
                dataset.cohort["prior_mi"].eq(0)
                & dataset.cohort["prior_stroke"].eq(0)
            ),
        ),
        "acs_history": (
            labels_coded,
            np.flatnonzero(dataset.cohort["acs_history"].eq(1)),
        ),
        "prior_pci": (
            labels_coded,
            np.flatnonzero(dataset.cohort["prior_pci"].eq(1)),
        ),
        "prior_cabg": (
            labels_coded,
            np.flatnonzero(dataset.cohort["prior_cabg"].eq(1)),
        ),
        "age_ge_75": (
            labels_coded,
            np.flatnonzero(dataset.cohort["age"].ge(75)),
        ),
    }
    output = {"model": "xgb_static_ts", "seed": 7, "analyses": {}}
    for name, (labels, rows) in analyses.items():
        events = int(labels[rows].sum())
        if events < 50 or len(rows) - events < 50:
            output["analyses"][name] = {
                "n": int(len(rows)),
                "events": events,
                "status": "insufficient_events",
            }
            continue
        probability = cv_predictions(dataset, labels, rows, config, 7)
        result = metric_block(labels[rows], probability)
        result.update({"n": int(len(rows)), "events": events, "status": "ok"})
        output["analyses"][name] = result
        print(
            f"{name}: n={len(rows):,}, events={events:,}, "
            f"AUROC={result['auroc']:.3f}",
            flush=True,
        )
    (RESULTS / "sensitivity_analysis.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

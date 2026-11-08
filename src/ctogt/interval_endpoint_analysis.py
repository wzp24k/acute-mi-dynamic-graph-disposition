"""Evaluate existing temporal predictions against interval-censored endpoints."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import PREDICTIONS, PROCESSED, RESULTS, ensure_dirs, load_config
from .train_evaluate import bootstrap_metric, metric_block


MODELS = ("xgb_static_ts", "transformer", "temporal_ensemble")


def align_prediction(cohort: str, model: str, endpoint: pd.DataFrame):
    if cohort == "mimic4":
        path = PREDICTIONS / f"mimic4_cv_{model}.npz"
        key = "probability"
    else:
        path = PREDICTIONS / f"{cohort}_external_{model}.npz"
        key = model
    with np.load(path, allow_pickle=True) as data:
        stay_id = data["stay_id"].astype(np.int64)
        probability = data[key].astype(float)
    lookup = endpoint.set_index("stay_id")
    aligned = lookup.reindex(stay_id)
    return aligned.reset_index(), probability


def main() -> None:
    ensure_dirs()
    config = load_config()
    output = {"cohorts": {}}
    for cohort in ("mimic4", "mimic3", "eicu", "eicu_anycad"):
        endpoint = pd.read_csv(PROCESSED / cohort / "interval_endpoint.csv")
        cohort_result = {
            "n": int(len(endpoint)),
            "coded_events": int(endpoint["macce7"].sum()),
            "confirmed_events": int(endpoint["macce_interval_confirmed"].sum()),
            "possible_overlap_events": int(
                endpoint["macce_interval_possible"].sum()
            ),
            "overlap_events": int(endpoint["macce_interval_overlap"].sum()),
            "models": {},
        }
        for model in MODELS:
            path = (
                PREDICTIONS / f"mimic4_cv_{model}.npz"
                if cohort == "mimic4"
                else PREDICTIONS / f"{cohort}_external_{model}.npz"
            )
            if not path.is_file():
                continue
            aligned, probability = align_prediction(cohort, model, endpoint)
            model_result = {}
            for outcome in (
                "macce7",
                "macce_interval_confirmed",
                "macce_interval_overlap",
            ):
                labels = aligned[outcome].astype(int).to_numpy()
                if labels.min() == labels.max():
                    model_result[outcome] = {
                        "status": "single_class",
                        "events": int(labels.sum()),
                    }
                    continue
                if labels.sum() < 50:
                    model_result[outcome] = {
                        "status": "insufficient_events",
                        "events": int(labels.sum()),
                    }
                    continue
                metrics = metric_block(labels, probability)
                metrics["events"] = int(labels.sum())
                metrics["auroc_ci"] = bootstrap_metric(
                    labels,
                    probability,
                    "auroc",
                    int(config["bootstrap_replicates"]),
                    7,
                )
                model_result[outcome] = metrics
            cohort_result["models"][model] = model_result
        output["cohorts"][cohort] = cohort_result

    m4 = output["cohorts"]["mimic4"]
    m4["confirmed_minus_coded"] = {
        model: {
            "auroc_difference": (
                m4["models"][model]["macce_interval_confirmed"]["auroc"]
                - m4["models"][model]["macce7"]["auroc"]
            )
        }
        for model in m4["models"]
        if "auroc" in m4["models"][model].get("macce_interval_confirmed", {})
    }
    (RESULTS / "interval_endpoint_analysis.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

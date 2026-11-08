"""Evaluate interval-endpoint sensitivity to empirical timing assumptions."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .build_interval_endpoint import (
    build_eicu,
    build_mimic3,
    build_mimic4,
    connect,
)
from .common import PREDICTIONS, RESULTS, ensure_dirs


MODELS = ("xgb_static_ts", "transformer", "temporal_ensemble")


def prediction(cohort: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    path = (
        PREDICTIONS / f"mimic4_cv_{model}.npz"
        if cohort == "mimic4"
        else PREDICTIONS / f"{cohort}_external_{model}.npz"
    )
    key = "probability" if cohort == "mimic4" else model
    with np.load(path, allow_pickle=True) as data:
        return data["stay_id"].astype(np.int64), data[key].astype(float)


def evaluate(frame: pd.DataFrame, cohort: str) -> dict[str, object]:
    lookup = frame.set_index("stay_id")
    output: dict[str, object] = {
        "n": int(len(frame)),
        "coded_events": int(frame["macce7"].sum()),
        "confirmed_events": int(frame["macce_interval_confirmed"].sum()),
        "anchor_overlap_events": int(frame["macce_interval_overlap"].sum()),
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
        stay_id, probability = prediction(cohort, model)
        aligned = lookup.reindex(stay_id)
        labels = aligned["macce_interval_overlap"].fillna(0).astype(int).to_numpy()
        if labels.sum() < 50 or labels.min() == labels.max():
            output["models"][model] = {
                "status": "insufficient_events",
                "events": int(labels.sum()),
            }
            continue
        output["models"][model] = {
            "events": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "auroc": float(roc_auc_score(labels, probability)),
            "auprc": float(average_precision_score(labels, probability)),
            "brier": float(brier_score_loss(labels, probability)),
        }
    return output


def main() -> None:
    ensure_dirs()
    con = connect()
    output: dict[str, object] = {
        "troponin_threshold_multiplier": {},
        "eicu_documentation_lag_hours": {},
    }

    for multiplier in (0.5, 1.0, 2.0):
        key = f"{multiplier:.1f}"
        output["troponin_threshold_multiplier"][key] = {}
        for cohort, builder in (("mimic4", build_mimic4), ("mimic3", build_mimic3)):
            frame = builder(
                con,
                troponin_threshold_multiplier=multiplier,
            )
            output["troponin_threshold_multiplier"][key][cohort] = evaluate(
                frame, cohort
            )
            print(
                f"[troponin x{multiplier:g}] {cohort}: "
                f"{int(frame['macce_interval_overlap'].sum())} overlap events",
                flush=True,
            )

    for lag in (0.0, 3.0, 6.0, 12.0):
        key = f"{lag:.1f}"
        output["eicu_documentation_lag_hours"][key] = {}
        for cohort in ("eicu", "eicu_anycad"):
            frame = build_eicu(
                con,
                cohort,
                documentation_lag_hours=lag,
            )
            output["eicu_documentation_lag_hours"][key][cohort] = evaluate(
                frame, cohort
            )
            print(
                f"[eICU lag {lag:g}h] {cohort}: "
                f"{int(frame['macce_interval_overlap'].sum())} overlap events",
                flush=True,
            )

    (RESULTS / "interval_threshold_sensitivity.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    rows = []
    for family, variants in output.items():
        for parameter, cohorts in variants.items():
            for cohort, result in cohorts.items():
                for model, metrics in result["models"].items():
                    rows.append(
                        {
                            "family": family,
                            "parameter": parameter,
                            "cohort": cohort,
                            "model": model,
                            "n": result["n"],
                            "coded_events": result["coded_events"],
                            "confirmed_events": result["confirmed_events"],
                            "anchor_overlap_events": result[
                                "anchor_overlap_events"
                            ],
                            "auroc": metrics.get("auroc", np.nan),
                            "auprc": metrics.get("auprc", np.nan),
                            "brier": metrics.get("brier", np.nan),
                        }
                    )
    pd.DataFrame(rows).to_csv(
        RESULTS / "interval_threshold_sensitivity.csv",
        index=False,
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

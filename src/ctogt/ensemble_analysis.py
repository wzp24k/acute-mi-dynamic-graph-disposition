"""Build the prespecified XGBoost + Transformer temporal ensemble."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import PREDICTIONS, RESULTS, ensure_dirs, load_config
from .train_evaluate import (
    bootstrap_metric,
    bootstrap_paired_auc,
    crossfit_platt,
    metric_block,
    net_benefit,
)


MODEL_NAME = "temporal_ensemble"
BASE_MODELS = ("xgb_static_ts", "transformer")


def load_development():
    labels = None
    components = {}
    for model in BASE_MODELS:
        path = PREDICTIONS / f"mimic4_cv_{model}.npz"
        with np.load(path, allow_pickle=True) as data:
            current = data["labels"].astype(int)
            if labels is None:
                labels = current
            if not np.array_equal(labels, current):
                raise ValueError("Development prediction labels are misaligned")
            components[model] = data["probability"].astype(float)
    ensemble = np.mean(list(components.values()), axis=0)
    return labels, components, ensemble


def load_external(name: str):
    labels = None
    components = {}
    site = None
    for model in BASE_MODELS:
        path = PREDICTIONS / f"{name}_external_{model}.npz"
        with np.load(path, allow_pickle=True) as data:
            current = data["labels"].astype(int)
            if labels is None:
                labels = current
                site = data["site"].astype(str)
            if not np.array_equal(labels, current):
                raise ValueError(f"{name} prediction labels are misaligned")
            components[model] = data[model].astype(float)
    ensemble = np.mean(list(components.values()), axis=0)
    return labels, components, ensemble, site


def main() -> None:
    ensure_dirs()
    config = load_config()
    labels, components, ensemble = load_development()
    metrics = metric_block(labels, ensemble)
    xgb = components["xgb_static_ts"]
    transformer = components["transformer"]
    record = {
        "dataset": "mimic4",
        "model": MODEL_NAME,
        "model_label": "Temporal ensemble: 50% XGB+TS + 50% Transformer",
        "n": int(len(labels)),
        "events": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "seeds": [7, 42, 2026],
        "folds": int(config["cv_folds"]),
        "metrics": metrics,
        "bootstrap": {
            "auroc_ci": bootstrap_metric(
                labels,
                ensemble,
                "auroc",
                int(config["bootstrap_replicates"]),
                7,
            ),
            "auprc_ci": bootstrap_metric(
                labels,
                ensemble,
                "auprc",
                int(config["bootstrap_replicates"]),
                8,
            ),
        },
        "paired_contrasts": {
            "vs_xgb_static_ts": bootstrap_paired_auc(
                labels,
                ensemble,
                xgb,
                int(config["bootstrap_replicates"]),
                9,
            ),
            "vs_transformer": bootstrap_paired_auc(
                labels,
                ensemble,
                transformer,
                int(config["bootstrap_replicates"]),
                10,
            ),
        },
        "weights": {"xgb_static_ts": 0.5, "transformer": 0.5},
    }
    (RESULTS / f"metrics_development_{MODEL_NAME}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        PREDICTIONS / f"mimic4_cv_{MODEL_NAME}.npz",
        labels=labels,
        stay_id=np.load(
            PREDICTIONS / "mimic4_cv_xgb_static_ts.npz", allow_pickle=True
        )["stay_id"].astype(np.int64),
        probability=ensemble.astype(np.float32),
        seed_probability=ensemble[None, :].astype(np.float32),
    )
    for cohort in ("mimic3", "eicu", "eicu_anycad"):
        external_labels, external_components, external_ensemble, site = (
            load_external(cohort)
        )
        recalibrated = crossfit_platt(
            external_labels,
            external_ensemble,
            site,
            int(config["cv_folds"]),
            7,
        )
        external_record = {
            "external": cohort,
            "n": int(len(external_labels)),
            "events": int(external_labels.sum()),
            "prevalence": float(external_labels.mean()),
            "model": MODEL_NAME,
            "metrics": {
                "unrecalibrated": metric_block(
                    external_labels, external_ensemble
                ),
                "recalibrated": metric_block(
                    external_labels, recalibrated
                ),
                "net_benefit": net_benefit(
                    external_labels, external_ensemble
                ).to_dict(orient="records"),
            },
        }
        if cohort.startswith("eicu"):
            site_frame = pd.DataFrame(
                {
                    "site": site,
                    "label": external_labels,
                    "probability": external_ensemble,
                }
            )
            summary = (
                site_frame.groupby("site", as_index=False)
                .agg(
                    n=("label", "size"),
                    events=("label", "sum"),
                    mean_prediction=("probability", "mean"),
                    observed=("label", "mean"),
                )
                .query("n >= @config['eicu_minimum_site_n']")
            )
            external_record["metrics"]["site_summary"] = summary.to_dict(
                orient="records"
            )
        (
            RESULTS / f"metrics_external_{cohort}_{MODEL_NAME}.json"
        ).write_text(
            json.dumps(external_record, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            PREDICTIONS / f"{cohort}_external_{MODEL_NAME}.npz",
            labels=external_labels,
            stay_id=np.load(
                PREDICTIONS / f"{cohort}_external_xgb_static_ts.npz",
                allow_pickle=True,
            )["stay_id"].astype(np.int64),
            site=site,
            **{
                MODEL_NAME: external_ensemble.astype(np.float32),
                f"{MODEL_NAME}__recalibrated": recalibrated.astype(
                    np.float32
                ),
            },
        )
        print(
            f"{cohort}: AUROC={metric_block(external_labels, external_ensemble)['auroc']:.3f}, "
            f"recalibrated AUROC="
            f"{metric_block(external_labels, recalibrated)['auroc']:.3f}",
            flush=True,
        )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

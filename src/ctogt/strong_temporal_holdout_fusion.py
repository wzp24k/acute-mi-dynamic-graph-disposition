"""Select fusion weight on 2017-2019, then test the locked model in 2020-2022."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from .common import PREDICTIONS, RESULTS, ensure_dirs, load_config
from .temporal_holdout import fit_transformer, load_years
from .train_evaluate import (
    bootstrap_paired_auc,
    load_dataset,
    metric_block,
)


def main() -> None:
    ensure_dirs()
    config = load_config()
    dataset = load_dataset("mimic4")
    groups = load_years(dataset)
    transformer_train_rows = np.flatnonzero(
        np.isin(groups, ["2008 - 2010", "2011 - 2013", "2014 - 2016"])
    )
    validation_rows = np.flatnonzero(groups == "2017 - 2019")
    transformer_validation = fit_transformer(
        dataset,
        transformer_train_rows,
        validation_rows,
        config,
        seeds=(7, 42, 2026),
    )
    with np.load(
        PREDICTIONS / "mimic4_internal_era_validation_xgb.npz",
        allow_pickle=True,
    ) as data:
        validation_labels = data["labels"].astype(int)
        strong_validation_xgb = data["enhanced_features_deeper"].astype(float)

    weights = (0.0, 0.25, 0.50, 0.75, 1.0)
    validation_records = []
    validation_probabilities = {}
    for weight in weights:
        probability = (
            weight * strong_validation_xgb
            + (1.0 - weight) * transformer_validation
        )
        metrics = metric_block(validation_labels, probability)
        validation_records.append(
            {
                "xgb_weight": weight,
                **metrics,
            }
        )
        validation_probabilities[f"weight_{weight:.2f}"] = probability
        print(
            f"[fusion validation] xgb_weight={weight:.2f} "
            f"AUROC={metrics['auroc']:.4f}",
            flush=True,
        )
    selected = max(validation_records, key=lambda record: record["auroc"])
    selected_weight = float(selected["xgb_weight"])

    np.savez_compressed(
        PREDICTIONS / "mimic4_internal_era_validation_fusion.npz",
        labels=validation_labels,
        stay_id=dataset.cohort["stay_id"].iloc[validation_rows].to_numpy(
            dtype=np.int64
        ),
        transformer=transformer_validation.astype(np.float32),
        strong_xgb=strong_validation_xgb.astype(np.float32),
        **{
            name: probability.astype(np.float32)
            for name, probability in validation_probabilities.items()
        },
    )

    with np.load(
        PREDICTIONS / "mimic4_temporal_holdout_strong.npz",
        allow_pickle=True,
    ) as data:
        labels = data["labels"].astype(int)
        baseline_xgb = data["baseline_xgb"].astype(float)
        strong_xgb = data["selected_xgb"].astype(float)
        transformer = data["transformer"].astype(float)
    locked_probability = (
        selected_weight * strong_xgb + (1.0 - selected_weight) * transformer
    )
    output = {
        "selection": {
            "development_groups": ["2008 - 2010", "2011 - 2013", "2014 - 2016"],
            "validation_groups": ["2017 - 2019"],
            "n": int(len(validation_rows)),
            "events": int(validation_labels.sum()),
            "candidates": validation_records,
            "locked_xgb_weight": selected_weight,
        },
        "final_holdout": {
            "n": int(len(labels)),
            "events": int(labels.sum()),
            "baseline_xgb": metric_block(labels, baseline_xgb),
            "strong_xgb": metric_block(labels, strong_xgb),
            "transformer": metric_block(labels, transformer),
            "locked_fusion": metric_block(labels, locked_probability),
            "locked_fusion_minus_baseline_xgb": bootstrap_paired_auc(
                labels,
                locked_probability,
                baseline_xgb,
                2000,
                2027,
            ),
            "locked_fusion_minus_transformer": bootstrap_paired_auc(
                labels,
                locked_probability,
                transformer,
                2000,
                2028,
            ),
        },
    }
    (RESULTS / "strong_temporal_holdout_fusion.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        PREDICTIONS / "mimic4_temporal_holdout_strong_fusion.npz",
        labels=labels,
        stay_id=np.load(
            PREDICTIONS / "mimic4_temporal_holdout.npz",
            allow_pickle=True,
        )["stay_id"].astype(np.int64),
        baseline_xgb=baseline_xgb.astype(np.float32),
        strong_xgb=strong_xgb.astype(np.float32),
        transformer=transformer.astype(np.float32),
        locked_fusion=locked_probability.astype(np.float32),
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

"""Select a three-model blend on 2017-2019 and test it once in 2020-2022."""
from __future__ import annotations

import json

import numpy as np

from .common import PREDICTIONS, RESULTS, ensure_dirs
from .train_evaluate import bootstrap_paired_auc, metric_block


def main() -> None:
    ensure_dirs()
    with np.load(
        PREDICTIONS / "mimic4_internal_era_validation_fusion.npz",
        allow_pickle=True,
    ) as fusion:
        labels = fusion["labels"].astype(int)
        strong_xgb = fusion["strong_xgb"].astype(float)
        transformer = fusion["transformer"].astype(float)
    with np.load(
        PREDICTIONS / "mimic4_internal_era_validation_strong_tabular.npz",
        allow_pickle=True,
    ) as tabular:
        extra_trees = tabular["extra_trees"].astype(float)

    candidates = []
    for xgb_weight in np.linspace(0, 1, 11):
        for transformer_weight in np.linspace(
            0, 1 - xgb_weight, int(round((1 - xgb_weight) * 10)) + 1
        ):
            extra_weight = 1 - xgb_weight - transformer_weight
            probability = (
                xgb_weight * strong_xgb
                + transformer_weight * transformer
                + extra_weight * extra_trees
            )
            metrics = metric_block(labels, probability)
            candidates.append(
                {
                    "strong_xgb_weight": float(xgb_weight),
                    "transformer_weight": float(transformer_weight),
                    "extra_trees_weight": float(extra_weight),
                    **metrics,
                }
            )
    selected = max(candidates, key=lambda record: record["auroc"])
    print(
        "Selected validation blend: "
        f"XGB={selected['strong_xgb_weight']:.2f}, "
        f"Transformer={selected['transformer_weight']:.2f}, "
        f"ExtraTrees={selected['extra_trees_weight']:.2f}, "
        f"AUROC={selected['auroc']:.4f}",
        flush=True,
    )

    with np.load(
        PREDICTIONS / "mimic4_temporal_holdout_strong_fusion.npz",
        allow_pickle=True,
    ) as fusion:
        final_labels = fusion["labels"].astype(int)
        final_baseline_xgb = fusion["baseline_xgb"].astype(float)
        final_strong_xgb = fusion["strong_xgb"].astype(float)
        final_transformer = fusion["transformer"].astype(float)
        previous_pair = fusion["locked_fusion"].astype(float)
        stay_id = fusion["stay_id"].astype(np.int64)
    with np.load(
        PREDICTIONS / "mimic4_temporal_holdout_strong_tabular.npz",
        allow_pickle=True,
    ) as tabular:
        final_extra_trees = tabular["selected"].astype(float)
    final_probability = (
        selected["strong_xgb_weight"] * final_strong_xgb
        + selected["transformer_weight"] * final_transformer
        + selected["extra_trees_weight"] * final_extra_trees
    )
    output = {
        "validation_n": int(len(labels)),
        "validation_events": int(labels.sum()),
        "selected_weights": {
            "strong_xgb": selected["strong_xgb_weight"],
            "transformer": selected["transformer_weight"],
            "extra_trees": selected["extra_trees_weight"],
        },
        "selected_validation_metrics": selected,
        "final_holdout": {
            "n": int(len(final_labels)),
            "events": int(final_labels.sum()),
            "baseline_xgb": metric_block(final_labels, final_baseline_xgb),
            "previous_pair_fusion": metric_block(final_labels, previous_pair),
            "locked_triple_fusion": metric_block(final_labels, final_probability),
            "locked_triple_minus_baseline_xgb": bootstrap_paired_auc(
                final_labels,
                final_probability,
                final_baseline_xgb,
                2000,
                2040,
            ),
            "locked_triple_minus_previous_pair": bootstrap_paired_auc(
                final_labels,
                final_probability,
                previous_pair,
                2000,
                2041,
            ),
        },
    }
    (RESULTS / "strong_fusion_triple.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        PREDICTIONS / "mimic4_temporal_holdout_strong_triple.npz",
        labels=final_labels,
        stay_id=stay_id,
        baseline_xgb=final_baseline_xgb.astype(np.float32),
        locked_triple=final_probability.astype(np.float32),
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

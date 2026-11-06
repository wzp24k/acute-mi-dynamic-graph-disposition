"""Aggregate multi-seed temporal-regularization anchor estimates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.common import RESULTS  # noqa: E402
from src.ami_dynamic.evaluation import (  # noqa: E402
    bootstrap_metric_ci,
    paired_bootstrap_delta,
    point_metrics,
)


SEEDS = (1, 7, 42, 123, 2024)
ANCHORS = (
    ("No TR", 0.0, "no_tr"),
    ("Scalar TR, lambda_s=0.01", 0.01, "scalar_0p01"),
    (
        "Scalar TR, prespecified lambda_s=0.2",
        0.2,
        "scalar_0p2",
    ),
    ("Scalar TR, lambda_s=1.5", 1.5, "scalar_1p5"),
)


def _prediction_path(seed: int, anchor: str) -> Path:
    if anchor == "no_tr":
        return RESULTS / "predictions" / f"mimic4_dggru_seed{seed}.npz"
    if anchor == "scalar_0p2":
        return RESULTS / "predictions" / f"mimic4_dggru_tr_seed{seed}.npz"
    lambda_name = "0p01" if anchor == "scalar_0p01" else "1p5"
    return (
        RESULTS
        / "sensitivity"
        / "temporal_controls"
        / f"smooth_drift_lambda{lambda_name}_seed{seed}.npz"
    )


def build_frame() -> pd.DataFrame:
    rows = []
    averaged_by_anchor: dict[str, np.ndarray] = {}
    for label, lambda_smooth, anchor in ANCHORS:
        probabilities = []
        seed_aucs = []
        labels_reference = None
        group_reference = None
        for seed in SEEDS:
            path = _prediction_path(seed, anchor)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {path}; run the temporal anchor before aggregation"
                )
            with np.load(path, allow_pickle=True) as payload:
                labels = payload["labels"].astype(int)
                groups = payload["subject_key"].astype(str)
                probability = payload["probabilities"].astype(np.float32)
            if labels_reference is None:
                labels_reference = labels
                group_reference = groups
            elif not (
                np.array_equal(labels, labels_reference)
                and np.array_equal(groups, group_reference)
            ):
                raise ValueError(f"Patient order differs for {path}")
            probabilities.append(probability)
            seed_aucs.append(
                float(point_metrics(labels, probability)["macro_auc"])
            )
        averaged = np.mean(np.stack(probabilities), axis=0)
        averaged_by_anchor[anchor] = averaged
        interval = bootstrap_metric_ci(
            labels_reference,
            averaged,
            metric="macro_auc",
            groups=group_reference,
            replicates=2000,
        )
        rows.append(
            {
                "configuration": label,
                "lambda_smooth": lambda_smooth,
                "seeds": len(SEEDS),
                "macro_auc": interval["estimate"],
                "macro_auc_ci_low": interval["ci_low"],
                "macro_auc_ci_high": interval["ci_high"],
                "seed_macro_auc_mean": float(np.mean(seed_aucs)),
                "seed_macro_auc_sd": float(np.std(seed_aucs, ddof=1)),
            }
        )
    frame = pd.DataFrame(rows)
    no_tr = frame.loc[frame["configuration"].eq("No TR")]
    if len(no_tr) != 1:
        raise ValueError("Expected exactly one No TR temporal anchor")
    reference_probability = averaged_by_anchor["no_tr"]
    deltas = [
        paired_bootstrap_delta(
            labels_reference,
            averaged_by_anchor[anchor],
            reference_probability,
            metric="macro_auc",
            groups=group_reference,
            replicates=2000,
        )
        for anchor in (
            "no_tr",
            "scalar_0p01",
            "scalar_0p2",
            "scalar_1p5",
        )
    ]
    frame["delta_macro_auc_vs_no_tr"] = [
        float(value["delta"]) for value in deltas
    ]
    frame["delta_ci_low"] = [float(value["ci_low"]) for value in deltas]
    frame["delta_ci_high"] = [float(value["ci_high"]) for value in deltas]
    frame["delta_bootstrap_replicates_valid"] = [
        int(value["bootstrap_replicates_valid"]) for value in deltas
    ]
    return frame


def write_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        output_dir / "table4b_multiseed_temporal_anchors.csv",
        index=False,
    )
    display = pd.DataFrame(
        {
            "Configuration": frame["configuration"],
            "Lambda_s": frame["lambda_smooth"],
            "Seeds": frame["seeds"],
            "Seed-averaged macro-AUC (95% CI)": [
                (
                    f"{float(estimate):.4f} "
                    f"({float(low):.4f} to {float(high):.4f})"
                )
                for estimate, low, high in frame[
                    [
                        "macro_auc",
                        "macro_auc_ci_low",
                        "macro_auc_ci_high",
                    ]
                ].itertuples(index=False, name=None)
            ],
            "Delta vs No TR (current minus No TR)": [
                f"{float(delta):+.6f}"
                for delta in frame["delta_macro_auc_vs_no_tr"]
            ],
            "Single-seed macro-AUC, mean (SD)": [
                f"{float(mean):.4f} (SD {float(sd):.4f})"
                for mean, sd in frame[
                    ["seed_macro_auc_mean", "seed_macro_auc_sd"]
                ].itertuples(index=False, name=None)
            ],
        }
    ).astype(str)
    display.to_markdown(
        output_dir / "table4b_multiseed_temporal_anchors.md",
        index=False,
        disable_numparse=True,
    )


def main() -> None:
    frame = build_frame()
    output = RESULTS / "tables"
    write_outputs(frame, output)
    report = {
        "seeds": list(SEEDS),
        "estimand": "macro-AUC of seed-averaged out-of-fold probabilities",
        "delta_definition": (
            "configuration macro_auc minus No TR macro_auc; positive values "
            "mean higher macro-AUC for the configuration. Confidence "
            "intervals use paired patient bootstrap resampling of the "
            "seed-averaged probabilities."
        ),
        "bootstrap": "2000 patient bootstrap replicates",
        "anchors": frame.to_dict(orient="records"),
    }
    (RESULTS / "audits" / "multiseed_temporal_anchors.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

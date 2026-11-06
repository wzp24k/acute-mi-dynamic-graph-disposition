"""Independently verify model contrast directions from cached predictions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ami_dynamic.train_evaluate import (  # noqa: E402
    TABULAR_MODELS,
    TEMPORAL_MODELS,
)

RESULTS = PROJECT_ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
SEEDS = (1, 7, 42, 123, 2024)
CONTRAST_PAIRS = {
    "primary_contrast": ("dggru_tr", "xgb_static_ts"),
    "regularization_contrast": ("dggru_tr", "dggru"),
    "time_series_increment": ("xgb_static_ts", "xgb_static"),
    "time_series_increment_mlp": ("mlp_static_ts", "mlp_static"),
    "regularization_increment": ("dggru_tr", "dggru"),
    "deep_vs_static_ts": ("dggru_tr", "xgb_static_ts"),
    "static_input_dggru_increment": ("dggru", "dggru_static"),
    "cumulative_curvature_increment": ("dggru_ctr", "dggru"),
    "same_static_input_mlp_vs_xgb": ("mlp_static", "xgb_static"),
    "same_full_input_xgb_vs_mlp": ("xgb_static_ts", "mlp_static_ts"),
    "same_full_input_mlp_vs_dggru": ("mlp_static_ts", "dggru"),
    "same_full_input_gru_vs_dggru": ("gru", "dggru"),
}


def _macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    normalized = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0)
    normalized /= normalized.sum(axis=1, keepdims=True)
    classes = np.arange(normalized.shape[1])
    return float(
        roc_auc_score(
            labels,
            normalized,
            multi_class="ovr",
            average="macro",
            labels=classes,
        )
    )


def _load_average(model: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    probabilities = []
    labels = None
    files = []
    for seed in SEEDS:
        path = PREDICTIONS / f"mimic4_{model}_seed{seed}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path.name)
        with np.load(path, allow_pickle=True) as data:
            seed_labels = data["labels"].astype(int)
            seed_probabilities = data["probabilities"]
            if labels is None:
                labels = seed_labels
            elif not np.array_equal(labels, seed_labels):
                raise ValueError(f"{path.name}: labels differ")
        if not np.allclose(seed_probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"{path.name}: probabilities do not sum to one")
        probabilities.append(seed_probabilities)
    if labels is None:
        raise RuntimeError(model)
    return labels, np.mean(np.stack(probabilities), axis=0), files


def main() -> None:
    if TABULAR_MODELS["xgb_static"]["input_kind"] != "static":
        raise ValueError("XGB-static does not use the static input matrix")
    if TABULAR_MODELS["mlp_static"]["input_kind"] != "static":
        raise ValueError("MLP-static does not use the static input matrix")
    if TABULAR_MODELS["xgb_static_ts"]["input_kind"] != "full":
        raise ValueError("XGB-static+TS does not use the static-plus-summary matrix")
    if TABULAR_MODELS["mlp_static_ts"]["input_kind"] != "full":
        raise ValueError("MLP-static+TS does not use the static-plus-summary matrix")
    if TEMPORAL_MODELS["dggru"]["input_kind"] != "full":
        raise ValueError("DG-GRU does not use the full static predictor matrix")
    if TEMPORAL_MODELS["dggru_static"]["input_kind"] != "full":
        raise ValueError(
            "DG-GRU-static does not match the DG-GRU static predictor matrix"
        )
    if TEMPORAL_MODELS["dggru_static"].get("use_sequence", True):
        raise ValueError("DG-GRU-static still receives the raw sequence input")

    model_names = sorted(
        {
            model
            for pair in CONTRAST_PAIRS.values()
            for model in pair
        }
    )
    loaded = {}
    labels = None
    for model in model_names:
        model_labels, probability, files = _load_average(model)
        if labels is None:
            labels = model_labels
        elif not np.array_equal(labels, model_labels):
            raise ValueError(f"{model}: prediction labels differ")
        loaded[model] = {
            "probability": probability,
            "files": files,
            "macro_auc": _macro_auc(labels, probability),
        }
    if labels is None:
        raise RuntimeError("No prediction models were loaded")

    metrics_path = RESULTS / "metrics_mimic4.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    reports = {}
    for key, (minus_model, reference_model) in CONTRAST_PAIRS.items():
        if key not in metrics:
            raise ValueError(f"{key} is missing from the pipeline metrics")
        record = metrics[key]
        if "delta" in record:
            pipeline_delta = float(record["delta"])
        elif "delta_macro_auc" in record:
            pipeline_delta = float(record["delta_macro_auc"]["delta"])
        else:
            raise ValueError(f"{key} does not contain a macro-AUC delta")
        recomputed_delta = (
            loaded[minus_model]["macro_auc"]
            - loaded[reference_model]["macro_auc"]
        )
        if not np.isclose(pipeline_delta, recomputed_delta, atol=1e-9):
            raise ValueError(
                f"{key}: pipeline delta {pipeline_delta} differs from "
                f"recomputed {minus_model} minus {reference_model} "
                f"{recomputed_delta}"
            )
        reports[key] = {
            "definition": f"{minus_model} minus {reference_model}",
            "pipeline_delta": pipeline_delta,
            "recomputed_delta": recomputed_delta,
        }

    threshold_path = (
        RESULTS
        / "ordinal_evaluation"
        / "table10_threshold_calibration.csv"
    )
    dca_path = (
        RESULTS
        / "ordinal_evaluation"
        / "external_dca_curves.csv"
    )
    threshold = pd.read_csv(threshold_path)
    dca = pd.read_csv(dca_path)
    negative_gap_rows = int(
        (
            threshold.loc[
                threshold["record_type"].eq("risk_stratum"),
                "calibration_gap",
            ]
            < 0
        ).sum()
    )
    negative_net_benefit_rows = int((dca["net_benefit"] < 0).sum())
    if negative_gap_rows == 0:
        raise ValueError("Expected negative calibration gaps were not retained")
    if negative_net_benefit_rows == 0:
        raise ValueError("Expected negative net-benefit values were not retained")

    output = {
        "definition": (
            "All contrast values equal the named first model minus the named "
            "second model. Values are not sign-flipped or absolute-transformed."
        ),
        "n": int(len(labels)),
        "seeds": list(SEEDS),
        "factorial_design": {
            "shared_static_feature_contract_mlp_vs_xgb": (
                "xgb_static and mlp_static share the static feature-column "
                "contract but use family-specific preprocessing and heads"
            ),
            "shared_static_plus_summary_contract_xgb_vs_mlp": (
                "xgb_static_ts and mlp_static_ts share the static-plus-summary "
                "feature-column contract but use family-specific preprocessing "
                "and heads"
            ),
            "strict_same_input_gru_vs_dggru": (
                "gru and dggru both receive input_kind=full and the raw "
                "sequence-plus-availability tensor; this is the strict "
                "same-input, different-architecture comparison"
            ),
            "dggru_sequence_toggle": (
                "dggru_static and dggru share input_kind=full; only "
                "dggru_static sets use_sequence=false, providing a "
                "sequence-plus-availability input toggle with fixed network "
                "structure"
            ),
        },
        "cached_files": {
            model: record["files"] for model, record in loaded.items()
        },
        "seed_averaged_macro_auc": {
            model: record["macro_auc"] for model, record in loaded.items()
        },
        "contrasts": reports,
        "negative_value_preservation": {
            "calibration_gap_rows_below_zero": negative_gap_rows,
            "external_net_benefit_rows_below_zero": negative_net_benefit_rows,
            "threshold_csv": str(threshold_path),
            "external_dca_csv": str(dca_path),
        },
        "status": "passed",
    }
    path = RESULTS / "audits" / "sign_convention_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

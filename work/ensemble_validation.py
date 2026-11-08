"""Bootstrap and sensitivity checks for the complementary-model ensemble."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
PRED = ROOT / "results" / "predictions"
RESULTS = ROOT / "results"


def load(path: Path, model: str):
    with np.load(path, allow_pickle=True) as data:
        labels = data["labels"].astype(int)
        values = data[model].astype(float)
    return labels, values


def auc_diff_ci(labels, first, second, seed=2026, replicates=4000):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        index = rng.integers(0, len(labels), len(labels))
        selected = labels[index]
        if len(np.unique(selected)) < 2:
            continue
        values.append(
            roc_auc_score(selected, first[index])
            - roc_auc_score(selected, second[index])
        )
    point = roc_auc_score(labels, first) - roc_auc_score(labels, second)
    return {
        "difference": float(point),
        "ci": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "p_two_sided_bootstrap": float(
            2 * min(np.mean(np.asarray(values) <= 0), np.mean(np.asarray(values) >= 0))
        ),
    }


def main() -> None:
    labels, xgb = load(PRED / "mimic4_cv_xgb_static_ts.npz", "probability")
    _, transformer = load(PRED / "mimic4_cv_transformer.npz", "probability")
    _, gru = load(PRED / "mimic4_cv_gru.npz", "probability")
    _, graph = load(PRED / "mimic4_cv_ct_graph.npz", "probability")

    candidates = {
        "xgb": xgb,
        "transformer": transformer,
        "gru": gru,
        "ct_graph": graph,
        "mean_xgb_transformer": (xgb + transformer) / 2,
        "mean_xgb_transformer_gru": (xgb + transformer + gru) / 3,
        "mean_xgb_transformer_graph": (xgb + transformer + graph) / 3,
        "rank_xgb_transformer": (
            rankdata(xgb) + rankdata(transformer)
        ) / (2 * len(xgb)),
    }
    weights = np.linspace(0, 1, 21)
    for weight in weights:
        candidates[f"weighted_xgb_{weight:.2f}"] = (
            weight * xgb + (1 - weight) * transformer
        )
    metrics = []
    for name, probability in candidates.items():
        metrics.append(
            {
                "name": name,
                "auroc": float(roc_auc_score(labels, probability)),
                "auprc": float(average_precision_score(labels, probability)),
                "brier": float(brier_score_loss(labels, probability)),
            }
        )
    metrics = sorted(metrics, key=lambda item: item["auroc"], reverse=True)
    selected = candidates["mean_xgb_transformer"]
    output = {
        "top_weight_sensitivity": metrics[:15],
        "ensemble_vs_xgb": auc_diff_ci(labels, selected, xgb),
        "ensemble_vs_transformer": auc_diff_ci(
            labels, selected, transformer, seed=2027
        ),
        "external": {},
    }
    for cohort in ("mimic3", "eicu", "eicu_anycad"):
        labels_ext, xgb_ext = load(
            PRED / f"{cohort}_external_xgb_static_ts.npz", "xgb_static_ts"
        )
        _, transformer_ext = load(
            PRED / f"{cohort}_external_transformer.npz", "transformer"
        )
        gru_path = PRED / f"{cohort}_external_gru.npz"
        gru_ext = None
        if gru_path.is_file():
            _, gru_ext = load(gru_path, "gru")
        ensemble_ext = (xgb_ext + transformer_ext) / 2
        cohort_output = {
            "xgb": float(roc_auc_score(labels_ext, xgb_ext)),
            "transformer": float(roc_auc_score(labels_ext, transformer_ext)),
            "gru": (
                float(roc_auc_score(labels_ext, gru_ext))
                if gru_ext is not None
                else None
            ),
            "mean_xgb_transformer": float(
                roc_auc_score(labels_ext, ensemble_ext)
            ),
            "mean_xgb_transformer_vs_xgb": auc_diff_ci(
                labels_ext, ensemble_ext, xgb_ext, seed=2030
            ),
        }
        if gru_ext is not None:
            triple_ext = (xgb_ext + transformer_ext + gru_ext) / 3
            cohort_output["mean_xgb_transformer_gru"] = float(
                roc_auc_score(labels_ext, triple_ext)
            )
        else:
            cohort_output["mean_xgb_transformer_gru"] = None
            cohort_output["note"] = "GRU prediction unavailable; triple average skipped."
        output["external"][cohort] = cohort_output
    (RESULTS / "ensemble_validation.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

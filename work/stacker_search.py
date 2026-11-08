"""Search simple, regularized stacking rules over saved OOF predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
PRED = ROOT / "results" / "predictions"
RESULTS = ROOT / "results"

ALL_MODELS = [
    "xgb_static_ts",
    "transformer",
    "gru",
    "tft",
    "ct_fixed",
    "ct_graph",
    "ctogt",
    "ct_random",
    "ct_reverse",
    "ct_permute",
    "ct_no_time",
]
SELECTED = ["xgb_static_ts", "transformer", "gru", "tft", "ct_fixed", "ct_graph", "ctogt"]


def load_development(models):
    labels = None
    values = {}
    for model in models:
        path = PRED / f"mimic4_cv_{model}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=True) as data:
            current = data["labels"].astype(int)
            if labels is None:
                labels = current
            if np.array_equal(labels, current):
                values[model] = data["probability"].astype(float)
    matrix = np.column_stack(
        [
            np.log(np.clip(values[model], 1e-6, 1 - 1e-6) / (1 - np.clip(values[model], 1e-6, 1 - 1e-6)))
            for model in values
        ]
    )
    return labels, values, matrix


def evaluate(labels, matrix, estimator, name):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    prediction = cross_val_predict(
        estimator,
        matrix,
        labels,
        cv=splitter,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    return prediction, {
        "name": name,
        "auroc": float(roc_auc_score(labels, prediction)),
        "auprc": float(average_precision_score(labels, prediction)),
    }


def external_matrix(models):
    output = {}
    labels_all = None
    matrix_all = None
    for cohort in ("mimic3", "eicu"):
        labels = None
        columns = []
        used = []
        for model in models:
            path = PRED / f"{cohort}_external_{model}.npz"
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=True) as data:
                current = data["labels"].astype(int)
                if labels is None:
                    labels = current
                if not np.array_equal(labels, current):
                    continue
                values = np.clip(data[model].astype(float), 1e-6, 1 - 1e-6)
                columns.append(np.log(values / (1 - values)))
                used.append(model)
        labels_all = labels
        matrix_all = np.column_stack(columns)
        output[cohort] = {
            "labels": labels_all,
            "matrix": matrix_all,
            "models": used,
        }
    return output


def main() -> None:
    outputs = {"development": [], "external": {}}
    estimators = {
        "logit_c0.05": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.05, solver="lbfgs", max_iter=3000),
        ),
        "logit_c0.2": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, solver="lbfgs", max_iter=3000),
        ),
        "logit_c1": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000),
        ),
        "hgb": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=120,
            max_leaf_nodes=7,
            l2_regularization=2.0,
            random_state=2026,
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=40,
            max_features=0.7,
            n_jobs=8,
            random_state=2026,
        ),
        "xgb": XGBClassifier(
            n_estimators=220,
            max_depth=2,
            learning_rate=0.025,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            min_child_weight=10,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=8,
            random_state=2026,
        ),
    }
    development_predictions = {}
    for model_set_name, models in (("all", ALL_MODELS), ("selected", SELECTED)):
        labels, values, matrix = load_development(models)
        for name, estimator in estimators.items():
            prediction, result = evaluate(
                labels, matrix, estimator, f"{model_set_name}:{name}"
            )
            outputs["development"].append(result)
            development_predictions[result["name"]] = prediction
    outputs["development"] = sorted(
        outputs["development"], key=lambda row: row["auroc"], reverse=True
    )

    external_models = SELECTED
    ext = external_matrix(external_models)
    ext_labels, _, ext_matrix = load_development(external_models)
    for estimator_name, estimator in estimators.items():
        outputs["external"][estimator_name] = {}
        for cohort, info in ext.items():
            try:
                fitted = estimator
                fitted.fit(ext_matrix, ext_labels)
                probability = fitted.predict_proba(info["matrix"])[:, 1]
                outputs["external"][estimator_name][cohort] = {
                    "auroc": float(roc_auc_score(info["labels"], probability)),
                    "auprc": float(
                        average_precision_score(info["labels"], probability)
                    ),
                }
            except Exception as error:
                outputs["external"][estimator_name][cohort] = {
                    "error": str(error)
                }
    (RESULTS / "stacker_search.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8"
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()

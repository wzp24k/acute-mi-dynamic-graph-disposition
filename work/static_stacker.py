"""Test whether static context adds to temporal probability stacking."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
PRED = ROOT / "results" / "predictions"


def main() -> None:
    cohort = pd.read_csv(ROOT / "data" / "processed" / "mimic4" / "cohort.csv")
    paths = [
        PRED / "mimic4_cv_xgb_static_ts.npz",
        PRED / "mimic4_cv_transformer.npz",
        PRED / "mimic4_cv_gru.npz",
        PRED / "mimic4_cv_ct_graph.npz",
    ]
    labels = None
    columns = []
    for path in paths:
        data = np.load(path)
        labels = data["labels"].astype(int)
        p = np.clip(data["probability"], 1e-6, 1 - 1e-6)
        columns.append(np.log(p / (1 - p)))
    static_columns = [
        "age",
        "sex_male",
        "prior_mi",
        "prior_stroke",
        "heart_failure",
        "atrial_fib",
        "diabetes",
        "hypertension",
        "ckd",
        "copd",
        "pad",
        "comorbidity_burden",
        "total_missing_channels",
    ]
    static = cohort[static_columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    for group_name, group in (
        ("probability_only", np.column_stack(columns)),
        ("probability_plus_static", np.column_stack([*columns, static])),
    ):
        for c in (0.03, 0.1, 0.3, 1.0):
            model = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                LogisticRegression(C=c, max_iter=4000),
            )
            prediction = cross_val_predict(
                model,
                group,
                labels,
                cv=StratifiedKFold(5, shuffle=True, random_state=2026),
                method="predict_proba",
                n_jobs=1,
            )[:, 1]
            print(group_name, c, roc_auc_score(labels, prediction))


if __name__ == "__main__":
    main()

"""Score completed two-reader endpoint adjudication and timing agreement."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ctogt.common import RESULTS


LABELS = (
    "confirmed_target",
    "probable_target",
    "pre_landmark",
    "after_horizon",
    "historical_or_not_event",
    "uncertain_or_insufficient",
)


def main() -> None:
    path = RESULTS / "manual_endpoint_review_structured.csv"
    review = pd.read_csv(path)
    required = ["reader1_label", "reader2_label", "consensus_label"]
    reviewable = review["review_status"].eq(
        "pending_independent_clinical_review"
    )
    source_unavailable = review["review_label"].eq("source_text_unavailable")
    complete_reviewable = review[required].notna().all(axis=1)
    for column in required:
        complete_reviewable &= review[column].astype(str).str.strip().ne("")
    scored = review[reviewable & complete_reviewable].copy()
    output = {
        "status": (
            "complete"
            if len(scored) == int(reviewable.sum())
            else "pending"
        ),
        "review_rows": int(len(review)),
        "reviewable_rows": int(reviewable.sum()),
        "source_text_unavailable_rows": int(source_unavailable.sum()),
        "completed_rows": int(len(scored)),
        "pending_rows": int(reviewable.sum() - len(scored)),
    }
    if scored.empty:
        output["note"] = (
            "No human two-reader labels are available yet. PPV, timing error, "
            "and inter-reader agreement are not estimable and are not imputed."
        )
    else:
        output["cohen_kappa"] = float(
            cohen_kappa_score(
                scored["reader1_label"],
                scored["reader2_label"],
                labels=list(LABELS),
            )
        )
        reference = scored["consensus_label"].isin(
            ["confirmed_target", "probable_target"]
        )
        index_positive = scored["review_status"].eq(
            "pending_independent_clinical_review"
        ) | scored["review_status"].eq("adjudicated")
        tp = int((reference & index_positive).sum())
        fp = int((~reference & index_positive).sum())
        output["ppv"] = float(tp / (tp + fp)) if tp + fp else np.nan
        output["time_window_agreement"] = float(
            scored["consensus_onset_window"]
            .eq(
                np.where(
                    scored["candidate_interval_start_hours"].ge(24)
                    & scored["candidate_interval_end_hours"].le(168),
                    "24h_to_day7",
                    "other",
                )
            )
            .mean()
        )
    (RESULTS / "manual_endpoint_review_scored.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

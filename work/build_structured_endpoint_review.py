"""Add a two-reader clinical adjudication schema to the endpoint review pack."""
from __future__ import annotations

import json

import pandas as pd

from ctogt.common import RESULTS


CHECKLIST_COLUMNS = (
    "reader1_label",
    "reader1_onset_window",
    "reader1_troponin_dynamic",
    "reader1_ecg_evidence",
    "reader1_angiography_or_pci",
    "reader1_stroke_imaging",
    "reader1_focal_deficit",
    "reader1_clinical_record_support",
    "reader1_confidence",
    "reader2_label",
    "reader2_onset_window",
    "reader2_troponin_dynamic",
    "reader2_ecg_evidence",
    "reader2_angiography_or_pci",
    "reader2_stroke_imaging",
    "reader2_focal_deficit",
    "reader2_clinical_record_support",
    "reader2_confidence",
    "consensus_label",
    "consensus_onset_window",
    "adjudication_notes",
)


def main() -> None:
    review = pd.read_csv(RESULTS / "manual_endpoint_review.csv")
    for column in CHECKLIST_COLUMNS:
        if column not in review:
            review[column] = ""
    review["adjudication_status"] = "pending_two_independent_clinical_readers"
    output = RESULTS / "manual_endpoint_review_structured.csv"
    review.to_csv(output, index=False)
    protocol = {
        "reviewable_records": int(
            review["review_status"].eq("pending_independent_clinical_review").sum()
        ),
        "source_text_unavailable": int(
            review["review_status"].eq("completed").sum()
        ),
        "reader_labels": [
            "confirmed_target",
            "probable_target",
            "pre_landmark",
            "after_horizon",
            "historical_or_not_event",
            "uncertain_or_insufficient",
        ],
        "required_ami_evidence": [
            "serial troponin rise or fall",
            "ECG ischemic change",
            "coronary angiography or PCI",
            "clinical record support",
        ],
        "required_stroke_evidence": [
            "acute imaging finding",
            "focal neurologic deficit",
            "clinical record support",
        ],
        "onset_windows": [
            "before_24h",
            "24h_to_day7",
            "after_day7",
            "uncertain",
        ],
        "agreement": "Cohen kappa will be computed after both readers finish.",
        "ppv": (
            "PPV will use the two-reader consensus target-window label as the "
            "reference and the structured candidate endpoint as the index test."
        ),
    }
    (RESULTS / "manual_endpoint_review_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()

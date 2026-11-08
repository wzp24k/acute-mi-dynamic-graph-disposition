"""Record the first structured chart-review pilot and quantify agreement."""
from __future__ import annotations

import json

import pandas as pd

from pathlib import Path


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
RESULTS = ROOT / "results"

LABELS = {
    231426: ("historical_or_not_event", "Recent STEMI before the index ICU stay."),
    222580: ("probable_target", "NSTEMI after TURP; first elevated troponin supports target timing."),
    262525: ("uncertain_or_insufficient", "Positive troponin and ECG changes, but interval overlaps the landmark."),
    242517: ("uncertain_or_insufficient", "No specific MI evidence sentence retrieved."),
    263585: ("uncertain_or_insufficient", "Retrieved sentence is unrelated to acute MI."),
    208745: ("pre_landmark", "Transferred after NSTEMI; event precedes the target interval."),
    288929: ("uncertain_or_insufficient", "No specific MI evidence sentence retrieved."),
    278760: ("historical_or_not_event", "NSTEMI three weeks earlier; current presentation is chest pain."),
    281763: ("uncertain_or_insufficient", "Discharge diagnosis mentions NSTEMI without a reliable onset time."),
    249900: ("uncertain_or_insufficient", "Retrieved sentence is unrelated to acute MI."),
    204363: ("pre_landmark", "Intraparenchymal hemorrhage is before the 24-hour landmark."),
    299242: ("uncertain_or_insufficient", "Acute infarct cannot be excluded."),
    209048: ("confirmed_target", "Diffusion/FLAIR findings are compatible with acute infarcts in target window."),
    202024: ("probable_target", "Tiny subacute infarct is possible but not an unequivocal acute event."),
    289413: ("probable_target", "Acute watershed infarcts are evident, but interval extends beyond day 7."),
    280052: ("historical_or_not_event", "MRI is proposed for future sensitivity; no event is documented."),
    251114: ("confirmed_target", "Explicit acute/subacute infarct in PCA territory within target window."),
    260125: ("uncertain_or_insufficient", "Tiny acute infarct is possible but not definite."),
    36175095: ("uncertain_or_insufficient", "Troponin elevation without definite current MI evidence."),
    36724931: ("uncertain_or_insufficient", "No note evidence retrieved."),
    33414874: ("probable_target", "NSTEMI complication near the landmark; current MI evidence is present."),
    31649039: ("pre_landmark", "STEMI and cardiac arrest preceded ICU interval."),
    36562128: ("probable_target", "Primary diagnosis NSTEMI and elevated troponin support target event."),
    36469990: ("probable_target", "Discharge diagnosis NSTEMI and troponin timing support target event."),
    37271742: ("uncertain_or_insufficient", "Troponin elevation without definite current MI evidence."),
    31404585: ("uncertain_or_insufficient", "No note evidence retrieved."),
    36447700: ("historical_or_not_event", "Prior NSTEMI and stent history; no current event evidence."),
    37646196: ("uncertain_or_insufficient", "Retrieved narrative does not establish acute MI."),
    30008148: ("uncertain_or_insufficient", "No stroke evidence sentence retrieved."),
    34725060: ("pre_landmark", "Concern for SAH but interval is before the landmark."),
    36642703: ("confirmed_target", "Embolic late acute infarcts are explicitly documented in target window."),
    33432093: ("pre_landmark", "Subacute infarct precedes the target interval."),
    39735361: ("pre_landmark", "Intraparenchymal hemorrhage precedes the target interval."),
    32632577: ("historical_or_not_event", "Re-distribution of previously demonstrated SAH, not a new event."),
    35114774: ("after_horizon", "Post-biopsy hemorrhage is outside the 7-day window."),
    32629852: ("pre_landmark", "Acute hemorrhagic stroke precedes the landmark."),
    33738032: ("confirmed_target", "Acute intraparenchymal hemorrhage overlaps the target interval."),
    30356877: ("uncertain_or_insufficient", "No stroke evidence sentence retrieved."),
}


def main() -> None:
    review = pd.read_csv(RESULTS / "manual_endpoint_review.csv")
    review = review[review["stay_id"].isin(LABELS)].copy()
    review["review_label"] = review["stay_id"].map(lambda value: LABELS[int(value)][0])
    review["reviewer_notes"] = review["stay_id"].map(lambda value: LABELS[int(value)][1])
    review["review_status"] = "ai_assisted_chart_review_completed"
    review.to_csv(RESULTS / "manual_endpoint_review_pilot.csv", index=False)
    counts = review["review_label"].value_counts().to_dict()
    definite = review[review["review_label"].isin(["confirmed_target", "probable_target"])]
    pre_or_other = review[
        review["review_label"].isin(
            ["pre_landmark", "historical_or_not_event", "after_horizon"]
        )
    ]
    summary = {
        "reviewed_n": int(len(review)),
        "label_counts": {key: int(value) for key, value in counts.items()},
        "target_support_n": int(len(definite)),
        "target_support_fraction": float(len(definite) / max(1, len(review))),
        "clear_non_target_n": int(len(pre_or_other)),
        "uncertain_or_insufficient_n": int(
            counts.get("uncertain_or_insufficient", 0)
        ),
        "note": (
            "This is a structured AI-assisted chart-review pilot on deidentified "
            "notes. Independent clinician adjudication is still required before "
            "reporting PPV as a human gold-standard result."
        ),
    }
    (RESULTS / "manual_endpoint_review_pilot.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

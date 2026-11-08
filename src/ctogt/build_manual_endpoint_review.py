"""Create a source-linked endpoint review packet for human adjudication."""
from __future__ import annotations

import json
import re

import duckdb
import numpy as np
import pandas as pd

from .common import EICU, MIMIC3, MIMIC4, PROCESSED, RESULTS, ensure_dirs


MI_TERMS = (
    "acute myocardial infarction",
    "acute mi",
    "nstemi",
    "stemi",
)
STROKE_TERMS = (
    "acute ischemic stroke",
    "ischemic stroke",
    "hemorrhagic stroke",
    "intracerebral hemorrhage",
    "intraparenchymal hemorrhage",
    "subarachnoid hemorrhage",
    "acute infarct",
)
NEGATIONS = ("no ", "without", "history of", "prior", "rule out", "negative")


def sentence_with_terms(text: str, terms: tuple[str, ...]) -> str:
    cleaned = str(text).replace("\n", " ")
    for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
        lowered = sentence.lower()
        if any(term in lowered for term in terms) and not any(
            negation in lowered for negation in NEGATIONS
        ):
            return sentence[:500]
    return ""


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(2026)
    frames = []

    m4 = pd.read_csv(PROCESSED / "mimic4" / "interval_endpoint.csv")
    m4_candidates = m4[
        m4["event_interval_support"].fillna("").str.contains("troponin|note")
    ].copy()
    if len(m4_candidates):
        sample = m4_candidates.iloc[
            rng.choice(len(m4_candidates), min(40, len(m4_candidates)), replace=False)
        ]
        frames.append(
            pd.DataFrame(
                {
                    "database": "mimic4",
                    "stay_id": sample["stay_id"].astype(int),
                    "component": "AMI",
                    "candidate_interval_start_hours": sample["interval_lower_hours"],
                    "candidate_interval_end_hours": sample["interval_upper_hours"],
                    "evidence_type": sample["event_interval_support"],
                    "evidence_text": "",
                    "review_label": "",
                    "reviewer_notes": "",
                }
            )
        )
    m4_stroke = m4[
        m4["event_interval_support"].fillna("").str.contains("radiology")
    ].copy()
    if len(m4_stroke):
        sample = m4_stroke.iloc[
            rng.choice(len(m4_stroke), min(40, len(m4_stroke)), replace=False)
        ]
        frames.append(
            pd.DataFrame(
                {
                    "database": "mimic4",
                    "stay_id": sample["stay_id"].astype(int),
                    "component": "stroke",
                    "candidate_interval_start_hours": sample["interval_lower_hours"],
                    "candidate_interval_end_hours": sample["interval_upper_hours"],
                    "evidence_type": sample["event_interval_support"],
                    "evidence_text": "",
                    "review_label": "",
                    "reviewer_notes": "",
                }
            )
        )

    m3 = pd.read_csv(PROCESSED / "mimic3" / "interval_endpoint.csv")
    m3_candidates = m3[
        m3["event_interval_support"].fillna("").str.contains("timed_note|troponin")
    ].copy()
    if len(m3_candidates):
        sample = m3_candidates.iloc[
            rng.choice(len(m3_candidates), min(20, len(m3_candidates)), replace=False)
        ]
        frames.append(
            pd.DataFrame(
                {
                    "database": "mimic3",
                    "stay_id": sample["stay_id"].astype(int),
                    "component": np.where(
                        sample["event_interval_support"].str.contains("stroke"),
                        "stroke",
                        "AMI",
                    ),
                    "candidate_interval_start_hours": sample["interval_lower_hours"],
                    "candidate_interval_end_hours": sample["interval_upper_hours"],
                    "evidence_type": sample["event_interval_support"],
                    "evidence_text": "",
                    "review_label": "",
                    "reviewer_notes": "",
                }
            )
        )

    eicu = pd.read_csv(PROCESSED / "eicu" / "interval_endpoint.csv")
    eicu_candidates = eicu[eicu["event_interval_support"].notna()].copy()
    if len(eicu_candidates):
        sample = eicu_candidates.iloc[
            rng.choice(len(eicu_candidates), min(20, len(eicu_candidates)), replace=False)
        ]
        frames.append(
            pd.DataFrame(
                {
                    "database": "eicu",
                    "stay_id": sample["stay_id"].astype(int),
                    "component": "AMI/stroke",
                    "candidate_interval_start_hours": sample["interval_lower_hours"],
                    "candidate_interval_end_hours": sample["interval_upper_hours"],
                    "evidence_type": sample["event_interval_support"],
                    "evidence_text": "eICU source text unavailable",
                    "review_label": "source_text_unavailable",
                    "reviewer_notes": (
                        "Current extract has diagnosisoffset but no note, "
                        "physicalExam, or pastHistory table."
                    ),
                }
            )
        )

    review = pd.concat(frames, ignore_index=True)
    con = duckdb.connect()
    con.execute("SET threads=8")
    m4_ids = review.loc[review["database"].eq("mimic4"), ["stay_id"]]
    m3_ids = review.loc[review["database"].eq("mimic3"), ["stay_id"]]
    if len(m4_ids):
        cohort4 = pd.read_csv(PROCESSED / "mimic4" / "cohort.csv", usecols=["stay_id", "hadm_id", "intime"])
        cohort4 = cohort4.merge(m4_ids, on="stay_id")
        con.register("review4", cohort4)
        discharge = MIMIC4 / "mimic iv note 2.2/note/discharge.csv.gz"
        radiology = MIMIC4 / "mimic iv note 2.2/note/radiology.csv.gz"
        con.execute(
            f"CREATE VIEW d4 AS SELECT * FROM read_csv_auto('{discharge.as_posix()}', header=true)"
        )
        con.execute(
            f"CREATE VIEW r4 AS SELECT * FROM read_csv_auto('{radiology.as_posix()}', header=true)"
        )
        notes = con.execute(
            """
            SELECT c.stay_id, 'discharge' AS source, n.charttime, n.text
            FROM d4 n JOIN review4 c USING (hadm_id)
            UNION ALL
            SELECT c.stay_id, 'radiology' AS source, n.charttime, n.text
            FROM r4 n JOIN review4 c USING (hadm_id)
            """
        ).fetchdf()
        notes.columns = [str(column).lower() for column in notes.columns]
        for index, row in review[review["database"].eq("mimic4")].iterrows():
            subset = notes[notes["stay_id"].eq(row["stay_id"])]
            terms = STROKE_TERMS if row["component"] == "stroke" else MI_TERMS
            snippets = [
                sentence_with_terms(text, terms) for text in subset["text"].tolist()
            ]
            snippets = [snippet for snippet in snippets if snippet]
            review.at[index, "evidence_text"] = snippets[0] if snippets else ""
    if len(m3_ids):
        cohort3 = pd.read_csv(PROCESSED / "mimic3" / "cohort.csv", usecols=["stay_id", "hadm_id", "intime"])
        cohort3 = cohort3.merge(m3_ids, on="stay_id")
        con.register("review3", cohort3)
        notes3 = MIMIC3 / "NOTEEVENTS.csv.gz"
        con.execute(
            f"CREATE VIEW n3 AS SELECT * FROM read_csv_auto('{notes3.as_posix()}', header=true)"
        )
        notes = con.execute(
            """
            SELECT c.stay_id, n.category AS source, n.charttime, n.chartdate, n.text
            FROM n3 n JOIN review3 c USING (hadm_id)
            """
        ).fetchdf()
        notes.columns = [str(column).lower() for column in notes.columns]
        for index, row in review[review["database"].eq("mimic3")].iterrows():
            subset = notes[notes["stay_id"].eq(row["stay_id"])]
            terms = STROKE_TERMS if row["component"] == "stroke" else MI_TERMS
            snippets = [
                sentence_with_terms(text, terms) for text in subset["text"].tolist()
            ]
            snippets = [snippet for snippet in snippets if snippet]
            review.at[index, "evidence_text"] = snippets[0] if snippets else ""
    review["review_status"] = np.where(
        review["review_label"].eq(""),
        "pending_independent_clinical_review",
        "completed",
    )
    review.to_csv(RESULTS / "manual_endpoint_review.csv", index=False)
    summary = {
        "rows": int(len(review)),
        "pending": int(review["review_status"].eq("pending_independent_clinical_review").sum()),
        "eicu_source_text_unavailable": int(
            review["review_label"].eq("source_text_unavailable").sum()
        ),
        "with_evidence_snippet": int(review["evidence_text"].str.len().gt(0).sum()),
    }
    (RESULTS / "manual_endpoint_review_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

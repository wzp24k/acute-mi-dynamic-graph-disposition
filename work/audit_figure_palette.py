"""Audit panel-level cohort colors and the revised manuscript figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from docx import Document
from PIL import Image


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")
FIGURE_DIR = ROOT / "results" / "figures_r"
PALETTE = {
    "MIMIC-IV deep blue": (31, 78, 121),
    "MIMIC-III light blue": (140, 184, 211),
    "eICU pink": (232, 138, 163),
}

images = {}
crops = {}
for path in sorted(FIGURE_DIR.glob("figure*.png")):
    array = np.asarray(Image.open(path).convert("RGB"))
    height, width = array.shape[:2]
    images[path.name] = {
        "palette_pixels": {
            name: int(np.all(array == colour, axis=2).sum())
            for name, colour in PALETTE.items()
        },
    }
    crops[path.name] = {
        "figure1a": array[
            int(0.05 * height):int(0.55 * height),
            :,
        ],
        "figure5d": array[
            int(0.50 * height):,
            int(0.50 * width):,
        ],
        "figure7d": array[
            int(0.55 * height):int(0.95 * height),
            int(0.50 * width):,
        ],
    }


def exact_colour_count(image: np.ndarray, colour: tuple[int, int, int]) -> int:
    return int(np.all(image == colour, axis=2).sum())

script = (ROOT / "work" / "replot_figures.R").read_text(encoding="utf-8")
figure1_svg = (FIGURE_DIR / "figure1_overall_roadmap.svg").read_text(
    encoding="utf-8"
)
document = Document(ROOT / "1108manuscript.docx")
text = "\n".join(paragraph.text for paragraph in document.paragraphs)
checks = {
    "no_legacy_orange_palette": not any(
        colour in script for colour in ("#B05A4A", "#E3B6AD")
    ),
    "pure_black_borders": (
        "#303030" not in script
        and script.count("geom_tile(colour = ink") >= 3
    ),
    "figure1_editable_svg": (
        "<text" in figure1_svg
        and "<rect" in figure1_svg
        and "<image" not in figure1_svg
        and "viewBox=" in figure1_svg
    ),
    "figure1_merged_layout": (
        "(A) Integrated cohort construction" in text
        and "(B) Recorded outcome components" in text
        and "(C) Model architecture" not in text
        and "(D) Temporal selection" not in text
    ),
    "cohort_palette_used": all(
        any(record["palette_pixels"][name] > 0 for record in images.values())
        for name in PALETTE
    ),
    "figure5d_incremental_value": (
        "(D) Incremental AUROC difference versus XGB+TS" in text
    ),
    "figure8c_invariance": (
        "(C) eICU documentation-lag invariance" in text
    ),
    "figure1a_database_colors": all(
        exact_colour_count(crops["figure1_overall_roadmap.png"]["figure1a"], colour)
        > 2_000
        for colour in PALETTE.values()
    ),
    "figure5d_diverging_colors": (
        exact_colour_count(
            crops["figure5_model_selection.png"]["figure5d"],
            PALETTE["MIMIC-IV deep blue"],
        ) > 500
        and exact_colour_count(
            crops["figure5_model_selection.png"]["figure5d"],
            PALETTE["eICU pink"],
        ) > 500
    ),
    "figure7d_database_colors": (
        exact_colour_count(
            crops["figure7_risk_and_decisions.png"]["figure7d"],
            PALETTE["MIMIC-III light blue"],
        ) > 100
        and exact_colour_count(
            crops["figure7_risk_and_decisions.png"]["figure7d"],
            PALETTE["eICU pink"],
        ) > 100
    ),
}
report = {"checks": checks, "images": images}
report["status"] = "PASS" if all(checks.values()) else "FAIL"
(ROOT / "results" / "figure_palette_audit.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)
print(json.dumps(report, indent=2))

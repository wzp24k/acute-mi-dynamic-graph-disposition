"""Audit cohort colors, Figure 5D caption, and absence of orange in figures."""
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
    "eICU pink": (214, 107, 160),
}

images = {}
for path in sorted(FIGURE_DIR.glob("figure*.png")):
    array = np.asarray(Image.open(path).convert("RGB"))
    images[path.name] = {
        "palette_pixels": {
            name: int(np.all(array == colour, axis=2).sum())
            for name, colour in PALETTE.items()
        },
    }

script = (ROOT / "work" / "replot_figures.R").read_text(encoding="utf-8")
document = Document(ROOT / "1108manuscript.docx")
text = "\n".join(paragraph.text for paragraph in document.paragraphs)
checks = {
    "no_legacy_orange_palette": not any(
        colour in script for colour in ("#B05A4A", "#E3B6AD")
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
}
report = {"checks": checks, "images": images}
report["status"] = "PASS" if all(checks.values()) else "FAIL"
(ROOT / "results" / "figure_palette_audit.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)
print(json.dumps(report, indent=2))

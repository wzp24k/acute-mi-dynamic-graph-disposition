"""Print a compact paragraph-and-table inventory of the reference DOCX."""
from __future__ import annotations

import json
from pathlib import Path


data = json.loads(Path("1.format.json").read_text(encoding="utf-8"))
for paragraph in data["paragraphs"]:
    first = paragraph["runs"][0] if paragraph["runs"] else {}
    print(
        f"{paragraph['index']:3d} "
        f"{paragraph['style']:<8} "
        f"size={first.get('size_pt')} "
        f"bold={first.get('bold')} "
        f"align={paragraph['alignment']} "
        f"text={paragraph['text'][:160]!r}"
    )
print("\nTABLES")
for table in data["tables"]:
    print(table)

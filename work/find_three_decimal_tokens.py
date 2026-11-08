"""List remaining values with three or more decimal places."""
from __future__ import annotations

import re
from collections import Counter

from docx import Document


document = Document("1108manuscript.docx")
texts = [paragraph.text for paragraph in document.paragraphs]
texts.extend(
    cell.text
    for table in document.tables
    for row in table.rows
    for cell in row.cells
)
values = []
for text in texts:
    values.extend(
        match.group(0)
        for match in re.finditer(r"(?<![\d.])\d+\.\d{3,}(?![\d.])", text)
    )
print(Counter(values))
for text in texts:
    if re.search(r"(?<![\d.])\d+\.\d{3,}(?![\d.])", text):
        print(repr(text[:300]))

"""Dump paragraphs and tables from the paper requirements DOCX."""
from __future__ import annotations

from docx import Document


document = Document("paper word.docx")
for index, paragraph in enumerate(document.paragraphs):
    if paragraph.text.strip():
        print(f"{index:03d} [{paragraph.style.name}] {paragraph.text}")
print(f"TABLES {len(document.tables)}")
for table_index, table in enumerate(document.tables):
    print(f"TABLE {table_index}")
    for row in table.rows:
        print(" | ".join(cell.text for cell in row.cells))

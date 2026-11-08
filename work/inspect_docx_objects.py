"""Inspect inline figures, table typography, and section XML details."""
from __future__ import annotations

from docx import Document


document = Document("1.docx")
print("INLINE_SHAPES")
for index, shape in enumerate(document.inline_shapes):
    print(index, round(shape.width.inches, 3), round(shape.height.inches, 3))

print("DRAWING_PARAGRAPHS")
for index, paragraph in enumerate(document.paragraphs):
    if paragraph._p.xpath(".//w:drawing"):
        print(index, paragraph.alignment, repr(paragraph.text[:200]))

print("TABLE_TYPOGRAPHY")
for table_index in (0, 2, 5):
    table = document.tables[table_index]
    print("TABLE", table_index, table.alignment, table.autofit)
    for row_index in (0, 1):
        for column_index in (0, min(1, len(table.columns) - 1)):
            paragraph = table.cell(row_index, column_index).paragraphs[0]
            run = paragraph.runs[0] if paragraph.runs else None
            print(
                table_index,
                row_index,
                column_index,
                paragraph.style.name,
                paragraph.alignment,
                paragraph.paragraph_format.line_spacing,
                paragraph.paragraph_format.space_before,
                paragraph.paragraph_format.space_after,
                run.font.name if run else None,
                run.font.size.pt if run and run.font.size else None,
                run.bold if run else None,
            )

print("SECTPR")
print(document.sections[0]._sectPr.xml)

print("NORMAL_STYLE")
print(document.styles["Normal"]._element.xml)

print("REFENTRY_STYLE")
print(document.styles["RefEntry"]._element.xml)

"""Inspect the structural and visual formatting of the reference DOCX."""
from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


def run_format(run) -> dict[str, object]:
    rpr = run._element.rPr
    east_asia = None
    if rpr is not None and rpr.rFonts is not None:
        east_asia = rpr.rFonts.get(qn("w:eastAsia"))
    return {
        "text": run.text[:120],
        "font": run.font.name,
        "east_asia_font": east_asia,
        "size_pt": run.font.size.pt if run.font.size else None,
        "bold": run.bold,
        "italic": run.italic,
        "underline": run.underline,
        "color": str(run.font.color.rgb) if run.font.color.rgb else None,
        "all_caps": run.font.all_caps,
        "small_caps": run.font.small_caps,
    }


def paragraph_record(paragraph, index: int) -> dict[str, object]:
    fmt = paragraph.paragraph_format
    section = paragraph.part.document.sections[-1]
    return {
        "index": index,
        "style": paragraph.style.name if paragraph.style else None,
        "text": paragraph.text[:300],
        "alignment": str(paragraph.alignment) if paragraph.alignment else None,
        "space_before_pt": (
            fmt.space_before.pt if fmt.space_before is not None else None
        ),
        "space_after_pt": (
            fmt.space_after.pt if fmt.space_after is not None else None
        ),
        "line_spacing": fmt.line_spacing,
        "line_spacing_rule": (
            str(fmt.line_spacing_rule) if fmt.line_spacing_rule else None
        ),
        "left_indent_pt": fmt.left_indent.pt if fmt.left_indent else None,
        "right_indent_pt": fmt.right_indent.pt if fmt.right_indent else None,
        "first_line_indent_pt": (
            fmt.first_line_indent.pt if fmt.first_line_indent else None
        ),
        "keep_with_next": fmt.keep_with_next,
        "page_break_before": fmt.page_break_before,
        "runs": [run_format(run) for run in paragraph.runs[:8]],
        "default_section": section.start_type,
    }


def table_record(table, index: int) -> dict[str, object]:
    widths = []
    for cell in table.rows[0].cells:
        widths.append(cell.width.inches if cell.width else None)
    return {
        "index": index,
        "style": table.style.name if table.style else None,
        "rows": len(table.rows),
        "columns": len(table.columns),
        "first_row": [cell.text[:120] for cell in table.rows[0].cells],
        "column_widths_inches": widths,
    }


def main() -> None:
    path = Path(sys.argv[1])
    document = Document(path)
    output: dict[str, object] = {
        "path": str(path),
        "sections": [],
        "paragraphs": [],
        "tables": [],
        "styles": {},
        "package_parts": [],
    }
    for section in document.sections:
        header_text = " | ".join(
            paragraph.text for paragraph in section.header.paragraphs
        )
        footer_text = " | ".join(
            paragraph.text for paragraph in section.footer.paragraphs
        )
        output["sections"].append(
            {
                "page_width_inches": section.page_width.inches,
                "page_height_inches": section.page_height.inches,
                "orientation": str(section.orientation),
                "left_margin_inches": section.left_margin.inches,
                "right_margin_inches": section.right_margin.inches,
                "top_margin_inches": section.top_margin.inches,
                "bottom_margin_inches": section.bottom_margin.inches,
                "header_distance_inches": section.header_distance.inches,
                "footer_distance_inches": section.footer_distance.inches,
                "different_first_page_header_footer": (
                    section.different_first_page_header_footer
                ),
                "header_text": header_text,
                "footer_text": footer_text,
            }
        )
    output["paragraphs"] = [
        paragraph_record(paragraph, index)
        for index, paragraph in enumerate(document.paragraphs)
        if paragraph.text.strip()
    ]
    output["tables"] = [
        table_record(table, index)
        for index, table in enumerate(document.tables)
    ]
    style_counts = Counter(
        paragraph.style.name for paragraph in document.paragraphs
        if paragraph.text.strip()
    )
    output["styles"]["paragraph_style_counts"] = dict(style_counts)
    style_samples = {}
    for style in document.styles:
        if style.type is None or style.name not in style_counts:
            continue
        font = getattr(style, "font", None)
        paragraph_format = getattr(style, "paragraph_format", None)
        style_samples[style.name] = {
            "type": str(style.type),
            "font": font.name if font else None,
            "size_pt": font.size.pt if font and font.size else None,
            "bold": font.bold if font else None,
            "italic": font.italic if font else None,
            "space_before_pt": (
                paragraph_format.space_before.pt
                if paragraph_format and paragraph_format.space_before
                else None
            ),
            "space_after_pt": (
                paragraph_format.space_after.pt
                if paragraph_format and paragraph_format.space_after
                else None
            ),
            "line_spacing": (
                paragraph_format.line_spacing if paragraph_format else None
            ),
        }
    output["styles"]["definitions"] = style_samples
    with zipfile.ZipFile(path) as package:
        output["package_parts"] = [
            {
                "name": info.filename,
                "size": info.file_size,
            }
            for info in package.infolist()
        ]
    output_path = path.with_suffix(".format.json")
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output_path)
    print(
        json.dumps(
            {
                "sections": output["sections"],
                "paragraph_style_counts": style_counts,
                "tables": output["tables"],
                "first_paragraphs": output["paragraphs"][:35],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

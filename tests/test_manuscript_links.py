"""Contract tests for in-text citation cross-references."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


ROOT = Path(r"D:\TT paper\1107Invariant Continuous-Time")


def test_citation_numbers_link_to_all_reference_bookmarks() -> None:
    document = Document(ROOT / "1108manuscript.docx")
    bookmarks = {
        element.get(qn("w:name"))
        for element in document.element.body.iter(qn("w:bookmarkStart"))
        if (element.get(qn("w:name")) or "").startswith("Ref")
    }
    anchors = {
        element.get(qn("w:anchor"))
        for element in document.element.body.iter(qn("w:hyperlink"))
        if element.get(qn("w:anchor"))
    }
    expected = {f"Ref{index}" for index in range(1, 46)}
    assert bookmarks == expected
    assert anchors == expected


def test_citation_groups_are_numerically_ascending() -> None:
    document = Document(ROOT / "1108manuscript.docx")
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    groups = re.findall(r"\[([0-9,\-]+)\]", text)
    assert groups
    assert all(
        [int(token.split("-", 1)[0]) for token in group.split(",")]
        == sorted(int(token.split("-", 1)[0]) for token in group.split(","))
        for group in groups
    )

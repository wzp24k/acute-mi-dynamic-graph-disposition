"""Audit abstract length, sentence length, numeric precision, and citations."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document


path = Path("1108manuscript.docx")
document = Document(path)
paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
text = "\n".join(paragraphs)
start = paragraphs.index("1. Introduction")
end = paragraphs.index("7. References")
body_paragraphs = []
for paragraph in document.paragraphs[start:end]:
    value = paragraph.text.strip()
    if not value:
        continue
    if paragraph.paragraph_format.keep_with_next:
        continue
    if value.startswith(("Table ", "Figure ", "Interpretation:")):
        continue
    body_paragraphs.append(value)
sentences = []
for value in body_paragraphs:
    sentences.extend(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", value)
        if sentence.strip()
    )
long_sentences = [
    (len(sentence.split()), sentence)
    for sentence in sentences
    if len(sentence.split()) > 30
]
abstract_start = paragraphs.index("Abstract")
abstract_end = next(
    index for index, value in enumerate(paragraphs)
    if value.startswith("Keywords:")
)
abstract = " ".join(paragraphs[abstract_start + 1:abstract_end])
table_text = "\n".join(
    cell.text
    for table in document.tables
    for row in table.rows
    for cell in row.cells
)
decimal_tokens = re.findall(r"\b\d+\.\d+\b", text + "\n" + table_text)
three_plus = [token for token in decimal_tokens if len(token.split(".")[1]) >= 3]
citations = re.findall(r"\[[0-9,\-–\s]+\]", text)

print("abstract_words", len(abstract.split()))
print("sentences", len(sentences))
print("over_30", len(long_sentences))
print("max_words", max((length for length, _ in long_sentences), default=0))
print("decimal_tokens", len(decimal_tokens), "three_plus", len(three_plus))
print("citation_tokens", len(citations))
print("citation_order", citations[:60])
for length, sentence in sorted(long_sentences, reverse=True)[:30]:
    print(length, sentence[:300])

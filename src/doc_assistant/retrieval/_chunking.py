"""法律文本分块：章节识别、文本切分、heading 注入。

独立于 DocumentVectorStore，可被 ingestion pipeline 直接复用。
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from doc_assistant.config.settings import settings

_LEGAL_SECTION_PATTERN = re.compile(
    r"^\s*("
    r"第[一二三四五六七八九十百千万\d]+[章节条款项]|"
    r"\d+(?:\.\d+)*[\.)、]?|"
    r"(?:Section|Article|Clause|Schedule|Exhibit|Appendix)\s+[\w\dIVXLC]+"
    r")\s*[:：.-]?\s*(.*)$",
    re.IGNORECASE,
)

INGESTION_CHUNK_SEPARATORS: tuple[str, ...] = (
    "\n第",
    "\nSection ",
    "\nArticle ",
    "\nClause ",
    "\nSchedule ",
    "\nExhibit ",
    "\n\n",
    "\n",
    "。 ",
    "；",
    ". ",
    "; ",
    ", ",
    " ",
    "",
)


def build_ingestion_text_splitter(
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=list(INGESTION_CHUNK_SEPARATORS),
    )


def split_documents_for_ingestion(
    documents: list[Document],
    *,
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[Document]:
    text_splitter = splitter or build_ingestion_text_splitter()
    chunks = text_splitter.split_documents(split_legal_sections(documents))
    for chunk in chunks:
        chunk.page_content = chunk_text_with_heading(
            chunk.page_content,
            chunk.metadata.get("section_heading"),
        )
    return chunks


def chunk_text_with_heading(text: str, heading: Any) -> str:
    heading_text = str(heading or "").strip()
    content = text or ""
    if not heading_text:
        return content
    if content.lstrip().startswith(heading_text):
        return content
    return f"{heading_text}\n{content}".strip()


def split_legal_sections(documents: list[Document]) -> list[Document]:
    section_documents = []
    for document in documents:
        text = document.page_content or ""
        blocks = _section_blocks(text)
        if len(blocks) <= 1:
            section_documents.append(document)
            continue

        for section_index, (heading, content) in enumerate(blocks):
            metadata = dict(document.metadata)
            metadata["section_index"] = section_index
            if heading:
                metadata["section_heading"] = heading[:180]
            section_documents.append(Document(page_content=content, metadata=metadata))

    return section_documents


def _section_blocks(text: str) -> list[tuple[str | None, str]]:
    blocks: list[tuple[str | None, str]] = []
    current_lines: list[str] = []
    current_heading: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_lines:
                current_lines.append("")
            continue

        heading = _legal_section_heading(line)
        if heading and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                blocks.append((current_heading, content))
            current_lines = [line]
            current_heading = heading
            continue

        if heading and not current_lines:
            current_heading = heading
        current_lines.append(line)

    content = "\n".join(current_lines).strip()
    if content:
        blocks.append((current_heading, content))

    return blocks or [(None, text)]


def _legal_section_heading(line: str) -> str | None:
    if len(line) > 220:
        return None
    match = _LEGAL_SECTION_PATTERN.match(line)
    if not match:
        return None
    return " ".join(line.split())

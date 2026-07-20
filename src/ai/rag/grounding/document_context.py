"""Format retrieved document chunks as prompt context and citations."""

from __future__ import annotations

from langchain_core.documents import Document

from ai.rag.schemas import Citation
from ai.utils.text import compact_text, optional_text


def format_document_context(
    documents: list[Document],
    *,
    prefix: str = "S",
) -> tuple[str, list[Citation]]:
    context_parts = []
    citations = []

    for index, document in enumerate(documents, start=1):
        source_id = f"{prefix}{index}"
        metadata = document.metadata or {}
        text = compact_text(document.page_content)
        page = metadata.get("page")
        chunk_id = metadata.get("chunk_id")
        section_heading = metadata.get("section_heading")
        retrieval_score = metadata.get("retrieval_score")
        retrieval_relevance = metadata.get("retrieval_relevance")
        file_name = metadata.get("file_name") or metadata.get("source") or "unknown"
        document_version = (
            metadata.get("document_version")
            if isinstance(metadata.get("document_version"), int)
            else None
        )
        page_number = page if isinstance(page, int) else None
        page_label = f"page {page_number + 1}" if page_number is not None else None
        section_part = f"；章节={section_heading}" if section_heading else ""
        page_part = str(page_number + 1) if page_number is not None else "未知"

        context_parts.append(
            f"[{source_id}] 文件={file_name}；页码={page_part}；"
            f"页索引={page}；分块={chunk_id}{section_part}\n{text}"
        )
        citations.append(
            Citation(
                source_id=source_id,
                file_name=str(file_name),
                page=page_number,
                chunk_id=chunk_id if isinstance(chunk_id, int) else None,
                preview=text[:500],
                source_type="document",
                file_id=optional_text(metadata.get("file_id")),
                document_key=optional_text(metadata.get("document_key")),
                document_version=document_version,
                page_label=page_label,
                section_heading=str(section_heading) if section_heading else None,
                exact_quote=text[:1200],
                retrieval_score=(
                    float(retrieval_score)
                    if isinstance(retrieval_score, int | float)
                    else None
                ),
                retrieval_relevance=(
                    float(retrieval_relevance)
                    if isinstance(retrieval_relevance, int | float)
                    else None
                ),
            )
        )

    return "\n\n".join(context_parts), citations

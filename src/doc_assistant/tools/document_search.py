"""Document-search tool schema and result normalization."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.documents import Document
from langchain_core.tools import InjectedToolCallId
from pydantic import BaseModel, Field, field_validator

from doc_assistant.utils.text import optional_text


class SearchDocumentsInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=10)
    tool_call_id: Annotated[str, InjectedToolCallId] = ""

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("query is required")
        return value


def document_search_result(document: Document) -> dict[str, Any]:
    metadata = document.metadata or {}
    content = " ".join(document.page_content.split())[:1600]
    page = metadata.get("page")
    chunk_id = metadata.get("chunk_id")
    section_heading = metadata.get("section_heading")
    retrieval_score = metadata.get("retrieval_score")
    retrieval_relevance = metadata.get("retrieval_relevance")
    file_name = str(metadata.get("file_name") or metadata.get("source") or "unknown")
    page_number = page if isinstance(page, int) else None
    return {
        "file_name": file_name,
        "file_id": optional_text(metadata.get("file_id")),
        "document_key": optional_text(metadata.get("document_key")),
        "document_version": (
            metadata.get("document_version")
            if isinstance(metadata.get("document_version"), int)
            else None
        ),
        "page": page_number,
        "page_label": f"page {page_number + 1}" if page_number is not None else None,
        "chunk_id": chunk_id if isinstance(chunk_id, int) else None,
        "section_heading": str(section_heading) if section_heading else None,
        "retrieval_score": retrieval_score if isinstance(retrieval_score, int | float) else None,
        "retrieval_relevance": (
            retrieval_relevance if isinstance(retrieval_relevance, int | float) else None
        ),
        "content": content,
    }


__all__ = [
    "SearchDocumentsInput",
    "document_search_result",
]

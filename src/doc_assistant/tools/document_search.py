"""Document-search tool definition and execution.

The tool owns its OpenAI-compatible schema, argument validation, retrieval,
and result normalization. Conversation-scoped citation identifiers remain the
responsibility of the calling service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Protocol

from langchain_core.documents import Document
from langchain_core.tools import InjectedToolCallId
from pydantic import BaseModel, Field, field_validator

from doc_assistant.retrieval.document_identity import document_identity
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


class DocumentSearchBackend(Protocol):
    """Minimal retrieval interface required by :class:`DocumentSearchTool`."""

    def search(self, query: str, k: int | None = None) -> list[Document]: ...


@dataclass(frozen=True)
class DocumentSearchHit:
    """A normalized document hit before a conversation source ID is assigned."""

    identity: str
    result: dict[str, Any]


@dataclass(frozen=True)
class DocumentSearchExecution:
    """Validated query and normalized hits produced by one tool execution."""

    query: str
    hits: tuple[DocumentSearchHit, ...]


class DocumentSearchTool:
    """Execute the ``search_documents`` tool against an injected vector store."""

    def __init__(self, backend: DocumentSearchBackend, *, default_top_k: int) -> None:
        self.backend = backend
        self.default_top_k = default_top_k

    def execute(self, query: str, top_k: int | None = None) -> DocumentSearchExecution:
        top_k = top_k or self.default_top_k
        documents = self.backend.search(query, k=top_k)
        hits = tuple(
            DocumentSearchHit(
                identity=document_identity(document),
                result=_document_result(document),
            )
            for document in documents
        )
        return DocumentSearchExecution(query=query, hits=hits)


def _document_result(document: Document) -> dict[str, Any]:
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
    "DocumentSearchBackend",
    "DocumentSearchExecution",
    "DocumentSearchHit",
    "DocumentSearchTool",
    "SearchDocumentsInput",
]

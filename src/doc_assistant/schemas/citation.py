from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from doc_assistant.memory.schemas import MemoryUsage


@dataclass(frozen=True)
class Citation:
    source_id: str
    file_name: str
    preview: str
    page: int | None = None
    chunk_id: int | None = None
    source_type: str = "document"
    file_id: str | None = None
    document_key: str | None = None
    document_version: int | None = None
    page_label: str | None = None
    section_heading: str | None = None
    exact_quote: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    retrieval_score: float | None = None
    retrieval_relevance: float | None = None

    def location_label(self) -> str:
        parts = []
        if self.page is not None:
            parts.append(self.page_label or f"page {self.page + 1}")
        if self.chunk_id is not None:
            parts.append(f"chunk {self.chunk_id}")
        if self.section_heading:
            parts.append(str(self.section_heading))
        return f" ({', '.join(parts)})" if parts else ""


@dataclass(frozen=True)
class QAAnswer:
    content: str
    citations: list[Citation] = field(default_factory=list)
    memories_used: list[MemoryUsage] = field(default_factory=list)
    confidence: str | None = None
    guard_warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestResult:
    file_id: str
    file_name: str
    document_count: int
    chunk_count: int
    document_key: str = ""
    document_version: int = 1
    file_extension: str = ""
    page_count: int | None = None
    skipped: bool = False
    warnings: list[str] = field(default_factory=list)

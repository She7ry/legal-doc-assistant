from __future__ import annotations

from dataclasses import dataclass, field

from doc_assistant.memory.schemas import MemoryUsage


@dataclass(frozen=True)
class Citation:
    source_id: str
    file_name: str
    preview: str
    page: int | None = None
    chunk_id: int | None = None

    def location_label(self) -> str:
        parts = []
        if self.page is not None:
            parts.append(f"page {self.page + 1}")
        if self.chunk_id is not None:
            parts.append(f"chunk {self.chunk_id}")
        return f" ({', '.join(parts)})" if parts else ""


@dataclass(frozen=True)
class QAAnswer:
    content: str
    citations: list[Citation] = field(default_factory=list)
    memories_used: list[MemoryUsage] = field(default_factory=list)


@dataclass(frozen=True)
class IngestResult:
    file_id: str
    file_name: str
    document_count: int
    chunk_count: int

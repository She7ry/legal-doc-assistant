from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CitationOut(BaseModel):
    source_id: str
    file_name: str
    page: int | None
    chunk_id: int | None
    preview: str
    location_label: str

    @classmethod
    def from_citation(cls, citation) -> "CitationOut":
        return cls(
            source_id=citation.source_id,
            file_name=citation.file_name,
            page=citation.page,
            chunk_id=citation.chunk_id,
            preview=citation.preview,
            location_label=citation.location_label(),
        )


class MemoryUsageOut(BaseModel):
    memory_id: str
    type: str
    key: str
    content: str
    source: str
    confidence: float
    scope: str
    score: float | None = None

    @classmethod
    def from_usage(cls, usage) -> "MemoryUsageOut":
        return cls(
            memory_id=usage.memory_id,
            type=usage.type,
            key=usage.key,
            content=usage.content,
            source=usage.source,
            confidence=usage.confidence,
            scope=usage.scope,
            score=usage.score,
        )


class AskResponse(BaseModel):
    content: str
    citations: list[CitationOut]
    memories_used: list[MemoryUsageOut] = Field(default_factory=list)


class WebSourceOut(BaseModel):
    source_id: str
    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    source: str | None = None

    @classmethod
    def from_source(cls, source) -> "WebSourceOut":
        return cls(
            source_id=source.source_id,
            title=source.title,
            url=source.url,
            snippet=source.snippet,
            published_at=source.published_at,
            source=source.source,
        )


class ToolCallOut(BaseModel):
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]

    @classmethod
    def from_trace(cls, trace) -> "ToolCallOut":
        return cls(
            tool_call_id=trace.tool_call_id,
            name=trace.name,
            arguments=trace.arguments,
            result=trace.result,
        )


class ToolChatResponse(BaseModel):
    content: str
    citations: list[CitationOut]
    web_sources: list[WebSourceOut] = Field(default_factory=list)
    tool_calls: list[ToolCallOut] = Field(default_factory=list)


class ClauseReviewResponse(BaseModel):
    content: str
    citations: list[CitationOut]


class ConflictCheckResponse(BaseModel):
    content: str
    citations: list[CitationOut]


class IngestResponse(BaseModel):
    file_id: str
    file_name: str
    document_count: int
    chunk_count: int


class IngestJobResponse(BaseModel):
    job_id: str
    status: str
    file_name: str
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: IngestResponse | None = None
    error: str | None = None


class DocumentInfo(BaseModel):
    file_name: str
    file_id: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class MemoryOut(BaseModel):
    memory_id: str
    scope: str
    type: str
    key: str
    content: str
    value: dict[str, Any] | None
    source: str
    confidence: float
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    visibility: str
    permissions: list[str]
    embedding_id: str | None
    supersedes_id: str | None
    status: str
    source_message_id: str | None
    conversation_id: str | None
    task_id: str | None

    @classmethod
    def from_memory(cls, memory) -> "MemoryOut":
        return cls(
            memory_id=memory.memory_id,
            scope=memory.scope,
            type=memory.type,
            key=memory.key,
            content=memory.content,
            value=memory.value_json,
            source=memory.source,
            confidence=memory.confidence,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            expires_at=memory.expires_at,
            visibility=memory.visibility,
            permissions=list(memory.permissions),
            embedding_id=memory.embedding_id,
            supersedes_id=memory.supersedes_id,
            status=memory.status,
            source_message_id=memory.source_message_id,
            conversation_id=memory.conversation_id,
            task_id=memory.task_id,
        )


class MemoryListResponse(BaseModel):
    memories: list[MemoryOut]
    total: int


class ErrorResponse(BaseModel):
    code: str
    detail: str
    request_id: str | None = None

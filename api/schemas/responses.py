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

    @classmethod
    def from_citation(cls, citation) -> "CitationOut":
        return cls(
            source_id=citation.source_id,
            file_name=citation.file_name,
            page=citation.page,
            chunk_id=citation.chunk_id,
            preview=citation.preview,
            location_label=citation.location_label(),
            source_type=getattr(citation, "source_type", "document"),
            file_id=getattr(citation, "file_id", None),
            document_key=getattr(citation, "document_key", None),
            document_version=getattr(citation, "document_version", None),
            page_label=getattr(citation, "page_label", None),
            section_heading=getattr(citation, "section_heading", None),
            exact_quote=getattr(citation, "exact_quote", None),
            char_start=getattr(citation, "char_start", None),
            char_end=getattr(citation, "char_end", None),
            retrieval_score=getattr(citation, "retrieval_score", None),
            retrieval_relevance=getattr(citation, "retrieval_relevance", None),
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
    confidence: str | None = None
    guard_warnings: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] | None = None


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
    confidence: str | None = None
    guard_warnings: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] | None = None


class ClauseRiskReasonOut(BaseModel):
    reason: str = ""
    citation: str | None = None


class ClauseReviewResponse(BaseModel):
    content: str
    citations: list[CitationOut]
    clause_type: str = ""
    normalized_clause_type: str = ""
    found: bool | None = None
    summary: str = ""
    risk_level: str = "Needs human review"
    risk_reasons: list[ClauseRiskReasonOut] = Field(default_factory=list)
    affected_party: str | None = None
    plain_language_explanation: str = ""
    questions_for_lawyer: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    needs_human_review: bool = True
    guard_warnings: list[str] = Field(default_factory=list)


class ConflictItemOut(BaseModel):
    topic: str = ""
    conflict_type: str = "ambiguous_relationship"
    severity: str = "Needs human review"
    contract_position: str = ""
    policy_position: str = ""
    why_conflict: str = ""
    recommended_action: str = ""
    contract_citations: list[str] = Field(default_factory=list)
    policy_citations: list[str] = Field(default_factory=list)
    needs_human_review: bool = True
    confidence: str | None = None


class ConflictCheckResponse(BaseModel):
    content: str
    citations: list[CitationOut]
    overall_status: str = "Insufficient information"
    conflicts: list[ConflictItemOut] = Field(default_factory=list)
    needs_human_review: bool = True
    guard_warnings: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    file_id: str
    file_name: str
    document_count: int
    chunk_count: int
    document_key: str = ""
    document_version: int = 1
    file_extension: str = ""
    page_count: int | None = None
    skipped: bool = False
    warnings: list[str] = Field(default_factory=list)


class IngestJobResponse(BaseModel):
    job_id: str
    status: str
    file_name: str
    stage: str = "uploaded"
    progress: int = 0
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: IngestResponse | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    file_name: str
    file_id: str
    document_key: str = ""
    document_version: int = 1
    file_extension: str = ""
    document_count: int = 0
    chunk_count: int = 0
    page_count: int | None = None
    indexed_at: str | None = None
    warning_count: int = 0


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

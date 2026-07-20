from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _AttrModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True, use_enum_values=True)


class UserResponse(_AttrModel):
    user_id: str
    username: str
    created_at: datetime


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
    def from_citation(cls, citation) -> CitationOut:
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


class AskResponse(BaseModel):
    content: str
    citations: list[CitationOut]
    confidence: str | None = None
    guard_warnings: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] | None = None


class WebSourceOut(_AttrModel):
    source_id: str
    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    source: str | None = None

class ToolCallOut(_AttrModel):
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]

class ToolChatResponse(AskResponse):
    web_sources: list[WebSourceOut] = Field(default_factory=list)
    tool_calls: list[ToolCallOut] = Field(default_factory=list)


class ConversationOut(_AttrModel):
    conversation_id: str
    title: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

class ConversationListResponse(BaseModel):
    conversations: list[ConversationOut]
    total: int
    offset: int = 0
    limit: int | None = None


class ConversationMessageOut(BaseModel):
    role: str
    content: str


class ConversationMessagesResponse(BaseModel):
    conversation_id: str
    messages: list[ConversationMessageOut] = Field(default_factory=list)


class AgentStepResultOut(_AttrModel):
    step_id: str
    title: str
    tool: str
    status: str
    summary: str
    citations: list[CitationOut] = Field(default_factory=list)
    evidence: dict[str, Any] | None = None
    guard_warnings: list[str] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)

    @field_validator("citations", mode="before")
    @classmethod
    def _citations(cls, value):
        return [CitationOut.from_citation(citation) for citation in value or []]


class AgentTaskResponse(_AttrModel):
    task_id: str
    status: str
    objective: str
    steps: list[AgentStepResultOut]
    human_review_required: bool = True
    report: str
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: str | None = None
    guard_warnings: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("citations", mode="before")
    @classmethod
    def _citations(cls, value):
        return [CitationOut.from_citation(citation) for citation in value or []]


class AgentTaskEventOut(_AttrModel):
    event_id: int
    task_id: str
    event_type: str
    stage: str
    progress: int
    message: str
    created_at: datetime
    step_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload", mode="before")
    @classmethod
    def _payload_or_empty(cls, value):
        return value or {}

class AgentTaskRecordResponse(_AttrModel):
    task_id: str
    status: str
    objective: str
    focus_areas: list[str] = Field(default_factory=list)
    user_role: str
    max_steps: int
    conversation_id: str | None = None
    stage: str
    progress: int
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: AgentTaskResponse | None = None
    error: str | None = None
    events: list[AgentTaskEventOut] = Field(default_factory=list)

    @field_validator("events", mode="before")
    @classmethod
    def _events_or_empty(cls, value):
        return value or []


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


class IngestResponse(_AttrModel):
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


class IngestJobResponse(_AttrModel):
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

    @field_validator("warnings", mode="before")
    @classmethod
    def _warnings_or_empty(cls, value):
        return value or []


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


class DocumentTextChunkOut(BaseModel):
    chunk_id: int | None = None
    text: str
    page: int | None = None
    page_label: str | None = None
    section_heading: str | None = None
    location_label: str = ""


class DocumentTextResponse(BaseModel):
    document: DocumentInfo
    chunks: list[DocumentTextChunkOut]
    total_chunks: int
    offset: int
    limit: int
    next_offset: int | None = None


class HealthCheckOut(BaseModel):
    name: str
    status: str
    detail: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
    auth_required: bool
    providers: dict[str, Any]
    features: dict[str, bool]
    limits: dict[str, Any]
    checks: list[HealthCheckOut]

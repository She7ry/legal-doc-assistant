from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from doc_assistant.matter._domain import _formal_report_blockers


class _AttrModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True, use_enum_values=True)


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

    @classmethod
    def from_source(cls, source) -> WebSourceOut:
        return cls.model_validate(source)


class ToolCallOut(_AttrModel):
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]

    @classmethod
    def from_trace(cls, trace) -> ToolCallOut:
        return cls.model_validate(trace)


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

    @classmethod
    def from_conversation(cls, conversation) -> ConversationOut:
        return cls.model_validate(conversation)


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


class AgentStepResultOut(BaseModel):
    step_id: str
    title: str
    tool: str
    status: str
    summary: str
    citations: list[CitationOut] = Field(default_factory=list)
    evidence: dict[str, Any] | None = None
    guard_warnings: list[str] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_step(cls, step) -> AgentStepResultOut:
        return cls(
            step_id=step.step_id,
            title=step.title,
            tool=step.tool,
            status=step.status,
            summary=step.summary,
            citations=[CitationOut.from_citation(c) for c in step.citations],
            evidence=step.evidence,
            guard_warnings=step.guard_warnings,
            output=step.output,
        )


# ------------------------------------------------------------------
# Shared Finding / Artifact bases to avoid field duplication
# ------------------------------------------------------------------

class _FindingBase(_AttrModel):
    finding_id: str
    category: str
    severity: str
    summary: str
    citations: list[str] = Field(default_factory=list)
    recommended_action: str = ""
    needs_human_review: bool = True
    source_step_id: str = ""
    clause_reference: str = ""
    evidence_coverage: str = "missing"
    support_level: str = "missing"
    unsupported_reason: str = ""
    source_quote: str = ""
    location_label: str = ""
    human_review_status: str = "pending"
    status: str = "open"


class MatterFindingRecordOut(_FindingBase):
    matter_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_task_id: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, finding) -> MatterFindingRecordOut:
        return cls.model_validate(finding)


class _ArtifactBase(_AttrModel):
    artifact_id: str
    artifact_type: str
    title: str
    summary: str
    items: list[dict[str, Any]] = Field(default_factory=list)
    source_finding_ids: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MatterArtifactRecordOut(_ArtifactBase):
    matter_id: str
    source_task_id: str
    version: int
    status: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, artifact) -> MatterArtifactRecordOut:
        return cls.model_validate(artifact)


class MatterRecordOut(_AttrModel):
    matter_id: str
    title: str
    status: str
    matter_profile: dict[str, Any]
    source_task_id: str
    latest_task_id: str
    created_at: datetime
    updated_at: datetime
    artifacts: list[MatterArtifactRecordOut] = Field(default_factory=list)
    findings: list[MatterFindingRecordOut] = Field(default_factory=list)
    formal_report_blockers: list[str] = Field(default_factory=list)
    can_generate_formal_report: bool = False

    @field_validator("artifacts", "findings", mode="before")
    @classmethod
    def _none_to_list(cls, value):
        return value or []

    @classmethod
    def from_record(cls, matter) -> MatterRecordOut:
        blockers = _formal_report_blockers(
            matter.matter_profile,
            matter.findings or [],
        )
        return cls.model_validate(matter).model_copy(
            update={
                "formal_report_blockers": blockers,
                "can_generate_formal_report": not blockers,
            }
        )


class MatterEventOut(_AttrModel):
    event_id: str
    matter_id: str
    event_type: str
    entity_type: str
    entity_id: str
    old_value: Any = None
    new_value: Any = None
    actor: str
    created_at: datetime

    @classmethod
    def from_record(cls, event) -> MatterEventOut:
        return cls.model_validate(event)


class MatterListResponse(BaseModel):
    matters: list[MatterRecordOut]
    total: int


class AgentTaskResponse(BaseModel):
    task_id: str
    status: str
    objective: str
    steps: list[AgentStepResultOut]
    findings: list[dict[str, Any]] = Field(default_factory=list)
    human_review_required: bool = True
    report: str
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: str | None = None
    guard_warnings: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] | None = None
    matter_profile: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_result(cls, result) -> AgentTaskResponse:
        return cls(
            task_id=result.task_id,
            status=result.status,
            objective=result.objective,
            steps=[AgentStepResultOut.from_step(s) for s in result.steps],
            findings=result.findings,
            human_review_required=result.human_review_required,
            report=result.report,
            citations=[CitationOut.from_citation(c) for c in result.citations],
            confidence=result.confidence,
            guard_warnings=result.guard_warnings,
            evidence=result.evidence,
            matter_profile=result.matter_profile,
            artifacts=result.artifacts,
            metadata=result.metadata,
        )


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

    @classmethod
    def from_event(cls, event) -> AgentTaskEventOut:
        return cls.model_validate(event)


class AgentTaskRecordResponse(BaseModel):
    task_id: str
    status: str
    objective: str
    focus_areas: list[str] = Field(default_factory=list)
    user_role: str
    max_steps: int
    conversation_id: str | None = None
    matter_id: str | None = None
    stage: str
    progress: int
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: AgentTaskResponse | None = None
    error: str | None = None
    events: list[AgentTaskEventOut] = Field(default_factory=list)

    @classmethod
    def from_record(cls, record) -> AgentTaskRecordResponse:
        result = AgentTaskResponse(**record.result) if record.result else None
        return cls(
            task_id=record.task_id,
            status=record.status.value if hasattr(record.status, "value") else record.status,
            objective=record.objective,
            focus_areas=record.focus_areas,
            user_role=record.user_role,
            max_steps=record.max_steps,
            conversation_id=record.conversation_id,
            matter_id=getattr(record, "matter_id", None),
            stage=record.stage,
            progress=record.progress,
            submitted_at=record.submitted_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            result=result,
            error=record.error,
            events=[AgentTaskEventOut.from_event(e) for e in record.events or []],
        )


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

    @classmethod
    def from_record(cls, record) -> IngestJobResponse:
        return cls(
            job_id=record.job_id,
            status=record.status.value,
            file_name=record.file_name,
            stage=record.stage,
            progress=record.progress,
            submitted_at=record.submitted_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            result=IngestResponse.model_validate(record.result) if record.result else None,
            error=record.error,
            warnings=record.warnings or [],
        )


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


class ErrorResponse(BaseModel):
    code: str
    detail: str
    request_id: str | None = None


class HealthCheckOut(BaseModel):
    name: str
    status: str
    detail: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
    auth_required: bool
    default_tenant_id: str
    providers: dict[str, Any]
    features: dict[str, bool]
    limits: dict[str, Any]
    checks: list[HealthCheckOut]

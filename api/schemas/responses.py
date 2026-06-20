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
    last_accessed_at: datetime | None = None
    access_count: int = 0
    superseded_conflicting: bool = False
    superseded_from_content: str | None = None

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
            last_accessed_at=getattr(usage, "last_accessed_at", None),
            access_count=getattr(usage, "access_count", 0),
            superseded_conflicting=getattr(usage, "superseded_conflicting", False),
            superseded_from_content=getattr(usage, "superseded_from_content", None),
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


class ToolChatResponse(AskResponse):
    web_sources: list[WebSourceOut] = Field(default_factory=list)
    tool_calls: list[ToolCallOut] = Field(default_factory=list)


class ConversationOut(BaseModel):
    conversation_id: str
    title: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    @classmethod
    def from_conversation(cls, conversation) -> "ConversationOut":
        return cls(
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            status=conversation.status,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=conversation.message_count,
        )


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


class AgentPlanStepOut(BaseModel):
    step_id: str
    title: str
    purpose: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False

    @classmethod
    def from_step(cls, step) -> "AgentPlanStepOut":
        return cls(
            step_id=step.step_id,
            title=step.title,
            purpose=step.purpose,
            tool=step.tool,
            arguments=step.arguments,
            requires_confirmation=step.requires_confirmation,
        )


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
    def from_step(cls, step) -> "AgentStepResultOut":
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

class _FindingBase(BaseModel):
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


class AgentFindingOut(_FindingBase):
    evidence: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_finding(cls, finding) -> "AgentFindingOut":
        return cls(
            finding_id=finding.finding_id,
            category=finding.category,
            severity=finding.severity,
            summary=finding.summary,
            citations=finding.citations,
            recommended_action=finding.recommended_action,
            needs_human_review=finding.needs_human_review,
            source_step_id=finding.source_step_id,
            clause_reference=getattr(finding, "clause_reference", ""),
            evidence_coverage=getattr(finding, "evidence_coverage", "missing"),
            support_level=getattr(finding, "support_level", "missing"),
            unsupported_reason=getattr(finding, "unsupported_reason", ""),
            source_quote=getattr(finding, "source_quote", ""),
            location_label=getattr(finding, "location_label", ""),
            human_review_status=getattr(finding, "human_review_status", "pending"),
            status=getattr(finding, "status", "open"),
            evidence=getattr(finding, "evidence", []),
        )


class MatterFindingRecordOut(_FindingBase):
    matter_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_task_id: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, finding) -> "MatterFindingRecordOut":
        return cls(
            finding_id=finding.finding_id,
            matter_id=finding.matter_id,
            category=finding.category,
            severity=finding.severity,
            summary=finding.summary,
            recommended_action=finding.recommended_action,
            citations=finding.citations,
            source_step_id=finding.source_step_id,
            clause_reference=finding.clause_reference,
            evidence_coverage=finding.evidence_coverage,
            support_level=finding.support_level,
            unsupported_reason=finding.unsupported_reason,
            source_quote=finding.source_quote,
            location_label=finding.location_label,
            needs_human_review=finding.needs_human_review,
            human_review_status=finding.human_review_status,
            status=finding.status,
            metadata=finding.metadata,
            source_task_id=finding.source_task_id,
            created_at=finding.created_at,
            updated_at=finding.updated_at,
        )


class MatterProfileOut(BaseModel):
    matter_id: str
    document_type: str = "Unknown"
    parties: list[str] = Field(default_factory=list)
    user_side: str = ""
    governing_law: str = ""
    jurisdiction: str = ""
    key_dates: list[dict[str, Any]] = Field(default_factory=list)
    review_scope: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    confidence: str = "Low"
    citations: list[str] = Field(default_factory=list)
    source_step_id: str = ""
    confirmation_gates: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_profile(cls, profile) -> "MatterProfileOut":
        return cls(
            matter_id=profile.matter_id,
            document_type=profile.document_type,
            parties=profile.parties,
            user_side=profile.user_side,
            governing_law=profile.governing_law,
            jurisdiction=profile.jurisdiction,
            key_dates=profile.key_dates,
            review_scope=profile.review_scope,
            open_questions=profile.open_questions,
            confidence=profile.confidence,
            citations=profile.citations,
            source_step_id=profile.source_step_id,
            confirmation_gates=getattr(profile, "confirmation_gates", []),
        )


class _ArtifactBase(BaseModel):
    artifact_id: str
    artifact_type: str
    title: str
    summary: str
    items: list[dict[str, Any]] = Field(default_factory=list)
    source_finding_ids: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentArtifactOut(_ArtifactBase):
    @classmethod
    def from_artifact(cls, artifact) -> "AgentArtifactOut":
        return cls(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            title=artifact.title,
            summary=artifact.summary,
            items=artifact.items,
            source_finding_ids=artifact.source_finding_ids,
            citations=artifact.citations,
            metadata=artifact.metadata,
        )


class MatterArtifactRecordOut(_ArtifactBase):
    matter_id: str
    source_task_id: str
    version: int
    status: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, artifact) -> "MatterArtifactRecordOut":
        return cls(
            artifact_id=artifact.artifact_id,
            matter_id=artifact.matter_id,
            artifact_type=artifact.artifact_type,
            title=artifact.title,
            summary=artifact.summary,
            items=artifact.items,
            source_finding_ids=artifact.source_finding_ids,
            citations=artifact.citations,
            metadata=artifact.metadata,
            source_task_id=artifact.source_task_id,
            version=artifact.version,
            status=artifact.status,
            created_at=artifact.created_at,
            updated_at=artifact.updated_at,
        )


class AgentConfirmationGateOut(BaseModel):
    gate_id: str
    gate_type: str
    title: str
    question: str
    status: str = "pending"
    priority: str = "normal"
    required: bool = True
    reason: str = ""
    related_finding_ids: list[str] = Field(default_factory=list)
    related_artifact_ids: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_gate(cls, gate) -> "AgentConfirmationGateOut":
        return cls(
            gate_id=gate.gate_id,
            gate_type=gate.gate_type,
            title=gate.title,
            question=gate.question,
            status=gate.status,
            priority=gate.priority,
            required=gate.required,
            reason=gate.reason,
            related_finding_ids=gate.related_finding_ids,
            related_artifact_ids=gate.related_artifact_ids,
            citations=gate.citations,
            metadata=gate.metadata,
        )


class MatterRecordOut(BaseModel):
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

    @classmethod
    def from_record(cls, matter) -> "MatterRecordOut":
        return cls(
            matter_id=matter.matter_id,
            title=matter.title,
            status=matter.status,
            matter_profile=matter.matter_profile,
            source_task_id=matter.source_task_id,
            latest_task_id=matter.latest_task_id,
            created_at=matter.created_at,
            updated_at=matter.updated_at,
            artifacts=[
                MatterArtifactRecordOut.from_record(a)
                for a in matter.artifacts or []
            ],
            findings=[
                MatterFindingRecordOut.from_record(f)
                for f in getattr(matter, "findings", None) or []
            ],
        )


class MatterEventOut(BaseModel):
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
    def from_record(cls, event) -> "MatterEventOut":
        return cls(
            event_id=event.event_id,
            matter_id=event.matter_id,
            event_type=event.event_type,
            entity_type=event.entity_type,
            entity_id=event.entity_id,
            old_value=event.old_value,
            new_value=event.new_value,
            actor=event.actor,
            created_at=event.created_at,
        )


class MatterListResponse(BaseModel):
    matters: list[MatterRecordOut]
    total: int


class AgentTaskResponse(BaseModel):
    task_id: str
    status: str
    objective: str
    plan: list[AgentPlanStepOut]
    steps: list[AgentStepResultOut]
    findings: list[AgentFindingOut]
    missing_information: list[str] = Field(default_factory=list)
    human_review_required: bool = True
    report: str
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: str | None = None
    guard_warnings: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] | None = None
    matter_profile: MatterProfileOut | None = None
    artifacts: list[AgentArtifactOut] = Field(default_factory=list)
    confirmation_gates: list[AgentConfirmationGateOut] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_result(cls, result) -> "AgentTaskResponse":
        matter_profile = (
            MatterProfileOut.from_profile(result.matter_profile)
            if result.matter_profile
            else None
        )
        return cls(
            task_id=result.task_id,
            status=result.status,
            objective=result.objective,
            plan=[AgentPlanStepOut.from_step(s) for s in result.plan],
            steps=[AgentStepResultOut.from_step(s) for s in result.steps],
            findings=[AgentFindingOut.from_finding(f) for f in result.findings],
            missing_information=result.missing_information,
            human_review_required=result.human_review_required,
            report=result.report,
            citations=[CitationOut.from_citation(c) for c in result.citations],
            confidence=result.confidence,
            guard_warnings=result.guard_warnings,
            evidence=result.evidence,
            matter_profile=matter_profile,
            artifacts=[AgentArtifactOut.from_artifact(a) for a in result.artifacts],
            confirmation_gates=[
                AgentConfirmationGateOut.from_gate(g)
                for g in getattr(result, "confirmation_gates", [])
            ],
            metadata=result.metadata,
        )


class AgentTaskEventOut(BaseModel):
    event_id: int
    task_id: str
    event_type: str
    stage: str
    progress: int
    message: str
    created_at: datetime
    step_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_event(cls, event) -> "AgentTaskEventOut":
        return cls(
            event_id=event.event_id,
            task_id=event.task_id,
            event_type=event.event_type,
            stage=event.stage,
            progress=event.progress,
            message=event.message,
            created_at=event.created_at,
            step_id=event.step_id,
            payload=event.payload or {},
        )


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
    def from_record(cls, record) -> "AgentTaskRecordResponse":
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

    @classmethod
    def from_record(cls, record) -> "IngestJobResponse":
        result = None
        if record.result is not None:
            result = IngestResponse(
                file_id=record.result.file_id,
                file_name=record.result.file_name,
                document_count=record.result.document_count,
                chunk_count=record.result.chunk_count,
                document_key=record.result.document_key,
                document_version=record.result.document_version,
                file_extension=record.result.file_extension,
                page_count=record.result.page_count,
                skipped=record.result.skipped,
                warnings=record.result.warnings,
            )
        return cls(
            job_id=record.job_id,
            status=record.status.value,
            file_name=record.file_name,
            stage=record.stage,
            progress=record.progress,
            submitted_at=record.submitted_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            result=result,
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
    last_accessed_at: datetime | None = None
    access_count: int = 0
    superseded_conflicting: bool = False
    superseded_from_content: str | None = None

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
            last_accessed_at=getattr(memory, "last_accessed_at", None),
            access_count=getattr(memory, "access_count", 0),
            superseded_conflicting=getattr(memory, "superseded_conflicting", False),
            superseded_from_content=getattr(memory, "superseded_from_content", None),
        )


class MemoryListResponse(BaseModel):
    memories: list[MemoryOut]
    total: int
    offset: int = 0
    limit: int | None = None


class MemoryBatchDeleteResponse(BaseModel):
    deleted: list[MemoryOut]
    not_found: list[str]
    total_deleted: int


class MemoryMaintenanceResponse(BaseModel):
    expired_stale: int = 0
    limit_stale: int = 0
    vector_deleted: int = 0
    vector_upserted: int = 0


class MemoryAccessStatsOut(BaseModel):
    tracked_memories: int = 0
    never_accessed: int = 0
    accessed: int = 0
    accessed_last_7d: int = 0
    accessed_last_30d: int = 0
    total_access_count: int = 0
    average_access_count: float = 0.0
    max_access_count: int = 0


class MemoryRetrievalStatsOut(BaseModel):
    total: int = 0
    with_memory: int = 0
    last_7d: int = 0
    last_30d: int = 0
    hit_rate: float = 0.0
    average_memory_count: float = 0.0
    average_document_count: float = 0.0
    last_retrieval_at: datetime | None = None
    selected_memory_source_counts: dict[str, int] = Field(default_factory=dict)
    selected_memory_source_ratios: dict[str, float] = Field(default_factory=dict)


class MemoryStatsResponse(BaseModel):
    tenant_id: str
    user_id: str
    generated_at: datetime
    total_memories: int = 0
    active_memories: int = 0
    stale_memories: int = 0
    deleted_memories: int = 0
    expired_active_memories: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    scope_counts: dict[str, int] = Field(default_factory=dict)
    type_counts: dict[str, int] = Field(default_factory=dict)
    average_confidence: float = 0.0
    average_active_confidence: float = 0.0
    access: MemoryAccessStatsOut = Field(default_factory=MemoryAccessStatsOut)
    retrievals: MemoryRetrievalStatsOut = Field(default_factory=MemoryRetrievalStatsOut)


class FeedbackMemoryAdjustmentOut(BaseModel):
    memory_id: str
    status: str
    previous_confidence: float | None = None
    new_confidence: float | None = None
    memory: MemoryOut | None = None

    @classmethod
    def from_adjustment(cls, adjustment) -> "FeedbackMemoryAdjustmentOut":
        return cls(
            memory_id=adjustment.memory_id,
            status=adjustment.status,
            previous_confidence=adjustment.previous_confidence,
            new_confidence=adjustment.new_confidence,
            memory=MemoryOut.from_memory(adjustment.memory) if adjustment.memory else None,
        )


class FeedbackResponse(BaseModel):
    feedback_id: str
    tenant_id: str
    user_id: str
    rating: int
    created_at: datetime
    conversation_id: str | None = None
    message_id: str | None = None
    memory_ids: list[str] = Field(default_factory=list)
    comment: str | None = None
    adjusted_memories: list[FeedbackMemoryAdjustmentOut] = Field(default_factory=list)

    @classmethod
    def from_feedback(cls, event, adjustments) -> "FeedbackResponse":
        return cls(
            feedback_id=event.feedback_id,
            tenant_id=event.tenant_id,
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            message_id=event.message_id,
            rating=event.rating,
            memory_ids=list(event.memory_ids),
            comment=event.comment,
            created_at=event.created_at,
            adjusted_memories=[
                FeedbackMemoryAdjustmentOut.from_adjustment(a)
                for a in adjustments
            ],
        )


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

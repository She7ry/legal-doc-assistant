from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    conversation_id: str | None = Field(default=None, max_length=128)
    task_id: str | None = Field(default=None, max_length=128)


class ToolChatRequest(AskRequest):
    enable_web_search: bool = False
    max_tool_iterations: int | None = Field(default=None, ge=1, le=10)


class ConversationCreateRequest(BaseModel):
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=200)


class ConversationUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    status: str | None = Field(default=None, pattern="^(active|archived)$")


class AgentTaskRequest(BaseModel):
    objective: str = Field(..., min_length=1, max_length=2000)
    focus_areas: list[str] = Field(default_factory=list, max_length=8)
    user_role: str = Field(default="ordinary", pattern="^(ordinary|lawyer)$")
    max_steps: int = Field(default=6, ge=3, le=10)
    conversation_id: str | None = Field(default=None, max_length=128)
    matter_id: str | None = Field(default=None, max_length=128)


class AgentTaskResumeRequest(BaseModel):
    objective: str | None = Field(default=None, min_length=1, max_length=2000)
    clarification_answers: list[str] = Field(default_factory=list, max_length=6)
    focus_areas: list[str] | None = Field(default=None, max_length=8)
    user_role: str | None = Field(default=None, pattern="^(ordinary|lawyer)$")
    max_steps: int | None = Field(default=None, ge=3, le=10)
    conversation_id: str | None = Field(default=None, max_length=128)
    matter_id: str | None = Field(default=None, max_length=128)


class MatterConfirmationGateUpdateRequest(BaseModel):
    status: str = Field(..., pattern="^(pending|approved|waived|needs_info)$")
    note: str | None = Field(default=None, max_length=1000)
    confirmed_value: str | None = Field(default=None, max_length=1000)


class MatterFormalReportCreateRequest(BaseModel):
    note: str | None = Field(default=None, max_length=1000)


class MatterArtifactUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    summary: str | None = Field(default=None, max_length=2000)
    items: list[dict[str, Any]] | None = Field(default=None, max_length=200)
    status: str | None = Field(
        default=None,
        pattern="^(draft|active|needs_review|approved|archived)$",
    )
    note: str | None = Field(default=None, max_length=1000)


class MatterFindingUpdateRequest(BaseModel):
    human_review_status: str = Field(
        ...,
        pattern="^(pending|approved|waived|needs_info|resolved)$",
    )
    note: str | None = Field(default=None, max_length=1000)


class ClauseReviewRequest(BaseModel):
    clause_type: str = Field(
        ...,
        min_length=1,
        max_length=200,
        examples=[
            "termination",
            "payment",
            "late fee",
            "auto renewal",
            "liability limitation",
            "indemnification",
            "confidentiality",
            "non-compete",
            "IP ownership",
            "data privacy",
            "governing law",
            "dispute resolution",
            "assignment",
            "audit rights",
            "notice",
        ],
    )
    top_k: int = Field(default=5, ge=1, le=20)


class ConflictCheckRequest(BaseModel):
    contract_query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query to retrieve contract excerpts",
        examples=["payment terms and obligations"],
    )
    policy_query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query to retrieve policy excerpts",
        examples=["payment policy and compliance requirements"],
    )
    top_k: int = Field(default=5, ge=1, le=20)

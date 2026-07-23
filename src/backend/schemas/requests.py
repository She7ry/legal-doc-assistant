from __future__ import annotations

from pydantic import BaseModel, Field, SecretStr, field_validator


class AuthRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: SecretStr

    @field_validator("username")
    @classmethod
    def strip_username(cls, value: str) -> str:
        return value.strip()

    @field_validator("password")
    @classmethod
    def validate_password_length(cls, value: SecretStr) -> SecretStr:
        if not 8 <= len(value.get_secret_value()) <= 128:
            raise ValueError("Password must be 8-128 characters.")
        return value


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    chat_history: list[ChatMessage] = Field(default_factory=list, max_length=50)
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


class AgentTaskResumeRequest(BaseModel):
    objective: str | None = Field(default=None, min_length=1, max_length=2000)
    clarification_answers: list[str] = Field(default_factory=list, max_length=6)
    focus_areas: list[str] | None = Field(default=None, max_length=8)
    user_role: str | None = Field(default=None, pattern="^(ordinary|lawyer)$")
    max_steps: int | None = Field(default=None, ge=3, le=10)
    conversation_id: str | None = Field(default=None, max_length=128)


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

"""记忆相关的类型定义与 dataclass（Conversation、MemoryRecord 等）。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final, Literal, get_args

MemoryScope = Literal["user", "org", "session", "task"]
MemoryType = Literal["preference", "fact", "task_state", "feedback", "correction"]
MemorySource = Literal["explicit", "inferred", "imported", "system_generated"]
MemoryStatus = Literal["active", "stale", "deleted"]
MemoryVisibility = Literal["private", "team", "org"]
ConversationStatus = Literal["active", "archived"]

VALID_MEMORY_SCOPES = set(get_args(MemoryScope))
VALID_MEMORY_TYPES = set(get_args(MemoryType))
VALID_MEMORY_SOURCES = set(get_args(MemorySource))
VALID_MEMORY_STATUSES = set(get_args(MemoryStatus))
VALID_MEMORY_VISIBILITIES = set(get_args(MemoryVisibility))
VALID_CONVERSATION_STATUSES = set(get_args(ConversationStatus))


@dataclass(frozen=True)
class _UnsetType:
    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final = _UnsetType()


def is_unset(value: object) -> bool:
    return value is UNSET


@dataclass(frozen=True)
class MessageRecord:
    """一条对话消息（user / assistant），持久化在 SQLite conversations 表关联下。"""

    message_id: str
    conversation_id: str
    tenant_id: str
    user_id: str
    role: str
    content: str
    created_at: datetime


@dataclass(frozen=True)
class ConversationRecord:
    """用户的一次聊天会话元数据（标题、状态、消息数），可归档。"""

    conversation_id: str
    tenant_id: str
    user_id: str
    title: str | None
    status: ConversationStatus
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


@dataclass(frozen=True)
class MemoryRecord:
    """一条结构化长期记忆（偏好、事实、任务状态等），带置信度与过期时间。

    scope 区分 user/org/session/task；status=stale 表示已被新任务结论取代。
    """

    memory_id: str
    tenant_id: str
    user_id: str
    scope: MemoryScope
    type: MemoryType
    key: str
    content: str
    value_json: dict[str, Any] | None
    source: MemorySource
    confidence: float
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    visibility: MemoryVisibility = "private"
    permissions: tuple[str, ...] = ("read", "write", "delete")
    embedding_id: str | None = None
    supersedes_id: str | None = None
    status: MemoryStatus = "active"
    source_message_id: str | None = None
    conversation_id: str | None = None
    task_id: str | None = None
    last_accessed_at: datetime | None = None
    access_count: int = 0
    superseded_conflicting: bool = False
    superseded_from_content: str | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return self.expires_at <= (now or datetime.now(timezone.utc))


@dataclass(frozen=True)
class MemoryCandidate:
    """检索到的候选记忆 + 相似度分数，供 prompt 注入前过滤。"""

    memory: MemoryRecord
    score: float | None = None
    retrieval_source: str | None = None


@dataclass(frozen=True)
class MemoryWriteIntent:
    """准备写入数据库的记忆草稿（来自规则引擎或 LLM 抽取），尚未持久化。"""

    type: MemoryType
    key: str
    content: str
    value_json: dict[str, Any] | None = None
    scope: MemoryScope = "user"
    source: MemorySource = "explicit"
    confidence: float = 0.95
    expires_at: datetime | None = None
    task_id: str | None = None


@dataclass(frozen=True)
class MemoryUsage:
    """本次回答实际使用到的记忆条目，返回给前端展示「参考了哪些记忆」。"""

    memory_id: str
    type: MemoryType
    key: str
    content: str
    source: MemorySource
    confidence: float
    scope: MemoryScope
    score: float | None = None
    last_accessed_at: datetime | None = None
    access_count: int = 0
    superseded_conflicting: bool = False
    superseded_from_content: str | None = None


@dataclass(frozen=True)
class FeedbackEventRecord:
    """用户对某次回答的 thumbs up/down 反馈，可联动调整相关记忆的置信度。"""

    feedback_id: str
    tenant_id: str
    user_id: str
    rating: int
    created_at: datetime
    conversation_id: str | None = None
    message_id: str | None = None
    memory_ids: tuple[str, ...] = ()
    comment: str | None = None


@dataclass(frozen=True)
class FeedbackMemoryAdjustment:
    """用户反馈触发后，对单条记忆置信度/状态的调整记录。"""

    memory_id: str
    status: str
    previous_confidence: float | None = None
    new_confidence: float | None = None
    memory: MemoryRecord | None = None


@dataclass(frozen=True)
class MemoryUpdate:
    """更新已有记忆时的部分字段 PATCH（UNSET 表示不修改该字段）。"""

    key: str | None = None
    content: str | None = None
    value_json: dict[str, Any] | None | _UnsetType = UNSET
    source: MemorySource | None = None
    confidence: float | None = None
    expires_at: datetime | None | _UnsetType = UNSET
    visibility: MemoryVisibility | None = None
    permissions: tuple[str, ...] | None = None
    status: MemoryStatus | None = None

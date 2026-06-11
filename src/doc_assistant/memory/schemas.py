from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final, Literal, get_args


MemoryScope = Literal["user", "org", "session", "task"]
MemoryType = Literal["preference", "fact", "task_state", "feedback", "correction"]
MemorySource = Literal["explicit", "inferred", "imported", "system_generated"]
MemoryStatus = Literal["active", "stale", "deleted"]
MemoryVisibility = Literal["private", "team", "org"]

VALID_MEMORY_SCOPES = set(get_args(MemoryScope))
VALID_MEMORY_TYPES = set(get_args(MemoryType))
VALID_MEMORY_SOURCES = set(get_args(MemorySource))
VALID_MEMORY_STATUSES = set(get_args(MemoryStatus))
VALID_MEMORY_VISIBILITIES = set(get_args(MemoryVisibility))


@dataclass(frozen=True)
class _UnsetType:
    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final = _UnsetType()


def is_unset(value: object) -> bool:
    return value is UNSET


@dataclass(frozen=True)
class MessageRecord:
    message_id: str
    conversation_id: str
    tenant_id: str
    user_id: str
    role: str
    content: str
    created_at: datetime


@dataclass(frozen=True)
class MemoryRecord:
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

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return self.expires_at <= (now or datetime.now(timezone.utc))


@dataclass(frozen=True)
class MemoryCandidate:
    memory: MemoryRecord
    score: float | None = None


@dataclass(frozen=True)
class MemoryWriteIntent:
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
    memory_id: str
    type: MemoryType
    key: str
    content: str
    source: MemorySource
    confidence: float
    scope: MemoryScope
    score: float | None = None


@dataclass(frozen=True)
class MemoryUpdate:
    key: str | None = None
    content: str | None = None
    value_json: dict[str, Any] | None | _UnsetType = UNSET
    source: MemorySource | None = None
    confidence: float | None = None
    expires_at: datetime | None | _UnsetType = UNSET
    visibility: MemoryVisibility | None = None
    permissions: tuple[str, ...] | None = None
    status: MemoryStatus | None = None

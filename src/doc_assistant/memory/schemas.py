from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal


MemoryScope = Literal["user", "org", "session", "task"]
MemoryType = Literal["preference", "fact", "task_state", "feedback", "correction"]
MemorySource = Literal["explicit", "inferred", "imported", "system_generated"]
MemoryStatus = Literal["active", "stale", "deleted"]
MemoryVisibility = Literal["private", "team", "org"]

VALID_MEMORY_SCOPES = {"user", "org", "session", "task"}
VALID_MEMORY_TYPES = {"preference", "fact", "task_state", "feedback", "correction"}
VALID_MEMORY_SOURCES = {"explicit", "inferred", "imported", "system_generated"}
VALID_MEMORY_STATUSES = {"active", "stale", "deleted"}
VALID_MEMORY_VISIBILITIES = {"private", "team", "org"}


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
    type: str
    key: str
    content: str
    source: str
    confidence: float
    scope: str
    score: float | None = None


@dataclass(frozen=True)
class MemoryUpdate:
    key: str | None = None
    content: str | None = None
    value_json: dict[str, Any] | None = None
    source: MemorySource | None = None
    confidence: float | None = None
    expires_at: datetime | None = None
    visibility: MemoryVisibility | None = None
    permissions: tuple[str, ...] | None = None
    status: MemoryStatus | None = None

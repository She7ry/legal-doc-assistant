from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from doc_assistant.config.settings import settings
from doc_assistant.memory.schemas import (
    MemoryCandidate,
    MemoryRecord,
    MemoryUpdate,
    MessageRecord,
    VALID_MEMORY_SCOPES,
    VALID_MEMORY_SOURCES,
    VALID_MEMORY_STATUSES,
    VALID_MEMORY_TYPES,
    VALID_MEMORY_VISIBILITIES,
)


class MemoryStore:
    """SQLite-backed repository for conversations and structured memories."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or settings.memory_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._ensure_schema()

    def ensure_user(self, tenant_id: str, user_id: str) -> None:
        now = _to_db_time(_utc_now())
        with self._connect() as connection, self._lock:
            connection.execute(
                """
                INSERT INTO users (tenant_id, user_id, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tenant_id, user_id)
                DO UPDATE SET updated_at = excluded.updated_at
                """,
                (tenant_id, user_id, now, now),
            )

    def ensure_conversation(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        title: str | None = None,
    ) -> None:
        self.ensure_user(tenant_id, user_id)
        now = _to_db_time(_utc_now())
        with self._connect() as connection, self._lock:
            connection.execute(
                """
                INSERT INTO conversations (
                    conversation_id, tenant_id, user_id, title, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 'active', ?, ?)
                ON CONFLICT(conversation_id)
                DO UPDATE SET updated_at = excluded.updated_at
                """,
                (conversation_id, tenant_id, user_id, title, now, now),
            )

    def add_message(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        message_id: str | None = None,
    ) -> MessageRecord:
        if role not in {"user", "assistant"}:
            raise ValueError("Message role must be 'user' or 'assistant'.")

        self.ensure_conversation(tenant_id, user_id, conversation_id)
        record = MessageRecord(
            message_id=message_id or uuid4().hex,
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            user_id=user_id,
            role=role,
            content=content,
            created_at=_utc_now(),
        )
        with self._connect() as connection, self._lock:
            connection.execute(
                """
                INSERT INTO messages (
                    message_id, conversation_id, tenant_id, user_id, role, content, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.message_id,
                    record.conversation_id,
                    record.tenant_id,
                    record.user_id,
                    record.role,
                    record.content,
                    _to_db_time(record.created_at),
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (_to_db_time(record.created_at), conversation_id),
            )
        return record

    def save_memory(self, memory: MemoryRecord) -> MemoryRecord:
        _validate_memory(memory)
        with self._connect() as connection, self._lock:
            connection.execute(
                """
                INSERT INTO memories (
                    memory_id, tenant_id, user_id, scope, type, key, content, value_json,
                    source, confidence, created_at, updated_at, expires_at, visibility,
                    permissions_json, embedding_id, supersedes_id, status, source_message_id,
                    conversation_id, task_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _memory_values(memory),
            )
        return memory

    def create_memory(
        self,
        *,
        tenant_id: str,
        user_id: str,
        scope: str,
        type: str,
        key: str,
        content: str,
        value_json: dict | None,
        source: str,
        confidence: float,
        expires_at: datetime | None = None,
        visibility: str = "private",
        permissions: tuple[str, ...] = ("read", "write", "delete"),
        embedding_id: str | None = None,
        supersedes_id: str | None = None,
        status: str = "active",
        source_message_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        memory_id: str | None = None,
    ) -> MemoryRecord:
        now = _utc_now()
        memory = MemoryRecord(
            memory_id=memory_id or uuid4().hex,
            tenant_id=tenant_id,
            user_id=user_id,
            scope=scope,  # type: ignore[arg-type]
            type=type,  # type: ignore[arg-type]
            key=_normalize_key(key),
            content=content.strip(),
            value_json=value_json,
            source=source,  # type: ignore[arg-type]
            confidence=confidence,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            visibility=visibility,  # type: ignore[arg-type]
            permissions=permissions,
            embedding_id=embedding_id,
            supersedes_id=supersedes_id,
            status=status,  # type: ignore[arg-type]
            source_message_id=source_message_id,
            conversation_id=conversation_id,
            task_id=task_id,
        )
        return self.save_memory(memory)

    def get_memory(self, tenant_id: str, user_id: str, memory_id: str) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM memories
                WHERE tenant_id = ?
                  AND memory_id = ?
                  AND (user_id = ? OR visibility IN ('team', 'org'))
                """,
                (tenant_id, memory_id, user_id),
            ).fetchone()
        return _row_to_memory(row) if row else None

    def get_memories_by_ids(
        self,
        tenant_id: str,
        user_id: str,
        memory_ids: list[str],
    ) -> list[MemoryRecord]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM memories
                WHERE tenant_id = ?
                  AND memory_id IN ({placeholders})
                  AND (user_id = ? OR visibility IN ('team', 'org'))
                """,
                (tenant_id, *memory_ids, user_id),
            ).fetchall()
        memories = [_row_to_memory(row) for row in rows]
        by_id = {memory.memory_id: memory for memory in memories}
        return [by_id[memory_id] for memory_id in memory_ids if memory_id in by_id]

    def list_memories(
        self,
        tenant_id: str,
        user_id: str,
        *,
        status: str | None = "active",
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        clauses = ["tenant_id = ?", "(user_id = ? OR visibility IN ('team', 'org'))"]
        values: list[object] = [tenant_id, user_id]
        if status is not None:
            clauses.append("status = ?")
            values.append(status)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM memories
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC
                """,
                values,
            ).fetchall()

        memories = [_row_to_memory(row) for row in rows]
        if include_expired:
            return memories
        return [memory for memory in memories if not memory.is_expired()]

    def find_active_memory_by_key(
        self,
        tenant_id: str,
        user_id: str,
        *,
        scope: str,
        type: str,
        key: str,
    ) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM memories
                WHERE tenant_id = ?
                  AND user_id = ?
                  AND scope = ?
                  AND type = ?
                  AND key = ?
                  AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (tenant_id, user_id, scope, type, _normalize_key(key)),
            ).fetchone()
        return _row_to_memory(row) if row else None

    def update_memory(
        self,
        tenant_id: str,
        user_id: str,
        memory_id: str,
        update: MemoryUpdate,
    ) -> MemoryRecord | None:
        current = self.get_memory(tenant_id, user_id, memory_id)
        if current is None or current.user_id != user_id:
            return None

        updated = MemoryRecord(
            memory_id=current.memory_id,
            tenant_id=current.tenant_id,
            user_id=current.user_id,
            scope=current.scope,
            type=current.type,
            key=_normalize_key(update.key) if update.key is not None else current.key,
            content=update.content.strip() if update.content is not None else current.content,
            value_json=update.value_json if update.value_json is not None else current.value_json,
            source=update.source if update.source is not None else current.source,
            confidence=update.confidence if update.confidence is not None else current.confidence,
            created_at=current.created_at,
            updated_at=_utc_now(),
            expires_at=update.expires_at if update.expires_at is not None else current.expires_at,
            visibility=update.visibility if update.visibility is not None else current.visibility,
            permissions=update.permissions if update.permissions is not None else current.permissions,
            embedding_id=current.embedding_id,
            supersedes_id=current.supersedes_id,
            status=update.status if update.status is not None else current.status,
            source_message_id=current.source_message_id,
            conversation_id=current.conversation_id,
            task_id=current.task_id,
        )
        _validate_memory(updated)
        with self._connect() as connection, self._lock:
            connection.execute(
                """
                UPDATE memories
                SET key = ?, content = ?, value_json = ?, source = ?, confidence = ?,
                    updated_at = ?, expires_at = ?, visibility = ?, permissions_json = ?,
                    status = ?
                WHERE memory_id = ? AND tenant_id = ? AND user_id = ?
                """,
                (
                    updated.key,
                    updated.content,
                    _json_dump(updated.value_json),
                    updated.source,
                    updated.confidence,
                    _to_db_time(updated.updated_at),
                    _to_db_time(updated.expires_at),
                    updated.visibility,
                    _json_dump(list(updated.permissions)),
                    updated.status,
                    updated.memory_id,
                    tenant_id,
                    user_id,
                ),
            )
        return updated

    def update_memory_embedding_id(
        self,
        tenant_id: str,
        user_id: str,
        memory_id: str,
        embedding_id: str | None,
    ) -> None:
        with self._connect() as connection, self._lock:
            connection.execute(
                """
                UPDATE memories
                SET embedding_id = ?, updated_at = ?
                WHERE memory_id = ? AND tenant_id = ? AND user_id = ?
                """,
                (_empty_to_none(embedding_id), _to_db_time(_utc_now()), memory_id, tenant_id, user_id),
            )

    def mark_memory_status(
        self,
        tenant_id: str,
        user_id: str,
        memory_id: str,
        status: str,
    ) -> MemoryRecord | None:
        if status not in VALID_MEMORY_STATUSES:
            raise ValueError(f"Invalid memory status: {status}")
        current = self.get_memory(tenant_id, user_id, memory_id)
        if current is None or current.user_id != user_id:
            return None
        return self.update_memory(
            tenant_id,
            user_id,
            memory_id,
            MemoryUpdate(status=status),  # type: ignore[arg-type]
        )

    def search_memories_lexical(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        *,
        limit: int = 5,
        min_confidence: float = 0.0,
    ) -> list[MemoryCandidate]:
        terms = [term.casefold() for term in query.split() if len(term.strip()) >= 2]
        memories = [
            memory
            for memory in self.list_memories(tenant_id, user_id, status="active")
            if memory.confidence >= min_confidence and not memory.is_expired()
        ]
        if not terms:
            return [MemoryCandidate(memory=memory, score=None) for memory in memories[:limit]]

        candidates: list[MemoryCandidate] = []
        for memory in memories:
            haystack = f"{memory.type} {memory.key} {memory.content}".casefold()
            score = sum(1 for term in terms if term in haystack) / len(terms)
            if score > 0:
                candidates.append(MemoryCandidate(memory=memory, score=score))

        candidates.sort(key=lambda candidate: (candidate.score or 0, candidate.memory.updated_at), reverse=True)
        return candidates[:limit]

    def log_retrieval(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        query: str,
        document_count: int,
        memory_count: int,
        selected_memory_ids: list[str],
    ) -> None:
        with self._connect() as connection, self._lock:
            connection.execute(
                """
                INSERT INTO retrieval_logs (
                    retrieval_id, tenant_id, user_id, conversation_id, query,
                    document_count, memory_count, selected_memory_ids_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    tenant_id,
                    user_id,
                    conversation_id,
                    query,
                    document_count,
                    memory_count,
                    _json_dump(selected_memory_ids),
                    _to_db_time(_utc_now()),
                ),
            )

    def _ensure_schema(self) -> None:
        with self._connect() as connection, self._lock:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    value_json TEXT,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    visibility TEXT NOT NULL DEFAULT 'private',
                    permissions_json TEXT NOT NULL,
                    embedding_id TEXT,
                    supersedes_id TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    source_message_id TEXT,
                    conversation_id TEXT,
                    task_id TEXT
                );

                CREATE TABLE IF NOT EXISTS task_states (
                    task_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT,
                    state_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT
                );

                CREATE TABLE IF NOT EXISTS retrieval_logs (
                    retrieval_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT,
                    query TEXT NOT NULL,
                    document_count INTEGER NOT NULL,
                    memory_count INTEGER NOT NULL,
                    selected_memory_ids_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback_events (
                    feedback_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT,
                    message_id TEXT,
                    rating INTEGER,
                    comment TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_subject
                    ON memories (tenant_id, user_id, status, scope, type, key);
                CREATE INDEX IF NOT EXISTS idx_memories_expiry
                    ON memories (tenant_id, status, expires_at);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages (conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_retrieval_logs_subject
                    ON retrieval_logs (tenant_id, user_id, created_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _memory_values(memory: MemoryRecord) -> tuple[object, ...]:
    return (
        memory.memory_id,
        memory.tenant_id,
        memory.user_id,
        memory.scope,
        memory.type,
        memory.key,
        memory.content,
        _json_dump(memory.value_json),
        memory.source,
        memory.confidence,
        _to_db_time(memory.created_at),
        _to_db_time(memory.updated_at),
        _to_db_time(memory.expires_at),
        memory.visibility,
        _json_dump(list(memory.permissions)),
        memory.embedding_id,
        memory.supersedes_id,
        memory.status,
        memory.source_message_id,
        memory.conversation_id,
        memory.task_id,
    )


def _row_to_memory(row: sqlite3.Row) -> MemoryRecord:
    permissions = tuple(_json_load(row["permissions_json"]) or ["read", "write", "delete"])
    return MemoryRecord(
        memory_id=row["memory_id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        scope=row["scope"],
        type=row["type"],
        key=row["key"],
        content=row["content"],
        value_json=_json_load(row["value_json"]),
        source=row["source"],
        confidence=float(row["confidence"]),
        created_at=_from_db_time(row["created_at"]),
        updated_at=_from_db_time(row["updated_at"]),
        expires_at=_from_db_time(row["expires_at"]) if row["expires_at"] else None,
        visibility=row["visibility"],
        permissions=permissions,
        embedding_id=row["embedding_id"],
        supersedes_id=row["supersedes_id"],
        status=row["status"],
        source_message_id=row["source_message_id"],
        conversation_id=row["conversation_id"],
        task_id=row["task_id"],
    )


def _validate_memory(memory: MemoryRecord) -> None:
    if memory.scope not in VALID_MEMORY_SCOPES:
        raise ValueError(f"Invalid memory scope: {memory.scope}")
    if memory.type not in VALID_MEMORY_TYPES:
        raise ValueError(f"Invalid memory type: {memory.type}")
    if memory.source not in VALID_MEMORY_SOURCES:
        raise ValueError(f"Invalid memory source: {memory.source}")
    if memory.status not in VALID_MEMORY_STATUSES:
        raise ValueError(f"Invalid memory status: {memory.status}")
    if memory.visibility not in VALID_MEMORY_VISIBILITIES:
        raise ValueError(f"Invalid memory visibility: {memory.visibility}")
    if not 0 <= memory.confidence <= 1:
        raise ValueError("Memory confidence must be between 0 and 1.")
    if not memory.key:
        raise ValueError("Memory key is required.")
    if not memory.content:
        raise ValueError("Memory content is required.")


def _normalize_key(value: str | None) -> str:
    key = (value or "").strip().lower().replace(" ", "_")
    return key[:120]


def _json_dump(value: object | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_load(value: str | None):
    if not value:
        return None
    return json.loads(value)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_db_time(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _from_db_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _empty_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


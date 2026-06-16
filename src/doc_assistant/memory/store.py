from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock, local
from uuid import uuid4

from doc_assistant.config.settings import settings
from doc_assistant.memory.schemas import (
    VALID_CONVERSATION_STATUSES,
    VALID_MEMORY_SCOPES,
    VALID_MEMORY_SOURCES,
    VALID_MEMORY_STATUSES,
    VALID_MEMORY_TYPES,
    VALID_MEMORY_VISIBILITIES,
    ConversationRecord,
    FeedbackEventRecord,
    MemoryCandidate,
    MemoryRecord,
    MemoryUpdate,
    MessageRecord,
    is_unset,
)

try:
    import jieba as _jieba
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent.
    _jieba = None

SCHEMA_VERSION = 2


class MemoryStore:
    """SQLite-backed repository for conversations and structured memories."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or settings.memory_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._local = local()
        self._ensure_schema()

    def ensure_user(self, tenant_id: str, user_id: str) -> None:
        with self._connect() as connection, self._lock:
            self._ensure_user_row(connection, tenant_id, user_id)

    def ensure_conversation(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        title: str | None = None,
    ) -> None:
        with self._connect() as connection, self._lock:
            self._ensure_user_row(connection, tenant_id, user_id)
            self._ensure_conversation_row(connection, tenant_id, user_id, conversation_id, title)

    def get_conversation(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
    ) -> ConversationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT c.*, COUNT(m.message_id) AS message_count
                FROM conversations c
                LEFT JOIN messages m
                  ON m.conversation_id = c.conversation_id
                 AND m.tenant_id = c.tenant_id
                 AND m.user_id = c.user_id
                WHERE c.tenant_id = ?
                  AND c.user_id = ?
                  AND c.conversation_id = ?
                GROUP BY c.conversation_id
                """,
                (tenant_id, user_id, conversation_id),
            ).fetchone()
        return _row_to_conversation(row) if row else None

    def list_conversations(
        self,
        tenant_id: str,
        user_id: str,
        *,
        status: str | None = "active",
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ConversationRecord]:
        _validate_conversation_status(status)
        clauses = ["c.tenant_id = ?", "c.user_id = ?"]
        values: list[object] = [tenant_id, user_id]
        if status is not None:
            clauses.append("c.status = ?")
            values.append(status)

        pagination = ""
        if limit is not None:
            pagination = " LIMIT ? OFFSET ?"
            values.extend([max(0, limit), max(0, offset)])
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT c.*, COUNT(m.message_id) AS message_count
                FROM conversations c
                LEFT JOIN messages m
                  ON m.conversation_id = c.conversation_id
                 AND m.tenant_id = c.tenant_id
                 AND m.user_id = c.user_id
                WHERE {' AND '.join(clauses)}
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC
                {pagination}
                """,
                values,
            ).fetchall()
        return [_row_to_conversation(row) for row in rows]

    def count_conversations(
        self,
        tenant_id: str,
        user_id: str,
        *,
        status: str | None = "active",
    ) -> int:
        _validate_conversation_status(status)
        clauses = ["tenant_id = ?", "user_id = ?"]
        values: list[object] = [tenant_id, user_id]
        if status is not None:
            clauses.append("status = ?")
            values.append(status)

        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS count FROM conversations
                WHERE {' AND '.join(clauses)}
                """,
                values,
            ).fetchone()
        return int(row["count"] if row else 0)

    def update_conversation(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        *,
        title: str | None = None,
        status: str | None = None,
    ) -> ConversationRecord | None:
        _validate_conversation_status(status)
        current = self.get_conversation(tenant_id, user_id, conversation_id)
        if current is None:
            return None

        updated_title = _normalize_title(title) if title is not None else current.title
        updated_status = status if status is not None else current.status
        now = _to_db_time(_utc_now())
        with self._connect() as connection, self._lock:
            connection.execute(
                """
                UPDATE conversations
                SET title = ?, status = ?, updated_at = ?
                WHERE tenant_id = ? AND user_id = ? AND conversation_id = ?
                """,
                (updated_title, updated_status, now, tenant_id, user_id, conversation_id),
            )
        return self.get_conversation(tenant_id, user_id, conversation_id)

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
            self._ensure_user_row(connection, tenant_id, user_id)
            self._ensure_conversation_row(connection, tenant_id, user_id, conversation_id)
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

    def list_messages(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        *,
        limit: int = 20,
    ) -> list[MessageRecord]:
        if limit <= 0:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM (
                    SELECT messages.*, rowid AS _message_rowid FROM messages
                    WHERE tenant_id = ?
                      AND user_id = ?
                      AND conversation_id = ?
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC, _message_rowid ASC
                """,
                (tenant_id, user_id, conversation_id, limit),
            ).fetchall()
        return [_row_to_message(row) for row in rows]

    def count_messages(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
    ) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM messages
                WHERE tenant_id = ?
                  AND user_id = ?
                  AND conversation_id = ?
                """,
                (tenant_id, user_id, conversation_id),
            ).fetchone()
        return int(row["count"] if row else 0)

    def _ensure_user_row(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        user_id: str,
    ) -> None:
        now = _to_db_time(_utc_now())
        connection.execute(
            """
            INSERT INTO users (tenant_id, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(tenant_id, user_id)
            DO UPDATE SET updated_at = excluded.updated_at
            """,
            (tenant_id, user_id, now, now),
        )

    def _ensure_conversation_row(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        title: str | None = None,
    ) -> None:
        now = _to_db_time(_utc_now())
        title = _normalize_title(title)
        existing = connection.execute(
            """
            SELECT tenant_id, user_id FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if existing and (existing["tenant_id"] != tenant_id or existing["user_id"] != user_id):
            raise ValueError("Conversation id belongs to a different tenant or user.")

        connection.execute(
            """
            INSERT INTO conversations (
                conversation_id, tenant_id, user_id, title, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, 'active', ?, ?)
            ON CONFLICT(conversation_id)
            DO UPDATE SET
                updated_at = excluded.updated_at,
                title = COALESCE(excluded.title, conversations.title)
            """,
            (conversation_id, tenant_id, user_id, title, now, now),
        )

    def _message_exists(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        user_id: str,
        message_id: str,
    ) -> bool:
        row = connection.execute(
            """
            SELECT 1 FROM messages
            WHERE tenant_id = ?
              AND user_id = ?
              AND message_id = ?
            LIMIT 1
            """,
            (tenant_id, user_id, message_id),
        ).fetchone()
        return row is not None

    def save_memory(self, memory: MemoryRecord) -> MemoryRecord:
        _validate_memory(memory)
        with self._connect() as connection, self._lock:
            connection.execute(
                """
                INSERT INTO memories (
                    memory_id, tenant_id, user_id, scope, type, key, content, value_json,
                    source, confidence, created_at, updated_at, expires_at, visibility,
                    permissions_json, embedding_id, supersedes_id, status, source_message_id,
                    conversation_id, task_id, last_accessed_at, access_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _memory_values(memory),
            )
            _upsert_memory_fts(connection, memory)
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
            last_accessed_at=None,
            access_count=0,
            superseded_conflicting=_superseded_conflicting(value_json),
            superseded_from_content=_superseded_from_content(value_json),
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
        limit: int | None = None,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        clauses = ["tenant_id = ?", "(user_id = ? OR visibility IN ('team', 'org'))"]
        values: list[object] = [tenant_id, user_id]
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            values.append(_to_db_time(_utc_now()))

        pagination = ""
        if limit is not None:
            pagination = " LIMIT ? OFFSET ?"
            values.extend([max(0, limit), max(0, offset)])
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM memories
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC
                {pagination}
                """,
                values,
            ).fetchall()

        return [_row_to_memory(row) for row in rows]

    def count_memories(
        self,
        tenant_id: str,
        user_id: str,
        *,
        status: str | None = "active",
        include_expired: bool = False,
    ) -> int:
        clauses = ["tenant_id = ?", "(user_id = ? OR visibility IN ('team', 'org'))"]
        values: list[object] = [tenant_id, user_id]
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            values.append(_to_db_time(_utc_now()))

        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS count FROM memories
                WHERE {' AND '.join(clauses)}
                """,
                values,
            ).fetchone()
        return int(row["count"] if row else 0)

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
            value_json=current.value_json if is_unset(update.value_json) else update.value_json,
            source=update.source if update.source is not None else current.source,
            confidence=update.confidence if update.confidence is not None else current.confidence,
            created_at=current.created_at,
            updated_at=_utc_now(),
            expires_at=current.expires_at if is_unset(update.expires_at) else update.expires_at,
            visibility=update.visibility if update.visibility is not None else current.visibility,
            permissions=update.permissions if update.permissions is not None else current.permissions,
            embedding_id=current.embedding_id,
            supersedes_id=current.supersedes_id,
            status=update.status if update.status is not None else current.status,
            source_message_id=current.source_message_id,
            conversation_id=current.conversation_id,
            task_id=current.task_id,
            last_accessed_at=current.last_accessed_at,
            access_count=current.access_count,
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
            _upsert_memory_fts(connection, updated)
        return updated

    def touch_memories(
        self,
        tenant_id: str,
        user_id: str,
        memory_ids: list[str],
    ) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        now = _to_db_time(_utc_now())
        with self._connect() as connection, self._lock:
            connection.execute(
                f"""
                UPDATE memories
                SET last_accessed_at = ?,
                    access_count = access_count + 1
                WHERE tenant_id = ?
                  AND user_id = ?
                  AND memory_id IN ({placeholders})
                """,
                (now, tenant_id, user_id, *memory_ids),
            )

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
                SET embedding_id = ?
                WHERE memory_id = ? AND tenant_id = ? AND user_id = ?
                """,
                (_empty_to_none(embedding_id), memory_id, tenant_id, user_id),
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
        updated = replace(current, status=status, updated_at=_utc_now())  # type: ignore[arg-type]
        _validate_memory(updated)
        with self._connect() as connection, self._lock:
            connection.execute(
                """
                UPDATE memories
                SET status = ?, updated_at = ?
                WHERE memory_id = ? AND tenant_id = ? AND user_id = ?
                """,
                (
                    updated.status,
                    _to_db_time(updated.updated_at),
                    updated.memory_id,
                    tenant_id,
                    user_id,
                ),
            )
            _upsert_memory_fts(connection, updated)
        return updated

    def mark_expired_memories_stale(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[MemoryRecord]:
        now = _to_db_time(_utc_now())
        with self._connect() as connection, self._lock:
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE tenant_id = ?
                  AND user_id = ?
                  AND status = 'active'
                  AND expires_at IS NOT NULL
                  AND expires_at <= ?
                """,
                (tenant_id, user_id, now),
            ).fetchall()
            if rows:
                connection.execute(
                    """
                    UPDATE memories
                    SET status = 'stale', updated_at = ?
                    WHERE tenant_id = ?
                      AND user_id = ?
                      AND status = 'active'
                      AND expires_at IS NOT NULL
                      AND expires_at <= ?
                    """,
                    (now, tenant_id, user_id, now),
                )
        return [replace(_row_to_memory(row), status="stale", updated_at=_from_db_time(now)) for row in rows]

    def mark_task_memories_stale(
        self,
        tenant_id: str,
        user_id: str,
        task_id: str,
    ) -> list[MemoryRecord]:
        now = _to_db_time(_utc_now())
        with self._connect() as connection, self._lock:
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE tenant_id = ?
                  AND user_id = ?
                  AND task_id = ?
                  AND scope = 'task'
                  AND status = 'active'
                """,
                (tenant_id, user_id, task_id),
            ).fetchall()
            if rows:
                connection.execute(
                    """
                    UPDATE memories
                    SET status = 'stale', updated_at = ?
                    WHERE tenant_id = ?
                      AND user_id = ?
                      AND task_id = ?
                      AND scope = 'task'
                      AND status = 'active'
                    """,
                    (now, tenant_id, user_id, task_id),
                )
        return [replace(_row_to_memory(row), status="stale", updated_at=_from_db_time(now)) for row in rows]

    def list_active_memories_for_user(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[MemoryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE tenant_id = ?
                  AND user_id = ?
                  AND status = 'active'
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY updated_at DESC
                """,
                (tenant_id, user_id, _to_db_time(_utc_now())),
            ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def list_vector_cleanup_memory_ids(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_id FROM memories
                WHERE tenant_id = ?
                  AND user_id = ?
                  AND (
                    status != 'active'
                    OR (expires_at IS NOT NULL AND expires_at <= ?)
                  )
                """,
                (tenant_id, user_id, _to_db_time(_utc_now())),
            ).fetchall()
        return [str(row["memory_id"]) for row in rows]

    def search_memories_lexical(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        *,
        limit: int = 5,
        min_confidence: float = 0.0,
    ) -> list[MemoryCandidate]:
        terms = _lexical_terms(query)
        clauses = [
            "tenant_id = ?",
            "(user_id = ? OR visibility IN ('team', 'org'))",
            "status = 'active'",
            "confidence >= ?",
            "(expires_at IS NULL OR expires_at > ?)",
        ]
        values: list[object] = [tenant_id, user_id, min_confidence, _to_db_time(_utc_now())]
        if terms:
            fts_results = self._search_memories_fts(
                tenant_id,
                user_id,
                terms,
                limit=limit,
                min_confidence=min_confidence,
            )
            if fts_results:
                return fts_results

            term_clauses = []
            for term in terms:
                term_clauses.append("LOWER(type || ' ' || key || ' ' || content) LIKE ?")
                values.append(f"%{term}%")
            clauses.append(f"({' OR '.join(term_clauses)})")

        sql_limit = limit if not terms else max(limit * 20, limit)
        values.append(sql_limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM memories
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        memories = [_row_to_memory(row) for row in rows]
        if not terms:
            return [
                MemoryCandidate(memory=memory, score=None, retrieval_source="structured")
                for memory in memories[:limit]
            ]

        candidates: list[MemoryCandidate] = []
        for memory in memories:
            haystack = f"{memory.type} {memory.key} {memory.content}".casefold()
            score = sum(1 for term in terms if term in haystack) / len(terms)
            if score > 0:
                candidates.append(MemoryCandidate(memory=memory, score=score, retrieval_source="lexical"))

        candidates.sort(key=lambda candidate: (candidate.score or 0, candidate.memory.updated_at), reverse=True)
        return candidates[:limit]

    def _search_memories_fts(
        self,
        tenant_id: str,
        user_id: str,
        terms: list[str],
        *,
        limit: int,
        min_confidence: float,
    ) -> list[MemoryCandidate]:
        match_query = _fts_match_query(terms)
        if not match_query:
            return []
        values: list[object] = [
            match_query,
            tenant_id,
            user_id,
            min_confidence,
            _to_db_time(_utc_now()),
            max(limit * 20, limit),
        ]
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT m.*, bm25(memories_fts) AS fts_rank
                    FROM memories_fts
                    JOIN memories AS m ON m.memory_id = memories_fts.memory_id
                    WHERE memories_fts MATCH ?
                      AND m.tenant_id = ?
                      AND (m.user_id = ? OR m.visibility IN ('team', 'org'))
                      AND m.status = 'active'
                      AND m.confidence >= ?
                      AND (m.expires_at IS NULL OR m.expires_at > ?)
                    ORDER BY fts_rank ASC, m.updated_at DESC
                    LIMIT ?
                    """,
                    values,
                ).fetchall()
        except sqlite3.OperationalError:
            return []

        candidates = []
        for row in rows:
            rank = row["fts_rank"]
            score = 1.0 / (1.0 + max(float(rank or 0.0), 0.0))
            candidates.append(
                MemoryCandidate(memory=_row_to_memory(row), score=score, retrieval_source="fts")
            )
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
        selected_memory_sources: dict[str, int] | None = None,
    ) -> None:
        with self._connect() as connection, self._lock:
            self._ensure_user_row(connection, tenant_id, user_id)
            connection.execute(
                """
                INSERT INTO retrieval_logs (
                    retrieval_id, tenant_id, user_id, conversation_id, query,
                    document_count, memory_count, selected_memory_ids_json,
                    memory_sources_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    _json_dump(selected_memory_sources or {}),
                    _to_db_time(_utc_now()),
                ),
            )

    def get_memory_stats(self, tenant_id: str, user_id: str) -> dict[str, object]:
        now = _utc_now()
        now_db = _to_db_time(now)
        since_7d = _to_db_time(now - timedelta(days=7))
        since_30d = _to_db_time(now - timedelta(days=30))
        with self._connect() as connection:
            status_rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM memories
                WHERE tenant_id = ? AND user_id = ?
                GROUP BY status
                """,
                (tenant_id, user_id),
            ).fetchall()
            scope_rows = connection.execute(
                """
                SELECT scope, COUNT(*) AS count
                FROM memories
                WHERE tenant_id = ? AND user_id = ?
                GROUP BY scope
                """,
                (tenant_id, user_id),
            ).fetchall()
            type_rows = connection.execute(
                """
                SELECT type, COUNT(*) AS count
                FROM memories
                WHERE tenant_id = ? AND user_id = ?
                GROUP BY type
                """,
                (tenant_id, user_id),
            ).fetchall()
            total_row = connection.execute(
                """
                SELECT COUNT(*) AS count, AVG(confidence) AS average_confidence
                FROM memories
                WHERE tenant_id = ? AND user_id = ?
                """,
                (tenant_id, user_id),
            ).fetchone()
            active_row = connection.execute(
                """
                SELECT COUNT(*) AS count, AVG(confidence) AS average_confidence
                FROM memories
                WHERE tenant_id = ?
                  AND user_id = ?
                  AND status = 'active'
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (tenant_id, user_id, now_db),
            ).fetchone()
            expired_row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM memories
                WHERE tenant_id = ?
                  AND user_id = ?
                  AND status = 'active'
                  AND expires_at IS NOT NULL
                  AND expires_at <= ?
                """,
                (tenant_id, user_id, now_db),
            ).fetchone()
            access_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS count,
                    SUM(CASE WHEN last_accessed_at IS NULL THEN 1 ELSE 0 END) AS never_accessed,
                    SUM(CASE WHEN last_accessed_at IS NOT NULL THEN 1 ELSE 0 END) AS accessed,
                    SUM(CASE WHEN last_accessed_at >= ? THEN 1 ELSE 0 END) AS accessed_last_7d,
                    SUM(CASE WHEN last_accessed_at >= ? THEN 1 ELSE 0 END) AS accessed_last_30d,
                    SUM(access_count) AS total_access_count,
                    AVG(access_count) AS average_access_count,
                    MAX(access_count) AS max_access_count
                FROM memories
                WHERE tenant_id = ?
                  AND user_id = ?
                  AND status = 'active'
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (since_7d, since_30d, tenant_id, user_id, now_db),
            ).fetchone()
            retrieval_row = connection.execute(
                """
                SELECT
                    COUNT(*) AS count,
                    SUM(CASE WHEN memory_count > 0 THEN 1 ELSE 0 END) AS with_memory,
                    SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS last_7d,
                    SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS last_30d,
                    AVG(memory_count) AS average_memory_count,
                    AVG(document_count) AS average_document_count,
                    MAX(created_at) AS last_retrieval_at
                FROM retrieval_logs
                WHERE tenant_id = ? AND user_id = ?
                """,
                (since_7d, since_30d, tenant_id, user_id),
            ).fetchone()
            source_rows = connection.execute(
                """
                SELECT memory_count, memory_sources_json
                FROM retrieval_logs
                WHERE tenant_id = ? AND user_id = ?
                """,
                (tenant_id, user_id),
            ).fetchall()

        status_counts = _count_map(status_rows, "status")
        for status in VALID_MEMORY_STATUSES:
            status_counts.setdefault(status, 0)
        total_retrievals = _row_int(retrieval_row, "count")
        retrievals_with_memory = _row_int(retrieval_row, "with_memory")
        source_counts = _memory_source_counts(source_rows)
        selected_source_total = sum(source_counts.values())
        source_ratios = {
            source: count / selected_source_total
            for source, count in source_counts.items()
            if selected_source_total > 0
        }
        return {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "generated_at": now,
            "total_memories": _row_int(total_row, "count"),
            "active_memories": _row_int(active_row, "count"),
            "stale_memories": status_counts.get("stale", 0),
            "deleted_memories": status_counts.get("deleted", 0),
            "expired_active_memories": _row_int(expired_row, "count"),
            "status_counts": status_counts,
            "scope_counts": _count_map(scope_rows, "scope"),
            "type_counts": _count_map(type_rows, "type"),
            "average_confidence": _row_float(total_row, "average_confidence"),
            "average_active_confidence": _row_float(active_row, "average_confidence"),
            "access": {
                "tracked_memories": _row_int(access_row, "count"),
                "never_accessed": _row_int(access_row, "never_accessed"),
                "accessed": _row_int(access_row, "accessed"),
                "accessed_last_7d": _row_int(access_row, "accessed_last_7d"),
                "accessed_last_30d": _row_int(access_row, "accessed_last_30d"),
                "total_access_count": _row_int(access_row, "total_access_count"),
                "average_access_count": _row_float(access_row, "average_access_count"),
                "max_access_count": _row_int(access_row, "max_access_count"),
            },
            "retrievals": {
                "total": total_retrievals,
                "with_memory": retrievals_with_memory,
                "last_7d": _row_int(retrieval_row, "last_7d"),
                "last_30d": _row_int(retrieval_row, "last_30d"),
                "hit_rate": retrievals_with_memory / total_retrievals
                if total_retrievals > 0
                else 0.0,
                "average_memory_count": _row_float(retrieval_row, "average_memory_count"),
                "average_document_count": _row_float(retrieval_row, "average_document_count"),
                "last_retrieval_at": _row_datetime(retrieval_row, "last_retrieval_at"),
                "selected_memory_source_counts": source_counts,
                "selected_memory_source_ratios": source_ratios,
            },
        }

    def record_feedback_event(
        self,
        *,
        tenant_id: str,
        user_id: str,
        rating: int,
        conversation_id: str | None = None,
        message_id: str | None = None,
        memory_ids: tuple[str, ...] = (),
        comment: str | None = None,
        feedback_id: str | None = None,
    ) -> FeedbackEventRecord:
        if rating not in {-1, 1}:
            raise ValueError("Feedback rating must be 1 or -1.")
        record = FeedbackEventRecord(
            feedback_id=feedback_id or uuid4().hex,
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            rating=rating,
            memory_ids=tuple(dict.fromkeys(memory_ids)),
            comment=comment.strip() if comment and comment.strip() else None,
            created_at=_utc_now(),
        )
        with self._connect() as connection, self._lock:
            self._ensure_user_row(connection, tenant_id, user_id)
            if conversation_id:
                self._ensure_conversation_row(connection, tenant_id, user_id, conversation_id)
            if message_id and not self._message_exists(connection, tenant_id, user_id, message_id):
                raise ValueError("Feedback message_id was not found for this tenant and user.")
            connection.execute(
                """
                INSERT INTO feedback_events (
                    feedback_id, tenant_id, user_id, conversation_id, message_id,
                    rating, comment, memory_ids_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.feedback_id,
                    record.tenant_id,
                    record.user_id,
                    record.conversation_id,
                    record.message_id,
                    record.rating,
                    record.comment,
                    _json_dump(list(record.memory_ids)),
                    _to_db_time(record.created_at),
                ),
            )
        return record

    def _ensure_schema(self) -> None:
        with self._connect() as connection, self._lock:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                );

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
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (tenant_id, user_id)
                        REFERENCES users(tenant_id, user_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (tenant_id, user_id)
                        REFERENCES users(tenant_id, user_id)
                        ON DELETE CASCADE
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
                    task_id TEXT,
                    last_accessed_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (source_message_id)
                        REFERENCES messages(message_id)
                        ON DELETE SET NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE SET NULL
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
                    expires_at TEXT,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE SET NULL,
                    FOREIGN KEY (tenant_id, user_id)
                        REFERENCES users(tenant_id, user_id)
                        ON DELETE CASCADE
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
                    memory_sources_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE SET NULL,
                    FOREIGN KEY (tenant_id, user_id)
                        REFERENCES users(tenant_id, user_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS feedback_events (
                    feedback_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT,
                    message_id TEXT,
                    rating INTEGER,
                    comment TEXT,
                    memory_ids_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE SET NULL,
                    FOREIGN KEY (message_id)
                        REFERENCES messages(message_id)
                        ON DELETE SET NULL,
                    FOREIGN KEY (tenant_id, user_id)
                        REFERENCES users(tenant_id, user_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_memories_subject
                    ON memories (tenant_id, user_id, status, scope, type, key);
                CREATE INDEX IF NOT EXISTS idx_memories_expiry
                    ON memories (tenant_id, status, expires_at);
                CREATE INDEX IF NOT EXISTS idx_conversations_subject
                    ON conversations (tenant_id, user_id, status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages (conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_retrieval_logs_subject
                    ON retrieval_logs (tenant_id, user_id, created_at);

                """
            )
            _ensure_memory_columns(connection)
            _ensure_retrieval_log_columns(connection)
            _ensure_feedback_event_columns(connection)
            _ensure_memory_fts_schema(connection)
            _rebuild_memory_fts(connection)
            connection.execute(
                """
                INSERT OR IGNORE INTO schema_migrations (version, name, applied_at)
                VALUES (?, ?, ?)
                """,
                (SCHEMA_VERSION, "memory_schema_v2", _to_db_time(_utc_now())),
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = self._thread_connection()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def _thread_connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            return connection

        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.commit()
        self._local.connection = connection
        return connection

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            return
        connection.close()
        self._local.connection = None


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
        _to_db_time(memory.last_accessed_at),
        memory.access_count,
    )


def _row_to_memory(row: sqlite3.Row) -> MemoryRecord:
    permissions = tuple(_json_load(row["permissions_json"]) or ["read", "write", "delete"])
    value_json = _json_load(row["value_json"])
    return MemoryRecord(
        memory_id=row["memory_id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        scope=row["scope"],
        type=row["type"],
        key=row["key"],
        content=row["content"],
        value_json=value_json,
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
        last_accessed_at=(
            _from_db_time(row["last_accessed_at"])
            if _row_has_column(row, "last_accessed_at") and row["last_accessed_at"]
            else None
        ),
        access_count=(
            int(row["access_count"])
            if _row_has_column(row, "access_count") and row["access_count"] is not None
            else 0
        ),
        superseded_conflicting=_superseded_conflicting(value_json),
        superseded_from_content=_superseded_from_content(value_json),
    )


def _row_to_conversation(row: sqlite3.Row) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=row["conversation_id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        title=row["title"],
        status=row["status"],
        created_at=_from_db_time(row["created_at"]),
        updated_at=_from_db_time(row["updated_at"]),
        message_count=(
            int(row["message_count"])
            if _row_has_column(row, "message_count") and row["message_count"] is not None
            else 0
        ),
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    return MessageRecord(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        role=row["role"],
        content=row["content"],
        created_at=_from_db_time(row["created_at"]),
    )


def _row_has_column(row: sqlite3.Row, name: str) -> bool:
    return name in dict(row)


def _count_map(rows: list[sqlite3.Row], key: str) -> dict[str, int]:
    return {str(row[key]): int(row["count"] or 0) for row in rows}


def _row_int(row: sqlite3.Row | None, key: str) -> int:
    if row is None or row[key] is None:
        return 0
    return int(row[key])


def _row_float(row: sqlite3.Row | None, key: str) -> float:
    if row is None or row[key] is None:
        return 0.0
    return float(row[key])


def _row_datetime(row: sqlite3.Row | None, key: str) -> datetime | None:
    if row is None or row[key] is None:
        return None
    return _from_db_time(str(row[key]))


def _memory_source_counts(rows: list[sqlite3.Row]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        memory_count = int(row["memory_count"] or 0)
        payload = _json_load(row["memory_sources_json"])
        if not isinstance(payload, dict):
            payload = {}
        if not payload and memory_count > 0:
            counts["unknown"] = counts.get("unknown", 0) + memory_count
            continue
        for source, count in payload.items():
            try:
                amount = int(count)
            except (TypeError, ValueError):
                continue
            if amount > 0:
                counts[str(source)] = counts.get(str(source), 0) + amount
    return counts


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


def _validate_conversation_status(status: str | None) -> None:
    if status is not None and status not in VALID_CONVERSATION_STATUSES:
        raise ValueError(f"Invalid conversation status: {status}")


def _normalize_title(value: str | None) -> str | None:
    title = (value or "").strip()
    return title[:200] or None


def _normalize_key(value: str | None) -> str:
    key = (value or "").strip().lower().replace(" ", "_")
    return key[:120]


def _lexical_terms(query: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    raw_terms = [*query.split(), *_jieba_terms(query)]
    for term in raw_terms:
        for normalized in _term_variants(term):
            if normalized not in seen:
                terms.append(normalized)
                seen.add(normalized)
    return terms


def _term_variants(term: str) -> list[str]:
    normalized = term.strip().casefold()
    if len(normalized) < 2:
        return []
    variants = [normalized]
    cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    for run in cjk_runs:
        variants.extend(run[index : index + 2] for index in range(len(run) - 1))
    return variants


def _fts_match_query(terms: list[str]) -> str:
    quoted = []
    for term in terms[:12]:
        clean = term.replace('"', " ").strip()
        if clean:
            quoted.append(f'"{clean}"')
    return " OR ".join(quoted)


def _memory_fts_text(memory: MemoryRecord) -> str:
    source = f"{memory.type} {memory.key} {memory.content}".casefold()
    segmented = " ".join(_jieba_terms(source))
    terms = _lexical_terms(source)
    if segmented:
        terms.append(segmented)
    return " ".join([source, *terms])


def _jieba_terms(text: str) -> list[str]:
    if _jieba is None:
        return []
    return [
        term.strip().casefold()
        for term in _jieba.cut(text)
        if len(term.strip()) >= 2
    ]


def _superseded_conflicting(value_json: object) -> bool:
    return isinstance(value_json, dict) and bool(value_json.get("_superseded_conflicting"))


def _superseded_from_content(value_json: object) -> str | None:
    if not isinstance(value_json, dict):
        return None
    value = value_json.get("_superseded_from")
    return str(value) if value else None


def _upsert_memory_fts(connection: sqlite3.Connection, memory: MemoryRecord) -> None:
    try:
        connection.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory.memory_id,))
        connection.execute(
            """
            INSERT INTO memories_fts (memory_id, type, key, content, search_text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                memory.memory_id,
                memory.type,
                memory.key,
                memory.content,
                _memory_fts_text(memory),
            ),
        )
    except sqlite3.OperationalError:
        return


def _ensure_memory_fts_schema(connection: sqlite3.Connection) -> None:
    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                memory_id UNINDEXED,
                type,
                key,
                content,
                search_text,
                tokenize='unicode61'
            )
            """
        )
    except sqlite3.OperationalError:
        return


def _ensure_memory_columns(connection: sqlite3.Connection) -> None:
    try:
        rows = connection.execute("PRAGMA table_info(memories)").fetchall()
    except sqlite3.OperationalError:
        return
    existing = {str(row["name"]) for row in rows}
    migrations = {
        "last_accessed_at": "ALTER TABLE memories ADD COLUMN last_accessed_at TEXT",
        "access_count": "ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0",
    }
    for column, statement in migrations.items():
        if column in existing:
            continue
        try:
            connection.execute(statement)
        except sqlite3.OperationalError:
            continue


def _ensure_retrieval_log_columns(connection: sqlite3.Connection) -> None:
    try:
        rows = connection.execute("PRAGMA table_info(retrieval_logs)").fetchall()
    except sqlite3.OperationalError:
        return
    existing = {str(row["name"]) for row in rows}
    if "memory_sources_json" in existing:
        return
    try:
        connection.execute(
            "ALTER TABLE retrieval_logs ADD COLUMN memory_sources_json TEXT NOT NULL DEFAULT '{}'"
        )
    except sqlite3.OperationalError:
        return


def _ensure_feedback_event_columns(connection: sqlite3.Connection) -> None:
    try:
        rows = connection.execute("PRAGMA table_info(feedback_events)").fetchall()
    except sqlite3.OperationalError:
        return
    existing = {str(row["name"]) for row in rows}
    if "memory_ids_json" in existing:
        return
    try:
        connection.execute(
            "ALTER TABLE feedback_events ADD COLUMN memory_ids_json TEXT NOT NULL DEFAULT '[]'"
        )
    except sqlite3.OperationalError:
        return


def _rebuild_memory_fts(connection: sqlite3.Connection) -> None:
    try:
        memory_count = connection.execute("SELECT COUNT(*) AS count FROM memories").fetchone()
        fts_count = connection.execute("SELECT COUNT(*) AS count FROM memories_fts").fetchone()
        if memory_count and fts_count and memory_count["count"] == fts_count["count"]:
            return
        rows = connection.execute("SELECT * FROM memories").fetchall()
    except sqlite3.OperationalError:
        return
    try:
        connection.execute("DELETE FROM memories_fts")
        for row in rows:
            _upsert_memory_fts(connection, _row_to_memory(row))
    except sqlite3.OperationalError:
        return


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

"""记忆模块 SQLite 持久层：对话、消息和记忆条目。

``MemoryStore`` 负责 CRUD 与 schema 迁移；``MemoryService`` 在其上封装
业务逻辑（检索、抽取、衰减）。向量语义检索见 ``memory/vector_store.py``。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, local
from uuid import uuid4

from ai.config.settings import settings
from ai.memory.schemas import (
    VALID_CONVERSATION_STATUSES,
    VALID_MEMORY_SCOPES,
    VALID_MEMORY_SOURCES,
    VALID_MEMORY_STATUSES,
    VALID_MEMORY_TYPES,
    VALID_MEMORY_VISIBILITIES,
    ConversationRecord,
    MemoryRecord,
    MemoryUpdate,
    MessageRecord,
    is_unset,
)


class MemoryStore:
    """记忆的 SQLite 底层仓库（不含业务策略）。

    表结构：users、conversations、messages、memories；
    MemoryService 调用本类做 CRUD，向量检索由 MemoryVectorStore 负责。
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or settings.memory_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._local = local()
        self._ensure_schema()

    def ensure_conversation(
        self,
        user_id: str,
        conversation_id: str,
        title: str | None = None,
    ) -> None:
        with self._connect() as connection, self._lock:
            self._ensure_user_row(connection, user_id)
            self._ensure_conversation_row(connection, user_id, conversation_id, title)

    def get_conversation(
        self,
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
                 AND m.user_id = c.user_id
                WHERE c.user_id = ?
                  AND c.conversation_id = ?
                GROUP BY c.conversation_id
                """,
                (user_id, conversation_id),
            ).fetchone()
        return _row_to_conversation(row) if row else None

    def list_conversations(
        self,
        user_id: str,
        *,
        status: str | None = "active",
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ConversationRecord]:
        _validate_conversation_status(status)
        clauses = ["c.user_id = ?"]
        values: list[object] = [user_id]
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
        user_id: str,
        *,
        status: str | None = "active",
    ) -> int:
        _validate_conversation_status(status)
        clauses = ["user_id = ?"]
        values: list[object] = [user_id]
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
        user_id: str,
        conversation_id: str,
        *,
        title: str | None = None,
        status: str | None = None,
    ) -> ConversationRecord | None:
        _validate_conversation_status(status)
        current = self.get_conversation(user_id, conversation_id)
        if current is None:
            return None

        updated_title = _normalize_title(title) if title is not None else current.title
        updated_status = status if status is not None else current.status
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection, self._lock:
            connection.execute(
                """
                UPDATE conversations
                SET title = ?, status = ?, updated_at = ?
                WHERE user_id = ? AND conversation_id = ?
                """,
                (updated_title, updated_status, now, user_id, conversation_id),
            )
        return self.get_conversation(user_id, conversation_id)

    def add_message(
        self,
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
            user_id=user_id,
            role=role,
            content=content,
            created_at=datetime.now(timezone.utc),
        )
        with self._connect() as connection, self._lock:
            self._ensure_user_row(connection, user_id)
            self._ensure_conversation_row(connection, user_id, conversation_id)
            connection.execute(
                """
                INSERT INTO messages (
                    message_id, conversation_id, user_id, role, content, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.message_id,
                    record.conversation_id,
                    record.user_id,
                    record.role,
                    record.content,
                    record.created_at.isoformat(),
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (record.created_at.isoformat(), conversation_id),
            )
        return record

    def list_messages(
        self,
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
                    WHERE user_id = ?
                      AND conversation_id = ?
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC, _message_rowid ASC
                """,
                (user_id, conversation_id, limit),
            ).fetchall()
        return [_row_to_message(row) for row in rows]

    def count_messages(
        self,
        user_id: str,
        conversation_id: str,
    ) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM messages
                WHERE user_id = ?
                  AND conversation_id = ?
                """,
                (user_id, conversation_id),
            ).fetchone()
        return int(row["count"] if row else 0)

    def _ensure_user_row(
        self,
        connection: sqlite3.Connection,
        user_id: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        connection.execute(
            """
            INSERT INTO users (user_id, created_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id)
            DO UPDATE SET updated_at = excluded.updated_at
            """,
            (user_id, now, now),
        )

    def _ensure_conversation_row(
        self,
        connection: sqlite3.Connection,
        user_id: str,
        conversation_id: str,
        title: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        title = _normalize_title(title)
        existing = connection.execute(
            """
            SELECT user_id FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if existing and existing["user_id"] != user_id:
            raise ValueError("Conversation id belongs to a different user.")

        connection.execute(
            """
            INSERT INTO conversations (
                conversation_id, user_id, title, status, created_at, updated_at
            )
            VALUES (?, ?, ?, 'active', ?, ?)
            ON CONFLICT(conversation_id)
            DO UPDATE SET
                updated_at = excluded.updated_at,
                title = COALESCE(excluded.title, conversations.title)
            """,
            (conversation_id, user_id, title, now, now),
        )

    def create_memory(
        self,
        *,
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
        supersedes_id: str | None = None,
        status: str = "active",
        source_message_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        memory_id: str | None = None,
    ) -> MemoryRecord:
        memory = _new_memory(
            user_id=user_id,
            scope=scope,
            type=type,
            key=key,
            content=content,
            value_json=value_json,
            source=source,
            confidence=confidence,
            expires_at=expires_at,
            visibility=visibility,
            supersedes_id=supersedes_id,
            status=status,
            source_message_id=source_message_id,
            conversation_id=conversation_id,
            task_id=task_id,
            memory_id=memory_id,
        )
        with self._connect() as connection, self._lock:
            _insert_memory(connection, memory)
        return memory

    def create_or_supersede_memory(
        self,
        *,
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
        source_message_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
    ) -> tuple[MemoryRecord, str | None, bool]:
        """Atomically reuse or supersede the active row for one memory key."""
        normalized_key = _normalize_key(key)
        with self._connect() as connection, self._lock:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND scope = ? AND type = ?
                  AND key = ? AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id, scope, type, normalized_key),
            ).fetchone()
            previous = _row_to_memory(row) if row else None
            if previous and _same_memory_values(
                previous,
                content=content,
                value_json=value_json,
                visibility=visibility,
                task_id=task_id,
                expires_at=expires_at,
            ):
                return previous, None, False

            memory = _new_memory(
                user_id=user_id,
                scope=scope,
                type=type,
                key=normalized_key,
                content=content,
                value_json=value_json,
                source=source,
                confidence=confidence,
                expires_at=expires_at,
                visibility=visibility,
                supersedes_id=previous.memory_id if previous else None,
                source_message_id=source_message_id,
                conversation_id=conversation_id,
                task_id=task_id,
            )
            if previous:
                connection.execute(
                    """
                    UPDATE memories
                    SET status = 'stale', updated_at = ?
                    WHERE memory_id = ? AND status = 'active'
                    """,
                    (memory.updated_at.isoformat(), previous.memory_id),
                )
            _insert_memory(connection, memory)
        return memory, previous.memory_id if previous else None, True

    def get_memory(self, user_id: str, memory_id: str) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ?
                  AND memory_id = ?
                """,
                (user_id, memory_id),
            ).fetchone()
        return _row_to_memory(row) if row else None

    def get_memories_by_ids(
        self,
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
                WHERE user_id = ?
                  AND memory_id IN ({placeholders})
                """,
                (user_id, *memory_ids),
            ).fetchall()
        memories = [_row_to_memory(row) for row in rows]
        by_id = {memory.memory_id: memory for memory in memories}
        return [by_id[memory_id] for memory_id in memory_ids if memory_id in by_id]

    def list_memories(
        self,
        user_id: str,
        *,
        status: str | None = "active",
        include_expired: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        clauses = ["user_id = ?"]
        values: list[object] = [user_id]
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            values.append(datetime.now(timezone.utc).isoformat())

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
        user_id: str,
        *,
        status: str | None = "active",
        include_expired: bool = False,
    ) -> int:
        clauses = ["user_id = ?"]
        values: list[object] = [user_id]
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            values.append(datetime.now(timezone.utc).isoformat())

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
                WHERE user_id = ?
                  AND scope = ?
                  AND type = ?
                  AND key = ?
                  AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id, scope, type, _normalize_key(key)),
            ).fetchone()
        return _row_to_memory(row) if row else None

    def update_memory(
        self,
        user_id: str,
        memory_id: str,
        update: MemoryUpdate,
    ) -> MemoryRecord | None:
        with self._connect() as connection, self._lock:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND memory_id = ?
                """,
                (user_id, memory_id),
            ).fetchone()
            if row is None:
                return None
            current = _row_to_memory(row)
            updated = MemoryRecord(
                memory_id=current.memory_id,
                user_id=current.user_id,
                scope=current.scope,
                type=current.type,
                key=_normalize_key(update.key) if update.key is not None else current.key,
                content=update.content.strip() if update.content is not None else current.content,
                value_json=current.value_json if is_unset(update.value_json) else update.value_json,
                source=update.source if update.source is not None else current.source,
                confidence=update.confidence
                if update.confidence is not None
                else current.confidence,
                created_at=current.created_at,
                updated_at=datetime.now(timezone.utc),
                expires_at=current.expires_at if is_unset(update.expires_at) else update.expires_at,
                visibility=update.visibility if update.visibility is not None else current.visibility,
                supersedes_id=current.supersedes_id,
                status=update.status if update.status is not None else current.status,
                source_message_id=current.source_message_id,
                conversation_id=current.conversation_id,
                task_id=current.task_id,
            )
            _validate_memory(updated)
            connection.execute(
                """
                UPDATE memories
                SET key = ?, content = ?, value_json = ?, source = ?, confidence = ?,
                    updated_at = ?, expires_at = ?, visibility = ?, status = ?
                WHERE memory_id = ? AND user_id = ?
                """,
                (
                    updated.key,
                    updated.content,
                    _json_dump(updated.value_json),
                    updated.source,
                    updated.confidence,
                    updated.updated_at.isoformat(),
                    updated.expires_at.isoformat() if updated.expires_at else None,
                    updated.visibility,
                    updated.status,
                    updated.memory_id,
                    user_id,
                ),
            )
        return updated

    def mark_memory_status(
        self,
        user_id: str,
        memory_id: str,
        status: str,
    ) -> MemoryRecord | None:
        if status not in VALID_MEMORY_STATUSES:
            raise ValueError(f"Invalid memory status: {status}")
        current = self.get_memory(user_id, memory_id)
        if current is None or current.user_id != user_id:
            return None
        updated = replace(
            current,
            status=status,
            updated_at=datetime.now(timezone.utc),
        )  # type: ignore[arg-type]
        _validate_memory(updated)
        with self._connect() as connection, self._lock:
            connection.execute(
                """
                UPDATE memories
                SET status = ?, updated_at = ?
                WHERE memory_id = ? AND user_id = ?
                """,
                (
                    updated.status,
                    updated.updated_at.isoformat(),
                    updated.memory_id,
                    user_id,
                ),
            )
        return updated

    def mark_expired_memories_stale(
        self,
        user_id: str,
    ) -> list[MemoryRecord]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection, self._lock:
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ?
                  AND status = 'active'
                  AND expires_at IS NOT NULL
                  AND expires_at <= ?
                """,
                (user_id, now),
            ).fetchall()
            if rows:
                connection.execute(
                    """
                    UPDATE memories
                    SET status = 'stale', updated_at = ?
                    WHERE user_id = ?
                      AND status = 'active'
                      AND expires_at IS NOT NULL
                      AND expires_at <= ?
                    """,
                    (now, user_id, now),
                )
        return [
            replace(_row_to_memory(row), status="stale", updated_at=datetime.fromisoformat(now))
            for row in rows
        ]

    def mark_task_memories_stale(
        self,
        user_id: str,
        task_id: str,
    ) -> list[MemoryRecord]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection, self._lock:
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ?
                  AND task_id = ?
                  AND scope = 'task'
                  AND status = 'active'
                """,
                (user_id, task_id),
            ).fetchall()
            if rows:
                connection.execute(
                    """
                    UPDATE memories
                    SET status = 'stale', updated_at = ?
                    WHERE user_id = ?
                      AND task_id = ?
                      AND scope = 'task'
                      AND status = 'active'
                    """,
                    (now, user_id, task_id),
                )
        return [
            replace(_row_to_memory(row), status="stale", updated_at=datetime.fromisoformat(now))
            for row in rows
        ]

    def list_active_memories_for_user(
        self,
        user_id: str,
    ) -> list[MemoryRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ?
                  AND status = 'active'
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY updated_at DESC
                """,
                (user_id, datetime.now(timezone.utc).isoformat()),
            ).fetchall()
        return [_row_to_memory(row) for row in rows]

    def list_vector_cleanup_memory_ids(
        self,
        user_id: str,
    ) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_id FROM memories
                WHERE user_id = ?
                  AND (
                    status != 'active'
                    OR (expires_at IS NOT NULL AND expires_at <= ?)
                  )
                """,
                (user_id, datetime.now(timezone.utc).isoformat()),
            ).fetchall()
        return [str(row["memory_id"]) for row in rows]

    def _ensure_schema(self) -> None:
        with self._connect() as connection, self._lock:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id)
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id)
                        REFERENCES users(user_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (user_id)
                        REFERENCES users(user_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
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
                    permissions_json TEXT NOT NULL DEFAULT '[]',
                    supersedes_id TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    source_message_id TEXT,
                    conversation_id TEXT,
                    task_id TEXT,
                    FOREIGN KEY (source_message_id)
                        REFERENCES messages(message_id)
                        ON DELETE SET NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_subject
                    ON memories (user_id, status, scope, type, key);
                CREATE INDEX IF NOT EXISTS idx_memories_expiry
                    ON memories (user_id, status, expires_at);
                CREATE INDEX IF NOT EXISTS idx_conversations_subject
                    ON conversations (user_id, status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages (conversation_id, created_at);
                """
            )
            now = datetime.now(timezone.utc).isoformat()
            connection.execute(
                """
                WITH ranked AS (
                    SELECT memory_id,
                           ROW_NUMBER() OVER (
                               PARTITION BY user_id, scope, type, key
                               ORDER BY updated_at DESC, rowid DESC
                           ) AS rank
                    FROM memories
                    WHERE status = 'active'
                )
                UPDATE memories
                SET status = 'stale', updated_at = ?
                WHERE memory_id IN (SELECT memory_id FROM ranked WHERE rank > 1)
                """,
                (now,),
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_active_key
                ON memories (user_id, scope, type, key)
                WHERE status = 'active'
                """
            )

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


def _new_memory(
    *,
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
    supersedes_id: str | None = None,
    status: str = "active",
    source_message_id: str | None = None,
    conversation_id: str | None = None,
    task_id: str | None = None,
    memory_id: str | None = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    memory = MemoryRecord(
        memory_id=memory_id or uuid4().hex,
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
        supersedes_id=supersedes_id,
        status=status,  # type: ignore[arg-type]
        source_message_id=source_message_id,
        conversation_id=conversation_id,
        task_id=task_id,
        superseded_conflicting=_superseded_conflicting(value_json),
        superseded_from_content=_superseded_from_content(value_json),
    )
    _validate_memory(memory)
    return memory


def _insert_memory(connection: sqlite3.Connection, memory: MemoryRecord) -> None:
    connection.execute(
        """
        INSERT INTO memories (
            memory_id, user_id, scope, type, key, content, value_json,
            source, confidence, created_at, updated_at, expires_at, visibility,
            permissions_json, supersedes_id, status, source_message_id,
            conversation_id, task_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _memory_values(memory),
    )


def _same_memory_values(
    memory: MemoryRecord,
    *,
    content: str,
    value_json: dict | None,
    visibility: str,
    task_id: str | None,
    expires_at: datetime | None,
) -> bool:
    return (
        memory.content == content.strip()
        and memory.value_json == value_json
        and memory.visibility == visibility
        and memory.task_id == task_id
        and memory.expires_at == expires_at
    )


def _memory_values(memory: MemoryRecord) -> tuple[object, ...]:
    return (
        memory.memory_id,
        memory.user_id,
        memory.scope,
        memory.type,
        memory.key,
        memory.content,
        _json_dump(memory.value_json),
        memory.source,
        memory.confidence,
        memory.created_at.isoformat(),
        memory.updated_at.isoformat(),
        memory.expires_at.isoformat() if memory.expires_at else None,
        memory.visibility,
        _json_dump([]),  # Legacy SQLite column retained for existing databases.
        memory.supersedes_id,
        memory.status,
        memory.source_message_id,
        memory.conversation_id,
        memory.task_id,
    )


def _row_to_memory(row: sqlite3.Row) -> MemoryRecord:
    value_json = _json_load(row["value_json"])
    return MemoryRecord(
        memory_id=row["memory_id"],
        user_id=row["user_id"],
        scope=row["scope"],
        type=row["type"],
        key=row["key"],
        content=row["content"],
        value_json=value_json,
        source=row["source"],
        confidence=float(row["confidence"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
        visibility=row["visibility"],
        supersedes_id=row["supersedes_id"],
        status=row["status"],
        source_message_id=row["source_message_id"],
        conversation_id=row["conversation_id"],
        task_id=row["task_id"],
        superseded_conflicting=_superseded_conflicting(value_json),
        superseded_from_content=_superseded_from_content(value_json),
    )


def _row_to_conversation(row: sqlite3.Row) -> ConversationRecord:
    return ConversationRecord(
        conversation_id=row["conversation_id"],
        user_id=row["user_id"],
        title=row["title"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        message_count=int(row["message_count"]),
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    return MessageRecord(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        user_id=row["user_id"],
        role=row["role"],
        content=row["content"],
        created_at=datetime.fromisoformat(row["created_at"]),
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


def _validate_conversation_status(status: str | None) -> None:
    if status is not None and status not in VALID_CONVERSATION_STATUSES:
        raise ValueError(f"Invalid conversation status: {status}")


def _normalize_title(value: str | None) -> str | None:
    title = (value or "").strip()
    return title[:200] or None


def _normalize_key(value: str | None) -> str:
    key = (value or "").strip().lower().replace(" ", "_")
    return key[:120]


def _superseded_conflicting(value_json: object) -> bool:
    return isinstance(value_json, dict) and bool(value_json.get("_superseded_conflicting"))


def _superseded_from_content(value_json: object) -> str | None:
    if not isinstance(value_json, dict):
        return None
    value = value_json.get("_superseded_from")
    return str(value) if value else None


def _json_dump(value: object | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_load(value: str | None):
    if not value:
        return None
    return json.loads(value)

from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from doc_assistant.config.settings import settings
from doc_assistant.memory.policy import extract_memory_write_intents
from doc_assistant.memory.schemas import (
    MemoryCandidate,
    MemoryRecord,
    MemoryUpdate,
    MemoryUsage,
    MemoryWriteIntent,
)
from doc_assistant.memory.store import MemoryStore
from doc_assistant.memory.vector_store import MemoryVectorStore

logger = logging.getLogger(__name__)


class MemoryService:
    def __init__(
        self,
        store: MemoryStore | None = None,
        vector_store: MemoryVectorStore | None = None,
    ) -> None:
        self.store = store or MemoryStore()
        self.vector_store = vector_store

    def ensure_context(self, tenant_id: str, user_id: str, conversation_id: str | None) -> str:
        resolved_conversation_id = conversation_id or uuid4().hex
        self.store.ensure_conversation(tenant_id, user_id, resolved_conversation_id)
        return resolved_conversation_id

    def record_user_message(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        content: str,
    ) -> str:
        message = self.store.add_message(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            role="user",
            content=content,
        )
        return message.message_id

    def record_assistant_message(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        content: str,
    ) -> str:
        message = self.store.add_message(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            role="assistant",
            content=content,
        )
        return message.message_id

    def write_memories_from_user_message(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        message_id: str,
        content: str,
    ) -> list[MemoryRecord]:
        intents = extract_memory_write_intents(content)
        return [
            self.create_memory_from_intent(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                source_message_id=message_id,
                intent=intent,
            )
            for intent in intents
        ]

    def create_memory_from_intent(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        source_message_id: str | None,
        intent: MemoryWriteIntent,
    ) -> MemoryRecord:
        return self.create_memory(
            tenant_id=tenant_id,
            user_id=user_id,
            scope=intent.scope,
            type=intent.type,
            key=intent.key,
            content=intent.content,
            value_json=intent.value_json,
            source=intent.source,
            confidence=intent.confidence,
            expires_at=intent.expires_at,
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            task_id=intent.task_id,
        )

    def create_memory(
        self,
        *,
        tenant_id: str,
        user_id: str,
        scope: str,
        type: str,
        key: str,
        content: str,
        value_json: dict | None = None,
        source: str = "explicit",
        confidence: float = 0.95,
        expires_at: datetime | None = None,
        visibility: str = "private",
        permissions: tuple[str, ...] = ("read", "write", "delete"),
        conversation_id: str | None = None,
        source_message_id: str | None = None,
        task_id: str | None = None,
    ) -> MemoryRecord:
        previous = self.store.find_active_memory_by_key(
            tenant_id,
            user_id,
            scope=scope,
            type=type,
            key=key,
        )
        memory = self.store.create_memory(
            tenant_id=tenant_id,
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
            permissions=permissions,
            supersedes_id=previous.memory_id if previous else None,
            source_message_id=source_message_id,
            conversation_id=conversation_id,
            task_id=task_id,
        )
        if previous:
            self.store.mark_memory_status(tenant_id, user_id, previous.memory_id, "stale")
            self._delete_vector(previous.memory_id)

        if self._upsert_vector(memory):
            self.store.update_memory_embedding_id(tenant_id, user_id, memory.memory_id, memory.memory_id)
            refreshed = self.store.get_memory(tenant_id, user_id, memory.memory_id)
            if refreshed:
                memory = refreshed
        return memory

    def list_memories(
        self,
        tenant_id: str,
        user_id: str,
        *,
        status: str | None = "active",
        include_expired: bool = False,
    ) -> list[MemoryRecord]:
        return self.store.list_memories(
            tenant_id,
            user_id,
            status=status,
            include_expired=include_expired,
        )

    def update_memory(
        self,
        tenant_id: str,
        user_id: str,
        memory_id: str,
        update: MemoryUpdate,
    ) -> MemoryRecord | None:
        updated = self.store.update_memory(tenant_id, user_id, memory_id, update)
        if updated is None:
            return None
        if updated.status == "active" and not updated.is_expired():
            if self._upsert_vector(updated):
                self.store.update_memory_embedding_id(tenant_id, user_id, memory_id, memory_id)
                refreshed = self.store.get_memory(tenant_id, user_id, memory_id)
                return refreshed or updated
        else:
            self._delete_vector(memory_id)
        return updated

    def delete_memory(self, tenant_id: str, user_id: str, memory_id: str) -> MemoryRecord | None:
        deleted = self.store.mark_memory_status(tenant_id, user_id, memory_id, "deleted")
        self._delete_vector(memory_id)
        return deleted

    def retrieve_relevant_memories(
        self,
        *,
        tenant_id: str,
        user_id: str,
        query: str,
        limit: int | None = None,
    ) -> list[MemoryCandidate]:
        search_limit = limit or settings.memory_top_k
        candidates = self._vector_search(user_id, query, search_limit)
        if candidates:
            score_by_id = {
                candidate.memory.memory_id: candidate.score
                for candidate in candidates
                if candidate.memory.memory_id
            }
            hydrated = self.store.get_memories_by_ids(tenant_id, user_id, list(score_by_id))
            candidates = [
                MemoryCandidate(memory=memory, score=score_by_id.get(memory.memory_id))
                for memory in hydrated
            ]
        else:
            candidates = self.store.search_memories_lexical(
                tenant_id,
                user_id,
                query,
                limit=search_limit,
                min_confidence=settings.memory_min_confidence,
            )

        filtered = [
            candidate
            for candidate in candidates
            if candidate.memory.status == "active"
            and not candidate.memory.is_expired()
            and candidate.memory.confidence >= settings.memory_min_confidence
            and _can_read_memory(candidate.memory, user_id)
        ]
        filtered.sort(
            key=lambda candidate: (
                candidate.score if candidate.score is not None else -1,
                candidate.memory.confidence,
                candidate.memory.updated_at,
            ),
            reverse=True,
        )
        return filtered[:search_limit]

    def log_retrieval(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str | None,
        query: str,
        document_count: int,
        memories: list[MemoryCandidate],
    ) -> None:
        self.store.log_retrieval(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            query=query,
            document_count=document_count,
            memory_count=len(memories),
            selected_memory_ids=[candidate.memory.memory_id for candidate in memories],
        )

    def format_for_prompt(self, candidates: list[MemoryCandidate]) -> str:
        if not candidates:
            return "No relevant user memory."

        lines = ["Relevant user memory:"]
        for candidate in candidates:
            memory = candidate.memory
            qualifier = f"{memory.type}:{memory.key}"
            lines.append(f"- {qualifier} ({memory.source}, confidence {memory.confidence:.2f}): {memory.content}")
        return "\n".join(lines)

    def usages_from_candidates(self, candidates: list[MemoryCandidate]) -> list[MemoryUsage]:
        return [
            MemoryUsage(
                memory_id=candidate.memory.memory_id,
                type=candidate.memory.type,
                key=candidate.memory.key,
                content=candidate.memory.content,
                source=candidate.memory.source,
                confidence=candidate.memory.confidence,
                scope=candidate.memory.scope,
                score=candidate.score,
            )
            for candidate in candidates
        ]

    def _vector_search(self, user_id: str, query: str, limit: int) -> list[MemoryCandidate]:
        if self.vector_store is None:
            return []
        try:
            return self.vector_store.search(query, user_id=user_id, k=limit)
        except Exception:
            logger.warning("Memory vector search failed; falling back to structured search", exc_info=True)
            return []

    def _upsert_vector(self, memory: MemoryRecord) -> bool:
        if self.vector_store is None:
            return False
        try:
            self.vector_store.upsert_memory(memory)
            return True
        except Exception:
            logger.warning("Memory vector upsert failed; memory remains in structured store", exc_info=True)
            return False

    def _delete_vector(self, memory_id: str) -> None:
        if self.vector_store is None:
            return
        self.vector_store.delete_memory(memory_id)


def _can_read_memory(memory: MemoryRecord, user_id: str) -> bool:
    return memory.user_id == user_id or memory.visibility in {"team", "org"}


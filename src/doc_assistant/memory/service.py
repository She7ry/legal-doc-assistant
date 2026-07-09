"""Lean memory service: explicit user memories plus conversation history."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from uuid import uuid4

from doc_assistant.config.settings import settings
from doc_assistant.memory.schemas import (
    ConversationRecord,
    MemoryCandidate,
    MemoryRecord,
    MemoryUpdate,
    MemoryUsage,
    MemoryWriteIntent,
    MessageRecord,
)
from doc_assistant.memory.store import MemoryStore
from doc_assistant.memory.vector_store import MemoryVectorStore

logger = logging.getLogger(__name__)

_RECENT_HISTORY_WITH_SUMMARY_LIMIT = 8
_SUMMARY_MAX_CHARS = 2000
_EXPLICIT_MEMORY_MARKERS = (
    "please remember",
    "remember that",
    "remember my",
    "remember",
    "from now on",
    "going forward",
    "always answer",
    "请记住",
    "帮我记住",
    "记住",
    "以后",
)


class MemoryService:
    """Small facade over SQLite memory rows and optional vector lookup."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        vector_store: MemoryVectorStore | None = None,
        memory_extractor=None,
        summary_model: object | None = None,
        summary_model_factory=None,
    ) -> None:
        del memory_extractor, summary_model, summary_model_factory
        self.store = store or MemoryStore()
        self.vector_store = vector_store

    def ensure_context(self, tenant_id: str, user_id: str, conversation_id: str | None) -> str:
        resolved_conversation_id = conversation_id or uuid4().hex
        self.store.ensure_conversation(tenant_id, user_id, resolved_conversation_id)
        return resolved_conversation_id

    def create_conversation(
        self,
        tenant_id: str,
        user_id: str,
        *,
        conversation_id: str | None = None,
        title: str | None = None,
    ) -> ConversationRecord:
        resolved_conversation_id = conversation_id or uuid4().hex
        self.store.ensure_conversation(tenant_id, user_id, resolved_conversation_id, title=title)
        conversation = self.store.get_conversation(tenant_id, user_id, resolved_conversation_id)
        if conversation is None:
            raise RuntimeError("Conversation could not be created.")
        return conversation

    def list_conversations(
        self,
        tenant_id: str,
        user_id: str,
        *,
        status: str | None = "active",
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ConversationRecord]:
        return self.store.list_conversations(
            tenant_id,
            user_id,
            status=status,
            limit=limit,
            offset=offset,
        )

    def count_conversations(
        self,
        tenant_id: str,
        user_id: str,
        *,
        status: str | None = "active",
    ) -> int:
        return self.store.count_conversations(tenant_id, user_id, status=status)

    def update_conversation(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        *,
        title: str | None = None,
        status: str | None = None,
    ) -> ConversationRecord | None:
        return self.store.update_conversation(
            tenant_id,
            user_id,
            conversation_id,
            title=title,
            status=status,
        )

    def record_user_message(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        content: str,
    ) -> str:
        return self.store.add_message(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            role="user",
            content=content,
        ).message_id

    def record_assistant_message(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        content: str,
    ) -> str:
        return self.store.add_message(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            role="assistant",
            content=content,
        ).message_id

    def load_conversation_history(
        self,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        limit: int = 20,
        *,
        include_summary: bool = True,
    ) -> list[dict[str, str]]:
        summary_memory = None
        if include_summary:
            summary_memory = self.store.find_active_memory_by_key(
                tenant_id,
                user_id,
                scope="session",
                type="task_state",
                key=_conversation_summary_key(conversation_id),
            )

        message_limit = max(0, limit)
        if summary_memory:
            message_limit = min(message_limit, _RECENT_HISTORY_WITH_SUMMARY_LIMIT)
        messages = self.store.list_messages(
            tenant_id,
            user_id,
            conversation_id,
            limit=message_limit,
        )

        history = []
        if summary_memory and summary_memory.content.strip():
            history.append({"role": "system", "content": summary_memory.content})
        history.extend(
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role in {"user", "assistant"} and message.content.strip()
        )
        return history

    def summarize_conversation_to_memory(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        limit: int = 40,
    ) -> MemoryRecord | None:
        messages = self.store.list_messages(
            tenant_id,
            user_id,
            conversation_id,
            limit=max(1, min(limit, 100)),
        )
        summary = _summarize_messages(messages)
        if not summary:
            return None
        return self.create_memory(
            tenant_id=tenant_id,
            user_id=user_id,
            scope="session",
            type="task_state",
            key=_conversation_summary_key(conversation_id),
            content=summary,
            value_json={
                "conversation_id": conversation_id,
                "message_count": self.store.count_messages(tenant_id, user_id, conversation_id),
                "summary": summary,
                "summary_method": "simple",
            },
            source="system_generated",
            confidence=0.7,
            conversation_id=conversation_id,
        )

    def write_memories_from_user_message(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        message_id: str,
        content: str,
    ) -> list[MemoryRecord]:
        return [
            self.create_memory_from_intent(
                tenant_id=tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                source_message_id=message_id,
                intent=intent,
            )
            for intent in extract_memory_write_intents(content)
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
        if previous and _same_memory(
            previous,
            content=content,
            value_json=value_json,
            visibility=visibility,
            task_id=task_id,
            expires_at=expires_at,
        ):
            return previous

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
            supersedes_id=previous.memory_id if previous else None,
            source_message_id=source_message_id,
            conversation_id=conversation_id,
            task_id=task_id,
        )
        if previous:
            self.store.mark_memory_status(tenant_id, user_id, previous.memory_id, "stale")
            self._delete_vector(previous.memory_id)
        self._upsert_vector(memory)
        return memory

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
        return self.store.list_memories(
            tenant_id,
            user_id,
            status=status,
            include_expired=include_expired,
            limit=limit,
            offset=offset,
        )

    def count_memories(
        self,
        tenant_id: str,
        user_id: str,
        *,
        status: str | None = "active",
        include_expired: bool = False,
    ) -> int:
        return self.store.count_memories(
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
            self._upsert_vector(updated)
        else:
            self._delete_vector(memory_id)
        return updated

    def delete_memory(self, tenant_id: str, user_id: str, memory_id: str) -> MemoryRecord | None:
        deleted = self.store.mark_memory_status(tenant_id, user_id, memory_id, "deleted")
        self._delete_vector(memory_id)
        return deleted

    def cleanup_expired_memories(self, tenant_id: str, user_id: str) -> list[MemoryRecord]:
        stale = self.store.mark_expired_memories_stale(tenant_id, user_id)
        for memory in stale:
            self._delete_vector(memory.memory_id)
        return stale

    def mark_task_memories_stale(
        self,
        tenant_id: str,
        user_id: str,
        task_id: str,
    ) -> list[MemoryRecord]:
        stale = self.store.mark_task_memories_stale(tenant_id, user_id, task_id)
        for memory in stale:
            self._delete_vector(memory.memory_id)
        return stale

    def enforce_memory_limit(self, tenant_id: str, user_id: str) -> list[MemoryRecord]:
        del tenant_id, user_id
        return []

    def repair_vector_index(self, tenant_id: str, user_id: str) -> dict[str, int]:
        if self.vector_store is None:
            return {"deleted": 0, "upserted": 0}
        deleted = 0
        for memory_id in self.store.list_vector_cleanup_memory_ids(tenant_id, user_id):
            self._delete_vector(memory_id)
            deleted += 1
        upserted = sum(
            1
            for memory in self.store.list_active_memories_for_user(tenant_id, user_id)
            if self._upsert_vector(memory)
        )
        return {"deleted": deleted, "upserted": upserted}

    def retrieve_relevant_memories(
        self,
        *,
        tenant_id: str,
        user_id: str,
        query: str,
        limit: int | None = None,
    ) -> list[MemoryCandidate]:
        search_limit = max(1, int(limit or settings.memory_top_k))
        return [
            candidate
            for candidate in self._hydrate_vector_candidates(
                tenant_id,
                user_id,
                self._vector_search(tenant_id, user_id, query, search_limit),
            )
            if _usable_candidate(candidate, user_id)
        ][:search_limit]

    def format_for_prompt(self, candidates: list[MemoryCandidate]) -> str:
        usable = sorted(candidates, key=_candidate_rank, reverse=True)
        if not usable:
            return "No relevant user memory."

        lines = [
            "Relevant memory for this user and tenant:",
            "Use high-confidence memory as context. Treat confidence below 0.70 as a hint, not a fact.",
        ]
        max_chars = max(200, int(settings.memory_prompt_max_tokens) * 4)
        for candidate in usable:
            memory = candidate.memory
            line = (
                f"- {memory.key} ({memory.source}, confidence {memory.confidence:.2f}): "
                f"{' '.join(memory.content.split())}"
            )
            if len("\n".join([*lines, line])) > max_chars:
                break
            lines.append(line[:500])
        return "\n".join(lines) if len(lines) > 2 else "No relevant user memory."

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
                superseded_conflicting=candidate.memory.superseded_conflicting,
                superseded_from_content=candidate.memory.superseded_from_content,
            )
            for candidate in candidates
        ]

    def _vector_search(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        limit: int,
    ) -> list[MemoryCandidate]:
        if self.vector_store is None:
            return []
        try:
            return self.vector_store.search(query, tenant_id=tenant_id, user_id=user_id, k=limit)
        except Exception:
            logger.warning("Memory vector search failed", exc_info=True)
            return []

    def _hydrate_vector_candidates(
        self,
        tenant_id: str,
        user_id: str,
        candidates: list[MemoryCandidate],
    ) -> list[MemoryCandidate]:
        hydrate_ids = [
            candidate.memory.memory_id
            for candidate in candidates
            if candidate.memory.memory_id and candidate.retrieval_source != "vector"
        ]
        hydrated = {
            memory.memory_id: memory
            for memory in self.store.get_memories_by_ids(tenant_id, user_id, hydrate_ids)
        }
        return [
            MemoryCandidate(
                memory=hydrated.get(candidate.memory.memory_id, candidate.memory),
                score=candidate.score,
                retrieval_source="vector",
            )
            for candidate in candidates
            if candidate.memory.memory_id
            and (candidate.retrieval_source == "vector" or candidate.memory.memory_id in hydrated)
        ]

    def _upsert_vector(self, memory: MemoryRecord) -> bool:
        if self.vector_store is None:
            return False
        try:
            self.vector_store.upsert_memory(memory)
            return True
        except Exception:
            logger.warning("Memory vector upsert failed; memory remains in SQLite", exc_info=True)
            return False

    def _delete_vector(self, memory_id: str) -> None:
        if self.vector_store is None:
            return
        self.vector_store.delete_memory(memory_id)


def extract_memory_write_intents(user_text: str) -> list[MemoryWriteIntent]:
    text = " ".join(user_text.split())
    if not text or not _has_explicit_memory_marker(text):
        return []
    content = _strip_memory_marker(text)
    return [
        MemoryWriteIntent(
            type="preference" if _looks_like_preference(part) else "fact",
            key=_infer_key(part),
            content=part,
            value_json={"text": part},
            source="explicit",
            confidence=0.95,
        )
        for part in _split_memory_parts(content)
        if len(part) >= 4
    ]


def _has_explicit_memory_marker(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in _EXPLICIT_MEMORY_MARKERS)


def _strip_memory_marker(text: str) -> str:
    patterns = (
        r"^\s*(please\s+)?remember\s+(that\s+|my\s+)?",
        r"^\s*from\s+now\s+on[:,]?\s*",
        r"^\s*going\s+forward[:,]?\s*",
        r"^\s*always\s+answer\s*[:,]?\s*",
        r"^\s*请?帮?我?记住[：:\s]*",
        r"^\s*以后(都|请)?[：:\s]*",
    )
    content = text.strip()
    for pattern in patterns:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE)
    return _clean_memory_text(content)


def _split_memory_parts(content: str) -> list[str]:
    content = re.sub(r"\s+(?:and also|also|and my)\s+", "；", content, flags=re.IGNORECASE)
    content = re.sub(r"(并且|而且|另外|同时)", "；", content)
    return [
        cleaned
        for part in re.split(r"[;；。]\s*", content)
        if (cleaned := _clean_memory_text(part))
    ]


def _clean_memory_text(text: str) -> str:
    return re.sub(r"^[ ：:。.]+|[ ：:。.]+$", "", text.strip())


def _looks_like_preference(text: str) -> bool:
    lowered = text.casefold()
    return any(
        term in lowered
        for term in (
            "answer",
            "reply",
            "response",
            "style",
            "prefer",
            "concise",
            "detailed",
            "language",
            "回答",
            "回复",
            "风格",
            "偏好",
            "简洁",
            "详细",
            "中文",
            "英文",
        )
    )


def _infer_key(content: str) -> str:
    lowered = content.casefold()
    if _looks_like_preference(content):
        return "answer_style"
    if any(term in lowered for term in ("role", "job", "company", "business", "职位", "岗位", "公司")):
        return "business_context"
    words = re.findall(r"[A-Za-z0-9_\-\u4e00-\u9fff]+", lowered)
    digest = hashlib.sha1(" ".join(words).encode("utf-8")).hexdigest()[:10] if words else ""
    return ("_".join(words[:6])[:48] + (f"_{digest}" if digest else ""))[:80] or "user_memory"


def _same_memory(
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


def _usable_candidate(candidate: MemoryCandidate, user_id: str) -> bool:
    memory = candidate.memory
    return (
        memory.status == "active"
        and not memory.is_expired()
        and memory.confidence >= settings.memory_min_confidence
        and (memory.user_id == user_id or memory.visibility in {"team", "org"})
    )


def _candidate_rank(candidate: MemoryCandidate) -> tuple[float, float, datetime, datetime]:
    memory = candidate.memory
    return (
        memory.confidence,
        candidate.score or 0.0,
        memory.updated_at,
        memory.created_at,
    )


def _summarize_messages(messages: list[MessageRecord]) -> str:
    parts = [
        f"{'User' if message.role == 'user' else 'Assistant'}: {' '.join(message.content.split())}"
        for message in messages
        if message.role in {"user", "assistant"} and message.content.strip()
    ]
    if not parts:
        return ""
    summary = "Conversation summary:\n" + "\n".join(parts)
    return summary[: _SUMMARY_MAX_CHARS - 3] + "..." if len(summary) > _SUMMARY_MAX_CHARS else summary


def _conversation_summary_key(conversation_id: str) -> str:
    return f"conversation_summary_{conversation_id[:40]}"

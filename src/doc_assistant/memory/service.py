"""用户记忆服务：对话历史、长期记忆写入/检索、自动摘要与衰减维护。

MemoryService 在 QA / ToolCalling 提问前注入相关记忆，在回答后提取新记忆候选。
底层 ``MemoryStore``（SQLite）存结构化记录，``MemoryVectorStore`` 做语义检索去重。
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

from doc_assistant.config.settings import settings
from doc_assistant.memory._conflict import (
    _clamp_confidence,
    _is_conflicting_memory_update,
    _is_equivalent_memory,
    _normalize_feedback_rating,
    _with_supersede_conflict_metadata,
)
from doc_assistant.memory._prompt import (
    _estimate_prompt_tokens,
    _format_memory_prompt_line,
    _prompt_candidate_rank,
    _truncate_to_prompt_tokens,
)
from doc_assistant.memory._retrieval import (
    _filter_memory_candidates,
    _is_semantic_duplicate_memory,
    _memory_similarity_query,
    _rrf_fuse_memory_candidates,
    _vector_candidate_needs_hydration,
)
from doc_assistant.memory.policy import (
    extract_memory_write_intents,
    extract_task_memory_write_intents,
)
from doc_assistant.memory.schemas import (
    ConversationRecord,
    FeedbackEventRecord,
    FeedbackMemoryAdjustment,
    MemoryCandidate,
    MemoryRecord,
    MemoryUpdate,
    MemoryUsage,
    MemoryWriteIntent,
    MessageRecord,
)
from doc_assistant.memory.maintenance import (
    _default_expires_at,
    _memory_retention_rank,
)
from doc_assistant.memory.store import MemoryStore
from doc_assistant.memory.summarization import (
    _conversation_summary_key,
    _summarize_conversation,
    _summarize_conversation_llm_structured,
    _summary_message_count,
)
from doc_assistant.memory.vector_store import MemoryVectorStore

logger = logging.getLogger(__name__)
_POSITIVE_FEEDBACK_CONFIDENCE_DELTA = 0.03
_NEGATIVE_FEEDBACK_CONFIDENCE_DELTA = -0.08
_RECENT_HISTORY_WITH_SUMMARY_LIMIT = 8


class MemoryService:
    """用户记忆与对话历史的业务编排层。

    主要能力：
    - 问答前：检索相关记忆、加载/合并对话历史，注入 prompt
    - 问答后：规则 + LLM 抽取新记忆、写入 SQLite 与向量索引
    - 维护：过期清理、置信度衰减、冲突记忆 supersede、自动摘要

    QA / ToolCalling / Agent 通过本类共享同一套记忆，而非各自读写数据库。
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        vector_store: MemoryVectorStore | None = None,
        memory_extractor: Callable[[str], list[MemoryWriteIntent]] | None = None,
        summary_model: object | None = None,
        summary_model_factory: Callable[[], object] | None = None,
    ) -> None:
        self.store = store or MemoryStore()
        self.vector_store = vector_store
        self.memory_extractor = memory_extractor
        self._summary_model = summary_model
        self._summary_model_factory = summary_model_factory
        self._maintenance_lock = Lock()
        self._maintenance_last_run: dict[tuple[str, str, str], datetime] = {}

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
        for message in messages:
            if message.role in {"user", "assistant"} and message.content.strip():
                history.append({"role": message.role, "content": message.content})
        return history

    def summarize_conversation_to_memory(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        limit: int = 40,
    ) -> MemoryRecord | None:
        total_message_count = self.store.count_messages(tenant_id, user_id, conversation_id)
        previous = self.store.find_active_memory_by_key(
            tenant_id,
            user_id,
            scope="session",
            type="task_state",
            key=_conversation_summary_key(conversation_id),
        )
        previous_count = _summary_message_count(previous)
        new_message_count = max(0, total_message_count - previous_count) if previous else total_message_count
        if previous and new_message_count <= 0:
            return previous
        message_limit = max(2, min(limit, 200))
        if previous and previous_count > 0:
            message_limit = max(0, min(message_limit, new_message_count))
        messages = self.store.list_messages(
            tenant_id,
            user_id,
            conversation_id,
            limit=message_limit,
        )
        summary, summary_method = self._summarize_conversation_messages(
            messages,
            previous_summary=previous.content if previous else None,
        )
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
                "message_count": total_message_count,
                "previous_message_count": previous_count if previous else 0,
                "incremental": bool(previous and previous_count > 0),
                "summary": summary,
                "summary_method": summary_method,
            },
            source="system_generated",
            confidence=0.7,
            conversation_id=conversation_id,
        )

    def _summarize_conversation_messages(
        self,
        messages: list[MessageRecord],
        *,
        previous_summary: str | None = None,
    ) -> tuple[str, str]:
        if settings.memory_llm_extraction_enabled:
            try:
                summary = _summarize_conversation_llm_structured(
                    messages,
                    self._summary_chat_model(),
                    previous_summary=previous_summary,
                )
                if summary:
                    return summary, "llm"
            except Exception:
                logger.debug("LLM conversation summary failed; falling back to rule summary.", exc_info=True)
        return _summarize_conversation(messages, previous_summary=previous_summary), "rule"

    def _summary_chat_model(self) -> object:
        if self._summary_model is None:
            if self._summary_model_factory is not None:
                self._summary_model = self._summary_model_factory()
            else:
                from doc_assistant.models.language_model import build_chat_model

                self._summary_model = build_chat_model()
        return self._summary_model

    def maybe_summarize_conversation(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
    ) -> MemoryRecord | None:
        threshold = int(getattr(settings, "memory_auto_summary_threshold", 0))
        if threshold <= 0:
            return None

        message_count = self.store.count_messages(tenant_id, user_id, conversation_id)
        if message_count < threshold:
            return None

        previous = self.store.find_active_memory_by_key(
            tenant_id,
            user_id,
            scope="session",
            type="task_state",
            key=_conversation_summary_key(conversation_id),
        )
        previous_count = _summary_message_count(previous)
        refresh_interval = max(1, int(getattr(settings, "memory_auto_summary_interval", 8)))
        if previous and message_count - previous_count < refresh_interval:
            return None

        return self.summarize_conversation_to_memory(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            limit=int(getattr(settings, "memory_auto_summary_window", 40)),
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
        intents = extract_memory_write_intents(content)
        if not intents and self.memory_extractor is not None:
            try:
                intents = self.memory_extractor(content)
            except Exception:
                logger.warning("External memory extractor failed; skipping inferred writes", exc_info=True)
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

    def write_memories_from_assistant_message(
        self,
        *,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        message_id: str,
        content: str,
        task_id: str | None = None,
    ) -> list[MemoryRecord]:
        if not task_id:
            return []
        intents = extract_task_memory_write_intents(content, task_id=task_id)
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
        expires_at = expires_at or _default_expires_at(scope)
        previous = self.store.find_active_memory_by_key(
            tenant_id,
            user_id,
            scope=scope,
            type=type,
            key=key,
        )
        if previous is None:
            previous = self._find_semantic_duplicate_memory(
                tenant_id=tenant_id,
                user_id=user_id,
                scope=scope,
                type=type,
                key=key,
                content=content,
            )
            if previous is not None:
                key = previous.key
        if previous and not previous.is_expired() and _is_equivalent_memory(
            previous,
            content=content,
            value_json=value_json,
            visibility=visibility,
            permissions=permissions,
            task_id=task_id,
            expires_at=expires_at,
        ):
            return previous
        superseded_conflicting = bool(
            previous
            and type == "preference"
            and previous.type == "preference"
            and _is_conflicting_memory_update(previous.content, content)
        )
        if superseded_conflicting:
            value_json = _with_supersede_conflict_metadata(value_json, previous.content)

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
        self._run_maintenance_if_due(
            tenant_id,
            user_id,
            "enforce_memory_limit",
            lambda: self.enforce_memory_limit(tenant_id, user_id),
        )
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
        max_active = int(getattr(settings, "memory_max_active_per_user", 0))
        if max_active <= 0:
            return []
        active = self.store.list_active_memories_for_user(tenant_id, user_id)
        overflow = len(active) - max_active
        if overflow <= 0:
            return []
        candidates = sorted(
            active,
            key=lambda memory: (
                _memory_retention_rank(memory),
                memory.last_accessed_at or memory.updated_at,
                memory.created_at,
            ),
        )
        stale: list[MemoryRecord] = []
        for memory in candidates[:overflow]:
            updated = self.store.mark_memory_status(tenant_id, user_id, memory.memory_id, "stale")
            if updated:
                stale.append(updated)
                self._delete_vector(memory.memory_id)
        return stale

    def repair_vector_index(self, tenant_id: str, user_id: str) -> dict[str, int]:
        if self.vector_store is None:
            return {"deleted": 0, "upserted": 0}
        deleted = 0
        for memory_id in self.store.list_vector_cleanup_memory_ids(tenant_id, user_id):
            self._delete_vector(memory_id)
            deleted += 1

        upserted = 0
        for memory in self.store.list_active_memories_for_user(tenant_id, user_id):
            if self._upsert_vector(memory):
                self.store.update_memory_embedding_id(tenant_id, user_id, memory.memory_id, memory.memory_id)
                upserted += 1
        return {"deleted": deleted, "upserted": upserted}

    def retrieve_relevant_memories(
        self,
        *,
        tenant_id: str,
        user_id: str,
        query: str,
        limit: int | None = None,
    ) -> list[MemoryCandidate]:
        self._run_maintenance_if_due(
            tenant_id,
            user_id,
            "cleanup_expired_memories",
            lambda: self.cleanup_expired_memories(tenant_id, user_id),
        )
        search_limit = limit or settings.memory_top_k
        vector_candidates = _filter_memory_candidates(
            self._hydrate_vector_candidates(
                tenant_id,
                user_id,
                self._vector_search(tenant_id, user_id, query, search_limit),
            ),
            user_id,
        )
        lexical_candidates = _filter_memory_candidates(
            self.store.search_memories_lexical(
                tenant_id,
                user_id,
                query,
                limit=search_limit,
                min_confidence=settings.memory_min_confidence,
            ),
            user_id,
        )
        selected = _rrf_fuse_memory_candidates(
            vector_candidates,
            lexical_candidates,
            limit=search_limit,
        )
        self.store.touch_memories(
            tenant_id,
            user_id,
            [candidate.memory.memory_id for candidate in selected if candidate.memory.user_id == user_id],
        )
        return selected

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
            selected_memory_sources=dict(
                Counter(candidate.retrieval_source or "unknown" for candidate in memories)
            ),
        )

    def get_memory_stats(self, tenant_id: str, user_id: str) -> dict[str, object]:
        return self.store.get_memory_stats(tenant_id, user_id)

    def record_feedback(
        self,
        *,
        tenant_id: str,
        user_id: str,
        rating: int | str,
        conversation_id: str | None = None,
        message_id: str | None = None,
        memory_ids: list[str] | tuple[str, ...] = (),
        comment: str | None = None,
    ) -> tuple[FeedbackEventRecord, list[FeedbackMemoryAdjustment]]:
        normalized_rating = _normalize_feedback_rating(rating)
        unique_memory_ids = tuple(dict.fromkeys(memory_id.strip() for memory_id in memory_ids if memory_id.strip()))
        event = self.store.record_feedback_event(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            rating=normalized_rating,
            memory_ids=unique_memory_ids,
            comment=comment,
        )
        adjustments: list[FeedbackMemoryAdjustment] = []
        delta = (
            _POSITIVE_FEEDBACK_CONFIDENCE_DELTA
            if normalized_rating > 0
            else _NEGATIVE_FEEDBACK_CONFIDENCE_DELTA
        )
        for memory_id in unique_memory_ids:
            memory = self.store.get_memory(tenant_id, user_id, memory_id)
            if memory is None or memory.user_id != user_id:
                adjustments.append(FeedbackMemoryAdjustment(memory_id=memory_id, status="not_found"))
                continue
            previous_confidence = memory.confidence
            new_confidence = round(_clamp_confidence(previous_confidence + delta), 6)
            updated = self.update_memory(
                tenant_id,
                user_id,
                memory_id,
                MemoryUpdate(confidence=new_confidence),
            )
            if updated is None:
                adjustments.append(FeedbackMemoryAdjustment(memory_id=memory_id, status="not_found"))
                continue
            adjustments.append(
                FeedbackMemoryAdjustment(
                    memory_id=memory_id,
                    status="adjusted",
                    previous_confidence=previous_confidence,
                    new_confidence=updated.confidence,
                    memory=updated,
                )
            )
        return event, adjustments

    def format_for_prompt(self, candidates: list[MemoryCandidate]) -> str:
        if not candidates:
            return "No relevant user memory."

        lines = [
            "Relevant memory for this user and tenant:",
            "Use high-confidence memory as context. Treat confidence below 0.70 as a hint, not a fact.",
        ]
        ranked_candidates = sorted(candidates, key=_prompt_candidate_rank, reverse=True)
        max_tokens = max(1, int(getattr(settings, "memory_prompt_max_tokens", 800)))
        used_tokens = _estimate_prompt_tokens("\n".join(lines))
        current_groups: set[str] = set()
        emitted = 0
        for candidate in ranked_candidates:
            memory = candidate.memory
            group = f"{memory.scope}/{memory.type}"
            additions: list[str] = []
            if group not in current_groups:
                additions.append(f"\n{group}:")
            additions.append(_format_memory_prompt_line(candidate))

            addition_text = "\n".join(additions)
            addition_tokens = _estimate_prompt_tokens(addition_text)
            if used_tokens + addition_tokens <= max_tokens:
                lines.extend(additions)
                used_tokens += addition_tokens
                current_groups.add(group)
                emitted += 1
                continue

            header_tokens = _estimate_prompt_tokens(additions[0]) if len(additions) > 1 else 0
            remaining_tokens = max_tokens - used_tokens - header_tokens
            if remaining_tokens >= 16:
                truncated_line = _truncate_to_prompt_tokens(additions[-1], remaining_tokens)
                if truncated_line:
                    if len(additions) > 1:
                        lines.append(additions[0])
                        used_tokens += header_tokens
                        current_groups.add(group)
                    lines.append(truncated_line)
                    used_tokens += _estimate_prompt_tokens(truncated_line)
                    emitted += 1
            break

        if emitted == 0:
            return "No relevant user memory."
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
                last_accessed_at=candidate.memory.last_accessed_at,
                access_count=candidate.memory.access_count,
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
            logger.warning("Memory vector search failed; falling back to structured search", exc_info=True)
            return []

    def _hydrate_vector_candidates(
        self,
        tenant_id: str,
        user_id: str,
        candidates: list[MemoryCandidate],
    ) -> list[MemoryCandidate]:
        if not candidates:
            return []
        hydrate_ids = [
            candidate.memory.memory_id
            for candidate in candidates
            if candidate.memory.memory_id and _vector_candidate_needs_hydration(candidate)
        ]
        hydrated_by_id = {}
        if hydrate_ids:
            hydrated_by_id = {
                memory.memory_id: memory
                for memory in self.store.get_memories_by_ids(tenant_id, user_id, hydrate_ids)
            }
        hydrated_candidates: list[MemoryCandidate] = []
        for candidate in candidates:
            memory_id = candidate.memory.memory_id
            if not memory_id:
                continue
            memory = hydrated_by_id.get(memory_id) if _vector_candidate_needs_hydration(candidate) else candidate.memory
            if memory is None:
                continue
            hydrated_candidates.append(
                MemoryCandidate(
                    memory=memory,
                    score=candidate.score,
                    retrieval_source="vector",
                )
            )
        return hydrated_candidates

    def _run_maintenance_if_due(
        self,
        tenant_id: str,
        user_id: str,
        kind: str,
        action: Callable[[], object],
    ) -> None:
        if not settings.memory_maintenance_enabled:
            return
        cooldown_seconds = int(getattr(settings, "memory_maintenance_cooldown_seconds", 300))
        now = datetime.now(timezone.utc)
        key = (tenant_id, user_id, kind)
        with self._maintenance_lock:
            previous = self._maintenance_last_run.get(key)
            if previous and cooldown_seconds > 0 and (now - previous).total_seconds() < cooldown_seconds:
                return
            self._maintenance_last_run[key] = now
        try:
            action()
        except Exception:
            logger.warning(
                "Memory maintenance failed; request will continue.",
                extra={"tenant_id": tenant_id, "user_id": user_id, "maintenance_kind": kind},
                exc_info=True,
            )

    def _find_semantic_duplicate_memory(
        self,
        *,
        tenant_id: str,
        user_id: str,
        scope: str,
        type: str,
        key: str,
        content: str,
    ) -> MemoryRecord | None:
        if self.vector_store is None or scope not in {"user", "org"}:
            return None
        threshold = float(getattr(settings, "memory_semantic_dedup_min_score", 0.88))
        if threshold <= 0:
            return None
        query = _memory_similarity_query(scope=scope, type=type, key=key, content=content)
        for candidate in self._vector_search(tenant_id, user_id, query, 3):
            score = candidate.score if candidate.score is not None else 0.0
            if score < threshold:
                continue
            memory = self.store.get_memory(tenant_id, user_id, candidate.memory.memory_id)
            if memory is None:
                continue
            if _is_semantic_duplicate_memory(memory, scope=scope, type=type, user_id=user_id):
                return memory
        return None

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


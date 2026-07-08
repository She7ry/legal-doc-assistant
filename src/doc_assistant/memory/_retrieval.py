"""Memory 检索、过滤与 RRF 融合工具。"""

from __future__ import annotations

from doc_assistant.config.settings import settings
from doc_assistant.memory.schemas import MemoryCandidate, MemoryRecord


def _can_read_memory(memory: MemoryRecord, user_id: str) -> bool:
    return memory.user_id == user_id or memory.visibility in {"team", "org"}


def _filter_memory_candidates(
    candidates: list[MemoryCandidate],
    user_id: str,
) -> list[MemoryCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.memory.status == "active"
        and not candidate.memory.is_expired()
        and candidate.memory.confidence >= settings.memory_min_confidence
        and _can_read_memory(candidate.memory, user_id)
    ]


def _memory_similarity_query(*, scope: str, type: str, key: str, content: str) -> str:
    return "\n".join(
        [
            f"scope: {scope}",
            f"type: {type}",
            f"key: {key}",
            f"content: {content}",
        ]
    )


def _is_semantic_duplicate_memory(
    memory: MemoryRecord,
    *,
    scope: str,
    type: str,
    user_id: str,
) -> bool:
    return (
        memory.user_id == user_id
        and memory.scope == scope
        and memory.type == type
        and memory.status == "active"
        and not memory.is_expired()
    )


def _vector_candidate_needs_hydration(candidate: MemoryCandidate) -> bool:
    return candidate.retrieval_source != "vector"

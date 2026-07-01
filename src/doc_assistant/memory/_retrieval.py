"""Memory 检索、过滤与 RRF 融合工具。"""

from __future__ import annotations

from datetime import datetime

from doc_assistant.config.settings import settings
from doc_assistant.memory.maintenance import _memory_retention_rank
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


def _candidate_rank(candidate: MemoryCandidate) -> tuple[float, float, datetime]:
    memory = candidate.memory
    relevance = candidate.score if candidate.score is not None else 0.0
    retention = _memory_retention_rank(memory)
    recency = memory.last_accessed_at or memory.updated_at
    return (relevance, retention, recency)


def _rrf_fuse_memory_candidates(
    vector_candidates: list[MemoryCandidate],
    lexical_candidates: list[MemoryCandidate],
    *,
    limit: int,
) -> list[MemoryCandidate]:
    if not vector_candidates and not lexical_candidates:
        return []

    rrf_k = float(settings.retrieval_rrf_k)
    fused: dict[str, dict[str, object]] = {}

    def add_candidates(candidates: list[MemoryCandidate], source: str, weight: float) -> None:
        for rank, candidate in enumerate(candidates, start=1):
            memory_id = candidate.memory.memory_id
            if not memory_id:
                continue
            entry = fused.setdefault(
                memory_id,
                {
                    "memory": candidate.memory,
                    "score": 0.0,
                    "sources": set(),
                    "best_source_score": 0.0,
                },
            )
            entry["score"] = float(entry["score"]) + weight / (rrf_k + rank)
            sources = entry["sources"]
            if isinstance(sources, set):
                sources.add(source)
            source_score = candidate.score if candidate.score is not None else 0.0
            entry["best_source_score"] = max(float(entry["best_source_score"]), source_score)

    add_candidates(
        vector_candidates,
        "vector",
        settings.retrieval_dense_weight,
    )
    add_candidates(
        lexical_candidates,
        "lexical",
        settings.retrieval_bm25_weight,
    )

    candidates: list[MemoryCandidate] = []
    for entry in fused.values():
        memory = entry["memory"]
        if not isinstance(memory, MemoryRecord):
            continue
        sources = entry["sources"]
        source_set = sources if isinstance(sources, set) else set()
        retrieval_source = "hybrid" if len(source_set) > 1 else next(iter(source_set), "unknown")
        rrf_score = float(entry["score"])
        candidates.append(
            MemoryCandidate(
                memory=memory,
                score=rrf_score,
                retrieval_source=retrieval_source,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.score if candidate.score is not None else 0.0,
            _memory_retention_rank(candidate.memory),
            candidate.memory.last_accessed_at or candidate.memory.updated_at,
        ),
        reverse=True,
    )
    return candidates[:limit]

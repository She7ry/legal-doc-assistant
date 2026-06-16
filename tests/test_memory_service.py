from __future__ import annotations

from datetime import datetime, timedelta, timezone

from langchain_core.documents import Document

from doc_assistant.memory import service as memory_service_module
from doc_assistant.memory.extraction import LLMMemoryExtractor
from doc_assistant.memory.policy import extract_memory_write_intents
from doc_assistant.memory.schemas import (
    MemoryCandidate,
    MemoryRecord,
    MemoryUpdate,
    MemoryWriteIntent,
)
from doc_assistant.memory.service import MemoryService
from doc_assistant.memory.store import MemoryStore
from doc_assistant.memory.vector_store import MemoryVectorStore
from doc_assistant.services.qa_service import DocumentQAService


class CaptureChatModel:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def invoke_messages(self, messages: list[dict[str, str]]) -> dict[str, str]:
        self.messages = messages
        return {"content": "The answer is grounded in the document [S1]."}


class SingleDocumentVectorStore:
    tenant_id = "tenant-a"

    def search(self, query: str, k: int | None = None) -> list[Document]:
        return [
            Document(
                page_content="Section 4 says notices must be sent within 10 business days.",
                metadata={"file_name": "contract.pdf", "page": 0, "chunk_id": 4},
            )
        ]


class EmptyDocumentVectorStore:
    tenant_id = "tenant-a"

    def search(self, query: str, k: int | None = None) -> list[Document]:
        return []


class RecordingChatModel:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def invoke_messages(self, messages: list[dict[str, str]]) -> dict[str, str]:
        self.messages = messages
        return {"content": "General answer."}


class SummaryChatModel:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def invoke_messages(self, messages: list[dict[str, str]]) -> dict[str, str]:
        self.messages = messages
        return {
            "content": (
                "Document type: supply agreement; key facts: Acme is party A and "
                "the term is 3 years; user concern: renewal notice; next step: "
                "confirm governing law."
            )
        }


class FakeMemoryVectorStore:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.upserted: list[str] = []

    def upsert_memory(self, memory) -> str:
        self.upserted.append(memory.memory_id)
        return memory.memory_id

    def delete_memory(self, memory_id: str) -> None:
        self.deleted.append(memory_id)


class StaleOnlyVectorStore:
    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id

    def search(self, query: str, *, tenant_id: str, user_id: str, k: int | None = None):
        del query, tenant_id, user_id, k
        placeholder = memory_record_factory(self.memory_id)
        return [MemoryCandidate(memory=placeholder, score=0.99)]


class OrderedMemoryVectorStore:
    def __init__(self, results: list[tuple[str, float]] | None = None) -> None:
        self.results = results or []
        self.deleted: list[str] = []
        self.upserted: list[str] = []

    def upsert_memory(self, memory) -> str:
        self.upserted.append(memory.memory_id)
        return memory.memory_id

    def delete_memory(self, memory_id: str) -> None:
        self.deleted.append(memory_id)

    def search(self, query: str, *, tenant_id: str, user_id: str, k: int | None = None):
        del query, tenant_id, user_id
        limit = k or len(self.results)
        return [
            MemoryCandidate(memory=memory_record_factory(memory_id), score=score)
            for memory_id, score in self.results[:limit]
        ]


def memory_record_factory(memory_id: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="stale_fact",
        content="Stale fact.",
        value_json=None,
        source="explicit",
        confidence=0.95,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def build_memory_service(tmp_path) -> MemoryService:
    return MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"), vector_store=None)


def test_memory_store_reuses_thread_connection(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")

    first = store._thread_connection()
    with store._connect() as second:
        assert second is first
    with store._connect() as third:
        assert third is first

    store.close()


def test_memory_write_policy_only_writes_explicit_long_term_memory(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("tenant-a", "user-a", "conversation-a")

    message_id = service.record_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="hello",
    )

    assert (
        service.write_memories_from_user_message(
            tenant_id="tenant-a",
            user_id="user-a",
            conversation_id=conversation_id,
            message_id=message_id,
            content="hello",
        )
        == []
    )

    explicit_message_id = service.record_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="请记住：以后回答用中文并保持简洁",
    )
    created = service.write_memories_from_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        message_id=explicit_message_id,
        content="请记住：以后回答用中文并保持简洁",
    )

    assert len(created) == 1
    assert created[0].type == "preference"
    assert created[0].key == "answer_style"
    assert created[0].source == "explicit"


def test_memory_supersedes_existing_active_key(tmp_path) -> None:
    service = build_memory_service(tmp_path)

    first = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer detailed answers.",
    )
    second = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer concise answers.",
    )

    active = service.list_memories("tenant-a", "user-a")
    all_memories = service.list_memories("tenant-a", "user-a", status=None, include_expired=True)

    assert [memory.memory_id for memory in active] == [second.memory_id]
    assert second.supersedes_id == first.memory_id
    assert {memory.status for memory in all_memories} == {"active", "stale"}


def test_memory_semantic_dedup_supersedes_similar_memory_with_different_key(tmp_path) -> None:
    vector_store = OrderedMemoryVectorStore()
    service = MemoryService(
        store=MemoryStore(tmp_path / "memory.sqlite3"),
        vector_store=vector_store,  # type: ignore[arg-type]
    )
    first = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="user_preference",
        content="Prefer concise Chinese answers.",
    )
    vector_store.results = [(first.memory_id, 0.95)]

    second = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Please answer briefly in Chinese.",
    )
    active = service.list_memories("tenant-a", "user-a")

    assert second.supersedes_id == first.memory_id
    assert second.key == first.key
    assert [memory.memory_id for memory in active] == [second.memory_id]


def test_memory_supersede_marks_obvious_preference_conflict(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    first = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Please answer in Chinese.",
    )
    second = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Please answer in English.",
    )

    assert second.supersedes_id == first.memory_id
    assert second.superseded_conflicting is True
    assert second.superseded_from_content == "Please answer in Chinese."

    prompt = service.format_for_prompt([MemoryCandidate(memory=second, score=0.95)])

    assert "recently updated from 'Please answer in Chinese.'" in prompt


def test_duplicate_memory_write_reuses_existing_active_memory(tmp_path) -> None:
    service = build_memory_service(tmp_path)

    first = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer concise answers.",
        value_json={"text": "Prefer concise answers."},
    )
    second = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer concise answers.",
        value_json={"text": "Prefer concise answers."},
    )

    active = service.list_memories("tenant-a", "user-a")
    all_memories = service.list_memories("tenant-a", "user-a", status=None, include_expired=True)

    assert second.memory_id == first.memory_id
    assert [memory.memory_id for memory in active] == [first.memory_id]
    assert len(all_memories) == 1


def test_duplicate_memory_write_ignores_internal_conflict_metadata(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Please answer in Chinese.",
    )
    first_english = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Please answer in English.",
    )
    second_english = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Please answer in English.",
    )

    all_memories = service.list_memories("tenant-a", "user-a", status=None, include_expired=True)

    assert second_english.memory_id == first_english.memory_id
    assert len(all_memories) == 2


def test_memory_retrieval_uses_structured_fallback_without_vector_store(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    memory = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer concise Chinese answers with implementation details.",
    )

    results = service.retrieve_relevant_memories(
        tenant_id="tenant-a",
        user_id="user-a",
        query="Please answer in Chinese with implementation details.",
    )

    assert len(results) == 1
    assert results[0].memory.memory_id == memory.memory_id


def test_memory_retrieval_tracks_access_stats(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    memory = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer concise Chinese answers with implementation details.",
    )

    service.retrieve_relevant_memories(
        tenant_id="tenant-a",
        user_id="user-a",
        query="Please answer in Chinese with implementation details.",
    )

    refreshed = service.store.get_memory("tenant-a", "user-a", memory.memory_id)
    assert refreshed is not None
    assert refreshed.access_count == 1
    assert refreshed.last_accessed_at is not None


def test_memory_maintenance_is_cooled_down_on_hot_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        memory_service_module,
        "settings",
        memory_service_module.settings.with_overrides(memory_maintenance_cooldown_seconds=300),
    )
    service = build_memory_service(tmp_path)
    cleanup_calls = 0
    enforce_calls = 0

    def cleanup(tenant_id: str, user_id: str):
        nonlocal cleanup_calls
        cleanup_calls += 1
        return []

    def enforce(tenant_id: str, user_id: str):
        nonlocal enforce_calls
        enforce_calls += 1
        return []

    service.cleanup_expired_memories = cleanup  # type: ignore[method-assign]
    service.enforce_memory_limit = enforce  # type: ignore[method-assign]

    service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="fact_one",
        content="First fact.",
    )
    service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="fact_two",
        content="Second fact.",
    )
    service.retrieve_relevant_memories(tenant_id="tenant-a", user_id="user-a", query="fact")
    service.retrieve_relevant_memories(tenant_id="tenant-a", user_id="user-a", query="fact")

    assert enforce_calls == 1
    assert cleanup_calls == 1


def test_memory_retrieval_uses_rrf_hybrid_ranking(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    hybrid = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="patent_license_focus",
        content="Patent license agreements often need indemnity review.",
    )
    vector_only = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="billing_preference",
        content="Invoices are reviewed by finance.",
    )
    lexical_only = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="patent_license_context",
        content="Patent license agreements are common.",
    )
    service.vector_store = OrderedMemoryVectorStore(
        [(vector_only.memory_id, 0.99), (hybrid.memory_id, 0.95)]
    )  # type: ignore[assignment]

    results = service.retrieve_relevant_memories(
        tenant_id="tenant-a",
        user_id="user-a",
        query="patent license indemnity",
        limit=3,
    )

    assert results[0].memory.memory_id == hybrid.memory_id
    assert results[0].retrieval_source == "hybrid"
    assert {candidate.memory.memory_id for candidate in results} >= {
        vector_only.memory_id,
        lexical_only.memory_id,
    }


def test_complete_vector_memory_candidate_does_not_require_sqlite_hydration(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    service = MemoryService(store=store, vector_store=None)
    vector_memory = memory_record_factory("vector-only-memory")

    class CompleteVectorStore:
        def search(self, query: str, *, tenant_id: str, user_id: str, k: int | None = None):
            del query, tenant_id, user_id, k
            return [
                MemoryCandidate(
                    memory=vector_memory,
                    score=0.91,
                    retrieval_source="vector",
                )
            ]

    def fail_hydration(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Vector candidate should not be hydrated from SQLite.")

    store.get_memories_by_ids = fail_hydration  # type: ignore[method-assign]
    service.vector_store = CompleteVectorStore()  # type: ignore[assignment]

    results = service.retrieve_relevant_memories(
        tenant_id="tenant-a",
        user_id="user-a",
        query="stale fact",
    )

    assert [candidate.memory.memory_id for candidate in results] == ["vector-only-memory"]


def test_memory_stats_empty_user(tmp_path) -> None:
    service = build_memory_service(tmp_path)

    stats = service.get_memory_stats("tenant-a", "user-a")

    assert stats["total_memories"] == 0
    assert stats["active_memories"] == 0
    assert stats["stale_memories"] == 0
    assert stats["deleted_memories"] == 0
    assert stats["expired_active_memories"] == 0
    assert stats["status_counts"] == {"active": 0, "deleted": 0, "stale": 0}
    assert stats["access"]["tracked_memories"] == 0
    assert stats["retrievals"]["total"] == 0
    assert stats["retrievals"]["hit_rate"] == 0.0


def test_memory_stats_counts_health_access_and_retrieval_sources(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    old_preference = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Please answer in Chinese.",
        confidence=0.8,
    )
    active_preference = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Please answer in English.",
        confidence=0.9,
    )
    active_fact = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="client_context",
        content="Acme is the client.",
        confidence=0.95,
    )
    deleted = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="deleted_context",
        content="Delete this context.",
        confidence=0.7,
    )
    service.delete_memory("tenant-a", "user-a", deleted.memory_id)
    expired = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="session",
        type="task_state",
        key="expired_session",
        content="Expired session state.",
        confidence=0.6,
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    service.store.touch_memories("tenant-a", "user-a", [active_fact.memory_id])
    service.store.touch_memories("tenant-a", "user-a", [active_fact.memory_id])
    service.log_retrieval(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=None,
        query="client context",
        document_count=2,
        memories=[
            MemoryCandidate(memory=active_fact, score=0.92, retrieval_source="vector"),
            MemoryCandidate(memory=active_preference, score=0.80, retrieval_source="fts"),
        ],
    )
    service.log_retrieval(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=None,
        query="unmatched",
        document_count=1,
        memories=[],
    )

    stats = service.get_memory_stats("tenant-a", "user-a")

    assert old_preference.status == "active"
    assert expired.is_expired()
    assert stats["total_memories"] == 5
    assert stats["active_memories"] == 2
    assert stats["stale_memories"] == 1
    assert stats["deleted_memories"] == 1
    assert stats["expired_active_memories"] == 1
    assert stats["status_counts"]["active"] == 3
    assert stats["scope_counts"] == {"session": 1, "user": 4}
    assert stats["type_counts"] == {"fact": 2, "preference": 2, "task_state": 1}
    assert stats["access"]["tracked_memories"] == 2
    assert stats["access"]["accessed"] == 1
    assert stats["access"]["never_accessed"] == 1
    assert stats["access"]["total_access_count"] == 2
    assert stats["access"]["max_access_count"] == 2
    assert stats["retrievals"]["total"] == 2
    assert stats["retrievals"]["with_memory"] == 1
    assert stats["retrievals"]["hit_rate"] == 0.5
    assert stats["retrievals"]["average_memory_count"] == 1.0
    assert stats["retrievals"]["selected_memory_source_counts"] == {"fts": 1, "vector": 1}
    assert stats["retrievals"]["selected_memory_source_ratios"] == {"vector": 0.5, "fts": 0.5}


def test_format_for_prompt_respects_total_token_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        memory_service_module,
        "settings",
        memory_service_module.settings.with_overrides(memory_prompt_max_tokens=95),
    )
    service = build_memory_service(tmp_path)
    high = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="important_contract_fact",
        content="High priority fact. " + ("A" * 600),
        confidence=0.95,
    )
    low = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="low_priority_fact",
        content="Low priority fact. " + ("B" * 600),
        confidence=0.56,
    )

    prompt = service.format_for_prompt(
        [
            MemoryCandidate(memory=low, score=0.99),
            MemoryCandidate(memory=high, score=0.10),
        ]
    )

    assert "important_contract_fact" in prompt
    assert "low_priority_fact" not in prompt
    assert len(prompt) < 420


def test_record_feedback_adjusts_linked_memory_confidence(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    memory = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="client_context",
        content="Acme is the client.",
        confidence=0.8,
    )

    positive_event, positive_adjustments = service.record_feedback(
        tenant_id="tenant-a",
        user_id="user-a",
        rating="positive",
        memory_ids=[memory.memory_id],
        comment="Useful memory.",
    )
    negative_event, negative_adjustments = service.record_feedback(
        tenant_id="tenant-a",
        user_id="user-a",
        rating=-1,
        memory_ids=[memory.memory_id, "missing-memory"],
    )
    refreshed = service.store.get_memory("tenant-a", "user-a", memory.memory_id)

    assert positive_event.rating == 1
    assert positive_event.comment == "Useful memory."
    assert positive_event.memory_ids == (memory.memory_id,)
    assert positive_adjustments[0].status == "adjusted"
    assert positive_adjustments[0].previous_confidence == 0.8
    assert positive_adjustments[0].new_confidence == 0.83
    assert negative_event.rating == -1
    assert [adjustment.status for adjustment in negative_adjustments] == ["adjusted", "not_found"]
    assert refreshed is not None
    assert round(refreshed.confidence, 2) == 0.75


def test_record_feedback_clamps_confidence_and_allows_unlinked_feedback(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    high = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer concise answers.",
        confidence=0.99,
    )
    low = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="low_confidence_fact",
        content="Low confidence fact.",
        confidence=0.02,
    )

    _, positive = service.record_feedback(
        tenant_id="tenant-a",
        user_id="user-a",
        rating=1,
        memory_ids=[high.memory_id],
    )
    unlinked_event, unlinked_adjustments = service.record_feedback(
        tenant_id="tenant-a",
        user_id="user-a",
        rating="negative",
        memory_ids=[],
    )
    _, negative = service.record_feedback(
        tenant_id="tenant-a",
        user_id="user-a",
        rating="negative",
        memory_ids=[low.memory_id],
    )

    assert positive[0].new_confidence == 1.0
    assert unlinked_event.memory_ids == ()
    assert unlinked_adjustments == []
    assert negative[0].new_confidence == 0.0


def test_memory_limit_marks_low_priority_memories_stale(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        memory_service_module,
        "settings",
        memory_service_module.settings.with_overrides(
            memory_max_active_per_user=2,
            memory_maintenance_cooldown_seconds=0,
        ),
    )
    service = build_memory_service(tmp_path)
    for index in range(3):
        service.create_memory(
            tenant_id="tenant-a",
            user_id="user-a",
            scope="user",
            type="fact",
            key=f"fact_{index}",
            content=f"Low priority fact {index}.",
            confidence=0.6,
        )

    active = service.list_memories("tenant-a", "user-a")
    all_memories = service.list_memories("tenant-a", "user-a", status=None, include_expired=True)

    assert len(active) == 2
    assert [memory.status for memory in all_memories].count("stale") == 1


def test_expired_memories_are_marked_stale_and_removed_from_vector(tmp_path) -> None:
    vector_store = FakeMemoryVectorStore()
    service = MemoryService(
        store=MemoryStore(tmp_path / "memory.sqlite3"),
        vector_store=vector_store,  # type: ignore[arg-type]
    )
    expired = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="session",
        type="fact",
        key="temporary_fact",
        content="Temporary fact.",
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )

    stale = service.cleanup_expired_memories("tenant-a", "user-a")

    assert [memory.memory_id for memory in stale] == [expired.memory_id]
    assert vector_store.deleted[-1] == expired.memory_id
    refreshed = service.store.get_memory("tenant-a", "user-a", expired.memory_id)
    assert refreshed is not None
    assert refreshed.status == "stale"


def test_memory_update_can_clear_nullable_fields(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    expires_at = datetime.now(timezone.utc) + timedelta(days=3)
    memory = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="matter",
        content="Matter is active.",
        value_json={"text": "Matter is active."},
        expires_at=expires_at,
    )

    unchanged = service.update_memory("tenant-a", "user-a", memory.memory_id, MemoryUpdate())
    assert unchanged is not None
    assert unchanged.value_json == {"text": "Matter is active."}
    assert unchanged.expires_at == expires_at

    cleared = service.update_memory(
        "tenant-a",
        "user-a",
        memory.memory_id,
        MemoryUpdate(value_json=None, expires_at=None),
    )

    assert cleared is not None
    assert cleared.value_json is None
    assert cleared.expires_at is None


def test_memory_list_supports_pagination_and_count(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    for index in range(3):
        service.create_memory(
            tenant_id="tenant-a",
            user_id="user-a",
            scope="user",
            type="fact",
            key=f"matter_{index}",
            content=f"Matter fact {index}.",
        )

    page = service.list_memories("tenant-a", "user-a", limit=2, offset=1)

    assert len(page) == 2
    assert service.count_memories("tenant-a", "user-a") == 3


def test_load_conversation_history_returns_recent_messages(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("tenant-a", "user-a", "conversation-a")
    service.record_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="First question",
    )
    service.record_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="First answer",
    )

    history = service.load_conversation_history(
        "tenant-a",
        "user-a",
        conversation_id,
        limit=10,
    )

    assert history == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]


def test_load_conversation_history_uses_summary_plus_recent_messages(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("tenant-a", "user-a", "conversation-a")
    service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="session",
        type="task_state",
        key="conversation_summary_conversation-a",
        content="Conversation summary: Acme Corp and Beta LLC negotiated Delaware law.",
        source="system_generated",
        confidence=0.7,
        conversation_id=conversation_id,
    )
    for index in range(12):
        service.record_user_message(
            tenant_id="tenant-a",
            user_id="user-a",
            conversation_id=conversation_id,
            content=f"Question {index}",
        )

    history = service.load_conversation_history(
        "tenant-a",
        "user-a",
        conversation_id,
        limit=20,
    )

    assert history[0] == {
        "role": "system",
        "content": "Conversation summary: Acme Corp and Beta LLC negotiated Delaware law.",
    }
    assert [message["content"] for message in history[1:]] == [
        f"Question {index}" for index in range(4, 12)
    ]


def test_load_conversation_history_can_return_raw_messages_without_summary(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("tenant-a", "user-a", "conversation-a")
    service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="session",
        type="task_state",
        key="conversation_summary_conversation-a",
        content="Conversation summary: Use only for model context.",
        source="system_generated",
        confidence=0.7,
        conversation_id=conversation_id,
    )
    for index in range(3):
        service.record_user_message(
            tenant_id="tenant-a",
            user_id="user-a",
            conversation_id=conversation_id,
            content=f"Raw question {index}",
        )

    history = service.load_conversation_history(
        "tenant-a",
        "user-a",
        conversation_id,
        limit=20,
        include_summary=False,
    )

    assert history == [
        {"role": "user", "content": f"Raw question {index}"} for index in range(3)
    ]


def test_summarize_conversation_creates_session_task_state_memory(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("tenant-a", "user-a", "conversation-a")
    service.record_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Please review the notice clause.",
    )
    service.record_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="The notice clause requires 10 business days of prior notice.",
    )

    memory = service.summarize_conversation_to_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
    )

    assert memory is not None
    assert memory.scope == "session"
    assert memory.type == "task_state"
    assert memory.key == "conversation_summary_conversation-a"
    assert "Please review the notice clause" in memory.content
    assert "10 business days" in memory.content


def test_summarize_conversation_extracts_legal_review_structure(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("tenant-a", "user-a", "conversation-a")
    service.record_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content=(
            "Please review the SaaS MSA between Acme Corp and Beta LLC. "
            "The effective date is January 15, 2026. Governing law is Delaware. "
            "Focus on uncapped indemnity and renewal notice."
        ),
    )
    service.record_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content=(
            "Key risks: uncapped indemnity for IP claims; renewal notice requires "
            "30 days; confirm whether New York venue conflicts with Delaware governing law."
        ),
    )

    memory = service.summarize_conversation_to_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
    )

    assert memory is not None
    assert "Key parties and entities" in memory.content
    assert "Acme Corp" in memory.content
    assert "Beta LLC" in memory.content
    assert "Key dates and deadlines" in memory.content
    assert "January 15, 2026" in memory.content
    assert "Governing law is Delaware" in memory.content
    assert "uncapped indemnity" in memory.content


def test_summarize_conversation_uses_llm_when_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        memory_service_module,
        "settings",
        memory_service_module.settings.with_overrides(memory_llm_extraction_enabled=True),
    )
    model = SummaryChatModel()
    service = MemoryService(
        store=MemoryStore(tmp_path / "memory.sqlite3"),
        vector_store=None,
        summary_model=model,
    )
    conversation_id = service.ensure_context("tenant-a", "user-a", "conversation-a")
    service.record_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Can you review this agreement? It lasts 3 years and Acme is party A.",
    )
    service.record_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="I will focus on renewal notice and governing law.",
    )

    memory = service.summarize_conversation_to_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
    )

    assert memory is not None
    assert "Acme is party A" in memory.content
    assert "term is 3 years" in memory.content
    assert memory.value_json is not None
    assert memory.value_json["summary_method"] == "llm"
    assert model.messages[0]["role"] == "system"


def test_summarize_conversation_supersedes_prior_summary(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("tenant-a", "user-a", "conversation-a")
    service.record_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="First question.",
    )
    first = service.summarize_conversation_to_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
    )
    service.record_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="First answer.",
    )

    second = service.summarize_conversation_to_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
    )
    active = service.list_memories("tenant-a", "user-a")
    all_memories = service.list_memories("tenant-a", "user-a", status=None, include_expired=True)

    assert first is not None
    assert second is not None
    assert second.supersedes_id == first.memory_id
    assert [memory.memory_id for memory in active] == [second.memory_id]
    assert {memory.status for memory in all_memories} == {"active", "stale"}


def test_summarize_conversation_incrementally_updates_prior_summary(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("tenant-a", "user-a", "conversation-a")
    service.record_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Review the services agreement between Acme Corp and Beta LLC.",
    )
    service.record_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Initial conclusion: governing law is Delaware.",
    )
    first = service.summarize_conversation_to_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
    )
    service.record_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="New issue: renewal notice deadline is 45 days.",
    )
    service.record_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Next step: confirm whether the 45 days are calendar or business days.",
    )

    second = service.summarize_conversation_to_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
    )

    assert first is not None
    assert second is not None
    assert second.supersedes_id == first.memory_id
    assert "Acme Corp" in second.content
    assert "45 days" in second.content
    assert second.value_json is not None
    assert second.value_json["previous_message_count"] == 2
    assert second.value_json["incremental"] is True


def test_maybe_summarize_conversation_respects_threshold_and_interval(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        memory_service_module,
        "settings",
        memory_service_module.settings.with_overrides(
            memory_auto_summary_threshold=4,
            memory_auto_summary_interval=2,
            memory_auto_summary_window=3,
        ),
    )
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("tenant-a", "user-a", "conversation-a")
    for index in range(3):
        service.record_user_message(
            tenant_id="tenant-a",
            user_id="user-a",
            conversation_id=conversation_id,
            content=f"Question {index}",
        )

    assert (
        service.maybe_summarize_conversation(
            tenant_id="tenant-a",
            user_id="user-a",
            conversation_id=conversation_id,
        )
        is None
    )

    service.record_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Answer 3",
    )
    first = service.maybe_summarize_conversation(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
    )
    assert first is not None
    assert first.value_json is not None
    assert first.value_json["message_count"] == 4

    service.record_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Question 4",
    )
    assert (
        service.maybe_summarize_conversation(
            tenant_id="tenant-a",
            user_id="user-a",
            conversation_id=conversation_id,
        )
        is None
    )

    service.record_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Answer 5",
    )
    second = service.maybe_summarize_conversation(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
    )

    assert second is not None
    assert second.supersedes_id == first.memory_id
    assert second.value_json is not None
    assert second.value_json["message_count"] == 6


def test_document_qa_merges_persisted_history_when_client_history_is_missing(tmp_path) -> None:
    memory_service = build_memory_service(tmp_path)
    conversation_id = memory_service.ensure_context("tenant-a", "user-a", "conversation-a")
    memory_service.record_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Earlier question about renewal.",
    )
    memory_service.record_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Earlier answer about renewal.",
    )
    chat_model = RecordingChatModel()
    service = DocumentQAService(
        vector_store=EmptyDocumentVectorStore(),
        chat_model=chat_model,
        memory_service=memory_service,
        tenant_id="tenant-a",
    )

    service.ask(
        "What should I review next for this contract?",
        user_id="user-a",
        conversation_id=conversation_id,
    )

    prompt = chat_model.messages[1]["content"]
    assert "Earlier question about renewal." in prompt
    assert "Earlier answer about renewal." in prompt


def test_document_qa_includes_conversation_summary_context(tmp_path) -> None:
    memory_service = build_memory_service(tmp_path)
    conversation_id = memory_service.ensure_context("tenant-a", "user-a", "conversation-a")
    memory_service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="session",
        type="task_state",
        key="conversation_summary_conversation-a",
        content="Conversation summary: Acme Corp is the counterparty; Delaware law applies.",
        source="system_generated",
        confidence=0.7,
        conversation_id=conversation_id,
    )
    for index in range(12):
        memory_service.record_user_message(
            tenant_id="tenant-a",
            user_id="user-a",
            conversation_id=conversation_id,
            content=f"Earlier raw question {index}",
        )
    chat_model = RecordingChatModel()
    service = DocumentQAService(
        vector_store=EmptyDocumentVectorStore(),
        chat_model=chat_model,
        memory_service=memory_service,
        tenant_id="tenant-a",
    )

    service.ask(
        "What should I review next for this contract?",
        user_id="user-a",
        conversation_id=conversation_id,
    )

    prompt = chat_model.messages[1]["content"]
    assert "Session summary: Conversation summary: Acme Corp is the counterparty" in prompt
    assert "Earlier raw question 0" not in prompt
    assert "Earlier raw question 4" in prompt


def test_document_qa_can_skip_persisted_history_for_agent_context(tmp_path) -> None:
    memory_service = build_memory_service(tmp_path)
    conversation_id = memory_service.ensure_context("tenant-a", "user-a", "conversation-a")
    memory_service.record_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Earlier user chat that should not be injected.",
    )
    memory_service.record_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Earlier assistant chat that should not be injected.",
    )
    chat_model = RecordingChatModel()
    service = DocumentQAService(
        vector_store=EmptyDocumentVectorStore(),
        chat_model=chat_model,
        memory_service=memory_service,
        tenant_id="tenant-a",
    )

    service.ask(
        "Build the obligation calendar.",
        chat_history=[{"role": "user", "content": "Agent objective: Review obligations."}],
        user_id="user-a",
        conversation_id=conversation_id,
        merge_persisted_history=False,
    )

    prompt = chat_model.messages[1]["content"]
    assert "Agent objective: Review obligations." in prompt
    assert "Earlier user chat that should not be injected." not in prompt
    assert "Earlier assistant chat that should not be injected." not in prompt


def test_memory_vector_search_filters_by_tenant_user_and_visibility() -> None:
    class FakeVectorBackend:
        def __init__(self) -> None:
            self.filter = None

        def similarity_search_with_relevance_scores(self, query, *, k, filter):
            self.filter = filter
            return []

    backend = FakeVectorBackend()
    vector_store = MemoryVectorStore.__new__(MemoryVectorStore)
    vector_store.tenant_id = "tenant-a"
    vector_store.vector_store = backend

    results = vector_store.search("answer style", tenant_id="tenant-b", user_id="user-a", k=3)

    assert results == []
    assert backend.filter == {
        "$and": [
            {"tenant_id": "tenant-b"},
            {"status": "active"},
            {
                "$or": [
                    {"user_id": "user-a"},
                    {"visibility": "team"},
                    {"visibility": "org"},
                ]
            },
        ]
    }


def test_memory_vector_search_reconstructs_full_metadata() -> None:
    now = datetime.now(timezone.utc).isoformat()

    class FakeVectorBackend:
        def similarity_search_with_relevance_scores(self, query, *, k, filter):
            del query, k, filter
            return [
                (
                    Document(
                        page_content="Prefer concise answers.",
                        metadata={
                            "memory_id": "memory-a",
                            "tenant_id": "tenant-a",
                            "user_id": "user-a",
                            "scope": "user",
                            "type": "preference",
                            "key": "answer_style",
                            "source": "explicit",
                            "confidence": 0.9,
                            "visibility": "private",
                            "status": "active",
                            "content": "Prefer concise answers.",
                            "value_json": '{"text":"Prefer concise answers."}',
                            "permissions_json": '["read","write"]',
                            "created_at": now,
                            "updated_at": now,
                        },
                    ),
                    0.94,
                )
            ]

    vector_store = MemoryVectorStore.__new__(MemoryVectorStore)
    vector_store.tenant_id = "tenant-a"
    vector_store.vector_store = FakeVectorBackend()

    results = vector_store.search("concise answers", tenant_id="tenant-a", user_id="user-a", k=1)

    assert len(results) == 1
    assert results[0].retrieval_source == "vector"
    assert results[0].memory.value_json == {"text": "Prefer concise answers."}
    assert results[0].memory.permissions == ("read", "write")


def test_memory_vector_search_discards_unreadable_backend_results() -> None:
    now = datetime.now(timezone.utc).isoformat()

    class FakeVectorBackend:
        def similarity_search_with_relevance_scores(self, query, *, k, filter):
            del query, k, filter
            return [
                (
                    Document(
                        page_content="Own private memory.",
                        metadata={
                            "memory_id": "own-private",
                            "tenant_id": "tenant-a",
                            "user_id": "user-a",
                            "visibility": "private",
                            "status": "active",
                            "confidence": 0.9,
                            "created_at": now,
                            "updated_at": now,
                        },
                    ),
                    0.9,
                ),
                (
                    Document(
                        page_content="Shared team memory.",
                        metadata={
                            "memory_id": "team-shared",
                            "tenant_id": "tenant-a",
                            "user_id": "user-b",
                            "visibility": "team",
                            "status": "active",
                            "confidence": 0.9,
                            "created_at": now,
                            "updated_at": now,
                        },
                    ),
                    0.8,
                ),
                (
                    Document(
                        page_content="Other user's private memory.",
                        metadata={
                            "memory_id": "other-private",
                            "tenant_id": "tenant-a",
                            "user_id": "user-b",
                            "visibility": "private",
                            "status": "active",
                            "confidence": 0.9,
                            "created_at": now,
                            "updated_at": now,
                        },
                    ),
                    0.7,
                ),
                (
                    Document(
                        page_content="Wrong tenant shared memory.",
                        metadata={
                            "memory_id": "wrong-tenant",
                            "tenant_id": "tenant-b",
                            "user_id": "user-a",
                            "visibility": "org",
                            "status": "active",
                            "confidence": 0.9,
                            "created_at": now,
                            "updated_at": now,
                        },
                    ),
                    0.6,
                ),
            ]

    vector_store = MemoryVectorStore.__new__(MemoryVectorStore)
    vector_store.tenant_id = "tenant-a"
    vector_store.vector_store = FakeVectorBackend()

    results = vector_store.search("answer style", tenant_id="tenant-a", user_id="user-a", k=4)

    assert [candidate.memory.memory_id for candidate in results] == ["own-private", "team-shared"]


def test_memory_policy_splits_multiple_explicit_intents() -> None:
    intents = extract_memory_write_intents(
        "请记住：以后回答用中文并保持简洁，并且我的职位是法务总监"
    )

    assert [intent.key for intent in intents] == ["answer_style", "business_context"]
    assert all("请记住" not in intent.content for intent in intents)


def test_memory_policy_strips_always_answer_marker() -> None:
    intents = extract_memory_write_intents("Always answer in concise Chinese.")

    assert len(intents) == 1
    assert intents[0].key == "answer_style"
    assert intents[0].content == "in concise Chinese"


def test_memory_policy_infers_business_context_without_explicit_marker() -> None:
    intents = extract_memory_write_intents("Our company mainly provides IP agency services.")

    assert len(intents) == 1
    assert intents[0].type == "fact"
    assert intents[0].key == "business_context"
    assert intents[0].source == "inferred"


def test_memory_policy_infers_legal_review_profile_without_explicit_marker() -> None:
    intents = extract_memory_write_intents(
        "We mainly review patent license agreements and focus on indemnity and liability clauses."
    )

    assert len(intents) == 1
    assert intents[0].type == "fact"
    assert intents[0].key == "review_profile"
    assert intents[0].source == "inferred"


def test_memory_policy_infers_chinese_future_answer_preference() -> None:
    intents = extract_memory_write_intents("\u8bf7\u7ed9\u6211\u7684\u56de\u590d\u90fd\u9644\u4e0a\u82f1\u6587\u5bf9\u7167")

    assert len(intents) == 1
    assert intents[0].type == "preference"
    assert intents[0].key == "answer_style"
    assert intents[0].source == "inferred"


def test_external_memory_extractor_can_fill_rule_gaps(tmp_path) -> None:
    def extractor(text: str) -> list[MemoryWriteIntent]:
        assert text == "Acme prefers arbitration in Singapore."
        return [
            MemoryWriteIntent(
                type="preference",
                key="dispute_resolution_preference",
                content=text,
                value_json={"text": text},
                source="inferred",
                confidence=0.68,
            )
        ]

    service = MemoryService(
        store=MemoryStore(tmp_path / "memory.sqlite3"),
        vector_store=None,
        memory_extractor=extractor,
    )
    conversation_id = service.ensure_context("tenant-a", "user-a", "conversation-a")
    message_id = service.record_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="Acme prefers arbitration in Singapore.",
    )

    created = service.write_memories_from_user_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        message_id=message_id,
        content="Acme prefers arbitration in Singapore.",
    )

    assert len(created) == 1
    assert created[0].key == "dispute_resolution_preference"
    assert created[0].source == "inferred"


def test_llm_memory_extractor_parses_inferred_memory() -> None:
    class FakeExtractionModel:
        def invoke_messages(self, messages):
            assert "commonly reviewed contract types" in messages[0]["content"]
            assert "recurring clause focus areas" in messages[0]["content"]
            assert messages[-1]["content"] == "甲方是我们客户，需要偏保护客户侧。"
            return {
                "content": (
                    '{"memories":[{"type":"fact","key":"client_side_context",'
                    '"content":"甲方是我们客户，需要偏保护客户侧。","confidence":0.72}]}'
                )
            }

    extractor = LLMMemoryExtractor(chat_model=FakeExtractionModel())

    intents = extractor("甲方是我们客户，需要偏保护客户侧。")

    assert len(intents) == 1
    assert intents[0].type == "fact"
    assert intents[0].key == "client_side_context"
    assert intents[0].source == "inferred"
    assert intents[0].confidence == 0.72


def test_vector_hydration_filter_falls_back_to_lexical_results(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    stale = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="old_context",
        content="Old stale context.",
    )
    active = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="arbitration_context",
        content="Acme prefers arbitration in Singapore.",
    )
    service.delete_memory("tenant-a", "user-a", stale.memory_id)
    service.vector_store = StaleOnlyVectorStore(stale.memory_id)  # type: ignore[assignment]

    results = service.retrieve_relevant_memories(
        tenant_id="tenant-a",
        user_id="user-a",
        query="arbitration Singapore",
    )

    assert [candidate.memory.memory_id for candidate in results] == [active.memory_id]


def test_assistant_task_fact_extraction_writes_task_memory(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("tenant-a", "user-a", "conversation-a")
    message_id = service.record_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content="The agreement is governed by New York law [S1].",
    )

    created = service.write_memories_from_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        message_id=message_id,
        content="The agreement is governed by New York law [S1].",
        task_id="task-a",
    )

    assert len(created) == 1
    assert created[0].scope == "task"
    assert created[0].source == "system_generated"
    assert created[0].task_id == "task-a"


def test_assistant_task_fact_extraction_allows_structured_facts_without_citation(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("tenant-a", "user-a", "conversation-a")
    message_id = service.record_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        content=(
            "The parties are Acme Corp and Beta LLC. "
            "The effective date is January 15, 2026. "
            "The governing law is Delaware."
        ),
    )

    created = service.write_memories_from_assistant_message(
        tenant_id="tenant-a",
        user_id="user-a",
        conversation_id=conversation_id,
        message_id=message_id,
        content=(
            "The parties are Acme Corp and Beta LLC. "
            "The effective date is January 15, 2026. "
            "The governing law is Delaware."
        ),
        task_id="task-a",
    )

    assert len(created) == 3
    assert all(memory.scope == "task" for memory in created)
    assert all(memory.confidence == 0.56 for memory in created)
    assert any("governing law is Delaware" in memory.content for memory in created)


def test_vector_repair_deletes_inactive_and_upserts_active_memories(tmp_path) -> None:
    vector_store = FakeMemoryVectorStore()
    service = MemoryService(
        store=MemoryStore(tmp_path / "memory.sqlite3"),
        vector_store=vector_store,  # type: ignore[arg-type]
    )
    active = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="active_fact",
        content="Active fact.",
    )
    deleted = service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="fact",
        key="deleted_fact",
        content="Deleted fact.",
    )
    service.delete_memory("tenant-a", "user-a", deleted.memory_id)

    result = service.repair_vector_index("tenant-a", "user-a")

    assert deleted.memory_id in vector_store.deleted
    assert active.memory_id in vector_store.upserted
    assert result["deleted"] >= 1
    assert result["upserted"] >= 1


def test_document_qa_separates_memory_from_retrieved_documents(tmp_path) -> None:
    memory_service = build_memory_service(tmp_path)
    memory_service.create_memory(
        tenant_id="tenant-a",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer concise Chinese answers.",
    )
    chat_model = CaptureChatModel()
    service = DocumentQAService(
        vector_store=SingleDocumentVectorStore(),
        chat_model=chat_model,
        memory_service=memory_service,
        tenant_id="tenant-a",
    )

    answer = service.ask(
        "What is the notice period?",
        user_id="user-a",
        conversation_id="conversation-a",
    )

    assert answer.citations
    assert answer.memories_used[0].key == "answer_style"
    assert chat_model.messages[0]["role"] == "system"
    assert chat_model.messages[0]["content"].strip()
    assert chat_model.messages[1]["role"] == "user"
    assert "<user_memory>" in chat_model.messages[1]["content"]
    assert "Prefer concise Chinese answers." in chat_model.messages[1]["content"]
    assert "<retrieved_documents>" in chat_model.messages[1]["content"]
    assert "Section 4 says notices must be sent" in chat_model.messages[1]["content"]

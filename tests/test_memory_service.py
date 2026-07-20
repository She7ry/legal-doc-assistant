from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Barrier, Lock, Thread
from time import sleep
from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from qdrant_client import QdrantClient, models

from ai.memory.schemas import MemoryCandidate, MemoryRecord, MemoryUpdate
from ai.memory.service import MemoryService, extract_memory_write_intents
from ai.memory.store import MemoryStore
from ai.memory.vector_store import MemoryVectorStore
from ai.rag.qa_service import DocumentQAService


class RecordingChatModel(BaseChatModel):
    response: str = "The answer is grounded in the document [S1]."
    messages: list[dict[str, str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-test"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del stop, run_manager, kwargs
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        self.messages = [
            {"role": roles.get(message.type, message.type), "content": str(message.content)}
            for message in messages
        ]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response))])


class SingleDocumentVectorStore:
    user_id = "user-a"

    def search(self, query: str, k: int | None = None) -> list[Document]:
        return [
            Document(
                page_content="Section 4 says notices must be sent within 10 business days.",
                metadata={"file_name": "contract.pdf", "page": 0, "chunk_id": 4},
            )
        ]


class EmptyDocumentVectorStore:
    user_id = "user-a"

    def search(self, query: str, k: int | None = None) -> list[Document]:
        return []


class FakeMemoryVectorStore:
    def __init__(self, results: list[tuple[str, float]] | None = None) -> None:
        self.results = results or []
        self.deleted: list[str] = []
        self.upserted: list[str] = []

    def upsert_memory(self, memory) -> str:
        self.upserted.append(memory.memory_id)
        return memory.memory_id

    def delete_memory(self, memory_id: str) -> bool:
        self.deleted.append(memory_id)
        return True

    def search(self, query: str, *, user_id: str, k: int | None = None):
        del query, user_id
        return [
            MemoryCandidate(
                memory=memory_record_factory(memory_id),
                score=score,
                retrieval_source="vector",
            )
            for memory_id, score in self.results[: k or len(self.results)]
        ]


class CoordinatedMemoryStore(MemoryStore):
    """Expose the old lock-free reads to a barrier for deterministic regressions."""

    def __init__(self, db_path, *, lookup_barrier=None, read_barrier=None) -> None:
        self.lookup_barrier = lookup_barrier
        self.read_barrier = read_barrier
        super().__init__(db_path)

    def find_active_memory_by_key(self, *args, **kwargs):
        result = super().find_active_memory_by_key(*args, **kwargs)
        if self.lookup_barrier:
            self.lookup_barrier.wait(timeout=5)
        return result

    def get_memory(self, *args, **kwargs):
        result = super().get_memory(*args, **kwargs)
        if self.read_barrier:
            self.read_barrier.wait(timeout=5)
        return result


class FakeMemoryEmbeddingModel:
    def embed_query(self, text: str) -> list[float]:
        lowered = text.casefold()
        return [
            float("concise" in lowered or "简洁" in lowered),
            float("style" in lowered or "answer" in lowered),
            1.0,
        ]


def build_memory_service(tmp_path) -> MemoryService:
    return MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"), vector_store=None)


def memory_record_factory(memory_id: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        user_id="user-a",
        scope="user",
        type="fact",
        key="stored_fact",
        content="Stored fact.",
        value_json=None,
        source="explicit",
        confidence=0.95,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def build_qdrant_memory_vector_store() -> MemoryVectorStore:
    vector_store = MemoryVectorStore.__new__(MemoryVectorStore)
    vector_store.user_id = "user-a"
    vector_store.collection_name = "memories"
    vector_store.vector_store = QdrantClient(":memory:")
    vector_store.embedding_model = FakeMemoryEmbeddingModel()
    vector_store._collection_lock = Lock()
    vector_store._validated_vector_size = None
    return vector_store


def test_memory_store_creates_lean_schema(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    MemoryStore(db_path).close()

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        memory_columns = {row[1] for row in connection.execute("PRAGMA table_info(memories)")}

    assert {"retrieval_logs", "feedback_events", "memories_fts"}.isdisjoint(tables)
    assert {"embedding_id", "last_accessed_at", "access_count"}.isdisjoint(memory_columns)
    assert {"task_id", "expires_at"}.issubset(memory_columns)


def test_memory_store_repairs_duplicate_active_keys_before_unique_index(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    store = MemoryStore(db_path)
    original = store.create_memory(
        user_id="user-a",
        scope="user",
        type="fact",
        key="duplicate",
        content="Older value.",
        value_json=None,
        source="explicit",
        confidence=0.9,
    )
    store.close()

    newer_at = (original.updated_at + timedelta(seconds=1)).isoformat()
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP INDEX idx_memories_active_key")
        connection.execute(
            """
            INSERT INTO memories (
                memory_id, user_id, scope, type, key, content, value_json,
                source, confidence, created_at, updated_at, expires_at, visibility,
                permissions_json, supersedes_id, status, source_message_id,
                conversation_id, task_id
            )
            SELECT 'newer', user_id, scope, type, key, 'Newer value.', value_json,
                   source, confidence, created_at, ?, expires_at, visibility,
                   permissions_json, supersedes_id, status, source_message_id,
                   conversation_id, task_id
            FROM memories WHERE memory_id = ?
            """,
            (newer_at, original.memory_id),
        )

    repaired = MemoryStore(db_path)
    active = repaired.list_memories("user-a")

    assert [memory.memory_id for memory in active] == ["newer"]
    assert repaired.get_memory("user-a", original.memory_id).status == "stale"


def test_explicit_memory_write_only(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("user-a", "conversation-a")
    message_id = service.record_user_message(
        user_id="user-a",
        conversation_id=conversation_id,
        content="hello",
    )

    assert (
        service.write_memories_from_user_message(
            user_id="user-a",
            conversation_id=conversation_id,
            message_id=message_id,
            content="hello",
        )
        == []
    )

    explicit_id = service.record_user_message(
        user_id="user-a",
        conversation_id=conversation_id,
        content="请记住：以后回答用中文并保持简洁",
    )
    created = service.write_memories_from_user_message(
        user_id="user-a",
        conversation_id=conversation_id,
        message_id=explicit_id,
        content="请记住：以后回答用中文并保持简洁",
    )

    assert len(created) == 1
    assert created[0].type == "preference"
    assert created[0].key == "answer_style"
    assert created[0].source == "explicit"


def test_memory_policy_splits_multiple_explicit_intents() -> None:
    intents = extract_memory_write_intents(
        "请记住：以后回答用中文并保持简洁，并且我的职位是法务总监"
    )

    assert [intent.key for intent in intents] == ["answer_style", "business_context"]
    assert all(intent.source == "explicit" for intent in intents)


def test_like_and_dislike_statements_are_exact_profile_preferences() -> None:
    liked = extract_memory_write_intents("我喜欢用项目符号")
    disliked = extract_memory_write_intents("我不喜欢用项目符号")

    assert liked[0].type == disliked[0].type == "preference"
    assert liked[0].key == disliked[0].key
    assert liked[0].value_json["polarity"] == "like"
    assert disliked[0].value_json["polarity"] == "dislike"


def test_implicit_memory_is_not_written(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("user-a", "conversation-a")
    message_id = service.record_user_message(
        user_id="user-a",
        conversation_id=conversation_id,
        content="Our company mainly provides IP agency services.",
    )

    assert (
        service.write_memories_from_user_message(
            user_id="user-a",
            conversation_id=conversation_id,
            message_id=message_id,
            content="Our company mainly provides IP agency services.",
        )
        == []
    )


def test_memory_supersedes_existing_active_key(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    first = service.create_memory(
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer detailed answers.",
    )
    second = service.create_memory(
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer concise answers.",
    )

    assert second.supersedes_id == first.memory_id
    assert [memory.memory_id for memory in service.list_memories("user-a")] == [second.memory_id]


def test_duplicate_memory_write_reuses_existing_active_memory(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    first = service.create_memory(
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer concise answers.",
        value_json={"text": "Prefer concise answers."},
    )
    second = service.create_memory(
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer concise answers.",
        value_json={"text": "Prefer concise answers."},
    )

    assert second.memory_id == first.memory_id
    assert len(service.list_memories("user-a", status=None, include_expired=True)) == 1


def test_concurrent_memory_creates_leave_one_active_key(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    lookup_barrier = Barrier(2)
    start = Barrier(2)
    services = [
        MemoryService(
            store=CoordinatedMemoryStore(db_path, lookup_barrier=lookup_barrier),
            vector_store=None,
        )
        for _ in range(2)
    ]
    errors: list[BaseException] = []

    def create(service: MemoryService, content: str) -> None:
        try:
            start.wait(timeout=5)
            service.create_memory(
                user_id="user-a",
                scope="user",
                type="preference",
                key="answer_style",
                content=content,
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        Thread(target=create, args=(services[0], "Prefer concise answers.")),
        Thread(target=create, args=(services[1], "Prefer detailed answers.")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    memories = MemoryStore(db_path).list_memories("user-a", status=None, include_expired=True)
    assert sum(memory.status == "active" for memory in memories) == 1


def test_concurrent_memory_updates_preserve_distinct_fields(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    memory = MemoryService(store=MemoryStore(db_path), vector_store=None).create_memory(
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Original answer style.",
        confidence=0.5,
    )
    read_barrier = Barrier(2)
    start = Barrier(2)
    stores = [
        CoordinatedMemoryStore(db_path, read_barrier=read_barrier),
        CoordinatedMemoryStore(db_path, read_barrier=read_barrier),
    ]
    errors: list[BaseException] = []

    def update(store: MemoryStore, patch: MemoryUpdate) -> None:
        try:
            start.wait(timeout=5)
            store.update_memory("user-a", memory.memory_id, patch)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        Thread(target=update, args=(stores[0], MemoryUpdate(content="Updated answer style."))),
        Thread(target=update, args=(stores[1], MemoryUpdate(confidence=0.95))),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    updated = MemoryStore(db_path).get_memory("user-a", memory.memory_id)
    assert updated is not None
    assert updated.content == "Updated answer style."
    assert updated.confidence == 0.95


def test_memory_retrieval_uses_vector_ranking(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    first = service.create_memory(
        user_id="user-a",
        scope="user",
        type="fact",
        key="patent_license_focus",
        content="Patent license agreements often need indemnity review.",
    )
    second = service.create_memory(
        user_id="user-a",
        scope="user",
        type="fact",
        key="billing_preference",
        content="Invoices are reviewed by finance.",
    )
    service.vector_store = FakeMemoryVectorStore(
        [(second.memory_id, 0.99), (first.memory_id, 0.95)]
    )

    results = service.retrieve_relevant_memories(
        user_id="user-a",
        query="patent license indemnity",
        limit=3,
    )

    assert [candidate.memory.memory_id for candidate in results] == [
        second.memory_id,
        first.memory_id,
    ]
    assert results[0].retrieval_source == "vector"


def test_profile_preferences_are_loaded_from_sqlite_without_vector_search(tmp_path) -> None:
    vector_store = FakeMemoryVectorStore()
    service = MemoryService(
        store=MemoryStore(tmp_path / "memory.sqlite3"),
        vector_store=vector_store,  # type: ignore[arg-type]
    )
    memory = service.create_memory(
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer concise answers.",
    )

    results = service.retrieve_relevant_memories(user_id="user-a", query="payment terms")

    assert [candidate.memory.memory_id for candidate in results] == [memory.memory_id]
    assert results[0].retrieval_source == "profile"
    assert vector_store.upserted == []


def test_vector_hydration_filters_stale_results(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    stale = service.create_memory(
        user_id="user-a",
        scope="user",
        type="fact",
        key="old_context",
        content="Old stale context.",
    )
    service.delete_memory("user-a", stale.memory_id)
    service.vector_store = FakeMemoryVectorStore([(stale.memory_id, 0.99)])

    assert (
        service.retrieve_relevant_memories(
            user_id="user-a",
            query="arbitration Singapore",
        )
        == []
    )


def test_format_for_prompt_keeps_high_priority_memory(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    memory = service.create_memory(
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer concise Chinese answers.",
        confidence=0.95,
    )

    prompt = service.format_for_prompt([MemoryCandidate(memory=memory, score=0.8)])

    assert "相关的用户记忆" in prompt
    assert "Prefer concise Chinese answers." in prompt


def test_expired_memories_are_marked_stale_and_removed_from_vector(tmp_path) -> None:
    vector_store = FakeMemoryVectorStore()
    service = MemoryService(
        store=MemoryStore(tmp_path / "memory.sqlite3"),
        vector_store=vector_store,  # type: ignore[arg-type]
    )
    expired = service.create_memory(
        user_id="user-a",
        scope="session",
        type="fact",
        key="temporary_fact",
        content="Temporary fact.",
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )

    stale = service.cleanup_expired_memories("user-a")

    assert [memory.memory_id for memory in stale] == [expired.memory_id]
    assert vector_store.deleted[-1] == expired.memory_id


def test_memory_update_can_clear_nullable_fields(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    memory = service.create_memory(
        user_id="user-a",
        scope="user",
        type="fact",
        key="matter_deadline",
        content="Deadline is Friday.",
        value_json={"date": "Friday"},
        expires_at=datetime.now(timezone.utc) + timedelta(days=1),
    )

    cleared = service.update_memory(
        "user-a",
        memory.memory_id,
        MemoryUpdate(value_json=None, expires_at=None),
    )

    assert cleared is not None
    assert cleared.value_json is None
    assert cleared.expires_at is None


def test_load_conversation_history_uses_summary_plus_recent_messages(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("user-a", "conversation-a")
    service.create_memory(
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
            user_id="user-a",
            conversation_id=conversation_id,
            content=f"Question {index}",
        )

    history = service.load_conversation_history("user-a", conversation_id, limit=20)

    assert history[0]["role"] == "system"
    assert [message["content"] for message in history[1:]] == [
        f"Question {index}" for index in range(4, 12)
    ]


def test_manual_conversation_summary_creates_session_memory(tmp_path) -> None:
    service = build_memory_service(tmp_path)
    conversation_id = service.ensure_context("user-a", "conversation-a")
    service.record_user_message(
        user_id="user-a",
        conversation_id=conversation_id,
        content="Please review the notice clause.",
    )
    service.record_assistant_message(
        user_id="user-a",
        conversation_id=conversation_id,
        content="The notice clause requires 10 business days of prior notice.",
    )

    memory = service.summarize_conversation_to_memory(
        user_id="user-a",
        conversation_id=conversation_id,
    )

    assert memory is not None
    assert memory.key == "conversation_summary_conversation-a"
    assert "Please review the notice clause" in memory.content
    assert "10 business days" in memory.content


def test_conversation_summary_is_automatically_indexed_every_ten_messages(tmp_path) -> None:
    vector_store = FakeMemoryVectorStore()
    service = MemoryService(
        store=MemoryStore(tmp_path / "memory.sqlite3"),
        vector_store=vector_store,  # type: ignore[arg-type]
    )
    conversation_id = service.ensure_context("user-a", "conversation-a")
    for index in range(5):
        service.record_user_message(
            user_id="user-a",
            conversation_id=conversation_id,
            content=f"Question {index}",
        )
        service.record_assistant_message(
            user_id="user-a",
            conversation_id=conversation_id,
            content=f"Answer {index}",
        )

    summary = next(
        memory
        for memory in service.list_memories("user-a")
        if memory.key == "conversation_summary_conversation-a"
    )
    assert summary.value_json["message_count"] == 10
    assert summary.memory_id in vector_store.upserted


def test_vector_repair_deletes_inactive_and_upserts_active_memories(tmp_path) -> None:
    vector_store = FakeMemoryVectorStore()
    service = MemoryService(
        store=MemoryStore(tmp_path / "memory.sqlite3"),
        vector_store=vector_store,  # type: ignore[arg-type]
    )
    active = service.create_memory(
        user_id="user-a",
        scope="user",
        type="fact",
        key="active_fact",
        content="Active fact.",
    )
    deleted = service.create_memory(
        user_id="user-a",
        scope="user",
        type="fact",
        key="deleted_fact",
        content="Deleted fact.",
    )
    service.delete_memory("user-a", deleted.memory_id)

    result = service.repair_vector_index("user-a")

    assert deleted.memory_id in vector_store.deleted
    assert active.memory_id in vector_store.upserted
    assert result["deleted"] >= 1
    assert result["upserted"] >= 1


def test_document_qa_separates_memory_from_retrieved_documents(tmp_path) -> None:
    memory_service = build_memory_service(tmp_path)
    memory = memory_service.create_memory(
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer concise Chinese answers.",
    )
    memory_service.vector_store = FakeMemoryVectorStore([(memory.memory_id, 0.95)])
    chat_model = RecordingChatModel()
    service = DocumentQAService(
        vector_store=SingleDocumentVectorStore(),
        chat_model=chat_model,
        memory_service=memory_service,
        user_id="user-a",
    )

    answer = service.ask(
        "What is the notice period?",
        user_id="user-a",
        conversation_id="conversation-a",
    )

    assert answer.citations
    assert answer.memories_used[0].key == "answer_style"
    assert "<user_memory>" in chat_model.messages[1]["content"]
    assert "<retrieved_documents>" in chat_model.messages[1]["content"]


def test_document_qa_merges_persisted_history_when_client_history_is_missing(tmp_path) -> None:
    memory_service = build_memory_service(tmp_path)
    conversation_id = memory_service.ensure_context("user-a", "conversation-a")
    memory_service.record_user_message(
        user_id="user-a",
        conversation_id=conversation_id,
        content="Earlier question about renewal.",
    )
    memory_service.record_assistant_message(
        user_id="user-a",
        conversation_id=conversation_id,
        content="Earlier answer about renewal.",
    )
    chat_model = RecordingChatModel()
    service = DocumentQAService(
        vector_store=EmptyDocumentVectorStore(),
        chat_model=chat_model,
        memory_service=memory_service,
        user_id="user-a",
    )

    service.ask(
        "What should I review next for this contract?",
        user_id="user-a",
        conversation_id=conversation_id,
    )

    prompt = chat_model.messages[1]["content"]
    assert "Earlier question about renewal." in prompt
    assert "Earlier answer about renewal." in prompt


def test_memory_vector_search_filters_by_user() -> None:
    vector_store = build_qdrant_memory_vector_store()
    vector_store.upsert_memory(memory_record_factory("user-a-private"))
    vector_store.upsert_memory(
        replace(
            memory_record_factory("user-b-private"),
            user_id="user-b",
            content="Concise answer style.",
        )
    )

    results = vector_store.search("answer style", user_id="user-a", k=3)

    assert [candidate.memory.memory_id for candidate in results] == ["user-a-private"]
    vector_store.vector_store.close()


def test_memory_vector_search_reconstructs_full_metadata() -> None:
    vector_store = build_qdrant_memory_vector_store()
    vector_store.upsert_memory(
        replace(
            memory_record_factory("memory-a"),
            type="preference",
            key="answer_style",
            content="Prefer concise answers.",
            confidence=0.9,
            value_json={"text": "Prefer concise answers."},
        )
    )

    results = vector_store.search("concise answers", user_id="user-a", k=1)

    assert len(results) == 1
    assert results[0].retrieval_source == "vector"
    assert results[0].memory.value_json == {"text": "Prefer concise answers."}
    vector_store.vector_store.close()


def test_memory_vector_search_discards_other_users_results() -> None:
    vector_store = build_qdrant_memory_vector_store()
    memories = [
        replace(memory_record_factory("own-private"), content="Concise own memory."),
        replace(
            memory_record_factory("other-private"),
            user_id="user-b",
            visibility="private",
            content="Concise private memory.",
        ),
    ]
    for memory in memories:
        vector_store.upsert_memory(memory)

    results = vector_store.search("answer style", user_id="user-a", k=4)

    assert {candidate.memory.memory_id for candidate in results} == {"own-private"}
    vector_store.vector_store.close()


def test_memory_search_initializes_payload_indexes_only_once() -> None:
    barrier = Barrier(2)

    class CoordinatedEmbeddingModel(FakeMemoryEmbeddingModel):
        def embed_query(self, text: str) -> list[float]:
            barrier.wait(timeout=5)
            return super().embed_query(text)

    class FakeClient:
        def __init__(self) -> None:
            self.collection_checks = 0
            self.payload_indexes: list[tuple[str, models.PayloadSchemaType]] = []
            self.queries = 0

        def collection_exists(self, _collection_name: str) -> bool:
            self.collection_checks += 1
            return True

        def get_collection(self, _collection_name: str):
            sleep(0.05)
            vectors = {"dense": SimpleNamespace(size=3)}
            return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))

        def create_payload_index(self, **kwargs) -> None:
            self.payload_indexes.append((kwargs["field_name"], kwargs["field_schema"]))

        def query_points(self, **_kwargs):
            self.queries += 1
            return SimpleNamespace(points=[])

    client = FakeClient()
    vector_store = MemoryVectorStore.__new__(MemoryVectorStore)
    vector_store.user_id = "user-a"
    vector_store.collection_name = "memories"
    vector_store.vector_store = client
    vector_store.embedding_model = CoordinatedEmbeddingModel()
    vector_store._collection_lock = Lock()
    vector_store._validated_vector_size = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                vector_store.search,
                "answer style",
                user_id="user-a",
                k=1,
            )
            for _ in range(2)
        ]
        assert [future.result() for future in futures] == [[], []]

    assert client.payload_indexes == [
        ("user_id", models.PayloadSchemaType.KEYWORD),
        ("status", models.PayloadSchemaType.KEYWORD),
    ]
    assert client.collection_checks == 1
    assert client.queries == 2


def test_memory_vector_delete_failure_is_reported(caplog) -> None:
    class FailingClient:
        def collection_exists(self, _collection_name: str) -> bool:
            raise RuntimeError("Qdrant unavailable")

    vector_store = MemoryVectorStore.__new__(MemoryVectorStore)
    vector_store.collection_name = "memories"
    vector_store.vector_store = FailingClient()

    with caplog.at_level("WARNING"):
        deleted = vector_store.delete_memory("memory-a")

    assert deleted is False
    assert "Memory vector delete failed" in caplog.text

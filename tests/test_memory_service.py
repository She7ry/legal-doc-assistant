from __future__ import annotations

from datetime import datetime, timedelta, timezone

from langchain_core.documents import Document

from doc_assistant.memory.policy import extract_memory_write_intents
from doc_assistant.memory.schemas import MemoryUpdate
from doc_assistant.memory.service import MemoryService
from doc_assistant.memory.store import MemoryStore
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


def build_memory_service(tmp_path) -> MemoryService:
    return MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"), vector_store=None)


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
    assert "legal document analysis assistant" in chat_model.messages[0]["content"].casefold()
    assert chat_model.messages[1]["role"] == "user"
    assert "<user_memory>" in chat_model.messages[1]["content"]
    assert "Prefer concise Chinese answers." in chat_model.messages[1]["content"]
    assert "<retrieved_documents>" in chat_model.messages[1]["content"]
    assert "Section 4 says notices must be sent" in chat_model.messages[1]["content"]

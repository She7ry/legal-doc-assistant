from __future__ import annotations

from langchain_core.documents import Document

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

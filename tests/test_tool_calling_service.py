from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from doc_assistant.memory.schemas import MemoryCandidate
from doc_assistant.memory.service import MemoryService
from doc_assistant.memory.store import MemoryStore
from doc_assistant.services.qa_service import DocumentQAService
from doc_assistant.services.tool_calling_service import ToolCallingChatService
from doc_assistant.tools.web_search import WebSearchResult


class SingleDocumentVectorStore:
    tenant_id = "default"

    def __init__(self) -> None:
        self.queries: list[tuple[str, int | None]] = []

    def search(self, query: str, k: int | None = None) -> list[Document]:
        self.queries.append((query, k))
        return [
            Document(
                page_content="Payment is due within 30 days after invoice approval.",
                metadata={"file_name": "supply-contract.pdf", "page": 2, "chunk_id": 7},
            )
        ]


class DuplicateDocumentVectorStore:
    tenant_id = "default"

    def search(self, query: str, k: int | None = None) -> list[Document]:
        document = Document(
            page_content="Payment is due within 30 days after invoice approval.",
            metadata={"file_name": "supply-contract.pdf", "page": 2, "chunk_id": 7},
        )
        return [document, document]


class DocumentToolModel(BaseChatModel):
    calls: int = 0
    messages: list[list] = Field(default_factory=list)
    bound_tool_names: list[str] = Field(default_factory=list)
    tool_name: str = "search_documents"
    tool_args: dict = Field(default_factory=lambda: {"query": "payment terms", "top_k": 2})
    final_content: str = "Payment must be made within 30 days [D1]."

    @property
    def _llm_type(self) -> str:
        return "tool-test"

    def bind_tools(self, tools, **kwargs):
        del kwargs
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del stop, run_manager, kwargs
        self.calls += 1
        self.messages.append(messages)
        if self.calls == 1:
            assert self.tool_name in self.bound_tool_names
            message = AIMessage(
                content="",
                tool_calls=[{"name": self.tool_name, "args": self.tool_args, "id": "call_tool"}],
            )
        else:
            assert any(isinstance(message, ToolMessage) for message in messages)
            message = AIMessage(content=self.final_content)
        return ChatResult(generations=[ChatGeneration(message=message)])


class MemoryAwareToolModel(DocumentToolModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.calls == 0:
            assert "Prefer concise answers." in str(messages[0].content)
        return super()._generate(messages, stop, run_manager, **kwargs)


class EmptyVectorStore:
    tenant_id = "default"

    def search(self, query: str, k: int | None = None) -> list[Document]:
        return []


class WebToolModel(DocumentToolModel):
    tool_name: str = "web_search"
    tool_args: dict = Field(
        default_factory=lambda: {"query": "supplier recent news", "max_results": 1}
    )
    final_content: str = "Recent public reporting should be treated as background [W1]."


class FakeWebSearchClient:
    def search(self, query: str, *, max_results: int, recency_days=None, domains=None):
        assert query == "supplier recent news"
        assert max_results == 1
        return [
            WebSearchResult(
                title="Supplier announces restructuring",
                url="https://news.example/supplier",
                snippet="The supplier announced a restructuring plan.",
                published_at="2026-06-01",
                source="news.example",
            )
        ]


def test_tool_calling_service_executes_search_documents_tool() -> None:
    vector_store = SingleDocumentVectorStore()
    model = DocumentToolModel()
    qa_service = DocumentQAService(vector_store=vector_store, chat_model=model)
    service = ToolCallingChatService(qa_service)

    answer = service.ask("What are the payment terms?")

    assert answer.content == "Payment must be made within 30 days [D1]."
    assert vector_store.queries == [("payment terms", 2)]
    assert answer.citations[0].source_id == "D1"
    assert answer.citations[0].file_name == "supply-contract.pdf"
    assert answer.tool_calls[0].name == "search_documents"
    assert answer.tool_calls[0].tool_call_id == "call_tool"
    assert answer.tool_calls[0].result["result_count"] == 1


def test_tool_calling_service_reuses_duplicate_document_source_ids() -> None:
    model = DocumentToolModel()
    qa_service = DocumentQAService(vector_store=DuplicateDocumentVectorStore(), chat_model=model)
    service = ToolCallingChatService(qa_service)

    answer = service.ask("What are the payment terms?")

    assert [citation.source_id for citation in answer.citations] == ["D1"]
    assert [item["source_id"] for item in answer.tool_calls[0].result["results"]] == ["D1", "D1"]


def test_tool_calling_service_executes_web_search_when_enabled() -> None:
    model = WebToolModel()
    qa_service = DocumentQAService(vector_store=EmptyVectorStore(), chat_model=model)
    service = ToolCallingChatService(qa_service, web_search_client=FakeWebSearchClient())

    answer = service.ask("Check recent supplier news.", enable_web_search=True)

    assert answer.content == "Recent public reporting should be treated as background [W1]."
    assert answer.web_sources[0].source_id == "W1"
    assert answer.web_sources[0].url == "https://news.example/supplier"
    assert answer.tool_calls[0].name == "web_search"


def test_tool_calling_service_uses_memory_context(tmp_path) -> None:
    memory_service = MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"), vector_store=None)
    memory = memory_service.create_memory(
        tenant_id="default",
        user_id="user-a",
        scope="user",
        type="preference",
        key="answer_style",
        content="Prefer concise answers.",
    )

    class StaticMemoryVectorStore:
        def search(self, query: str, *, tenant_id: str, user_id: str, k: int | None = None):
            del query, tenant_id, user_id, k
            return [MemoryCandidate(memory=memory, score=0.95, retrieval_source="vector")]

        def upsert_memory(self, candidate) -> str:
            return candidate.memory_id

        def delete_memory(self, memory_id: str) -> None:
            del memory_id

    memory_service.vector_store = StaticMemoryVectorStore()  # type: ignore[assignment]
    model = MemoryAwareToolModel()
    qa_service = DocumentQAService(
        vector_store=SingleDocumentVectorStore(),
        chat_model=model,
        memory_service=memory_service,
        tenant_id="default",
    )
    service = ToolCallingChatService(qa_service)

    answer = service.ask(
        "Please give concise payment terms.",
        user_id="user-a",
        conversation_id="conversation-a",
    )

    assert answer.memories_used[0].key == "answer_style"
    history = memory_service.load_conversation_history(
        "default",
        "user-a",
        "conversation-a",
        limit=5,
    )
    assert [message["role"] for message in history] == ["user", "assistant"]


def test_tool_calling_service_does_not_auto_summarize_conversation(tmp_path) -> None:
    memory_service = MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"), vector_store=None)
    model = DocumentToolModel()
    qa_service = DocumentQAService(
        vector_store=SingleDocumentVectorStore(),
        chat_model=model,
        memory_service=memory_service,
        tenant_id="default",
    )
    service = ToolCallingChatService(qa_service)

    service.ask(
        "What are the payment terms?",
        user_id="user-a",
        conversation_id="conversation-a",
    )

    memories = memory_service.list_memories("default", "user-a")

    assert all(memory.key != "conversation_summary_conversation-a" for memory in memories)

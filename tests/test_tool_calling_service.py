from __future__ import annotations

import json

from langchain_core.documents import Document

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


class DocumentToolModel:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[list[dict]] = []

    def invoke_messages(self, messages, tools=None, tool_choice=None):
        self.calls += 1
        self.messages.append(messages)
        if self.calls == 1:
            assert tools
            assert tools[0]["function"]["name"] == "search_documents"
            return {
                "tool_calls": [
                    {
                        "id": "call_docs",
                        "type": "function",
                        "function": {
                            "name": "search_documents",
                            "arguments": json.dumps({"query": "payment terms", "top_k": 2}),
                        },
                    }
                ]
            }
        assert any(message["role"] == "tool" and "D1" in message["content"] for message in messages)
        return {"content": "Payment must be made within 30 days [D1]."}


class EmptyVectorStore:
    tenant_id = "default"

    def search(self, query: str, k: int | None = None) -> list[Document]:
        return []


class WebToolModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke_messages(self, messages, tools=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            assert any(tool["function"]["name"] == "web_search" for tool in tools)
            return {
                "tool_calls": [
                    {
                        "id": "call_web",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps(
                                {"query": "supplier recent news", "max_results": 1}
                            ),
                        },
                    }
                ]
            }
        assert any(message["role"] == "tool" and "W1" in message["content"] for message in messages)
        return {"content": "Recent public reporting should be treated as background [W1]."}


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
    assert answer.tool_calls[0].result["result_count"] == 1


def test_tool_calling_service_executes_web_search_when_enabled() -> None:
    model = WebToolModel()
    qa_service = DocumentQAService(vector_store=EmptyVectorStore(), chat_model=model)
    service = ToolCallingChatService(qa_service, web_search_client=FakeWebSearchClient())

    answer = service.ask("Check recent supplier news.", enable_web_search=True)

    assert answer.content == "Recent public reporting should be treated as background [W1]."
    assert answer.web_sources[0].source_id == "W1"
    assert answer.web_sources[0].url == "https://news.example/supplier"
    assert answer.tool_calls[0].name == "web_search"

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from doc_assistant.tools.document_search import (
    DocumentSearchTool,
    SearchDocumentsInput,
)


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, k: int | None = None) -> list[Document]:
        self.calls.append((query, k))
        return [
            Document(
                page_content="  Payment is due\nwithin 30 days.  ",
                metadata={"file_name": "contract.pdf", "page": 1, "chunk_id": 3},
            )
        ]


def test_document_search_tool_executes_and_normalizes_results() -> None:
    backend = RecordingBackend()
    tool = DocumentSearchTool(backend, default_top_k=4)

    arguments = SearchDocumentsInput(query=" payment terms ", top_k=10)
    execution = tool.execute(arguments.query, arguments.top_k)

    assert backend.calls == [("payment terms", 10)]
    assert execution.query == "payment terms"
    assert execution.hits[0].result == {
        "file_name": "contract.pdf",
        "file_id": None,
        "document_key": None,
        "document_version": None,
        "page": 1,
        "page_label": "page 2",
        "chunk_id": 3,
        "section_heading": None,
        "retrieval_score": None,
        "retrieval_relevance": None,
        "content": "Payment is due within 30 days.",
    }


def test_document_search_tool_rejects_an_empty_query() -> None:
    with pytest.raises(ValueError, match="query is required"):
        SearchDocumentsInput(query="  ")

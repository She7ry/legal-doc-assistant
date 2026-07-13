from __future__ import annotations

import pytest
from langchain_core.documents import Document

from doc_assistant.tools.document_search import (
    SearchDocumentsInput,
    document_search_result,
)


def test_document_search_result_normalizes_document() -> None:
    result = document_search_result(
        Document(
            page_content="  Payment is due\nwithin 30 days.  ",
            metadata={"file_name": "contract.pdf", "page": 1, "chunk_id": 3},
        )
    )

    assert result == {
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

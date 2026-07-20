from langchain_core.documents import Document

from ai.rag.retrieval.document_identity import document_identity
from ai.utils.json import parse_json_object
from ai.utils.text import (
    as_text_list,
    compact_text,
    dedupe_texts,
    optional_text,
)


def test_text_helpers_normalize_and_dedupe_scalars() -> None:
    assert compact_text("  alpha\n beta  ") == "alpha beta"
    assert compact_text(3) == "3"
    assert compact_text({"unsupported": "value"}) == ""
    assert optional_text("  alpha  ") == "alpha"
    assert optional_text([]) is None
    assert as_text_list("  alpha\n beta  ") == ["alpha beta"]
    assert as_text_list([" Alpha ", "", 2, None]) == ["Alpha", "2"]
    assert dedupe_texts([" Alpha ", "alpha", " beta\nvalue "]) == [
        "Alpha",
        "beta value",
    ]


def test_json_helpers_parse_persisted_and_llm_output() -> None:
    assert parse_json_object('{"status": "ok"}') == {"status": "ok"}
    assert parse_json_object("[]") is None
    assert parse_json_object("not json") is None


def test_document_identity_uses_canonical_chunk_metadata() -> None:
    first = Document(
        page_content="first rendering",
        metadata={"file_id": "f1", "document_version": 2, "page": 0, "chunk_id": 3},
    )
    same_chunk = Document(
        page_content="second rendering",
        metadata={"file_id": "f1", "document_version": 2, "page": 0, "chunk_id": 3},
    )
    next_page = Document(
        page_content="first rendering",
        metadata={"file_id": "f1", "document_version": 2, "page": 1, "chunk_id": 3},
    )

    assert document_identity(first) == document_identity(same_chunk)
    assert document_identity(first) != document_identity(next_page)


def test_document_identity_falls_back_to_record_id_or_content() -> None:
    document = Document(page_content="content", metadata={})

    assert document_identity(document, fallback_id="record-1") == document_identity(
        Document(page_content="changed", metadata={}), fallback_id="record-1"
    )
    assert document_identity(document) != document_identity(
        Document(page_content="changed", metadata={})
    )

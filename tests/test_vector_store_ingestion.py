from __future__ import annotations

from langchain_core.documents import Document

from doc_assistant.retrieval.vector_store import _split_legal_sections, document_key_for_file_name


def test_document_key_is_stable_for_case_and_whitespace() -> None:
    assert document_key_for_file_name(" Contract.DOCX ") == document_key_for_file_name("contract.docx")


def test_split_legal_sections_preserves_headings_in_metadata() -> None:
    document = Document(
        page_content="Intro text\n\nSection 1 Term\nThe term is one year.\n\n2. Payment\nPay monthly.",
        metadata={"file_name": "contract.txt"},
    )

    sections = _split_legal_sections([document])

    headings = [section.metadata.get("section_heading") for section in sections]
    assert "Section 1 Term" in headings
    assert "2. Payment" in headings

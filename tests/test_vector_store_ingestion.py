from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document

from doc_assistant.retrieval import vector_store as vector_store_module
from doc_assistant.retrieval.vector_store import (
    DocumentVectorStore,
    _bm25_rank,
    _chunk_text_with_heading,
    _split_legal_sections,
    document_key_for_file_name,
)


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


def test_chunk_text_with_heading_preserves_section_context() -> None:
    assert _chunk_text_with_heading("Payment must be made monthly.", "2. Payment") == (
        "2. Payment\nPayment must be made monthly."
    )
    assert _chunk_text_with_heading("2. Payment\nPayment must be made monthly.", "2. Payment") == (
        "2. Payment\nPayment must be made monthly."
    )


def test_bm25_rank_matches_exact_legal_terms() -> None:
    records = [
        {
            "id": "payment",
            "metadata": {"file_name": "contract.pdf", "chunk_id": 1},
            "document": "Invoices are payable within 30 calendar days.",
        },
        {
            "id": "damages",
            "metadata": {"file_name": "contract.pdf", "chunk_id": 2},
            "document": "Liquidated damages are capped at 10% of the delayed shipment value.",
        },
    ]

    ranked = _bm25_rank("liquidated damages 10% cap", records, k=2)

    assert ranked[0][2] == "damages"


class FakeChroma:
    def __init__(self) -> None:
        self.vector_filter = None
        self.record_filter = None

    def similarity_search_with_relevance_scores(self, query, k, filter=None):
        self.vector_filter = filter
        return [
            (
                Document(
                    page_content="Invoices are payable within 30 calendar days.",
                    metadata={
                        "active": True,
                        "document_key": "contract",
                        "file_id": "file-a",
                        "document_version": 1,
                        "chunk_id": 1,
                    },
                ),
                0.8,
            )
        ]

    def get(self, where=None, include=None):
        self.record_filter = where
        return {
            "ids": ["damages"],
            "metadatas": [
                {
                    "active": True,
                    "document_key": "contract",
                    "file_id": "file-a",
                    "document_version": 1,
                    "chunk_id": 2,
                }
            ],
            "documents": [
                "Liquidated damages are capped at 10% of the delayed shipment value."
            ],
        }


def test_hybrid_search_pushes_active_filter_and_fuses_bm25(monkeypatch) -> None:
    fake_chroma = FakeChroma()
    store = object.__new__(DocumentVectorStore)
    store.vector_store = fake_chroma

    monkeypatch.setattr(
        vector_store_module,
        "settings",
        SimpleNamespace(
            top_k=1,
            retrieval_mode="hybrid",
            retrieval_fetch_k=10,
            retrieval_min_relevance=0.0,
            retrieval_rrf_k=60,
            retrieval_dense_weight=1.0,
            retrieval_bm25_weight=1.0,
            retrieval_mmr_lambda=1.0,
        ),
    )

    results = store.search("liquidated damages 10% cap", k=1)

    assert fake_chroma.vector_filter == {"active": True}
    assert fake_chroma.record_filter == {"active": True}
    assert "Liquidated damages" in results[0].page_content
    assert results[0].metadata["bm25_rank"] == 1
    assert results[0].metadata["rerank_score"] > 0
    assert results[0].metadata["retrieval_score"] > 0

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document

from doc_assistant.retrieval import vector_store as vector_store_module
from doc_assistant.retrieval.bm25_index import BM25Document, PersistentBM25Index
from doc_assistant.retrieval.vector_store import (
    DocumentVectorStore,
    _bm25_rank,
    _chunk_text_with_heading,
    _split_legal_sections,
    _tokenize_for_search,
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


def test_persistent_bm25_index_searches_without_full_chroma_scan(tmp_path) -> None:
    index = PersistentBM25Index(tmp_path / "bm25.sqlite3")
    index.add_document(
        BM25Document(
            doc_id="damages",
            tokens=_tokenize_for_search(
                "contract.pdf Liquidated damages are capped at 10% of shipment value."
            ),
            document="Liquidated damages are capped at 10% of shipment value.",
            metadata={
                "active": True,
                "document_key": "contract",
                "file_id": "file-a",
                "document_version": 1,
                "chunk_id": 2,
            },
        )
    )

    class ExplodingChroma:
        def get(self, **_kwargs):
            raise AssertionError("BM25 search should not scan Chroma records.")

    store = object.__new__(DocumentVectorStore)
    store.vector_store = ExplodingChroma()
    store._bm25_index = index
    store._bm25_rebuild_attempted = True
    store.tenant_id = "default"

    ranked = store._bm25_candidates("liquidated damages cap", fetch_k=1)

    assert ranked[0][2] == "damages"
    assert "Liquidated damages" in ranked[0][0].page_content


def test_batch_embed_and_add_preserves_chunk_order(monkeypatch) -> None:
    class FakeEmbeddingFunction:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return [[float(text.rsplit(" ", 1)[-1])] for text in texts]

    class FakeCollection:
        def __init__(self) -> None:
            self.payload = None

        def add(self, **payload) -> None:
            self.payload = payload

    class FakeVectorStore:
        def __init__(self) -> None:
            self._embedding_function = FakeEmbeddingFunction()
            self._collection = FakeCollection()

        def add_documents(self, *_args, **_kwargs) -> None:
            raise AssertionError("Manual batch embedding should bypass add_documents.")

    fake_vector_store = FakeVectorStore()
    store = object.__new__(DocumentVectorStore)
    store.vector_store = fake_vector_store
    monkeypatch.setattr(
        vector_store_module,
        "settings",
        SimpleNamespace(embedding_batch_size=2, embedding_max_workers=2),
    )

    chunks = [
        Document(page_content=f"chunk {index}", metadata={"chunk_id": index})
        for index in range(4)
    ]
    ids = [f"doc-{index}" for index in range(4)]

    store._batch_embed_and_add(chunks, ids)

    assert fake_vector_store._collection.payload["ids"] == ids
    assert fake_vector_store._collection.payload["documents"] == [
        "chunk 0",
        "chunk 1",
        "chunk 2",
        "chunk 3",
    ]
    assert fake_vector_store._collection.payload["embeddings"] == [
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ]


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


def test_search_cache_avoids_repeating_dense_and_bm25_work(monkeypatch) -> None:
    fake_chroma = FakeChroma()
    store = object.__new__(DocumentVectorStore)
    store.vector_store = fake_chroma
    store.tenant_id = "default"
    store._query_cache = vector_store_module._QueryCache(ttl_seconds=300, max_size=8)
    store._bm25_index = None
    calls = {"dense": 0, "bm25": 0}

    def dense(query: str, *, fetch_k: int):
        calls["dense"] += 1
        return [
            (
                Document(
                    page_content="Cached dense result.",
                    metadata={
                        "active": True,
                        "document_key": "contract",
                        "file_id": "file-a",
                        "document_version": 1,
                        "chunk_id": 1,
                    },
                ),
                0.9,
            )
        ]

    def bm25(query: str, *, fetch_k: int):
        calls["bm25"] += 1
        return []

    monkeypatch.setattr(store, "_dense_candidates", dense)
    monkeypatch.setattr(store, "_bm25_candidates", bm25)
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
            retrieval_rerank_mode="none",
            retrieval_rerank_weight=0,
        ),
    )

    first = store.search("payment terms", k=1)
    second = store.search("payment terms", k=1)

    assert first[0].page_content == second[0].page_content
    assert calls == {"dense": 1, "bm25": 1}

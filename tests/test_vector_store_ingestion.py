from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from doc_assistant.retrieval import _ingestion as ingestion_module
from doc_assistant.retrieval import _qdrant_retriever as retriever_module
from doc_assistant.retrieval._chunking import (
    chunk_text_with_heading,
    split_legal_sections,
)
from doc_assistant.retrieval._ingestion import DocumentIngester, document_key_for_file_name
from doc_assistant.retrieval._qdrant_repository import QdrantDocumentRepository
from doc_assistant.retrieval._qdrant_retriever import QdrantRetriever, QueryCache
from doc_assistant.retrieval.vector_store import DocumentVectorStore


class FakeEmbeddingModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        return [
            float("payment" in lowered),
            float("damages" in lowered),
            1.0,
        ]


def test_document_key_is_stable_for_case_and_whitespace() -> None:
    assert document_key_for_file_name(" Contract.DOCX ") == document_key_for_file_name(
        "contract.docx"
    )


def test_ingest_file_replaces_active_version_and_invalidates_cache(monkeypatch) -> None:
    class FakeRepository:
        def __init__(self) -> None:
            self.deleted_ids: list[str] = []
            self.embedded: list[tuple[list[Document], list[str]]] = []
            self.deactivated: list[tuple[list[str], list[dict]]] = []

        def records_for_document_key(self, _document_key: str) -> list[dict]:
            return [existing]

        def delete(self, ids: list[str]) -> None:
            self.deleted_ids.extend(ids)

        def embed_and_add(self, chunks, ids, *, batch_size, max_workers) -> None:
            del batch_size, max_workers
            self.embedded.append((chunks, ids))

        def update_metadatas(self, ids, metadatas) -> None:
            self.deactivated.append((ids, metadatas))

    existing = {
        "id": "contract:old:v1:0",
        "metadata": {
            "active": True,
            "document_key": document_key_for_file_name("Contract.pdf"),
            "document_version": 1,
            "file_id": "old-hash",
        },
        "document": "Old text.",
    }
    repository = FakeRepository()
    cache_clear_count = 0

    def clear_cache() -> None:
        nonlocal cache_clear_count
        cache_clear_count += 1

    ingester = DocumentIngester(
        tenant_id="tenant-a",
        repository=repository,  # type: ignore[arg-type]
        splitter=object(),  # type: ignore[arg-type]
        invalidate_cache=clear_cache,
    )
    monkeypatch.setattr(ingestion_module, "file_sha256", lambda _path: "new-hash")
    monkeypatch.setattr(
        ingestion_module,
        "load_documents",
        lambda _path: [Document(page_content="New text.", metadata={"page": 0})],
    )
    monkeypatch.setattr(
        ingestion_module,
        "split_documents_for_ingestion",
        lambda documents, *, splitter: [
            Document(page_content="New chunk.", metadata=dict(documents[0].metadata))
        ],
    )

    result = ingester.ingest_file(Path("Contract.pdf"))

    assert result.document_version == 2
    assert result.skipped is False
    assert repository.embedded[0][0][0].metadata["active"] is True
    assert repository.embedded[0][0][0].metadata["document_version"] == 2
    assert repository.deactivated[0][0] == [existing["id"]]
    assert repository.deactivated[0][1][0]["superseded_by_file_id"] == "new-hash"
    assert cache_clear_count == 1


def test_ingest_rolls_back_qdrant_chunks_when_upsert_fails() -> None:
    class FailingRepository:
        def __init__(self) -> None:
            self.deleted: list[list[str]] = []

        def delete(self, ids: list[str]) -> None:
            self.deleted.append(list(ids))

        def embed_and_add(self, *_args, **_kwargs) -> None:
            raise RuntimeError("Qdrant unavailable")

    repository = FailingRepository()
    ingester = DocumentIngester(
        tenant_id="tenant-a",
        repository=repository,  # type: ignore[arg-type]
        splitter=object(),  # type: ignore[arg-type]
        invalidate_cache=lambda: None,
    )

    with pytest.raises(RuntimeError, match="Qdrant unavailable"):
        ingester._write_chunks(
            [Document(page_content="chunk", metadata={})],
            ["chunk-1"],
            active_records=[],
            file_id="file-a",
            progress_callback=None,
            warnings=[],
        )

    assert repository.deleted == [["chunk-1"], ["chunk-1"]]


def test_split_legal_sections_preserves_headings_in_metadata() -> None:
    document = Document(
        page_content=(
            "Intro text\n\nSection 1 Term\nThe term is one year.\n\n2. Payment\nPay monthly."
        ),
        metadata={"file_name": "contract.txt"},
    )

    sections = split_legal_sections([document])

    headings = [section.metadata.get("section_heading") for section in sections]
    assert "Section 1 Term" in headings
    assert "2. Payment" in headings


def test_chunk_text_with_heading_preserves_section_context() -> None:
    assert chunk_text_with_heading("Payment must be made monthly.", "2. Payment") == (
        "2. Payment\nPayment must be made monthly."
    )
    assert chunk_text_with_heading("2. Payment\nPayment must be made monthly.", "2. Payment") == (
        "2. Payment\nPayment must be made monthly."
    )


def test_qdrant_sparse_bm25_matches_exact_legal_terms() -> None:
    client = QdrantClient(":memory:")
    repository = QdrantDocumentRepository(client, "documents", FakeEmbeddingModel())
    repository.embed_and_add(
        [
            Document(
                page_content="Invoices are payable within 30 calendar days.",
                metadata={"active": True, "file_name": "contract.pdf"},
            ),
            Document(
                page_content="Liquidated damages are capped at 10% of shipment value.",
                metadata={"active": True, "file_name": "contract.pdf"},
            ),
        ],
        ["payment", "damages"],
        batch_size=2,
        max_workers=1,
    )

    results = repository.search("liquidated damages 10% cap", limit=1, mode="bm25")

    assert results[0].page_content.startswith("Liquidated damages")
    assert results[0].metadata["retrieval_mode"] == "bm25"
    client.close()


def test_qdrant_embed_and_add_preserves_chunk_order() -> None:
    class OrderedEmbeddingModel:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(list(texts))
            return [[float(text.rsplit(" ", 1)[-1]), 1.0] for text in texts]

        def embed_query(self, text: str) -> list[float]:
            del text
            return [0.0, 1.0]

    client = QdrantClient(":memory:")
    embedding_model = OrderedEmbeddingModel()
    repository = QdrantDocumentRepository(client, "documents", embedding_model)
    chunks = [
        Document(page_content=f"chunk {index}", metadata={"chunk_id": index, "active": True})
        for index in range(4)
    ]
    ids = [f"doc-{index}" for index in range(4)]

    repository.embed_and_add(chunks, ids, batch_size=2, max_workers=2)
    records = sorted(repository.all_records(), key=lambda record: record["metadata"]["chunk_id"])

    assert [record["id"] for record in records] == ids
    assert [record["document"] for record in records] == [f"chunk {index}" for index in range(4)]
    assert embedding_model.calls == [["chunk 0", "chunk 1"], ["chunk 2", "chunk 3"]]
    client.close()


def test_qdrant_hybrid_search_filters_inactive_and_fuses_dense_sparse() -> None:
    client = QdrantClient(":memory:")
    repository = QdrantDocumentRepository(client, "documents", FakeEmbeddingModel())
    repository.embed_and_add(
        [
            Document(
                page_content="Liquidated damages are capped at 10%.",
                metadata={"active": True, "file_name": "contract.pdf"},
            ),
            Document(
                page_content="Liquidated damages were removed from the old draft.",
                metadata={"active": False, "file_name": "contract-v1.pdf"},
            ),
            Document(
                page_content="Invoices are payable monthly.",
                metadata={"active": True, "file_name": "contract.pdf"},
            ),
        ],
        ["damages", "old-damages", "payment"],
        batch_size=3,
        max_workers=1,
    )

    results = repository.search("liquidated damages cap", limit=3, mode="hybrid")

    assert results[0].page_content.startswith("Liquidated damages are capped")
    assert all("old draft" not in result.page_content for result in results)
    assert results[0].metadata["qdrant_score"] > 0
    client.close()


def test_search_cache_avoids_repeating_qdrant_query(monkeypatch) -> None:
    class FakeRepository:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, query: str, *, limit: int, mode: str) -> list[Document]:
            del query, limit, mode
            self.calls += 1
            return [Document(page_content="Cached result.", metadata={})]

    repository = FakeRepository()
    retriever = QdrantRetriever(
        repository=repository,  # type: ignore[arg-type]
        tenant_id="default",
        cache=QueryCache(ttl_seconds=300, max_size=8),
    )
    monkeypatch.setattr(
        retriever_module,
        "settings",
        SimpleNamespace(
            top_k=1,
            retrieval_mode="hybrid",
            retrieval_fetch_k=10,
            retrieval_min_relevance=0.0,
            retrieval_mmr_lambda=0.85,
        ),
    )

    first = retriever.search("payment terms", k=1)
    second = retriever.search("payment terms", k=1)

    assert first[0].page_content == second[0].page_content
    assert repository.calls == 1


class FakeCatalogRepository:
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def matching_records(
        self,
        *,
        document_key: str | None,
        file_id: str | None,
        include_documents: bool,
    ) -> list[dict]:
        del include_documents
        return [
            record
            for record in self.records
            if (not document_key or record["metadata"].get("document_key") == document_key)
            and (not file_id or record["metadata"].get("file_id") == file_id)
        ]


def test_get_document_text_returns_latest_active_chunks_in_order() -> None:
    records = [
        {
            "id": "v1-0",
            "metadata": {
                "active": False,
                "document_key": "contract",
                "file_id": "old-file",
                "file_name": "contract.pdf",
                "document_version": 1,
                "chunk_id": 0,
            },
            "document": "Old text.",
        },
        {
            "id": "v2-1",
            "metadata": {
                "active": True,
                "document_key": "contract",
                "file_id": "new-file",
                "file_name": "contract.pdf",
                "document_version": 2,
                "file_extension": ".pdf",
                "indexed_at": "2026-06-17T00:00:00+00:00",
                "document_count": 2,
                "page_count": 2,
                "warning_count": 1,
                "page": 1,
                "chunk_id": 1,
                "section_heading": "2. Payment",
            },
            "document": "Payment is due in 30 days.",
        },
        {
            "id": "v2-0",
            "metadata": {
                "active": True,
                "document_key": "contract",
                "file_id": "new-file",
                "file_name": "contract.pdf",
                "document_version": 2,
                "file_extension": ".pdf",
                "indexed_at": "2026-06-17T00:00:00+00:00",
                "document_count": 2,
                "page_count": 2,
                "page": 0,
                "chunk_id": 0,
            },
            "document": "Intro text.",
        },
    ]
    store = object.__new__(DocumentVectorStore)
    store._repository = FakeCatalogRepository(records)  # type: ignore[assignment]

    preview = store.get_document_text(document_key="contract")

    assert preview is not None
    assert preview["document"]["file_id"] == "new-file"
    assert preview["document"]["document_version"] == 2
    assert [chunk["chunk_id"] for chunk in preview["chunks"]] == [0, 1]
    assert preview["chunks"][1]["location_label"] == "page 2, chunk 1, 2. Payment"


def test_get_document_text_can_load_requested_inactive_version() -> None:
    records = [
        {
            "id": "v1-0",
            "metadata": {
                "active": False,
                "document_key": "contract",
                "file_id": "old-file",
                "file_name": "contract.pdf",
                "document_version": 1,
                "chunk_id": 0,
            },
            "document": "Old text.",
        },
        {
            "id": "v2-0",
            "metadata": {
                "active": True,
                "document_key": "contract",
                "file_id": "new-file",
                "file_name": "contract.pdf",
                "document_version": 2,
                "chunk_id": 0,
            },
            "document": "New text.",
        },
    ]
    store = object.__new__(DocumentVectorStore)
    store._repository = FakeCatalogRepository(records)  # type: ignore[assignment]

    preview = store.get_document_text(document_key="contract", document_version=1)

    assert preview is not None
    assert preview["document"]["file_id"] == "old-file"
    assert preview["chunks"][0]["text"] == "Old text."

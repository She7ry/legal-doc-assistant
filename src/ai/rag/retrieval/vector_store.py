"""Public facade for document indexing, retrieval, listing, and preview."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from ai.config.settings import settings
from ai.llm import build_embedding_model
from ai.qdrant import collection_name_for_user, shared_qdrant_client
from ai.rag.ingestion.chunking import (
    INGESTION_CHUNK_SEPARATORS,
    build_ingestion_text_splitter,
    split_documents_for_ingestion,
)
from ai.rag.ingestion.indexer import (
    DocumentIngester,
    ProgressCallback,
    document_key_for_file_name,
)
from ai.rag.ingestion.loader import load_documents
from ai.rag.retrieval.catalog import (
    get_document_text as catalog_get_document_text,
)
from ai.rag.retrieval.catalog import list_documents as catalog_list_documents
from ai.rag.retrieval.repository import QdrantDocumentRepository
from ai.rag.retrieval.retriever import QdrantRetriever
from ai.rag.schemas import IngestResult

__all__ = [
    "INGESTION_CHUNK_SEPARATORS",
    "DocumentVectorStore",
    "collection_name_for_user",
    "document_key_for_file_name",
    "split_documents_for_ingestion",
]


class DocumentVectorStore:
    """Stable application-facing API over the document index components."""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: Path | None = None,
        user_id: str = "local",
    ) -> None:
        self.user_id = user_id
        effective_collection_name = collection_name or collection_name_for_user(
            settings.collection_name,
            self.user_id,
        )
        effective_persist_directory = Path(persist_directory or settings.vector_store_dir)
        self.vector_store = shared_qdrant_client(effective_persist_directory)
        self.splitter = build_ingestion_text_splitter()

        repository = QdrantDocumentRepository(
            self.vector_store,
            effective_collection_name,
            build_embedding_model(),
        )
        self._repository = repository
        self._retriever = QdrantRetriever(
            repository=repository,
            user_id=self.user_id,
        )
        self._ingester = DocumentIngester(
            user_id=self.user_id,
            repository=repository,
            splitter=self.splitter,
            invalidate_cache=self._retriever.clear_cache,
        )

    def split_documents(self, file_path: Path) -> list[Document]:
        return split_documents_for_ingestion(load_documents(file_path), splitter=self.splitter)

    def ingest_file(
        self,
        file_path: Path,
        file_name: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IngestResult:
        return self._ingester.ingest_file(
            file_path,
            file_name=file_name,
            progress_callback=progress_callback,
        )

    def search(self, query: str, k: int | None = None) -> list[Document]:
        return self._retriever.search(query, k=k)

    def list_documents(self) -> list[dict[str, Any]]:
        return catalog_list_documents(self._repository)

    def get_document_text(
        self,
        *,
        document_key: str | None = None,
        file_id: str | None = None,
        document_version: int | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any] | None:
        return catalog_get_document_text(
            self._repository,
            document_key=document_key,
            file_id=file_id,
            document_version=document_version,
            offset=offset,
            limit=limit,
        )

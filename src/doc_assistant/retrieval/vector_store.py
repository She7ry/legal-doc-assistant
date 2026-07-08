"""Public facade for document indexing, retrieval, listing, and preview."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from doc_assistant.config.settings import settings
from doc_assistant.ingestion.document_loader import load_documents
from doc_assistant.models.language_model import build_embedding_model
from doc_assistant.retrieval._chunking import (
    INGESTION_CHUNK_SEPARATORS,
    split_documents_for_ingestion,
)
from doc_assistant.retrieval._document_catalog import (
    get_document_text as catalog_get_document_text,
)
from doc_assistant.retrieval._document_catalog import list_documents as catalog_list_documents
from doc_assistant.retrieval._ingestion import (
    DocumentIngester,
    ProgressCallback,
    document_key_for_file_name,
)
from doc_assistant.retrieval._qdrant_backend import shared_qdrant_client
from doc_assistant.retrieval._qdrant_repository import QdrantDocumentRepository
from doc_assistant.retrieval._qdrant_retriever import QdrantRetriever
from doc_assistant.schemas.citation import IngestResult

_COLLECTION_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_COLLECTION_NAME_LENGTH = 63

__all__ = [
    "INGESTION_CHUNK_SEPARATORS",
    "DocumentVectorStore",
    "collection_name_for_tenant",
    "document_key_for_file_name",
    "split_documents_for_ingestion",
]


class DocumentVectorStore:
    """Stable application-facing API over the document index components."""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: Path | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self.tenant_id = tenant_id or settings.default_tenant_id
        effective_collection_name = collection_name or collection_name_for_tenant(
            settings.collection_name,
            self.tenant_id,
        )
        effective_persist_directory = Path(persist_directory or settings.vector_store_dir)
        self.vector_store = shared_qdrant_client(effective_persist_directory)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=list(INGESTION_CHUNK_SEPARATORS),
        )

        repository = QdrantDocumentRepository(
            self.vector_store,
            effective_collection_name,
            build_embedding_model(),
        )
        self._repository = repository
        self._retriever = QdrantRetriever(
            repository=repository,
            tenant_id=self.tenant_id,
        )
        self._ingester = DocumentIngester(
            tenant_id=self.tenant_id,
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
    ) -> dict[str, Any] | None:
        return catalog_get_document_text(
            self._repository,
            document_key=document_key,
            file_id=file_id,
            document_version=document_version,
        )


def collection_name_for_tenant(base_name: str, tenant_id: str | None) -> str:
    base = _sanitize_collection_component(base_name)
    if not tenant_id or tenant_id == settings.default_tenant_id:
        return base

    tenant = _sanitize_collection_component(tenant_id)
    collection_name = f"{base}_{tenant}"
    if len(collection_name) <= _MAX_COLLECTION_NAME_LENGTH:
        return collection_name

    digest = hashlib.sha1(tenant_id.encode("utf-8")).hexdigest()[:12]
    available_base_length = _MAX_COLLECTION_NAME_LENGTH - len(digest) - 1
    return f"{base[:available_base_length].rstrip('_-')}_{digest}"


def _sanitize_collection_component(value: str) -> str:
    sanitized = _COLLECTION_COMPONENT_PATTERN.sub("_", value.strip()).strip("_-")
    if len(sanitized) < 3:
        sanitized = f"{sanitized or 'col'}_collection"
    return sanitized[:_MAX_COLLECTION_NAME_LENGTH].strip("_-") or "col_collection"

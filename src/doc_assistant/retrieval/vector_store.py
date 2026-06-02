from __future__ import annotations

import hashlib
import logging
import re
import threading
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from doc_assistant.config.settings import settings
from doc_assistant.ingestion.document_loader import file_sha256, load_documents
from doc_assistant.models.language_model import build_embedding_model
from doc_assistant.schemas.citation import IngestResult

logger = logging.getLogger(__name__)

_COLLECTION_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_COLLECTION_NAME_LENGTH = 63


class DocumentVectorStore:
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
        self.vector_store = Chroma(
            collection_name=effective_collection_name,
            embedding_function=build_embedding_model(),
            persist_directory=str(persist_directory or settings.vector_store_dir),
        )
        self._write_lock = threading.Lock()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )

    def ingest_file(self, file_path: Path, file_name: str | None = None) -> IngestResult:
        file_path = Path(file_path)
        display_name = file_name or file_path.name
        file_id = file_sha256(file_path)
        documents = load_documents(file_path)

        for document in documents:
            document.metadata["file_id"] = file_id
            document.metadata["file_name"] = display_name
            document.metadata["tenant_id"] = self.tenant_id

        chunks = self.splitter.split_documents(documents)
        ids = []
        for index, chunk in enumerate(chunks):
            chunk.metadata["file_id"] = file_id
            chunk.metadata["file_name"] = display_name
            chunk.metadata["tenant_id"] = self.tenant_id
            chunk.metadata["chunk_id"] = index
            ids.append(f"{file_id}:{index}")

        if ids:
            with self._write_lock:
                try:
                    self.vector_store.delete(ids=ids)
                except Exception as exc:
                    logger.warning(
                        "Failed to delete existing chunks before re-ingest",
                        extra={"file_id": file_id, "tenant_id": self.tenant_id},
                        exc_info=True,
                    )
                    raise RuntimeError("Failed to prepare vector store for document re-ingest.") from exc

                self.vector_store.add_documents(chunks, ids=ids)

        return IngestResult(
            file_id=file_id,
            file_name=display_name,
            document_count=len(documents),
            chunk_count=len(chunks),
        )

    def search(self, query: str, k: int | None = None) -> list[Document]:
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k or settings.top_k})
        return retriever.invoke(query)


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

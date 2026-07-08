"""Cached application-facing retrieval over Qdrant Query API."""

from __future__ import annotations

import hashlib
import logging
import threading
import time

from langchain_core.documents import Document

from doc_assistant.config.settings import settings
from doc_assistant.observability import traced_operation
from doc_assistant.retrieval._qdrant_repository import QdrantDocumentRepository

logger = logging.getLogger(__name__)


class QueryCache:
    def __init__(self, *, ttl_seconds: int, max_size: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._values: dict[str, tuple[float, list[Document]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> list[Document] | None:
        if self._ttl_seconds <= 0 or self._max_size <= 0:
            return None
        with self._lock:
            entry = self._values.get(key)
            if entry is None:
                return None
            stored_at, documents = entry
            if time.monotonic() - stored_at > self._ttl_seconds:
                self._values.pop(key, None)
                return None
            self._values[key] = (time.monotonic(), documents)
            return [_copy_document(document) for document in documents]

    def set(self, key: str, documents: list[Document]) -> None:
        if self._ttl_seconds <= 0 or self._max_size <= 0:
            return
        with self._lock:
            if len(self._values) >= self._max_size and key not in self._values:
                oldest_key = min(self._values, key=lambda item: self._values[item][0])
                self._values.pop(oldest_key, None)
            self._values[key] = (
                time.monotonic(),
                [_copy_document(document) for document in documents],
            )

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


class QdrantRetriever:
    def __init__(
        self,
        *,
        repository: QdrantDocumentRepository,
        tenant_id: str,
        cache: QueryCache | None = None,
    ) -> None:
        self.repository = repository
        self.tenant_id = tenant_id
        self.cache = cache or QueryCache(
            ttl_seconds=max(0, int(settings.retrieval_cache_ttl_seconds)),
            max_size=max(0, int(settings.retrieval_cache_max_size)),
        )

    def search(self, query: str, k: int | None = None) -> list[Document]:
        top_k = max(1, int(k or settings.top_k))
        mode = str(settings.retrieval_mode or "hybrid").strip().lower()
        if mode not in {"hybrid", "dense", "vector", "bm25", "sparse"}:
            logger.warning("Unknown retrieval mode %r; falling back to hybrid.", mode)
            mode = "hybrid"
        cache_key = self._query_cache_key(query, top_k, mode)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        with traced_operation(
            "vector_search",
            tenant_id=self.tenant_id,
            top_k=top_k,
            query=query[:120],
            backend="qdrant",
            retrieval_mode=mode,
        ):
            documents = self.repository.search(query, limit=top_k, mode=mode)
        self.cache.set(cache_key, documents)
        return [_copy_document(document) for document in documents]

    def clear_cache(self) -> None:
        self.cache.clear()

    def _query_cache_key(self, query: str, top_k: int, mode: str) -> str:
        parts = [
            self.tenant_id,
            mode,
            str(top_k),
            str(settings.retrieval_fetch_k),
            str(settings.retrieval_min_relevance),
            str(settings.retrieval_mmr_lambda),
            query.strip(),
        ]
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _copy_document(document: Document) -> Document:
    return Document(
        page_content=document.page_content,
        metadata=dict(document.metadata or {}),
    )

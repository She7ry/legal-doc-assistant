"""Qdrant repository for document chunks and hybrid retrieval."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, TypedDict

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from ai.config.settings import settings
from ai.qdrant import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ensure_dense_collection,
    ensure_payload_indexes,
    point_id,
)
from ai.rag.retrieval.backend import (
    DOCUMENT_PAYLOAD_KEY,
    RECORD_ID_PAYLOAD_KEY,
    active_filter,
    clean_metadata,
    field_filter,
    sparse_document_vector,
    sparse_query_vector,
)


class VectorRecord(TypedDict):
    id: str
    metadata: dict[str, Any]
    document: str


_DOCUMENT_PAYLOAD_INDEXES = {
    "active": models.PayloadSchemaType.BOOL,
    "document_key": models.PayloadSchemaType.KEYWORD,
    "file_id": models.PayloadSchemaType.KEYWORD,
    "document_version": models.PayloadSchemaType.INTEGER,
    "chunk_id": models.PayloadSchemaType.INTEGER,
}


class QdrantDocumentRepository:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_model: Any,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._collection_lock = Lock()
        self._payload_indexes_initialized = False
        self._validated_vector_size: int | None = None

    def all_records(self, *, include_documents: bool = True) -> list[VectorRecord]:
        return self._scroll(include_documents=include_documents)

    def records_for_document_key(self, document_key: str) -> list[VectorRecord]:
        return self._scroll(
            include_documents=False,
            query_filter=field_filter(document_key=document_key),
        )

    def matching_records(
        self,
        *,
        document_key: str | None,
        file_id: str | None,
        include_documents: bool,
        document_version: int | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[VectorRecord], int, VectorRecord | None]:
        if not self._ensure_payload_indexes_if_collection_exists():
            return [], 0, None

        identity_filter = field_filter(
            document_key=document_key,
            file_id=file_id,
            document_version=document_version,
        )
        document_filter = models.Filter(
            must=[
                *([identity_filter] if identity_filter is not None else []),
                *([active_filter()] if document_version is None else []),
            ]
        )
        total = int(
            self.client.count(
                collection_name=self.collection_name,
                count_filter=document_filter,
                exact=True,
            ).count
        )
        if total == 0:
            return [], 0, None

        page_filter = models.Filter(
            must=[
                document_filter,
                models.FieldCondition(
                    key="chunk_id",
                    range=models.Range(gte=offset, lt=offset + limit),
                ),
            ]
        )
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=page_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        records = [
            self._record_from_payload(
                dict(point.payload or {}),
                include_documents=include_documents,
            )
            for point in points
        ]
        if records:
            return records, total, records[0]

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=document_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        representative = (
            self._record_from_payload(dict(points[0].payload or {}), include_documents=False)
            if points
            else None
        )
        return records, total, representative

    def active_records(self, *, include_documents: bool = True) -> list[VectorRecord]:
        return self._scroll(
            include_documents=include_documents,
            query_filter=active_filter(),
        )

    def delete(self, ids: list[str]) -> None:
        if not ids or not self.client.collection_exists(self.collection_name):
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[point_id(record_id) for record_id in ids],
            wait=True,
        )

    def set_payload(
        self,
        ids: list[str],
        payload: dict[str, str | int | float | bool],
    ) -> None:
        if not ids:
            return
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=clean_metadata(payload),
            points=[point_id(record_id) for record_id in ids],
            wait=True,
        )

    def embed_query(self, query: str) -> list[float]:
        return [float(value) for value in self.embedding_model.embed_query(query)]

    def embed_and_add(
        self,
        chunks: list[Document],
        ids: list[str],
        *,
        batch_size: int,
        max_workers: int,
    ) -> None:
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self._embed_documents(
            texts,
            batch_size=max(1, batch_size),
            max_workers=max(1, max_workers),
        )
        if not embeddings:
            return
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Embedding provider returned {len(embeddings)} embeddings for {len(texts)} chunks."
            )

        self._ensure_collection(len(embeddings[0]))
        points = []
        for record_id, text, chunk, embedding in zip(ids, texts, chunks, embeddings, strict=True):
            metadata = clean_metadata(chunk.metadata or {})
            payload: dict[str, Any] = {
                **metadata,
                RECORD_ID_PAYLOAD_KEY: record_id,
                DOCUMENT_PAYLOAD_KEY: text,
            }
            points.append(
                models.PointStruct(
                    id=point_id(record_id),
                    vector={
                        DENSE_VECTOR_NAME: embedding,
                        SPARSE_VECTOR_NAME: sparse_document_vector(text, metadata),
                    },
                    payload=payload,
                )
            )

        resolved_batch_size = max(1, batch_size)
        for index in range(0, len(points), resolved_batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[index : index + resolved_batch_size],
                wait=True,
            )

    def search(
        self,
        query: str,
        *,
        limit: int,
        mode: str,
    ) -> list[Document]:
        fetch_k = max(limit, int(settings.retrieval_fetch_k), limit * 5)
        query_filter = active_filter()
        sparse_query = sparse_query_vector(query)
        diversity = max(0.0, min(1.0, 1.0 - float(settings.retrieval_mmr_lambda)))

        if mode in {"bm25", "sparse"}:
            if not sparse_query.indices:
                return []
            if not self._ensure_payload_indexes_if_collection_exists():
                return []
            query_object: Any = models.NearestQuery(
                nearest=sparse_query,
                mmr=models.Mmr(diversity=diversity, candidates_limit=fetch_k),
            )
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_object,
                using=SPARSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        else:
            dense_query = self.embed_query(query)
            self._ensure_collection(len(dense_query))
            nearest = models.NearestQuery(
                nearest=dense_query,
                mmr=models.Mmr(diversity=diversity, candidates_limit=fetch_k),
            )
            if mode in {"dense", "vector"} or not sparse_query.indices:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=nearest,
                    using=DENSE_VECTOR_NAME,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
            else:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=nearest,
                            using=DENSE_VECTOR_NAME,
                            filter=query_filter,
                            limit=fetch_k,
                        ),
                        models.Prefetch(
                            query=sparse_query,
                            using=SPARSE_VECTOR_NAME,
                            filter=query_filter,
                            limit=fetch_k,
                        ),
                    ],
                    query=models.RrfQuery(rrf=models.Rrf()),
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )

        minimum = max(0.0, float(settings.retrieval_min_relevance))
        return [
            self._document_from_scored_point(point, mode=mode)
            for point in response.points
            if _normalized_relevance(float(point.score), mode) >= minimum
        ]

    def _ensure_collection(self, vector_size: int) -> None:
        with self._collection_lock:
            if self._validated_vector_size == vector_size:
                return
            ensure_dense_collection(
                self.client,
                self.collection_name,
                vector_size,
                with_sparse=True,
                payload_indexes=(
                    {} if self._payload_indexes_initialized else _DOCUMENT_PAYLOAD_INDEXES
                ),
            )
            self._payload_indexes_initialized = True
            self._validated_vector_size = vector_size

    def _ensure_payload_indexes_if_collection_exists(self) -> bool:
        with self._collection_lock:
            if self._payload_indexes_initialized:
                return True
            if not self.client.collection_exists(self.collection_name):
                return False
            ensure_payload_indexes(
                self.client,
                self.collection_name,
                _DOCUMENT_PAYLOAD_INDEXES,
            )
            self._payload_indexes_initialized = True
            return True

    def _embed_documents(
        self,
        texts: list[str],
        *,
        batch_size: int,
        max_workers: int,
    ) -> list[list[float]]:
        if max_workers == 1 or len(texts) <= batch_size:
            return [
                list(map(float, vector)) for vector in self.embedding_model.embed_documents(texts)
            ]
        batches = [texts[index : index + batch_size] for index in range(0, len(texts), batch_size)]
        embedded_batches: list[list[list[float]] | None] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as executor:
            futures = {
                executor.submit(self.embedding_model.embed_documents, batch): index
                for index, batch in enumerate(batches)
            }
            for future, index in ((future, futures[future]) for future in futures):
                embedded_batches[index] = [list(map(float, vector)) for vector in future.result()]
        return [vector for batch in embedded_batches if batch is not None for vector in batch]

    def _scroll(
        self,
        *,
        include_documents: bool,
        query_filter: models.Filter | None = None,
    ) -> list[VectorRecord]:
        if query_filter is not None:
            if not self._ensure_payload_indexes_if_collection_exists():
                return []
        elif not self.client.collection_exists(self.collection_name):
            return []
        records: list[VectorRecord] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                records.append(
                    self._record_from_payload(payload, include_documents=include_documents)
                )
            if offset is None:
                break
        return records

    @staticmethod
    def _record_from_payload(
        payload: dict[str, Any],
        *,
        include_documents: bool,
    ) -> VectorRecord:
        record_id = str(payload.pop(RECORD_ID_PAYLOAD_KEY, ""))
        document = str(payload.pop(DOCUMENT_PAYLOAD_KEY, "")) if include_documents else ""
        return {"id": record_id, "metadata": payload, "document": document}

    @staticmethod
    def _document_from_scored_point(point: models.ScoredPoint, *, mode: str) -> Document:
        payload = dict(point.payload or {})
        payload.pop(RECORD_ID_PAYLOAD_KEY, None)
        text = str(payload.pop(DOCUMENT_PAYLOAD_KEY, ""))
        score = float(point.score)
        relevance = _normalized_relevance(score, mode)
        payload.update(
            {
                "retrieval_score": score,
                "retrieval_relevance": relevance,
                "qdrant_score": score,
                "retrieval_mode": mode,
            }
        )
        return Document(page_content=text, metadata=clean_metadata(payload))


def _normalized_relevance(score: float, mode: str) -> float:
    if mode in {"bm25", "sparse"}:
        return score / (1.0 + score) if score > 0 else 0.0
    return max(0.0, min(1.0, score))

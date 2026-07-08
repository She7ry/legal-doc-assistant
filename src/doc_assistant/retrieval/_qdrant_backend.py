"""Shared Qdrant client, collection, payload, and sparse-vector utilities."""

from __future__ import annotations

import hashlib
import threading
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient, models

from doc_assistant.config.settings import settings
from doc_assistant.ingestion.document_loader import INGEST_WARNINGS_METADATA_KEY
from doc_assistant.retrieval._search_utils import _tokenize_for_search

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"
DOCUMENT_PAYLOAD_KEY = "_document"
RECORD_ID_PAYLOAD_KEY = "_record_id"

_clients: dict[str, QdrantClient] = {}
_clients_lock = threading.Lock()
_collections_lock = threading.Lock()


def shared_qdrant_client(persist_directory: Path) -> QdrantClient:
    """Return one process-wide client per remote endpoint or local storage path."""
    url = str(settings.qdrant_url or "").strip()
    if url:
        cache_key = f"url:{url}"
    else:
        resolved_path = Path(persist_directory).resolve()
        resolved_path.mkdir(parents=True, exist_ok=True)
        cache_key = f"path:{resolved_path}"

    with _clients_lock:
        existing = _clients.get(cache_key)
        if existing is not None:
            return existing

        if url:
            client = QdrantClient(
                url=url,
                api_key=str(settings.qdrant_api_key) or None,
                prefer_grpc=bool(settings.qdrant_prefer_grpc),
            )
        else:
            client = QdrantClient(path=str(resolved_path))
        _clients[cache_key] = client
        return client


def ensure_dense_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    *,
    with_sparse: bool,
) -> None:
    """Create a collection lazily after the embedding dimension is known."""
    with _collections_lock:
        if client.collection_exists(collection_name):
            _validate_vector_size(client, collection_name, vector_size)
            return

        sparse_config = None
        if with_sparse:
            sparse_config = {
                SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
            }
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config=sparse_config,
        )


def _validate_vector_size(
    client: QdrantClient,
    collection_name: str,
    expected_size: int,
) -> None:
    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors
    params = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else vectors
    actual_size = int(params.size) if params is not None else 0
    if actual_size != expected_size:
        raise RuntimeError(
            f"Qdrant collection {collection_name!r} uses vector size {actual_size}, "
            f"but the configured embedding model returned {expected_size}. "
            "Use a new collection name or re-index the collection."
        )


def point_id(record_id: str) -> UUID:
    """Map application string IDs to Qdrant-compatible deterministic UUIDs."""
    return uuid5(NAMESPACE_URL, f"legal-doc-assistant:{record_id}")


def clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if key == INGEST_WARNINGS_METADATA_KEY or value is None:
            continue
        if isinstance(value, str | int | float | bool):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def metadata_is_active(metadata: dict[str, Any]) -> bool:
    return metadata.get("active", True) is not False


def active_filter() -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="active",
                match=models.MatchValue(value=True),
            )
        ]
    )


def field_filter(**values: str | int | float | bool | None) -> models.Filter | None:
    conditions = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in values.items()
        if value is not None
    ]
    return models.Filter(must=conditions) if conditions else None


def sparse_document_vector(text: str, metadata: dict[str, Any]) -> models.SparseVector:
    search_text = _sparse_search_text(text, metadata)
    tokens = _tokenize_for_search(search_text)
    if not tokens:
        return models.SparseVector(indices=[], values=[])

    token_counts = Counter(tokens)
    document_length = len(tokens)
    k1 = float(settings.retrieval_bm25_k1)
    b = float(settings.retrieval_bm25_b)
    average_length = float(settings.retrieval_bm25_average_length)
    normalizer = k1 * (1 - b + b * document_length / max(average_length, 1.0))

    weights: dict[int, float] = {}
    for token, frequency in token_counts.items():
        index = _sparse_token_index(token)
        value = frequency * (k1 + 1) / (frequency + normalizer)
        weights[index] = weights.get(index, 0.0) + value
    indices = sorted(weights)
    return models.SparseVector(indices=indices, values=[weights[index] for index in indices])


def sparse_query_vector(text: str) -> models.SparseVector:
    token_counts = Counter(_tokenize_for_search(text))
    weights: dict[int, float] = {}
    for token, frequency in token_counts.items():
        index = _sparse_token_index(token)
        weights[index] = weights.get(index, 0.0) + float(frequency)
    indices = sorted(weights)
    return models.SparseVector(indices=indices, values=[weights[index] for index in indices])


def _sparse_search_text(text: str, metadata: dict[str, Any]) -> str:
    file_name = str(metadata.get("file_name") or metadata.get("source") or "")
    heading = str(metadata.get("section_heading") or "")
    return "\n".join(part for part in (file_name, text, heading, heading) if part)


def _sparse_token_index(token: str) -> int:
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)

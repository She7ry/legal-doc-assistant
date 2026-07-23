"""Shared Qdrant client, collection initialization, and identifiers."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient, models

from ai.config.settings import settings

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

logger = logging.getLogger(__name__)

_COLLECTION_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_COLLECTION_NAME_LENGTH = 63
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
    payload_indexes: dict[str, models.PayloadSchemaType],
    collection_exists: bool | None = None,
) -> None:
    """Create a collection lazily after the embedding dimension is known."""
    with _collections_lock:
        exists = (
            collection_exists
            if collection_exists is not None
            else client.collection_exists(collection_name)
        )
        if exists:
            _validate_vector_size(client, collection_name, vector_size)
        else:
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

        _create_payload_indexes(client, collection_name, payload_indexes)


def ensure_payload_indexes(
    client: QdrantClient,
    collection_name: str,
    payload_indexes: dict[str, models.PayloadSchemaType],
) -> None:
    with _collections_lock:
        _create_payload_indexes(client, collection_name, payload_indexes)


def _create_payload_indexes(
    client: QdrantClient,
    collection_name: str,
    payload_indexes: dict[str, models.PayloadSchemaType],
) -> None:
    for field_name, field_schema in payload_indexes.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )
        except Exception:
            logger.warning(
                "Failed to create Qdrant payload index %r for collection %r",
                field_name,
                collection_name,
                exc_info=True,
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


def collection_name_for_user(base_name: str, user_id: str) -> str:
    base = _sanitize_collection_component(base_name)
    user = _sanitize_collection_component(user_id)
    collection_name = f"{base}_{user}"
    if len(collection_name) <= _MAX_COLLECTION_NAME_LENGTH:
        return collection_name

    digest = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:12]
    available_base_length = _MAX_COLLECTION_NAME_LENGTH - len(digest) - 1
    return f"{base[:available_base_length].rstrip('_-')}_{digest}"


def _sanitize_collection_component(value: str) -> str:
    sanitized = _COLLECTION_COMPONENT_PATTERN.sub("_", value.strip()).strip("_-")
    if len(sanitized) < 3:
        sanitized = f"{sanitized or 'col'}_collection"
    return sanitized[:_MAX_COLLECTION_NAME_LENGTH].strip("_-") or "col_collection"

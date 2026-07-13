"""Qdrant-backed semantic index for durable memories."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from qdrant_client import models

from doc_assistant.config.settings import settings
from doc_assistant.memory.schemas import MemoryCandidate, MemoryRecord
from doc_assistant.models.language_model import build_embedding_model
from doc_assistant.retrieval._qdrant_backend import (
    DENSE_VECTOR_NAME,
    DOCUMENT_PAYLOAD_KEY,
    RECORD_ID_PAYLOAD_KEY,
    ensure_dense_collection,
    point_id,
    shared_qdrant_client,
)
from doc_assistant.retrieval.vector_store import collection_name_for_tenant
from doc_assistant.utils.json import parse_json_object

logger = logging.getLogger(__name__)

_MEMORY_PAYLOAD_INDEXES = {
    "tenant_id": models.PayloadSchemaType.KEYWORD,
    "user_id": models.PayloadSchemaType.KEYWORD,
    "status": models.PayloadSchemaType.KEYWORD,
    "visibility": models.PayloadSchemaType.KEYWORD,
}


class MemoryVectorStore:
    """Dedicated Qdrant collection for memory deduplication and retrieval."""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: Path | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self.tenant_id = tenant_id or settings.default_tenant_id
        self.collection_name = collection_name or collection_name_for_tenant(
            settings.memory_collection_name,
            self.tenant_id,
        )
        self.vector_store = shared_qdrant_client(
            Path(persist_directory or settings.memory_vector_store_dir)
        )
        self.embedding_model = build_embedding_model()
        self._collection_lock = Lock()
        self._validated_vector_size: int | None = None

    def upsert_memory(self, memory: MemoryRecord) -> str:
        if memory.status != "active" or memory.is_expired():
            self.delete_memory(memory.memory_id)
            return memory.memory_id

        metadata = {
            "memory_id": memory.memory_id,
            "tenant_id": memory.tenant_id,
            "user_id": memory.user_id,
            "scope": memory.scope,
            "type": memory.type,
            "key": memory.key,
            "source": memory.source,
            "confidence": memory.confidence,
            "visibility": memory.visibility,
            "status": memory.status,
            "content": memory.content,
            "value_json": json.dumps(memory.value_json, ensure_ascii=False)
            if memory.value_json is not None
            else "",
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "expires_at": memory.expires_at.isoformat() if memory.expires_at else "",
            "conversation_id": memory.conversation_id or "",
            "task_id": memory.task_id or "",
        }
        text = _memory_embedding_text(memory)
        embedding = [float(value) for value in self.embedding_model.embed_query(text)]
        self._ensure_collection(len(embedding))
        self.vector_store.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id(memory.memory_id),
                    vector={DENSE_VECTOR_NAME: embedding},
                    payload={
                        **metadata,
                        RECORD_ID_PAYLOAD_KEY: memory.memory_id,
                        DOCUMENT_PAYLOAD_KEY: text,
                    },
                )
            ],
            wait=True,
        )
        return memory.memory_id

    def delete_memory(self, memory_id: str) -> bool:
        try:
            if not self.vector_store.collection_exists(self.collection_name):
                return True
            self.vector_store.delete(
                collection_name=self.collection_name,
                points_selector=[point_id(memory_id)],
                wait=True,
            )
            return True
        except Exception:
            logger.warning(
                "Memory vector delete failed",
                extra={"memory_id": memory_id},
                exc_info=True,
            )
            return False

    def search(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        user_id: str,
        k: int | None = None,
    ) -> list[MemoryCandidate]:
        search_k = max(1, int(k or settings.memory_top_k))
        resolved_tenant_id = tenant_id or self.tenant_id
        embedding = [float(value) for value in self.embedding_model.embed_query(query)]
        self._ensure_collection(len(embedding))
        response = self.vector_store.query_points(
            collection_name=self.collection_name,
            query=embedding,
            using=DENSE_VECTOR_NAME,
            query_filter=_readable_memory_filter(resolved_tenant_id, user_id),
            limit=search_k,
            with_payload=True,
        )
        candidates: list[MemoryCandidate] = []
        for point in response.points:
            metadata = dict(point.payload or {})
            metadata.pop(RECORD_ID_PAYLOAD_KEY, None)
            embedded_text = str(metadata.pop(DOCUMENT_PAYLOAD_KEY, ""))
            has_complete_metadata = "value_json" in metadata
            now = datetime.now(timezone.utc)
            created_at = metadata.get("created_at")
            updated_at = metadata.get("updated_at")
            expires_at = metadata.get("expires_at")
            memory = MemoryRecord(
                memory_id=str(metadata.get("memory_id") or ""),
                tenant_id=str(metadata.get("tenant_id") or resolved_tenant_id),
                user_id=str(metadata.get("user_id") or user_id),
                scope=str(metadata.get("scope") or "user"),  # type: ignore[arg-type]
                type=str(metadata.get("type") or "preference"),  # type: ignore[arg-type]
                key=str(metadata.get("key") or ""),
                content=str(metadata.get("content") or embedded_text),
                value_json=_metadata_json_dict(metadata.get("value_json")),
                source=str(metadata.get("source") or "explicit"),  # type: ignore[arg-type]
                confidence=float(metadata.get("confidence") or 0),
                created_at=datetime.fromisoformat(str(created_at)) if created_at else now,
                updated_at=datetime.fromisoformat(str(updated_at)) if updated_at else now,
                expires_at=datetime.fromisoformat(str(expires_at)) if expires_at else None,
                visibility=str(metadata.get("visibility") or "private"),  # type: ignore[arg-type]
                status=str(metadata.get("status") or "active"),  # type: ignore[arg-type]
                conversation_id=_metadata_text(metadata.get("conversation_id")),
                task_id=_metadata_text(metadata.get("task_id")),
            )
            if not _metadata_memory_is_readable(memory, resolved_tenant_id, user_id):
                continue
            candidates.append(
                MemoryCandidate(
                    memory=memory,
                    score=float(point.score),
                    retrieval_source="vector" if has_complete_metadata else "vector_partial",
                )
            )
        return candidates

    def _ensure_collection(self, vector_size: int) -> None:
        with self._collection_lock:
            if self._validated_vector_size == vector_size:
                return
            ensure_dense_collection(
                self.vector_store,
                self.collection_name,
                vector_size,
                with_sparse=False,
                payload_indexes=_MEMORY_PAYLOAD_INDEXES,
            )
            self._validated_vector_size = vector_size


def _memory_embedding_text(memory: MemoryRecord) -> str:
    return "\n".join(
        [
            f"scope: {memory.scope}",
            f"type: {memory.type}",
            f"key: {memory.key}",
            f"source: {memory.source}",
            f"content: {memory.content}",
        ]
    )


def _readable_memory_filter(tenant_id: str, user_id: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id),
            ),
            models.FieldCondition(
                key="status",
                match=models.MatchValue(value="active"),
            ),
        ],
        should=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
            models.FieldCondition(key="visibility", match=models.MatchValue(value="team")),
            models.FieldCondition(key="visibility", match=models.MatchValue(value="org")),
        ],
    )


def _metadata_memory_is_readable(memory: MemoryRecord, tenant_id: str, user_id: str) -> bool:
    return (
        memory.tenant_id == tenant_id
        and memory.status == "active"
        and not memory.is_expired()
        and (memory.user_id == user_id or memory.visibility in {"team", "org"})
    )


def _metadata_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _metadata_json_dict(value: object | None) -> dict | None:
    return parse_json_object(value)

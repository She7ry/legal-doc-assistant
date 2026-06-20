"""记忆向量索引：与文档 RAG 索引分离，避免用户偏好被当作合同证据检索。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma

from doc_assistant.config.settings import settings
from doc_assistant.memory.schemas import MemoryCandidate, MemoryRecord
from doc_assistant.models.language_model import build_embedding_model
from doc_assistant.retrieval.vector_store import collection_name_for_tenant

logger = logging.getLogger(__name__)


class MemoryVectorStore:
    """专用 Chroma 集合，仅存 MemoryRecord 的 embedding，供语义去重与检索。"""

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: Path | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self.tenant_id = tenant_id or settings.default_tenant_id
        effective_collection_name = collection_name or collection_name_for_tenant(
            settings.memory_collection_name,
            self.tenant_id,
        )
        self.vector_store = Chroma(
            collection_name=effective_collection_name,
            embedding_function=build_embedding_model(),
            persist_directory=str(persist_directory or settings.memory_vector_store_dir),
        )

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
            "value_json": json.dumps(memory.value_json, ensure_ascii=False) if memory.value_json is not None else "",
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "expires_at": memory.expires_at.isoformat() if memory.expires_at else "",
            "permissions_json": json.dumps(list(memory.permissions), ensure_ascii=False),
            "conversation_id": memory.conversation_id or "",
            "task_id": memory.task_id or "",
            "last_accessed_at": memory.last_accessed_at.isoformat() if memory.last_accessed_at else "",
            "access_count": memory.access_count,
        }
        text = _memory_embedding_text(memory)
        self.vector_store.add_texts([text], metadatas=[metadata], ids=[memory.memory_id])
        return memory.memory_id

    def delete_memory(self, memory_id: str) -> None:
        try:
            self.vector_store.delete(ids=[memory_id])
        except Exception:
            logger.debug("Memory vector delete failed", extra={"memory_id": memory_id}, exc_info=True)

    def search(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        user_id: str,
        k: int | None = None,
    ) -> list[MemoryCandidate]:
        search_k = k or settings.memory_top_k
        resolved_tenant_id = tenant_id or self.tenant_id
        docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=search_k,
            filter=_readable_memory_filter(resolved_tenant_id, user_id),
        )
        candidates: list[MemoryCandidate] = []
        for document, score in docs_and_scores:
            metadata = document.metadata or {}
            has_complete_metadata = "value_json" in metadata and "permissions_json" in metadata
            memory = MemoryRecord(
                memory_id=str(metadata.get("memory_id") or ""),
                tenant_id=str(metadata.get("tenant_id") or resolved_tenant_id),
                user_id=str(metadata.get("user_id") or user_id),
                scope=str(metadata.get("scope") or "user"),  # type: ignore[arg-type]
                type=str(metadata.get("type") or "preference"),  # type: ignore[arg-type]
                key=str(metadata.get("key") or ""),
                content=str(metadata.get("content") or document.page_content),
                value_json=_metadata_json_dict(metadata.get("value_json")),
                source=str(metadata.get("source") or "explicit"),  # type: ignore[arg-type]
                confidence=float(metadata.get("confidence") or 0),
                created_at=_metadata_datetime(metadata.get("created_at")) or _utc_now(),
                updated_at=_metadata_datetime(metadata.get("updated_at")) or _utc_now(),
                expires_at=_metadata_datetime(metadata.get("expires_at")),
                visibility=str(metadata.get("visibility") or "private"),  # type: ignore[arg-type]
                permissions=_metadata_permissions(metadata.get("permissions_json")),
                status=str(metadata.get("status") or "active"),  # type: ignore[arg-type]
                conversation_id=_metadata_text(metadata.get("conversation_id")),
                task_id=_metadata_text(metadata.get("task_id")),
                last_accessed_at=_metadata_datetime(metadata.get("last_accessed_at")),
                access_count=int(metadata.get("access_count") or 0),
            )
            if not _metadata_memory_is_readable(memory, resolved_tenant_id, user_id):
                continue
            candidates.append(
                MemoryCandidate(
                    memory=memory,
                    score=score,
                    retrieval_source="vector" if has_complete_metadata else "vector_partial",
                )
            )
        return candidates


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


def _readable_memory_filter(tenant_id: str, user_id: str) -> dict[str, object]:
    return {
        "$and": [
            {"tenant_id": tenant_id},
            {"status": "active"},
            {
                "$or": [
                    {"user_id": user_id},
                    {"visibility": "team"},
                    {"visibility": "org"},
                ]
            },
        ]
    }


def _metadata_memory_is_readable(memory: MemoryRecord, tenant_id: str, user_id: str) -> bool:
    return (
        memory.tenant_id == tenant_id
        and memory.status == "active"
        and not memory.is_expired()
        and (memory.user_id == user_id or memory.visibility in {"team", "org"})
    )


def _metadata_datetime(value: object | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _metadata_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _metadata_json_dict(value: object | None) -> dict | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _metadata_permissions(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ("read", "write", "delete")
    text = str(value).strip()
    if not text:
        return ("read", "write", "delete")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return ("read", "write", "delete")
    if not isinstance(parsed, list):
        return ("read", "write", "delete")
    permissions = tuple(str(permission) for permission in parsed if str(permission).strip())
    return permissions or ("read", "write", "delete")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

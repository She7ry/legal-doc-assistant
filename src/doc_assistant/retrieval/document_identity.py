"""Canonical identity generation for retrieved document chunks."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.documents import Document

from doc_assistant.utils.text import optional_text


def document_identity(document: Document, fallback_id: str | None = None) -> str:
    """Return a deterministic identity used to deduplicate retrieval results."""
    metadata = document.metadata or {}
    stable_identity: dict[str, Any] = {
        "file_id": optional_text(metadata.get("file_id")),
        "document_key": optional_text(metadata.get("document_key")),
        "document_version": _json_scalar(metadata.get("document_version")),
        "page": _json_scalar(metadata.get("page")),
        "chunk_id": _json_scalar(metadata.get("chunk_id")),
    }
    if any(value is not None for value in stable_identity.values()):
        stable_identity["source"] = optional_text(
            metadata.get("source") or metadata.get("file_name")
        )
        payload = stable_identity
    elif fallback_id:
        payload = {"fallback_id": fallback_id}
    else:
        payload = {
            "source": optional_text(metadata.get("source") or metadata.get("file_name")),
            "content": document.page_content or "",
        }

    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

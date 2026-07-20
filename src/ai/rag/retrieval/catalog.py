"""Document listing and preview projections over indexed chunk records."""

from __future__ import annotations

from typing import Any

from ai.rag.retrieval.backend import metadata_is_active
from ai.rag.retrieval.repository import (
    QdrantDocumentRepository,
    VectorRecord,
)


def list_documents(repository: QdrantDocumentRepository) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in repository.all_records(include_documents=False):
        metadata = record["metadata"]
        if not metadata_is_active(metadata):
            continue
        key = str(metadata.get("document_key") or metadata.get("file_id") or record["id"])
        version = metadata_int(metadata, "document_version", 1)
        current = grouped.get(key)
        if current is None or version > int(current["document_version"]):
            grouped[key] = _document_summary(metadata, key=key, version=version)

        if grouped[key]["document_version"] == version:
            grouped[key]["chunk_count"] += 1

    return sorted(
        grouped.values(),
        key=lambda item: str(item.get("indexed_at") or ""),
        reverse=True,
    )


def get_document_text(
    repository: QdrantDocumentRepository,
    *,
    document_key: str | None = None,
    file_id: str | None = None,
    document_version: int | None = None,
    offset: int = 0,
    limit: int = 100,
) -> dict[str, Any] | None:
    resolved_document_key = (document_key or "").strip()
    resolved_file_id = (file_id or "").strip()
    if not resolved_document_key and not resolved_file_id:
        raise ValueError("Provide document_key or file_id.")

    records, total_chunks, representative = repository.matching_records(
        document_key=resolved_document_key or None,
        file_id=resolved_file_id or None,
        include_documents=True,
        document_version=document_version,
        offset=offset,
        limit=limit,
    )
    if representative is None:
        return None

    records.sort(key=document_preview_sort_key)
    first_metadata = representative["metadata"]
    summary = _document_summary(
        first_metadata,
        key=str(
            first_metadata.get("document_key") or resolved_document_key or resolved_file_id
        ),
        version=metadata_int(first_metadata, "document_version", document_version or 1),
    )
    summary["file_id"] = str(first_metadata.get("file_id") or resolved_file_id)
    summary["chunk_count"] = total_chunks
    chunks = [
        {
            "chunk_id": optional_metadata_int(record["metadata"], "chunk_id"),
            "text": record["document"],
            "page": optional_metadata_int(record["metadata"], "page"),
            "page_label": page_label(record["metadata"]),
            "section_heading": optional_metadata_str(record["metadata"], "section_heading"),
            "location_label": record_location_label(record["metadata"]),
        }
        for record in records
    ]
    return {
        "document": summary,
        "chunks": chunks,
        "total_chunks": total_chunks,
        "offset": offset,
        "limit": limit,
        "next_offset": offset + limit if offset + limit < total_chunks else None,
    }


def _document_summary(
    metadata: dict[str, Any],
    *,
    key: str,
    version: int,
) -> dict[str, Any]:
    return {
        "file_id": str(metadata.get("file_id") or ""),
        "file_name": str(metadata.get("file_name") or metadata.get("source") or "unknown"),
        "document_key": key,
        "document_version": version,
        "file_extension": str(metadata.get("file_extension") or ""),
        "indexed_at": metadata.get("indexed_at"),
        "document_count": metadata_int(metadata, "document_count", 1),
        "page_count": optional_metadata_int(metadata, "page_count"),
        "chunk_count": 0,
        "warning_count": metadata_int(metadata, "warning_count", 0),
    }


def metadata_int(metadata: dict[str, Any], key: str, default: int) -> int:
    value = metadata.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default


def optional_metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata_int(metadata, key, -1)
    return value if value >= 0 else None


def optional_metadata_str(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def page_label(metadata: dict[str, Any]) -> str | None:
    existing = optional_metadata_str(metadata, "page_label")
    if existing:
        return existing
    page = optional_metadata_int(metadata, "page")
    return f"page {page + 1}" if page is not None else None


def record_location_label(metadata: dict[str, Any]) -> str:
    parts = []
    resolved_page_label = page_label(metadata)
    chunk_id = optional_metadata_int(metadata, "chunk_id")
    section_heading = optional_metadata_str(metadata, "section_heading")
    if resolved_page_label:
        parts.append(resolved_page_label)
    if chunk_id is not None:
        parts.append(f"chunk {chunk_id}")
    if section_heading:
        parts.append(section_heading)
    return ", ".join(parts)


def document_preview_sort_key(record: VectorRecord) -> tuple[int, int, str]:
    metadata = record["metadata"]
    chunk_id = optional_metadata_int(metadata, "chunk_id")
    page = optional_metadata_int(metadata, "page")
    return (
        chunk_id if chunk_id is not None else 1_000_000_000,
        page if page is not None else 1_000_000_000,
        record["id"],
    )

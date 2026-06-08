from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from doc_assistant.config.settings import settings
from doc_assistant.ingestion.document_loader import (
    INGEST_WARNINGS_METADATA_KEY,
    file_sha256,
    load_documents,
)
from doc_assistant.models.language_model import build_embedding_model
from doc_assistant.schemas.citation import IngestResult

logger = logging.getLogger(__name__)

_COLLECTION_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_COLLECTION_NAME_LENGTH = 63
_LEGAL_SECTION_PATTERN = re.compile(
    r"^\s*("
    r"第[一二三四五六七八九十百千万\d]+[章节条款项]|"
    r"\d+(?:\.\d+)*[\.)、]?|"
    r"(?:Section|Article|Clause|Schedule|Exhibit|Appendix)\s+[\w\dIVXLC]+"
    r")\s*[:：.-]?\s*(.*)$",
    re.IGNORECASE,
)
ProgressCallback = Callable[[str, int, str | None], None]


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
            separators=[
                "\n第",
                "\nSection ",
                "\nArticle ",
                "\nClause ",
                "\nSchedule ",
                "\nExhibit ",
                "\n\n",
                "\n",
                "。 ",
                "；",
                ". ",
                "; ",
                ", ",
                " ",
                "",
            ],
        )

    def ingest_file(
        self,
        file_path: Path,
        file_name: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IngestResult:
        file_path = Path(file_path)
        display_name = file_name or file_path.name
        file_id = file_sha256(file_path)
        document_key = document_key_for_file_name(display_name)
        existing_records = self._records_for_document_key(document_key)
        active_records = [
            record for record in existing_records if _metadata_is_active(record["metadata"])
        ]
        active_same_content = [
            record for record in active_records if record["metadata"].get("file_id") == file_id
        ]

        if active_same_content:
            version = max(_metadata_int(record["metadata"], "document_version", 1) for record in active_same_content)
            warning = "Document content is already indexed as the active version; existing chunks were reused."
            _report_progress(progress_callback, "completed", 100, warning)
            first_metadata = active_same_content[0]["metadata"]
            return IngestResult(
                file_id=file_id,
                file_name=display_name,
                document_count=_metadata_int(first_metadata, "document_count", 1),
                chunk_count=len(active_same_content),
                document_key=document_key,
                document_version=version,
                file_extension=file_path.suffix.lower(),
                page_count=_optional_metadata_int(first_metadata, "page_count"),
                skipped=True,
                warnings=[warning],
            )

        version = max(
            [_metadata_int(record["metadata"], "document_version", 0) for record in existing_records]
            or [0]
        ) + 1

        _report_progress(progress_callback, "parsing", 20)
        documents = load_documents(file_path)
        warnings = _collect_warnings(documents)
        if not any((document.page_content or "").strip() for document in documents):
            raise ValueError(
                "No extractable text was found in the uploaded document. "
                "For scanned PDFs, enable OCR and install OCR dependencies."
            )

        page_count = _page_count(documents)
        indexed_at = datetime.now(timezone.utc).isoformat()
        replaced_file_ids = sorted(
            {
                str(record["metadata"].get("file_id"))
                for record in active_records
                if record["metadata"].get("file_id")
            }
        )

        for document in documents:
            document.metadata["file_id"] = file_id
            document.metadata["file_name"] = display_name
            document.metadata["tenant_id"] = self.tenant_id
            document.metadata["document_key"] = document_key
            document.metadata["document_version"] = version
            document.metadata["file_extension"] = file_path.suffix.lower()

        _report_progress(progress_callback, "chunking", 40)
        section_documents = _split_legal_sections(documents)
        chunks = self.splitter.split_documents(section_documents)
        ids = []
        for index, chunk in enumerate(chunks):
            chunk.metadata["file_id"] = file_id
            chunk.metadata["file_name"] = display_name
            chunk.metadata["tenant_id"] = self.tenant_id
            chunk.metadata["chunk_id"] = index
            chunk.metadata["document_key"] = document_key
            chunk.metadata["document_version"] = version
            chunk.metadata["active"] = True
            chunk.metadata["indexed_at"] = indexed_at
            chunk.metadata["file_extension"] = file_path.suffix.lower()
            chunk.metadata["document_count"] = len(documents)
            if page_count is not None:
                chunk.metadata["page_count"] = page_count
            if replaced_file_ids:
                chunk.metadata["replaces_file_ids"] = ",".join(replaced_file_ids)
            if warnings:
                chunk.metadata["warning_count"] = len(warnings)
            chunk.metadata = _clean_metadata(chunk.metadata)
            ids.append(f"{document_key}:{file_id}:v{version}:{index}")

        if ids:
            _report_progress(progress_callback, "embedding", 70)
            with self._write_lock:
                try:
                    self.vector_store.delete(ids=ids)
                except Exception as exc:
                    logger.warning(
                        "Failed to clear stale chunks for this ingest version",
                        extra={"file_id": file_id, "tenant_id": self.tenant_id},
                        exc_info=True,
                    )
                    raise RuntimeError("Failed to prepare vector store for document ingest.") from exc

                _report_progress(progress_callback, "indexing", 85)
                self.vector_store.add_documents(chunks, ids=ids)
                try:
                    self._deactivate_records(active_records, superseded_by_file_id=file_id)
                except Exception as exc:
                    warning = (
                        "New document version was indexed, but older versions could not be "
                        f"marked inactive: {exc}"
                    )
                    logger.warning(
                        "Failed to deactivate older document versions",
                        extra={"file_id": file_id, "tenant_id": self.tenant_id},
                        exc_info=True,
                    )
                    warnings.append(warning)
                    _report_progress(progress_callback, "indexing", 92, warning)

        _report_progress(progress_callback, "completed", 100)
        return IngestResult(
            file_id=file_id,
            file_name=display_name,
            document_count=len(documents),
            chunk_count=len(chunks),
            document_key=document_key,
            document_version=version,
            file_extension=file_path.suffix.lower(),
            page_count=page_count,
            skipped=False,
            warnings=warnings,
        )

    def search(self, query: str, k: int | None = None) -> list[Document]:
        top_k = k or settings.top_k
        latest_versions = self._latest_active_versions()
        candidates = self.vector_store.similarity_search(query, k=max(top_k * 5, top_k + 20))
        active_documents = []
        for document in candidates:
            metadata = document.metadata or {}
            if not _metadata_is_active(metadata):
                continue

            document_key = metadata.get("document_key")
            if document_key:
                version = _metadata_int(metadata, "document_version", 1)
                if version < latest_versions.get(str(document_key), version):
                    continue

            active_documents.append(document)
            if len(active_documents) >= top_k:
                break

        return active_documents

    def list_documents(self) -> list[dict[str, Any]]:
        records = self._all_records()
        grouped: dict[str, dict[str, Any]] = {}
        for record in records:
            metadata = record["metadata"]
            if not _metadata_is_active(metadata):
                continue

            key = str(metadata.get("document_key") or metadata.get("file_id") or record["id"])
            version = _metadata_int(metadata, "document_version", 1)
            current = grouped.get(key)
            if current is None or version > int(current["document_version"]):
                grouped[key] = {
                    "file_id": str(metadata.get("file_id") or ""),
                    "file_name": str(metadata.get("file_name") or metadata.get("source") or "unknown"),
                    "document_key": key,
                    "document_version": version,
                    "file_extension": str(metadata.get("file_extension") or ""),
                    "indexed_at": metadata.get("indexed_at"),
                    "document_count": _metadata_int(metadata, "document_count", 1),
                    "page_count": _optional_metadata_int(metadata, "page_count"),
                    "chunk_count": 0,
                    "warning_count": _metadata_int(metadata, "warning_count", 0),
                }

            if grouped[key]["document_version"] == version:
                grouped[key]["chunk_count"] += 1

        return sorted(
            grouped.values(),
            key=lambda item: str(item.get("indexed_at") or ""),
            reverse=True,
        )

    def _all_records(self) -> list[dict[str, Any]]:
        collection = self.vector_store.get(include=["metadatas", "documents"])
        ids = collection.get("ids") or []
        metadatas = collection.get("metadatas") or []
        documents = collection.get("documents") or []
        records = []
        for index, record_id in enumerate(ids):
            records.append(
                {
                    "id": record_id,
                    "metadata": metadatas[index] or {},
                    "document": documents[index] if index < len(documents) else "",
                }
            )
        return records

    def _records_for_document_key(self, document_key: str) -> list[dict[str, Any]]:
        return [
            record
            for record in self._all_records()
            if record["metadata"].get("document_key") == document_key
        ]

    def _latest_active_versions(self) -> dict[str, int]:
        latest: dict[str, int] = {}
        for record in self._all_records():
            metadata = record["metadata"]
            if not _metadata_is_active(metadata):
                continue
            document_key = metadata.get("document_key")
            if not document_key:
                continue
            latest[str(document_key)] = max(
                latest.get(str(document_key), 0),
                _metadata_int(metadata, "document_version", 1),
            )
        return latest

    def _deactivate_records(
        self,
        records: list[dict[str, Any]],
        *,
        superseded_by_file_id: str,
    ) -> None:
        if not records:
            return

        ids = []
        metadatas = []
        for record in records:
            metadata = dict(record["metadata"])
            metadata["active"] = False
            metadata["superseded_by_file_id"] = superseded_by_file_id
            ids.append(record["id"])
            metadatas.append(_clean_metadata(metadata))

        self.vector_store._collection.update(ids=ids, metadatas=metadatas)


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


def document_key_for_file_name(file_name: str) -> str:
    normalized = " ".join(file_name.casefold().strip().split()) or "uploaded_document"
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _report_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    progress: int,
    warning: str | None = None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(stage, max(0, min(progress, 100)), warning)


def _collect_warnings(documents: list[Document]) -> list[str]:
    warnings = []
    seen = set()
    for document in documents:
        raw = document.metadata.get(INGEST_WARNINGS_METADATA_KEY)
        if not isinstance(raw, list):
            continue
        for warning in raw:
            text = str(warning).strip()
            if text and text not in seen:
                warnings.append(text)
                seen.add(text)
    return warnings


def _page_count(documents: list[Document]) -> int | None:
    pages = {
        document.metadata.get("page")
        for document in documents
        if isinstance(document.metadata.get("page"), int)
    }
    if pages:
        return len(pages)
    return len(documents) if documents else None


def _split_legal_sections(documents: list[Document]) -> list[Document]:
    section_documents = []
    for document in documents:
        text = document.page_content or ""
        blocks = _section_blocks(text)
        if len(blocks) <= 1:
            section_documents.append(document)
            continue

        for section_index, (heading, content) in enumerate(blocks):
            metadata = dict(document.metadata)
            metadata["section_index"] = section_index
            if heading:
                metadata["section_heading"] = heading[:180]
            section_documents.append(Document(page_content=content, metadata=metadata))

    return section_documents


def _section_blocks(text: str) -> list[tuple[str | None, str]]:
    blocks: list[tuple[str | None, str]] = []
    current_lines: list[str] = []
    current_heading: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_lines:
                current_lines.append("")
            continue

        heading = _legal_section_heading(line)
        if heading and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                blocks.append((current_heading, content))
            current_lines = [line]
            current_heading = heading
            continue

        if heading and not current_lines:
            current_heading = heading
        current_lines.append(line)

    content = "\n".join(current_lines).strip()
    if content:
        blocks.append((current_heading, content))

    return blocks or [(None, text)]


def _legal_section_heading(line: str) -> str | None:
    if len(line) > 220:
        return None
    match = _LEGAL_SECTION_PATTERN.match(line)
    if not match:
        return None
    return " ".join(line.split())


def _metadata_is_active(metadata: dict[str, Any]) -> bool:
    return metadata.get("active", True) is not False


def _metadata_int(metadata: dict[str, Any], key: str, default: int) -> int:
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


def _optional_metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = _metadata_int(metadata, key, -1)
    return value if value >= 0 else None


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if key == INGEST_WARNINGS_METADATA_KEY or value is None:
            continue
        if isinstance(value, str | int | float | bool):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean

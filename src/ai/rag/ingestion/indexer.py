"""Document parsing, chunk indexing, and version lifecycle management."""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai.config.settings import settings
from ai.rag.ingestion.chunking import split_documents_for_ingestion
from ai.rag.ingestion.loader import (
    INGEST_WARNINGS_METADATA_KEY,
    file_sha256,
    load_documents,
)
from ai.rag.retrieval.backend import clean_metadata, metadata_is_active
from ai.rag.retrieval.catalog import metadata_int, optional_metadata_int
from ai.rag.retrieval.repository import (
    QdrantDocumentRepository,
    VectorRecord,
)
from ai.rag.schemas import IngestResult

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, str | None], None]


class DocumentIngester:
    def __init__(
        self,
        *,
        user_id: str,
        repository: QdrantDocumentRepository,
        splitter: RecursiveCharacterTextSplitter,
        invalidate_cache: Callable[[], None],
    ) -> None:
        self.user_id = user_id
        self.repository = repository
        self.splitter = splitter
        self.invalidate_cache = invalidate_cache
        self._lock = threading.Lock()

    def ingest_file(
        self,
        file_path: Path,
        file_name: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IngestResult:
        # Version selection and activation must be serialized with the writes.
        with self._lock:
            return self._ingest_file(
                Path(file_path),
                file_name=file_name,
                progress_callback=progress_callback,
            )

    def _ingest_file(
        self,
        file_path: Path,
        *,
        file_name: str | None,
        progress_callback: ProgressCallback | None,
    ) -> IngestResult:
        display_name = file_name or file_path.name
        file_id = file_sha256(file_path)
        document_key = document_key_for_file_name(display_name)
        existing_records = self.repository.records_for_document_key(document_key)
        active_records = [
            record for record in existing_records if metadata_is_active(record["metadata"])
        ]
        active_same_content = [
            record for record in active_records if record["metadata"].get("file_id") == file_id
        ]
        if _records_form_complete_version(active_same_content):
            return _skipped_result(
                file_path=file_path,
                file_id=file_id,
                display_name=display_name,
                document_key=document_key,
                records=active_same_content,
                progress_callback=progress_callback,
            )
        if active_same_content:
            invalid_ids = {record["id"] for record in active_same_content}
            self.repository.delete(sorted(invalid_ids))
            active_records = [record for record in active_records if record["id"] not in invalid_ids]

        version = (
            max(
                [
                    metadata_int(record["metadata"], "document_version", 0)
                    for record in existing_records
                ]
                or [0]
            )
            + 1
        )
        report_progress(progress_callback, "parsing", 20)
        documents = load_documents(file_path)
        warnings = collect_warnings(documents)
        if not any((document.page_content or "").strip() for document in documents):
            raise ValueError(
                "No extractable text was found in the uploaded document. "
                "For scanned PDFs, enable OCR and install OCR dependencies."
            )

        page_count = count_pages(documents)
        indexed_at = datetime.now(timezone.utc).isoformat()
        replaced_file_ids = sorted(
            {
                str(record["metadata"].get("file_id"))
                for record in active_records
                if record["metadata"].get("file_id")
            }
        )
        base_metadata = {
            "file_id": file_id,
            "file_name": display_name,
            "user_id": self.user_id,
            "document_key": document_key,
            "document_version": version,
            "file_extension": file_path.suffix.lower(),
        }
        for document in documents:
            document.metadata.update(base_metadata)

        report_progress(progress_callback, "chunking", 40)
        chunks = split_documents_for_ingestion(documents, splitter=self.splitter)
        ids = self._prepare_chunks(
            chunks,
            base_metadata=base_metadata,
            document_count=len(documents),
            page_count=page_count,
            indexed_at=indexed_at,
            replaced_file_ids=replaced_file_ids,
            warning_count=len(warnings),
        )
        if ids:
            self._write_chunks(
                chunks,
                ids,
                active_records=active_records,
                file_id=file_id,
                progress_callback=progress_callback,
                warnings=warnings,
            )

        report_progress(progress_callback, "completed", 100)
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

    def _prepare_chunks(
        self,
        chunks: list[Document],
        *,
        base_metadata: dict[str, str | int],
        document_count: int,
        page_count: int | None,
        indexed_at: str,
        replaced_file_ids: list[str],
        warning_count: int,
    ) -> list[str]:
        ids = []
        for index, chunk in enumerate(chunks):
            chunk.metadata.update(base_metadata)
            chunk.metadata.update(
                {
                    "chunk_id": index,
                    "active": False,
                    "expected_chunk_count": len(chunks),
                    "indexed_at": indexed_at,
                    "document_count": document_count,
                }
            )
            if page_count is not None:
                chunk.metadata["page_count"] = page_count
            if replaced_file_ids:
                chunk.metadata["replaces_file_ids"] = ",".join(replaced_file_ids)
            if warning_count:
                chunk.metadata["warning_count"] = warning_count
            chunk.metadata = clean_metadata(chunk.metadata)
            ids.append(
                f"{base_metadata['document_key']}:{base_metadata['file_id']}:"
                f"v{base_metadata['document_version']}:{index}"
            )
        return ids

    def _write_chunks(
        self,
        chunks: list[Document],
        ids: list[str],
        *,
        active_records: list[VectorRecord],
        file_id: str,
        progress_callback: ProgressCallback | None,
        warnings: list[str],
    ) -> None:
        report_progress(progress_callback, "embedding", 70)
        try:
            self.repository.delete(ids)
        except Exception as exc:
            logger.warning(
                "Failed to clear stale chunks for this ingest version",
                extra={"file_id": file_id, "user_id": self.user_id},
                exc_info=True,
            )
            raise RuntimeError("Failed to prepare vector store for document ingest.") from exc

        report_progress(progress_callback, "indexing", 85)
        try:
            self.repository.embed_and_add(
                chunks,
                ids,
                batch_size=settings.embedding_batch_size,
                max_workers=settings.embedding_max_workers,
            )
            self.repository.set_payload(ids, {"active": True})
        except Exception:
            self._rollback_new_chunks(ids, file_id=file_id)
            raise

        try:
            self._deactivate_records(active_records, superseded_by_file_id=file_id)
        except Exception:
            warning = (
                "New document version was indexed, but older versions could not be marked inactive."
            )
            logger.warning(
                "Failed to deactivate older document versions",
                extra={"file_id": file_id, "user_id": self.user_id},
                exc_info=True,
            )
            warnings.append(warning)
            report_progress(progress_callback, "indexing", 92, warning)
        self.invalidate_cache()

    def _rollback_new_chunks(self, ids: list[str], *, file_id: str) -> None:
        try:
            self.repository.delete(ids)
        except Exception:
            logger.error(
                "Failed to roll back partially indexed document chunks",
                extra={"file_id": file_id, "user_id": self.user_id},
                exc_info=True,
            )

    def _deactivate_records(
        self,
        records: list[VectorRecord],
        *,
        superseded_by_file_id: str,
    ) -> None:
        if not records:
            return
        self.repository.set_payload(
            [record["id"] for record in records],
            {"active": False, "superseded_by_file_id": superseded_by_file_id},
        )


def document_key_for_file_name(file_name: str) -> str:
    normalized = " ".join(file_name.casefold().strip().split()) or "uploaded_document"
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def report_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    progress: int,
    warning: str | None = None,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, max(0, min(progress, 100)), warning)


def collect_warnings(documents: list[Document]) -> list[str]:
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


def count_pages(documents: list[Document]) -> int | None:
    pages = {
        document.metadata.get("page")
        for document in documents
        if isinstance(document.metadata.get("page"), int)
    }
    if pages:
        return len(pages)
    return len(documents) if documents else None


def _records_form_complete_version(records: list[VectorRecord]) -> bool:
    expected_counts = {
        optional_metadata_int(record["metadata"], "expected_chunk_count")
        for record in records
    }
    if len(expected_counts) != 1:
        return False
    expected = next(iter(expected_counts), None)
    if expected is None or expected <= 0 or len(records) != expected:
        return False
    chunk_ids = {
        optional_metadata_int(record["metadata"], "chunk_id") for record in records
    }
    return chunk_ids == set(range(expected))


def _skipped_result(
    *,
    file_path: Path,
    file_id: str,
    display_name: str,
    document_key: str,
    records: list[VectorRecord],
    progress_callback: ProgressCallback | None,
) -> IngestResult:
    version = max(metadata_int(record["metadata"], "document_version", 1) for record in records)
    warning = (
        "Document content is already indexed as the active version; existing chunks were reused."
    )
    report_progress(progress_callback, "completed", 100, warning)
    metadata = records[0]["metadata"]
    return IngestResult(
        file_id=file_id,
        file_name=display_name,
        document_count=metadata_int(metadata, "document_count", 1),
        chunk_count=len(records),
        document_key=document_key,
        document_version=version,
        file_extension=file_path.suffix.lower(),
        page_count=optional_metadata_int(metadata, "page_count"),
        skipped=True,
        warnings=[warning],
    )

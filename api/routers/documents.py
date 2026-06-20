from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from api.dependencies import JobStoreDep, TenantIdDep, VectorStoreDep, require_api_key
from api.jobs import IngestJobRecord, IngestJobStore
from api.schemas.responses import (
    DocumentInfo,
    DocumentListResponse,
    DocumentTextResponse,
    IngestJobResponse,
)
from api.task_queue import submit_background_task
from doc_assistant.config.settings import settings
from doc_assistant.ingestion.document_loader import SUPPORTED_EXTENSIONS, save_uploaded_file
from doc_assistant.retrieval.vector_store import DocumentVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(require_api_key)],
)


@router.post(
    "/ingest",
    response_model=IngestJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and index a document",
)
def ingest_document(
    vector_store: VectorStoreDep,
    tenant_id: TenantIdDep,
    job_store: JobStoreDep,
    file: UploadFile = File(...),
) -> IngestJobResponse:
    suffix = f".{file.filename.rsplit('.', 1)[-1].lower()}" if file.filename and "." in file.filename else ""
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    content = _read_upload_content(file)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    file_name = file.filename or "uploaded_document"
    saved_path = save_uploaded_file(file_name, content, tenant_id=tenant_id)
    job = job_store.create(tenant_id=tenant_id, file_name=file_name, saved_path=saved_path)
    enqueue_ingest_job(job, vector_store, job_store)
    return IngestJobResponse.from_record(job)


@router.get(
    "/jobs/{job_id}",
    response_model=IngestJobResponse,
    summary="Get an ingest job status",
)
def get_ingest_job(
    job_id: str,
    tenant_id: TenantIdDep,
    job_store: JobStoreDep,
) -> IngestJobResponse:
    record = job_store.get(job_id, tenant_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingest job not found.")
    return IngestJobResponse.from_record(record)


@router.get(
    "/text",
    response_model=DocumentTextResponse,
    summary="Get indexed document text chunks for preview",
)
def get_document_text(
    vector_store: VectorStoreDep,
    document_key: str | None = Query(default=None, min_length=1, max_length=128),
    file_id: str | None = Query(default=None, min_length=1, max_length=128),
    document_version: int | None = Query(default=None, ge=1),
) -> DocumentTextResponse:
    try:
        preview = vector_store.get_document_text(
            document_key=document_key,
            file_id=file_id,
            document_version=document_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to load document text: {exc}",
        ) from exc

    if preview is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document text not found.")
    return DocumentTextResponse(**preview)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List indexed documents",
)
def list_documents(vector_store: VectorStoreDep) -> DocumentListResponse:
    try:
        indexed_documents = vector_store.list_documents()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to query vector store: {exc}",
        ) from exc

    docs = [DocumentInfo(**document) for document in indexed_documents]
    return DocumentListResponse(documents=docs, total=len(docs))


def _read_upload_content(file: UploadFile) -> bytes:
    content = bytearray()
    while True:
        chunk = file.file.read(1024 * 1024)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file exceeds {settings.max_upload_bytes} bytes.",
            )
    return bytes(content)


def _run_ingest_job(
    job_id: str,
    saved_path,
    file_name: str,
    vector_store: DocumentVectorStore,
    job_store: IngestJobStore,
) -> None:
    job_store.mark_running(job_id)

    def update_progress(stage: str, progress: int, warning: str | None = None) -> None:
        job_store.update_progress(job_id, stage, progress, warning)

    try:
        result = vector_store.ingest_file(
            saved_path,
            file_name=file_name,
            progress_callback=update_progress,
        )
    except Exception as exc:
        logger.exception("Document ingest job failed", extra={"job_id": job_id})
        job_store.mark_failed(job_id, f"Failed to ingest document: {exc}")
        return

    job_store.mark_succeeded(job_id, result)


def enqueue_ingest_job(
    record: IngestJobRecord,
    vector_store: DocumentVectorStore,
    job_store: IngestJobStore,
) -> bool:
    return submit_background_task(
        f"ingest:{record.job_id}",
        _run_ingest_job,
        record.job_id,
        record.saved_path,
        record.file_name,
        vector_store,
        job_store,
    )

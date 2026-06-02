from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from uuid import uuid4

from doc_assistant.schemas.citation import IngestResult


class IngestJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class IngestJobRecord:
    job_id: str
    tenant_id: str
    file_name: str
    saved_path: Path
    status: IngestJobStatus
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: IngestResult | None = None
    error: str | None = None


class IngestJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, IngestJobRecord] = {}
        self._lock = Lock()

    def create(self, tenant_id: str, file_name: str, saved_path: Path) -> IngestJobRecord:
        record = IngestJobRecord(
            job_id=uuid4().hex,
            tenant_id=tenant_id,
            file_name=file_name,
            saved_path=saved_path,
            status=IngestJobStatus.QUEUED,
            submitted_at=_utc_now(),
        )
        with self._lock:
            self._jobs[record.job_id] = record
        return replace(record)

    def get(self, job_id: str, tenant_id: str) -> IngestJobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.tenant_id != tenant_id:
                return None
            return replace(record)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = IngestJobStatus.RUNNING
            record.started_at = _utc_now()

    def mark_succeeded(self, job_id: str, result: IngestResult) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = IngestJobStatus.SUCCEEDED
            record.completed_at = _utc_now()
            record.result = result
            record.error = None

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = IngestJobStatus.FAILED
            record.completed_at = _utc_now()
            record.error = error


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

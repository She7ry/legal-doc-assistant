from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import sqlite3
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
    stage: str = "uploaded"
    progress: int = 5
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: IngestResult | None = None
    error: str | None = None
    warnings: list[str] | None = None


class IngestJobStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else None
        self._jobs: dict[str, IngestJobRecord] = {}
        self._lock = Lock()
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    def create(self, tenant_id: str, file_name: str, saved_path: Path) -> IngestJobRecord:
        record = IngestJobRecord(
            job_id=uuid4().hex,
            tenant_id=tenant_id,
            file_name=file_name,
            saved_path=saved_path,
            status=IngestJobStatus.QUEUED,
            submitted_at=_utc_now(),
            warnings=[],
        )
        with self._lock:
            if self.db_path:
                self._insert_record(record)
            else:
                self._jobs[record.job_id] = record
        return replace(record)

    def get(self, job_id: str, tenant_id: str) -> IngestJobRecord | None:
        with self._lock:
            record = self._get_record(job_id) if self.db_path else self._jobs.get(job_id)
            if record is None or record.tenant_id != tenant_id:
                return None
            return replace(record)

    def mark_running(self, job_id: str, stage: str = "parsing", progress: int = 15) -> None:
        with self._lock:
            record = self._require_record(job_id)
            record.status = IngestJobStatus.RUNNING
            record.stage = stage
            record.progress = _clamp_progress(progress)
            record.started_at = _utc_now()
            self._save_record(record)

    def update_progress(
        self,
        job_id: str,
        stage: str,
        progress: int,
        warning: str | None = None,
    ) -> None:
        with self._lock:
            record = self._require_record(job_id)
            record.stage = stage
            record.progress = _clamp_progress(progress)
            if warning:
                warnings = list(record.warnings or [])
                if warning not in warnings:
                    warnings.append(warning)
                record.warnings = warnings
            self._save_record(record)

    def mark_succeeded(self, job_id: str, result: IngestResult) -> None:
        with self._lock:
            record = self._require_record(job_id)
            record.status = IngestJobStatus.SUCCEEDED
            record.stage = "completed"
            record.progress = 100
            record.completed_at = _utc_now()
            record.result = result
            record.error = None
            warnings = list(record.warnings or [])
            for warning in result.warnings:
                if warning not in warnings:
                    warnings.append(warning)
            record.warnings = warnings
            self._save_record(record)

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            record = self._require_record(job_id)
            record.status = IngestJobStatus.FAILED
            record.stage = "failed"
            record.progress = 100
            record.completed_at = _utc_now()
            record.error = error
            self._save_record(record)

    def _require_record(self, job_id: str) -> IngestJobRecord:
        record = self._get_record(job_id) if self.db_path else self._jobs[job_id]
        if record is None:
            raise KeyError(job_id)
        return record

    def _save_record(self, record: IngestJobRecord) -> None:
        if self.db_path:
            self._update_record(record)
        else:
            self._jobs[record.job_id] = record

    def _connect(self) -> sqlite3.Connection:
        if self.db_path is None:
            raise RuntimeError("Ingest job store is not configured for SQLite.")
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    saved_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    submitted_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    result_json TEXT,
                    error TEXT,
                    warnings_json TEXT NOT NULL
                )
                """
            )

    def _insert_record(self, record: IngestJobRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ingest_jobs (
                    job_id, tenant_id, file_name, saved_path, status, stage, progress,
                    submitted_at, started_at, completed_at, result_json, error, warnings_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _record_to_row(record),
            )

    def _update_record(self, record: IngestJobRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingest_jobs
                SET tenant_id = ?, file_name = ?, saved_path = ?, status = ?, stage = ?,
                    progress = ?, submitted_at = ?, started_at = ?, completed_at = ?,
                    result_json = ?, error = ?, warnings_json = ?
                WHERE job_id = ?
                """,
                (
                    record.tenant_id,
                    record.file_name,
                    str(record.saved_path),
                    record.status.value,
                    record.stage,
                    record.progress,
                    _datetime_to_db(record.submitted_at),
                    _datetime_to_db(record.started_at),
                    _datetime_to_db(record.completed_at),
                    _result_to_json(record.result),
                    record.error,
                    json.dumps(record.warnings or [], ensure_ascii=False),
                    record.job_id,
                ),
            )

    def _get_record(self, job_id: str) -> IngestJobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingest_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return _row_to_record(row) if row else None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp_progress(value: int) -> int:
    return max(0, min(int(value), 100))


def _record_to_row(record: IngestJobRecord) -> tuple[object, ...]:
    return (
        record.job_id,
        record.tenant_id,
        record.file_name,
        str(record.saved_path),
        record.status.value,
        record.stage,
        record.progress,
        _datetime_to_db(record.submitted_at),
        _datetime_to_db(record.started_at),
        _datetime_to_db(record.completed_at),
        _result_to_json(record.result),
        record.error,
        json.dumps(record.warnings or [], ensure_ascii=False),
    )


def _result_to_json(result: IngestResult | None) -> str | None:
    if result is None:
        return None
    return json.dumps(asdict(result), ensure_ascii=False)


def _result_from_json(value: str | None) -> IngestResult | None:
    if not value:
        return None
    data = json.loads(value)
    if "warnings" not in data or data["warnings"] is None:
        data["warnings"] = []
    return IngestResult(**data)


def _datetime_to_db(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _datetime_from_db(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _row_to_record(row: sqlite3.Row) -> IngestJobRecord:
    return IngestJobRecord(
        job_id=row["job_id"],
        tenant_id=row["tenant_id"],
        file_name=row["file_name"],
        saved_path=Path(row["saved_path"]),
        status=IngestJobStatus(row["status"]),
        stage=row["stage"],
        progress=row["progress"],
        submitted_at=_datetime_from_db(row["submitted_at"]) or _utc_now(),
        started_at=_datetime_from_db(row["started_at"]),
        completed_at=_datetime_from_db(row["completed_at"]),
        result=_result_from_json(row["result_json"]),
        error=row["error"],
        warnings=json.loads(row["warnings_json"] or "[]"),
    )

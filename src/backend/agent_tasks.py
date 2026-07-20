from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from backend.store_helpers import (
    clamp_progress,
    datetime_from_db,
    datetime_to_db,
    json_or_none,
    utc_now,
)


class AgentTaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class AgentTaskEventRecord:
    event_id: int
    task_id: str
    event_type: str
    stage: str
    progress: int
    message: str
    created_at: datetime
    step_id: str | None = None
    payload: dict[str, Any] | None = None


@dataclass
class AgentTaskRecord:
    task_id: str
    user_id: str
    objective: str
    focus_areas: list[str]
    user_role: str
    max_steps: int
    conversation_id: str | None
    status: AgentTaskStatus
    stage: str
    progress: int
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    events: list[AgentTaskEventRecord] | None = None


class AgentTaskStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._lock = Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create(
        self,
        *,
        user_id: str,
        objective: str,
        focus_areas: list[str],
        user_role: str,
        max_steps: int,
        conversation_id: str | None,
    ) -> AgentTaskRecord:
        task_id = uuid4().hex
        record = AgentTaskRecord(
            task_id=task_id,
            user_id=user_id,
            objective=objective,
            focus_areas=list(focus_areas),
            user_role=user_role,
            max_steps=max_steps,
            conversation_id=conversation_id,
            status=AgentTaskStatus.QUEUED,
            stage="queued",
            progress=0,
            submitted_at=utc_now(),
            events=[],
        )
        with self._lock:
            self._insert_record(record)
            event = self._append_event(
                record.task_id,
                event_type="queued",
                stage="queued",
                progress=0,
                message="Agent task queued.",
            )
            record.events = [event]
        return self._copy_record(record)

    def get(self, task_id: str, user_id: str) -> AgentTaskRecord | None:
        with self._lock:
            record = self._get_record(task_id)
            if record is None or record.user_id != user_id:
                return None
            events = self._get_events(task_id, after_event_id=0)
            return self._copy_record(replace(record, events=events))

    def events_after(
        self,
        task_id: str,
        user_id: str,
        after_event_id: int,
    ) -> list[AgentTaskEventRecord] | None:
        with self._lock:
            record = self._get_record(task_id)
            if record is None or record.user_id != user_id:
                return None
            return [replace(event) for event in self._get_events(task_id, after_event_id)]

    def list_restartable(self, *, limit: int = 100) -> list[AgentTaskRecord]:
        with self._lock:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT * FROM agent_tasks
                    WHERE status = ?
                    ORDER BY submitted_at ASC
                    LIMIT ?
                    """,
                    (AgentTaskStatus.QUEUED.value, max(1, min(limit, 500))),
                ).fetchall()
            return [self._copy_record(_row_to_record(row)) for row in rows]

    def claim(self, task_id: str) -> bool:
        started_at = utc_now()
        with self._lock:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE agent_tasks
                    SET status = ?, stage = ?, progress = ?, started_at = ?
                    WHERE task_id = ? AND status = ?
                    """,
                    (
                        AgentTaskStatus.RUNNING.value,
                        "answering",
                        5,
                        datetime_to_db(started_at),
                        task_id,
                        AgentTaskStatus.QUEUED.value,
                    ),
                )
                if cursor.rowcount != 1:
                    return False
                self._insert_event_with_connection(
                    connection,
                    task_id,
                    event_type="running",
                    stage="answering",
                    progress=5,
                    message="Agent task started.",
                    step_id=None,
                    payload=None,
                )
            return True

    def update_progress(
        self,
        task_id: str,
        *,
        event_type: str,
        stage: str,
        progress: int,
        message: str,
        step_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            record = self._require_record(task_id)
            record.stage = stage
            record.progress = clamp_progress(progress)
            self._save_record(record)
            self._append_event(
                task_id,
                event_type=event_type,
                stage=stage,
                progress=record.progress,
                message=message,
                step_id=step_id,
                payload=payload,
            )

    def mark_succeeded(
        self,
        task_id: str,
        result: dict[str, Any],
    ) -> None:
        with self._lock:
            record = self._require_record(task_id)
            record.status = AgentTaskStatus.SUCCEEDED
            record.stage = "completed"
            record.progress = 100
            record.completed_at = utc_now()
            record.result = result
            record.error = None
            self._save_record(record)
            self._append_event(
                task_id,
                event_type="succeeded",
                stage="completed",
                progress=100,
                message="Agent task completed.",
                payload={"status": result.get("status")},
            )

    def mark_needs_input(self, task_id: str, questions: list[str]) -> None:
        with self._lock:
            record = self._require_record(task_id)
            record.status = AgentTaskStatus.NEEDS_INPUT
            record.stage = "needs_input"
            record.progress = 0
            record.result = None
            record.error = None
            self._save_record(record)
            self._append_event(
                task_id,
                event_type="needs_input",
                stage="needs_input",
                progress=0,
                message="需要补充信息后再运行 Agent 任务。",
                payload={"questions": questions[:3]},
            )

    def resume_with_input(
        self,
        task_id: str,
        *,
        objective: str,
        focus_areas: list[str],
        user_role: str,
        max_steps: int,
        conversation_id: str | None,
        clarification_answers: list[str],
    ) -> AgentTaskRecord:
        with self._lock:
            record = self._require_record(task_id)
            record.objective = objective
            record.focus_areas = list(focus_areas)
            record.user_role = user_role
            record.max_steps = max_steps
            record.conversation_id = conversation_id
            record.status = AgentTaskStatus.QUEUED
            record.stage = "queued"
            record.progress = 0
            record.started_at = None
            record.completed_at = None
            record.result = None
            record.error = None
            self._save_record(record)
            self._append_event(
                task_id,
                event_type="input_received",
                stage="queued",
                progress=0,
                message="Received supplemental input for Agent task.",
                payload={"answers": clarification_answers[:6]},
            )
            self._append_event(
                task_id,
                event_type="queued",
                stage="queued",
                progress=0,
                message="Agent task re-queued after supplemental input.",
            )
            events = self._get_events(task_id, after_event_id=0)
            return self._copy_record(replace(record, events=events))

    def mark_failed(self, task_id: str, error: str) -> None:
        with self._lock:
            record = self._require_record(task_id)
            record.status = AgentTaskStatus.FAILED
            record.stage = "failed"
            record.progress = 100
            record.completed_at = utc_now()
            record.error = error
            self._save_record(record)
            self._append_event(
                task_id,
                event_type="failed",
                stage="failed",
                progress=100,
                message=error,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_record(self, task_id: str) -> AgentTaskRecord:
        record = self._get_record(task_id)
        if record is None:
            raise KeyError(task_id)
        return record

    def _save_record(self, record: AgentTaskRecord) -> None:
        self._update_record(record)

    def _append_event(
        self,
        task_id: str,
        *,
        event_type: str,
        stage: str,
        progress: int,
        message: str,
        step_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AgentTaskEventRecord:
        return self._insert_event(
            task_id,
            event_type=event_type,
            stage=stage,
            progress=progress,
            message=message,
            step_id=step_id,
            payload=payload,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    task_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    focus_areas_json TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    max_steps INTEGER NOT NULL,
                    conversation_id TEXT,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    submitted_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    result_json TEXT,
                    error TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_task_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    step_id TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES agent_tasks(task_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_task_events_task_event
                ON agent_task_events(task_id, event_id)
                """
            )

    def _insert_record(self, record: AgentTaskRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_tasks (
                    task_id, user_id, objective, focus_areas_json, user_role,
                    max_steps, conversation_id, status, stage, progress, submitted_at,
                    started_at, completed_at, result_json, error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _record_to_row(record),
            )

    def _update_record(self, record: AgentTaskRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE agent_tasks
                SET user_id = ?, objective = ?, focus_areas_json = ?,
                    user_role = ?, max_steps = ?, conversation_id = ?,
                    status = ?, stage = ?, progress = ?, submitted_at = ?, started_at = ?,
                    completed_at = ?, result_json = ?, error = ?
                WHERE task_id = ?
                """,
                (
                    record.user_id,
                    record.objective,
                    json.dumps(record.focus_areas, ensure_ascii=False),
                    record.user_role,
                    record.max_steps,
                    record.conversation_id,
                    record.status.value,
                    record.stage,
                    record.progress,
                    datetime_to_db(record.submitted_at),
                    datetime_to_db(record.started_at),
                    datetime_to_db(record.completed_at),
                    json_or_none(record.result),
                    record.error,
                    record.task_id,
                ),
            )

    def _get_record(self, task_id: str) -> AgentTaskRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return _row_to_record(row) if row else None

    def _insert_event(
        self,
        task_id: str,
        *,
        event_type: str,
        stage: str,
        progress: int,
        message: str,
        step_id: str | None,
        payload: dict[str, Any] | None,
    ) -> AgentTaskEventRecord:
        with self._connect() as connection:
            return self._insert_event_with_connection(
                connection,
                task_id,
                event_type=event_type,
                stage=stage,
                progress=progress,
                message=message,
                step_id=step_id,
                payload=payload,
            )

    def _insert_event_with_connection(
        self,
        connection: sqlite3.Connection,
        task_id: str,
        *,
        event_type: str,
        stage: str,
        progress: int,
        message: str,
        step_id: str | None,
        payload: dict[str, Any] | None,
    ) -> AgentTaskEventRecord:
        created_at = utc_now()
        progress = clamp_progress(progress)
        cursor = connection.execute(
            """
            INSERT INTO agent_task_events (
                task_id, event_type, stage, progress, message, step_id,
                payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                event_type,
                stage,
                progress,
                message,
                step_id,
                json.dumps(payload or {}, ensure_ascii=False),
                datetime_to_db(created_at),
            ),
        )
        return AgentTaskEventRecord(
            event_id=int(cursor.lastrowid),
            task_id=task_id,
            event_type=event_type,
            stage=stage,
            progress=progress,
            message=message,
            step_id=step_id,
            payload=payload or {},
            created_at=created_at,
        )

    def _get_events(self, task_id: str, after_event_id: int) -> list[AgentTaskEventRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM agent_task_events
                WHERE task_id = ? AND event_id > ?
                ORDER BY event_id ASC
                """,
                (task_id, after_event_id),
            ).fetchall()
        return [_event_row_to_record(row) for row in rows]

    @staticmethod
    def _copy_record(record: AgentTaskRecord) -> AgentTaskRecord:
        return replace(
            record,
            focus_areas=list(record.focus_areas),
            result=dict(record.result) if record.result else None,
            events=[replace(event) for event in record.events or []],
        )


def _record_to_row(record: AgentTaskRecord) -> tuple[object, ...]:
    return (
        record.task_id,
        record.user_id,
        record.objective,
        json.dumps(record.focus_areas, ensure_ascii=False),
        record.user_role,
        record.max_steps,
        record.conversation_id,
        record.status.value,
        record.stage,
        record.progress,
        datetime_to_db(record.submitted_at),
        datetime_to_db(record.started_at),
        datetime_to_db(record.completed_at),
        json_or_none(record.result),
        record.error,
    )


def _row_to_record(row: sqlite3.Row) -> AgentTaskRecord:
    return AgentTaskRecord(
        task_id=row["task_id"],
        user_id=row["user_id"],
        objective=row["objective"],
        focus_areas=json.loads(row["focus_areas_json"] or "[]"),
        user_role=row["user_role"],
        max_steps=row["max_steps"],
        conversation_id=row["conversation_id"],
        status=AgentTaskStatus(row["status"]),
        stage=row["stage"],
        progress=row["progress"],
        submitted_at=datetime_from_db(row["submitted_at"]) or utc_now(),
        started_at=datetime_from_db(row["started_at"]),
        completed_at=datetime_from_db(row["completed_at"]),
        result=json.loads(row["result_json"]) if row["result_json"] else None,
        error=row["error"],
        events=[],
    )


def _event_row_to_record(row: sqlite3.Row) -> AgentTaskEventRecord:
    return AgentTaskEventRecord(
        event_id=row["event_id"],
        task_id=row["task_id"],
        event_type=row["event_type"],
        stage=row["stage"],
        progress=row["progress"],
        message=row["message"],
        step_id=row["step_id"],
        payload=json.loads(row["payload_json"] or "{}"),
        created_at=datetime_from_db(row["created_at"]) or utc_now(),
    )

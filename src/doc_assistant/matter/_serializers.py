"""sqlite3.Row → dataclass 序列化转换函数。"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from doc_assistant.matter._utils import _datetime_from_db, _utc_now
from doc_assistant.matter.schemas import (
    MatterArtifactRecord,
    MatterEventRecord,
    MatterFindingRecord,
    MatterRecord,
)


def _row_to_matter(row: sqlite3.Row) -> MatterRecord:
    return MatterRecord(
        matter_id=row["matter_id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        title=row["title"],
        status=row["status"],
        matter_profile=json.loads(row["matter_profile_json"] or "{}"),
        source_task_id=row["source_task_id"],
        latest_task_id=row["latest_task_id"],
        created_at=_datetime_from_db(row["created_at"]) or _utc_now(),
        updated_at=_datetime_from_db(row["updated_at"]) or _utc_now(),
    )


def _row_to_artifact(row: sqlite3.Row) -> MatterArtifactRecord:
    return MatterArtifactRecord(
        artifact_id=row["artifact_id"],
        matter_id=row["matter_id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        artifact_type=row["artifact_type"],
        title=row["title"],
        summary=row["summary"],
        items=json.loads(row["items_json"] or "[]"),
        source_finding_ids=json.loads(row["source_finding_ids_json"] or "[]"),
        citations=json.loads(row["citations_json"] or "[]"),
        metadata=json.loads(row["metadata_json"] or "{}"),
        source_task_id=row["source_task_id"],
        version=int(row["version"]),
        status=row["status"],
        created_at=_datetime_from_db(row["created_at"]) or _utc_now(),
        updated_at=_datetime_from_db(row["updated_at"]) or _utc_now(),
    )


def _row_to_finding(row: sqlite3.Row) -> MatterFindingRecord:
    return MatterFindingRecord(
        finding_id=row["finding_id"],
        matter_id=row["matter_id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        category=row["category"],
        severity=row["severity"],
        summary=row["summary"],
        recommended_action=row["recommended_action"],
        citations=json.loads(row["citations_json"] or "[]"),
        source_step_id=row["source_step_id"],
        clause_reference=row["clause_reference"],
        evidence_coverage=row["evidence_coverage"],
        support_level=row["support_level"],
        unsupported_reason=row["unsupported_reason"],
        source_quote=row["source_quote"],
        location_label=row["location_label"],
        needs_human_review=bool(row["needs_human_review"]),
        human_review_status=row["human_review_status"],
        status=row["status"],
        metadata=json.loads(row["metadata_json"] or "{}"),
        source_task_id=row["source_task_id"],
        created_at=_datetime_from_db(row["created_at"]) or _utc_now(),
        updated_at=_datetime_from_db(row["updated_at"]) or _utc_now(),
    )


def _row_to_event(row: sqlite3.Row) -> MatterEventRecord:
    return MatterEventRecord(
        event_id=row["event_id"],
        matter_id=row["matter_id"],
        tenant_id=row["tenant_id"],
        user_id=row["user_id"],
        event_type=row["event_type"],
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        old_value=_json_value(row["old_value_json"]),
        new_value=_json_value(row["new_value_json"]),
        actor=row["actor"],
        created_at=_datetime_from_db(row["created_at"]) or _utc_now(),
    )


def _json_value(value: str | None) -> Any:
    """安全解析 JSON 字符串，解析失败则返回原值。"""
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value

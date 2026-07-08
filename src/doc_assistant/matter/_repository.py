"""SQLite statements and record mapping used by MatterStore."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime
from typing import Any
from uuid import uuid4

from doc_assistant.matter import _sql as sql
from doc_assistant.matter._domain import _finding_status
from doc_assistant.matter._serializers import (
    _row_to_artifact,
    _row_to_event,
    _row_to_finding,
    _row_to_matter,
)
from doc_assistant.matter._utils import (
    _as_bool,
    _as_dict,
    _as_dict_list,
    _as_text_list,
    _clean_text,
)
from doc_assistant.matter.schemas import (
    MatterArtifactRecord,
    MatterEventRecord,
    MatterFindingRecord,
    MatterRecord,
)


class MatterRepository:
    """Perform persistence operations on a caller-owned transaction connection."""

    def get_matter_row(
        self, connection: sqlite3.Connection, matter_id: str, tenant_id: str, user_id: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            sql.SELECT_MATTER_BY_IDS, (matter_id, tenant_id, user_id),
        ).fetchone()

    def get_matter(
        self,
        connection: sqlite3.Connection,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        *,
        include_artifacts: bool = False,
        include_findings: bool = False,
    ) -> MatterRecord | None:
        row = self.get_matter_row(connection, matter_id, tenant_id, user_id)
        if row is None:
            return None
        record = _row_to_matter(row)
        artifacts = self.list_artifacts(connection, matter_id, tenant_id, user_id) if include_artifacts else None
        findings = self.list_findings(connection, matter_id, tenant_id, user_id) if include_findings else None
        return replace(record, artifacts=artifacts, findings=findings)

    def list_matters(
        self, connection: sqlite3.Connection, tenant_id: str, user_id: str, limit: int,
    ) -> list[MatterRecord]:
        rows = connection.execute(
            sql.SELECT_MATTERS_BY_USER, (tenant_id, user_id, max(1, min(limit, 200))),
        ).fetchall()
        return [_row_to_matter(row) for row in rows]

    def upsert_matter(
        self,
        connection: sqlite3.Connection,
        *,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        title: str,
        status: str,
        profile: dict[str, Any],
        task_id: str,
        created_at: datetime,
        updated_at: datetime,
    ) -> None:
        connection.execute(
            sql.UPSERT_MATTER,
            (
                matter_id,
                tenant_id,
                user_id,
                title,
                status,
                json.dumps(profile, ensure_ascii=False),
                task_id,
                task_id,
                created_at.isoformat(),
                updated_at.isoformat(),
            ),
        )

    def update_matter_profile(
        self,
        connection: sqlite3.Connection,
        *,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        status: str,
        profile: dict[str, Any],
        now: datetime,
    ) -> None:
        connection.execute(
            sql.UPDATE_MATTER_STATUS_AND_PROFILE,
            (
                status,
                json.dumps(profile, ensure_ascii=False),
                now.isoformat(),
                matter_id,
                tenant_id,
                user_id,
            ),
        )

    def touch_matter(
        self,
        connection: sqlite3.Connection,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        now: datetime,
    ) -> None:
        connection.execute(
            sql.UPDATE_MATTER_UPDATED_AT,
            (now.isoformat(), matter_id, tenant_id, user_id),
        )

    def get_artifact(
        self,
        connection: sqlite3.Connection,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        artifact_id: str,
    ) -> MatterArtifactRecord | None:
        row = connection.execute(
            sql.SELECT_ARTIFACT_BY_ID, (matter_id, tenant_id, user_id, artifact_id),
        ).fetchone()
        return _row_to_artifact(row) if row is not None else None

    def list_artifacts(
        self, connection: sqlite3.Connection, matter_id: str, tenant_id: str, user_id: str,
    ) -> list[MatterArtifactRecord]:
        rows = connection.execute(
            sql.SELECT_ARTIFACTS_BY_MATTER, (matter_id, tenant_id, user_id),
        ).fetchall()
        return [_row_to_artifact(row) for row in rows]

    def list_artifact_versions(
        self,
        connection: sqlite3.Connection,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        artifact_id: str,
    ) -> list[MatterArtifactRecord]:
        rows = connection.execute(
            sql.SELECT_ARTIFACT_VERSIONS, (matter_id, tenant_id, user_id, artifact_id),
        ).fetchall()
        return [_row_to_artifact(row) for row in rows]

    def upsert_artifact(
        self,
        connection: sqlite3.Connection,
        *,
        tenant_id: str,
        user_id: str,
        matter_id: str,
        source_task_id: str,
        artifact: dict[str, Any],
        existing: MatterArtifactRecord | None,
        now: datetime,
    ) -> MatterArtifactRecord | None:
        artifact_id = _clean_text(artifact.get("artifact_id")) or _clean_text(
            artifact.get("artifact_type")
        )
        if not artifact_id:
            return None
        version = existing.version + 1 if existing else 1
        created_at = existing.created_at if existing else now
        connection.execute(
            sql.UPSERT_ARTIFACT,
            (
                artifact_id,
                matter_id,
                tenant_id,
                user_id,
                _clean_text(artifact.get("artifact_type")) or "custom",
                _clean_text(artifact.get("title")) or artifact_id,
                _clean_text(artifact.get("summary")),
                json.dumps(_as_dict_list(artifact.get("items")), ensure_ascii=False),
                json.dumps(_as_text_list(artifact.get("source_finding_ids")), ensure_ascii=False),
                json.dumps(_as_text_list(artifact.get("citations")), ensure_ascii=False),
                json.dumps(_as_dict(artifact.get("metadata")), ensure_ascii=False),
                source_task_id,
                version,
                _clean_text(artifact.get("status")) or "active",
                created_at.isoformat(),
                now.isoformat(),
            ),
        )
        self.record_artifact_version(connection, matter_id, tenant_id, user_id, artifact_id)
        return self.get_artifact(connection, matter_id, tenant_id, user_id, artifact_id)

    def update_artifact(
        self,
        connection: sqlite3.Connection,
        *,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        artifact_id: str,
        values: dict[str, Any],
        now: datetime,
    ) -> MatterArtifactRecord:
        connection.execute(
            sql.UPDATE_ARTIFACT,
            (
                values["title"],
                values["summary"],
                json.dumps(values["items"], ensure_ascii=False),
                json.dumps(values["metadata"], ensure_ascii=False),
                values["version"],
                values["status"],
                now.isoformat(),
                matter_id,
                tenant_id,
                user_id,
                artifact_id,
            ),
        )
        self.record_artifact_version(connection, matter_id, tenant_id, user_id, artifact_id)
        updated = self.get_artifact(connection, matter_id, tenant_id, user_id, artifact_id)
        if updated is None:
            raise KeyError(artifact_id)
        return updated

    def record_artifact_version(
        self,
        connection: sqlite3.Connection,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        artifact_id: str,
    ) -> None:
        connection.execute(
            sql.INSERT_CURRENT_ARTIFACT_VERSION,
            (matter_id, tenant_id, user_id, artifact_id),
        )

    def get_finding(
        self,
        connection: sqlite3.Connection,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        finding_id: str,
    ) -> MatterFindingRecord | None:
        row = connection.execute(
            sql.SELECT_FINDING_ROW_BY_ID, (tenant_id, user_id, matter_id, finding_id),
        ).fetchone()
        return _row_to_finding(row) if row is not None else None

    def list_findings(
        self, connection: sqlite3.Connection, matter_id: str, tenant_id: str, user_id: str,
    ) -> list[MatterFindingRecord]:
        rows = connection.execute(
            sql.SELECT_FINDINGS_BY_MATTER, (matter_id, tenant_id, user_id),
        ).fetchall()
        return [_row_to_finding(row) for row in rows]

    def upsert_finding(
        self,
        connection: sqlite3.Connection,
        *,
        tenant_id: str,
        user_id: str,
        matter_id: str,
        source_task_id: str,
        finding: dict[str, Any],
        existing: MatterFindingRecord | None,
        metadata: dict[str, Any],
        now: datetime,
    ) -> MatterFindingRecord | None:
        finding_id = _clean_text(finding.get("finding_id"))
        if not finding_id:
            return None
        incoming_human_status = _clean_text(finding.get("human_review_status")) or "pending"
        human_review_status = (
            existing.human_review_status
            if existing and existing.human_review_status in {"approved", "waived", "resolved"}
            else incoming_human_status
        )
        needs_human_review = _as_bool(finding.get("needs_human_review"), default=True)
        evidence_coverage = _clean_text(finding.get("evidence_coverage")) or "missing"
        status_value = _finding_status(
            needs_human_review=needs_human_review,
            human_review_status=human_review_status,
            evidence_coverage=evidence_coverage,
        )
        created_at = existing.created_at if existing else now
        connection.execute(
            sql.UPSERT_FINDING,
            (
                finding_id,
                matter_id,
                tenant_id,
                user_id,
                _clean_text(finding.get("category")) or "Finding",
                _clean_text(finding.get("severity")) or "Needs human review",
                _clean_text(finding.get("summary")),
                _clean_text(finding.get("recommended_action")),
                json.dumps(_as_text_list(finding.get("citations")), ensure_ascii=False),
                _clean_text(finding.get("source_step_id")),
                _clean_text(finding.get("clause_reference")),
                evidence_coverage,
                _clean_text(finding.get("support_level")) or "missing",
                _clean_text(finding.get("unsupported_reason")),
                _clean_text(finding.get("source_quote")),
                _clean_text(finding.get("location_label")),
                1 if needs_human_review else 0,
                human_review_status,
                status_value,
                json.dumps(metadata, ensure_ascii=False),
                source_task_id,
                created_at.isoformat(),
                now.isoformat(),
            ),
        )
        return self.get_finding(connection, matter_id, tenant_id, user_id, finding_id)

    def update_finding_reviews(
        self,
        connection: sqlite3.Connection,
        *,
        tenant_id: str,
        user_id: str,
        matter_id: str,
        finding_ids: list[str],
        human_review_status: str,
        decision: dict[str, Any],
        now: datetime,
    ) -> list[tuple[MatterFindingRecord, MatterFindingRecord]]:
        changes: list[tuple[MatterFindingRecord, MatterFindingRecord]] = []
        for finding_id in finding_ids:
            old = self.get_finding(connection, matter_id, tenant_id, user_id, finding_id)
            if old is None:
                continue
            metadata = dict(old.metadata)
            decisions = _as_dict_list(metadata.get("decisions"))
            metadata["decisions"] = [*decisions[-19:], decision]
            metadata["last_decision"] = decision
            status_value = _finding_status(
                needs_human_review=old.needs_human_review,
                human_review_status=human_review_status,
                evidence_coverage=old.evidence_coverage,
            )
            connection.execute(
                sql.UPDATE_FINDING_REVIEW,
                (
                    human_review_status,
                    status_value,
                    json.dumps(metadata, ensure_ascii=False),
                    now.isoformat(),
                    tenant_id,
                    user_id,
                    matter_id,
                    finding_id,
                ),
            )
            new = self.get_finding(connection, matter_id, tenant_id, user_id, finding_id)
            if new is not None:
                changes.append((old, new))
        return changes

    def list_events(
        self,
        connection: sqlite3.Connection,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        limit: int,
    ) -> list[MatterEventRecord]:
        rows = connection.execute(
            sql.SELECT_EVENTS_BY_MATTER,
            (matter_id, tenant_id, user_id, max(1, min(limit, 500))),
        ).fetchall()
        return [_row_to_event(row) for row in rows]

    def emit_event(
        self,
        connection: sqlite3.Connection,
        *,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        event_type: str,
        entity_type: str,
        entity_id: str,
        old_value: Any,
        new_value: Any,
        actor: str,
        created_at: datetime,
    ) -> None:
        connection.execute(
            sql.INSERT_EVENT,
            (
                uuid4().hex,
                matter_id,
                tenant_id,
                user_id,
                event_type,
                entity_type,
                entity_id,
                json.dumps(old_value, ensure_ascii=False, default=str),
                json.dumps(new_value, ensure_ascii=False, default=str),
                _clean_text(actor) or "system",
                created_at.isoformat(),
            ),
        )

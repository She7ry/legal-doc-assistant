"""Public facade for persisted legal matters."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from doc_assistant.config.settings import settings
from doc_assistant.matter._database import MatterDatabase
from doc_assistant.matter._domain import (
    _apply_confirmation_gate_decision,
    _artifact_snapshot,
    _build_artifact_update,
    _build_formal_report_artifact,
    _formal_report_blockers,
    _human_review_status_for_gate_status,
    _matter_status,
    _matter_title,
    _merge_agent_artifact,
    _merge_agent_profile,
    _merge_finding_metadata,
)
from doc_assistant.matter._repository import MatterRepository
from doc_assistant.matter._utils import (
    _as_dict_list,
    _clean_text,
)
from doc_assistant.matter.schemas import (
    MatterArtifactRecord,
    MatterEventRecord,
    MatterFindingRecord,
    MatterRecord,
)


class MatterStore:
    """Coordinate matter use cases while delegating domain and SQLite details."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or settings.matter_db_path)
        self._database = MatterDatabase(self.db_path)
        self._repository = MatterRepository()

    def upsert_from_agent_result(
        self,
        *,
        tenant_id: str,
        user_id: str,
        matter_id: str,
        result: dict[str, Any],
    ) -> MatterRecord:
        """Persist one immutable Agent task result, preserving later human decisions."""
        incoming_profile = result.get("matter_profile")
        if not isinstance(incoming_profile, dict):
            incoming_profile = {
                "matter_id": matter_id,
                "open_questions": ["Matter profile was not produced."],
            }
        incoming_profile = {**incoming_profile, "matter_id": matter_id}
        task_id = _clean_text(result.get("task_id")) or matter_id
        now = datetime.now(timezone.utc)

        with self._database.connection(write=True) as connection:
            existing = self._repository.get_matter(
                connection, matter_id, tenant_id, user_id,
            )
            if existing is not None and existing.latest_task_id == task_id:
                loaded = self._repository.get_matter(
                    connection,
                    matter_id,
                    tenant_id,
                    user_id,
                    include_artifacts=True,
                    include_findings=True,
                )
                if loaded is None:
                    raise RuntimeError("Matter disappeared during Agent result retry.")
                return loaded

            profile = _merge_agent_profile(
                incoming_profile,
                existing.matter_profile if existing else None,
            )
            self._repository.upsert_matter(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                title=_matter_title(result, profile),
                status=_matter_status(profile),
                profile=profile,
                task_id=task_id,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            self._repository.emit_event(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                event_type="matter_profile_upserted",
                entity_type="matter",
                entity_id=matter_id,
                old_value=existing.matter_profile if existing else None,
                new_value=profile,
                actor=user_id,
                created_at=now,
            )
            self._sync_artifacts(
                connection,
                tenant_id=tenant_id,
                user_id=user_id,
                matter_id=matter_id,
                task_id=task_id,
                artifacts=_as_dict_list(result.get("artifacts")),
                now=now,
            )
            self._sync_findings(
                connection,
                tenant_id=tenant_id,
                user_id=user_id,
                matter_id=matter_id,
                task_id=task_id,
                findings=_as_dict_list(result.get("findings")),
                now=now,
            )

        loaded = self.get(
            matter_id, tenant_id, user_id, include_artifacts=True, include_findings=True,
        )
        if loaded is None:
            raise RuntimeError("Matter was not persisted.")
        return loaded

    def get(
        self,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        *,
        include_artifacts: bool = False,
        include_findings: bool = False,
    ) -> MatterRecord | None:
        with self._database.connection() as connection:
            return self._repository.get_matter(
                connection,
                matter_id,
                tenant_id,
                user_id,
                include_artifacts=include_artifacts,
                include_findings=include_findings,
            )

    def list(
        self, tenant_id: str, user_id: str, *, limit: int = 50,
    ) -> list[MatterRecord]:
        with self._database.connection() as connection:
            return self._repository.list_matters(connection, tenant_id, user_id, limit)

    def list_artifacts(
        self, matter_id: str, tenant_id: str, user_id: str,
    ) -> list[MatterArtifactRecord] | None:
        with self._database.connection() as connection:
            if self._repository.get_matter_row(connection, matter_id, tenant_id, user_id) is None:
                return None
            return self._repository.list_artifacts(connection, matter_id, tenant_id, user_id)

    def list_artifact_versions(
        self,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        artifact_id: str,
    ) -> list[MatterArtifactRecord] | None:
        with self._database.connection() as connection:
            if self._repository.get_matter_row(connection, matter_id, tenant_id, user_id) is None:
                return None
            return self._repository.list_artifact_versions(
                connection, matter_id, tenant_id, user_id, _clean_text(artifact_id),
            )

    def list_findings(
        self, matter_id: str, tenant_id: str, user_id: str,
    ) -> list[MatterFindingRecord] | None:
        with self._database.connection() as connection:
            if self._repository.get_matter_row(connection, matter_id, tenant_id, user_id) is None:
                return None
            return self._repository.list_findings(connection, matter_id, tenant_id, user_id)

    def list_events(
        self,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        *,
        limit: int = 100,
    ) -> list[MatterEventRecord] | None:
        with self._database.connection() as connection:
            if self._repository.get_matter_row(connection, matter_id, tenant_id, user_id) is None:
                return None
            return self._repository.list_events(
                connection, matter_id, tenant_id, user_id, limit,
            )

    def update_artifact(
        self,
        *,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        artifact_id: str,
        title: str | None = None,
        summary: str | None = None,
        items: list[dict[str, Any]] | None = None,
        status: str | None = None,
        note: str | None = None,
        updated_by: str | None = None,
    ) -> MatterRecord | None:
        now = datetime.now(timezone.utc)
        normalized_id = _clean_text(artifact_id)
        actor = _clean_text(updated_by) or user_id
        with self._database.connection(write=True) as connection:
            if self._repository.get_matter_row(connection, matter_id, tenant_id, user_id) is None:
                return None
            old_artifact = self._repository.get_artifact(
                connection, matter_id, tenant_id, user_id, normalized_id,
            )
            if old_artifact is None:
                raise KeyError(normalized_id)
            values, old_value, new_value = _build_artifact_update(
                old_artifact,
                title=title,
                summary=summary,
                items=items,
                status=status,
                note=note,
                actor=actor,
                now=now,
            )
            self._repository.update_artifact(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                artifact_id=normalized_id,
                values=values,
                now=now,
            )
            self._repository.touch_matter(connection, matter_id, tenant_id, user_id, now)
            self._repository.emit_event(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                event_type="artifact_updated",
                entity_type="artifact",
                entity_id=normalized_id,
                old_value=old_value,
                new_value=new_value,
                actor=actor,
                created_at=now,
            )
        return self.get(
            matter_id, tenant_id, user_id, include_artifacts=True, include_findings=True,
        )

    def update_confirmation_gate(
        self,
        *,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        gate_id: str,
        status: str,
        note: str | None = None,
        confirmed_value: str | None = None,
        decided_by: str | None = None,
    ) -> MatterRecord | None:
        now = datetime.now(timezone.utc)
        normalized_gate_id = _clean_text(gate_id)
        normalized_status = _clean_text(status)
        actor = _clean_text(decided_by) or user_id
        with self._database.connection(write=True) as connection:
            matter = self._repository.get_matter(connection, matter_id, tenant_id, user_id)
            if matter is None:
                return None
            profile, old_gate, new_gate, finding_ids, decision = (
                _apply_confirmation_gate_decision(
                    matter.matter_profile,
                    gate_id=normalized_gate_id,
                    status=normalized_status,
                    note=note,
                    confirmed_value=confirmed_value,
                    decided_by=actor,
                    now=now,
                )
            )
            finding_changes = self._repository.update_finding_reviews(
                connection,
                tenant_id=tenant_id,
                user_id=user_id,
                matter_id=matter_id,
                finding_ids=finding_ids,
                human_review_status=_human_review_status_for_gate_status(normalized_status),
                decision=decision,
                now=now,
            )
            self._repository.update_matter_profile(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                status=_matter_status(profile),
                profile=profile,
                now=now,
            )
            self._repository.emit_event(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                event_type="confirmation_gate_updated",
                entity_type="confirmation_gate",
                entity_id=normalized_gate_id,
                old_value=old_gate,
                new_value=new_gate,
                actor=actor,
                created_at=now,
            )
            for old_finding, new_finding in finding_changes:
                self._repository.emit_event(
                    connection,
                    matter_id=matter_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    event_type="finding_review_updated_from_gate",
                    entity_type="finding",
                    entity_id=new_finding.finding_id,
                    old_value=asdict(old_finding),
                    new_value=asdict(new_finding),
                    actor=actor,
                    created_at=now,
                )
        return self.get(
            matter_id, tenant_id, user_id, include_artifacts=True, include_findings=True,
        )

    def update_finding_decision(
        self,
        *,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        finding_id: str,
        human_review_status: str,
        note: str | None = None,
        decided_by: str | None = None,
    ) -> MatterRecord | None:
        now = datetime.now(timezone.utc)
        normalized_id = _clean_text(finding_id)
        normalized_status = _clean_text(human_review_status)
        actor = _clean_text(decided_by) or user_id
        decision = {
            "status": normalized_status,
            "note": _clean_text(note),
            "decided_by": actor,
            "decided_at": now.isoformat(),
        }
        with self._database.connection(write=True) as connection:
            if self._repository.get_matter_row(connection, matter_id, tenant_id, user_id) is None:
                return None
            changes = self._repository.update_finding_reviews(
                connection,
                tenant_id=tenant_id,
                user_id=user_id,
                matter_id=matter_id,
                finding_ids=[normalized_id],
                human_review_status=normalized_status,
                decision=decision,
                now=now,
            )
            if not changes:
                raise KeyError(normalized_id)
            old_finding, new_finding = changes[0]
            self._repository.touch_matter(connection, matter_id, tenant_id, user_id, now)
            self._repository.emit_event(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                event_type="finding_decision_updated",
                entity_type="finding",
                entity_id=normalized_id,
                old_value=asdict(old_finding),
                new_value=asdict(new_finding),
                actor=actor,
                created_at=now,
            )
        return self.get(
            matter_id, tenant_id, user_id, include_artifacts=True, include_findings=True,
        )

    def create_formal_report_artifact(
        self,
        *,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        requested_by: str | None = None,
        note: str | None = None,
    ) -> MatterRecord | None:
        now = datetime.now(timezone.utc)
        actor = _clean_text(requested_by) or user_id
        with self._database.connection(write=True) as connection:
            matter = self._repository.get_matter(connection, matter_id, tenant_id, user_id)
            if matter is None:
                return None
            findings = self._repository.list_findings(connection, matter_id, tenant_id, user_id)
            blockers = _formal_report_blockers(matter.matter_profile, findings)
            if blockers:
                raise ValueError("; ".join(blockers))
            artifacts = self._repository.list_artifacts(connection, matter_id, tenant_id, user_id)
            source_task_id = matter.latest_task_id or matter.source_task_id or matter_id
            artifact_data = _build_formal_report_artifact(
                matter_id=matter_id,
                profile=matter.matter_profile,
                findings=findings,
                artifacts=artifacts,
                source_task_id=source_task_id,
                generated_by=actor,
                generated_at=now.isoformat(),
                note=note,
            )
            old_artifact = self._repository.get_artifact(
                connection, matter_id, tenant_id, user_id, "formal_report",
            )
            new_artifact = self._repository.upsert_artifact(
                connection,
                tenant_id=tenant_id,
                user_id=user_id,
                matter_id=matter_id,
                source_task_id=source_task_id,
                artifact=artifact_data,
                existing=old_artifact,
                now=now,
            )
            if new_artifact is None:
                raise RuntimeError("Formal report artifact was not persisted.")
            self._repository.touch_matter(connection, matter_id, tenant_id, user_id, now)
            self._repository.emit_event(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                event_type="artifact_upserted",
                entity_type="artifact",
                entity_id="formal_report",
                old_value=_artifact_snapshot(old_artifact) if old_artifact else None,
                new_value=_artifact_snapshot(new_artifact),
                actor=actor,
                created_at=now,
            )
            self._repository.emit_event(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                event_type="formal_report_created",
                entity_type="artifact",
                entity_id="formal_report",
                old_value=_artifact_snapshot(old_artifact) if old_artifact else None,
                new_value=_artifact_snapshot(new_artifact),
                actor=actor,
                created_at=now,
            )
        return self.get(
            matter_id, tenant_id, user_id, include_artifacts=True, include_findings=True,
        )

    def _sync_artifacts(
        self,
        connection: sqlite3.Connection,
        *,
        tenant_id: str,
        user_id: str,
        matter_id: str,
        task_id: str,
        artifacts: list[dict[str, Any]],
        now: datetime,
    ) -> None:
        for artifact in artifacts:
            artifact_id = _clean_text(artifact.get("artifact_id")) or _clean_text(
                artifact.get("artifact_type")
            )
            if not artifact_id:
                continue
            existing = self._repository.get_artifact(
                connection, matter_id, tenant_id, user_id, artifact_id,
            )
            merged = _merge_agent_artifact(artifact, existing)
            updated = self._repository.upsert_artifact(
                connection,
                tenant_id=tenant_id,
                user_id=user_id,
                matter_id=matter_id,
                source_task_id=task_id,
                artifact=merged,
                existing=existing,
                now=now,
            )
            if updated is None:
                continue
            self._repository.emit_event(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                event_type="artifact_upserted",
                entity_type="artifact",
                entity_id=artifact_id,
                old_value=_artifact_snapshot(existing) if existing else None,
                new_value=_artifact_snapshot(updated),
                actor=user_id,
                created_at=now,
            )

    def _sync_findings(
        self,
        connection: sqlite3.Connection,
        *,
        tenant_id: str,
        user_id: str,
        matter_id: str,
        task_id: str,
        findings: list[dict[str, Any]],
        now: datetime,
    ) -> None:
        for finding in findings:
            finding_id = _clean_text(finding.get("finding_id"))
            if not finding_id:
                continue
            existing = self._repository.get_finding(
                connection, matter_id, tenant_id, user_id, finding_id,
            )
            updated = self._repository.upsert_finding(
                connection,
                tenant_id=tenant_id,
                user_id=user_id,
                matter_id=matter_id,
                source_task_id=task_id,
                finding=finding,
                existing=existing,
                metadata=_merge_finding_metadata(finding, existing),
                now=now,
            )
            if updated is None:
                continue
            self._repository.emit_event(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                event_type="finding_upserted",
                entity_type="finding",
                entity_id=finding_id,
                old_value=asdict(existing) if existing else None,
                new_value=asdict(updated),
                actor=user_id,
                created_at=now,
            )


__all__ = [
    "MatterArtifactRecord",
    "MatterEventRecord",
    "MatterFindingRecord",
    "MatterRecord",
    "MatterStore",
]

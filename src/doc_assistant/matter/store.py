"""案件（matter）SQLite 持久化：档案、finding、artifact 与审计事件。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Any
from uuid import uuid4

from doc_assistant.config.settings import settings

# 内部模块
from doc_assistant.matter import _sql as sql
from doc_assistant.matter._serializers import (
    _row_to_artifact,
    _row_to_event,
    _row_to_finding,
    _row_to_matter,
)
from doc_assistant.matter._helpers import (
    _apply_gate_profile_decision,
    _finding_status,
    _formal_report_blockers,
    _human_review_status_for_gate_status,
    _matter_status,
    _matter_title,
)
from doc_assistant.matter._utils import (
    _as_bool,
    _as_dict,
    _as_dict_list,
    _as_text_list,
    _clean_text,
    _datetime_from_db,
    _datetime_to_db,
    _dedupe_texts,
    _utc_now,
)
# 重新导出以保持向后兼容
from doc_assistant.matter.schemas import (  # noqa: F401  # re-export
    MatterArtifactRecord,
    MatterEventRecord,
    MatterFindingRecord,
    MatterRecord,
)


class MatterStore:
    """案件数据的 SQLite 仓库。

    职责：创建/更新 matter、同步 Agent 产出的 finding 与 artifact、
    记录审计事件、支持按 tenant/user 查询与版本化 artifact。
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or settings.matter_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    # ── public API ────────────────────────────────────────────

    def upsert_from_agent_result(
        self,
        *,
        tenant_id: str,
        user_id: str,
        matter_id: str,
        result: dict[str, Any],
    ) -> MatterRecord:
        """将 Agent 任务结果写入/更新 matter。

        source_task_id 记录首次创建该 matter 的任务；
        latest_task_id 随每次 upsert 更新，表示最近一次写入来源。
        finding/artifact 按 ID 去重合并，并记录变更事件供审计追溯。
        """
        profile = result.get("matter_profile")
        if not isinstance(profile, dict):
            profile = {"matter_id": matter_id, "open_questions": ["Matter profile was not produced."]}
        profile = {**profile, "matter_id": matter_id}
        task_id = _clean_text(result.get("task_id")) or matter_id
        now = _utc_now()
        title = _matter_title(result, profile)

        with self._connect() as connection, self._lock:
            existing = connection.execute(
                sql.SELECT_MATTER_EXISTING_FOR_UPSERT,
                (tenant_id, user_id, matter_id),
            ).fetchone()
            created_at = _datetime_from_db(existing["created_at"]) if existing else now
            old_profile = json.loads(existing["matter_profile_json"] or "{}") if existing else None
            connection.execute(
                sql.UPSERT_MATTER,
                (
                    matter_id,
                    tenant_id,
                    user_id,
                    title,
                    _matter_status(profile),
                    json.dumps(profile, ensure_ascii=False),
                    task_id,
                    task_id,
                    _datetime_to_db(created_at or now),
                    _datetime_to_db(now),
                ),
            )
            self._emit_event(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                event_type="matter_profile_upserted",
                entity_type="matter",
                entity_id=matter_id,
                old_value=old_profile,
                new_value=profile,
                actor=user_id,
                created_at=now,
            )

            for artifact in _as_dict_list(result.get("artifacts")):
                self._upsert_artifact_row(
                    connection,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    matter_id=matter_id,
                    source_task_id=task_id,
                    artifact=artifact,
                    now=now,
                )

            for finding in _as_dict_list(result.get("findings")):
                self._upsert_finding_row(
                    connection,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    matter_id=matter_id,
                    source_task_id=task_id,
                    finding=finding,
                    now=now,
                )

        loaded = self.get(
            matter_id,
            tenant_id,
            user_id,
            include_artifacts=True,
            include_findings=True,
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
        """按 matter_id 读取案件；可选一并加载 artifacts / findings 列表。"""
        with self._connect() as connection:
            row = connection.execute(
                sql.SELECT_MATTER_BY_IDS,
                (matter_id, tenant_id, user_id),
            ).fetchone()
            if row is None:
                return None
            record = _row_to_matter(row)
            if not include_artifacts:
                artifacts = None
            else:
                artifacts = self._list_artifacts_with_connection(
                    connection,
                    matter_id=matter_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                )
            if not include_findings:
                findings = None
            else:
                findings = self._list_findings_with_connection(
                    connection,
                    matter_id=matter_id,
                    tenant_id=tenant_id,
                    user_id=user_id,
                )
            return replace(record, artifacts=artifacts, findings=findings)

    def list(
        self,
        tenant_id: str,
        user_id: str,
        *,
        limit: int = 50,
    ) -> list[MatterRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                sql.SELECT_MATTERS_BY_USER,
                (tenant_id, user_id, max(1, min(limit, 200))),
            ).fetchall()
        return [_row_to_matter(row) for row in rows]

    def list_artifacts(
        self,
        matter_id: str,
        tenant_id: str,
        user_id: str,
    ) -> list[MatterArtifactRecord] | None:
        if self.get(matter_id, tenant_id, user_id) is None:
            return None
        with self._connect() as connection:
            return self._list_artifacts_with_connection(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
            )

    def list_findings(
        self,
        matter_id: str,
        tenant_id: str,
        user_id: str,
    ) -> list[MatterFindingRecord] | None:
        if self.get(matter_id, tenant_id, user_id) is None:
            return None
        with self._connect() as connection:
            return self._list_findings_with_connection(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
            )

    def list_events(
        self,
        matter_id: str,
        tenant_id: str,
        user_id: str,
        *,
        limit: int = 100,
    ) -> list[MatterEventRecord] | None:
        if self.get(matter_id, tenant_id, user_id) is None:
            return None
        with self._connect() as connection:
            rows = connection.execute(
                sql.SELECT_EVENTS_BY_MATTER,
                (matter_id, tenant_id, user_id, max(1, min(limit, 500))),
            ).fetchall()
        return [_row_to_event(row) for row in rows]

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
        now = _utc_now()
        normalized_artifact_id = _clean_text(artifact_id)
        actor = _clean_text(updated_by) or user_id

        with self._connect() as connection, self._lock:
            if self.get(matter_id, tenant_id, user_id) is None:
                return None
            row = connection.execute(
                sql.SELECT_ARTIFACT_BY_ID,
                (matter_id, tenant_id, user_id, normalized_artifact_id),
            ).fetchone()
            if row is None:
                raise KeyError(normalized_artifact_id)

            old_artifact = _row_to_artifact(row)
            next_title = _clean_text(title) if title is not None else old_artifact.title
            next_summary = _clean_text(summary) if summary is not None else old_artifact.summary
            next_items = items if items is not None else old_artifact.items
            next_status = _clean_text(status) or old_artifact.status
            next_version = old_artifact.version + 1
            metadata = dict(old_artifact.metadata)
            edit = {
                "status": next_status,
                "note": _clean_text(note),
                "updated_by": actor,
                "updated_at": _datetime_to_db(now),
                "version": next_version,
            }
            edits = _as_dict_list(metadata.get("edits"))
            metadata["edits"] = [*edits[-19:], edit]
            metadata["last_edit"] = edit

            connection.execute(
                sql.UPDATE_ARTIFACT,
                (
                    next_title or old_artifact.title,
                    next_summary,
                    json.dumps(_as_dict_list(next_items), ensure_ascii=False),
                    json.dumps(metadata, ensure_ascii=False),
                    next_version,
                    next_status,
                    _datetime_to_db(now),
                    matter_id,
                    tenant_id,
                    user_id,
                    normalized_artifact_id,
                ),
            )
            connection.execute(
                sql.UPDATE_MATTER_UPDATED_AT,
                (_datetime_to_db(now), matter_id, tenant_id, user_id),
            )
            self._emit_event(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                event_type="artifact_updated",
                entity_type="artifact",
                entity_id=normalized_artifact_id,
                old_value={
                    "title": old_artifact.title,
                    "summary": old_artifact.summary,
                    "items": old_artifact.items,
                    "status": old_artifact.status,
                    "version": old_artifact.version,
                },
                new_value={
                    "title": next_title or old_artifact.title,
                    "summary": next_summary,
                    "items": _as_dict_list(next_items),
                    "status": next_status,
                    "version": next_version,
                    "note": _clean_text(note),
                },
                actor=actor,
                created_at=now,
            )

        return self.get(
            matter_id,
            tenant_id,
            user_id,
            include_artifacts=True,
            include_findings=True,
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
        now = _utc_now()
        normalized_status = _clean_text(status)
        normalized_gate_id = _clean_text(gate_id)

        with self._connect() as connection, self._lock:
            row = connection.execute(
                sql.SELECT_MATTER_BY_IDS,
                (matter_id, tenant_id, user_id),
            ).fetchone()
            if row is None:
                return None

            profile = json.loads(row["matter_profile_json"] or "{}")
            gates = _as_dict_list(profile.get("confirmation_gates"))
            gate = next(
                (
                    item
                    for item in gates
                    if _clean_text(item.get("gate_id")) == normalized_gate_id
                ),
                None,
            )
            if gate is None:
                raise KeyError(normalized_gate_id)

            metadata = _as_dict(gate.get("metadata")).copy()
            decision = {
                "status": normalized_status,
                "note": _clean_text(note),
                "decided_by": _clean_text(decided_by) or user_id,
                "decided_at": _datetime_to_db(now),
            }
            decisions = _as_dict_list(metadata.get("decisions"))
            metadata["decisions"] = [*decisions[-19:], decision]
            metadata["last_decision"] = decision

            gate["status"] = normalized_status
            gate["metadata"] = metadata
            gate["updated_at"] = _datetime_to_db(now)
            if normalized_status in {"approved", "waived"}:
                gate["decided_by"] = decision["decided_by"]
                gate["decided_at"] = decision["decided_at"]
            else:
                gate.pop("decided_by", None)
                gate.pop("decided_at", None)

            profile["confirmation_gates"] = gates
            _apply_gate_profile_decision(
                profile,
                gate=gate,
                status=normalized_status,
                confirmed_value=confirmed_value,
                decision=decision,
            )
            related_finding_ids = _as_text_list(gate.get("related_finding_ids"))
            if related_finding_ids:
                self._update_finding_review_rows(
                    connection,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    matter_id=matter_id,
                    finding_ids=related_finding_ids,
                    human_review_status=_human_review_status_for_gate_status(normalized_status),
                    decision=decision,
                    now=now,
                )
            connection.execute(
                sql.UPDATE_MATTER_STATUS_AND_PROFILE,
                (
                    _matter_status(profile),
                    json.dumps(profile, ensure_ascii=False),
                    _datetime_to_db(now),
                    matter_id,
                    tenant_id,
                    user_id,
                ),
            )
            self._emit_event(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                event_type="confirmation_gate_updated",
                entity_type="confirmation_gate",
                entity_id=normalized_gate_id,
                old_value=None,
                new_value=gate,
                actor=decision["decided_by"],
                created_at=now,
            )

        return self.get(
            matter_id,
            tenant_id,
            user_id,
            include_artifacts=True,
            include_findings=True,
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
        now = _utc_now()
        normalized_finding_id = _clean_text(finding_id)
        normalized_status = _clean_text(human_review_status)
        decision = {
            "status": normalized_status,
            "note": _clean_text(note),
            "decided_by": _clean_text(decided_by) or user_id,
            "decided_at": _datetime_to_db(now),
        }

        with self._connect() as connection, self._lock:
            if self.get(matter_id, tenant_id, user_id) is None:
                return None
            row = connection.execute(
                sql.SELECT_FINDING_BY_ID,
                (matter_id, tenant_id, user_id, normalized_finding_id),
            ).fetchone()
            if row is None:
                raise KeyError(normalized_finding_id)
            self._update_finding_review_rows(
                connection,
                tenant_id=tenant_id,
                user_id=user_id,
                matter_id=matter_id,
                finding_ids=[normalized_finding_id],
                human_review_status=normalized_status,
                decision=decision,
                now=now,
            )
            connection.execute(
                sql.UPDATE_MATTER_UPDATED_AT,
                (_datetime_to_db(now), matter_id, tenant_id, user_id),
            )
            self._emit_event(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                event_type="finding_decision_updated",
                entity_type="finding",
                entity_id=normalized_finding_id,
                old_value=None,
                new_value=decision,
                actor=decision["decided_by"],
                created_at=now,
            )

        return self.get(
            matter_id,
            tenant_id,
            user_id,
            include_artifacts=True,
            include_findings=True,
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
        now = _utc_now()
        with self._connect() as connection, self._lock:
            row = connection.execute(
                sql.SELECT_MATTER_BY_IDS,
                (matter_id, tenant_id, user_id),
            ).fetchone()
            if row is None:
                return None

            profile = json.loads(row["matter_profile_json"] or "{}")
            findings = self._list_findings_with_connection(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
            )
            blockers = _formal_report_blockers(profile, findings)
            if blockers:
                raise ValueError("; ".join(blockers))

            existing_artifacts = self._list_artifacts_with_connection(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
            )
            source_artifact_ids = [
                artifact.artifact_id
                for artifact in existing_artifacts
                if artifact.artifact_id != "formal_report"
            ]
            gates = _as_dict_list(profile.get("confirmation_gates"))
            source_task_id = _clean_text(row["latest_task_id"]) or _clean_text(row["source_task_id"])
            generated_by = _clean_text(requested_by) or user_id
            generated_at = _datetime_to_db(now)
            artifact = {
                "artifact_id": "formal_report",
                "artifact_type": "formal_report",
                "title": "Formal report record",
                "summary": (
                    "Confirmation gates were resolved and the current matter artifacts "
                    "were approved for formal use."
                ),
                "items": [
                    {
                        "item_id": "formal-report-1",
                        "matter_id": matter_id,
                        "document_type": _clean_text(profile.get("document_type")) or "Unknown",
                        "status": "approved_for_formal_use",
                        "generated_by": generated_by,
                        "generated_at": generated_at,
                        "source_task_id": source_task_id,
                        "source_artifact_ids": source_artifact_ids,
                        "finding_count": len(findings),
                        "gate_count": len(gates),
                        "note": _clean_text(note),
                    }
                ],
                "source_finding_ids": [finding.finding_id for finding in findings],
                "citations": _dedupe_texts(
                    [
                        source_id
                        for gate in gates
                        for source_id in _as_text_list(gate.get("citations"))
                    ]
                    + [source_id for finding in findings for source_id in finding.citations]
                ),
                "metadata": {
                    "matter_id": matter_id,
                    "generated_by": generated_by,
                    "generated_at": generated_at,
                    "source_task_id": source_task_id,
                    "source_artifact_ids": source_artifact_ids,
                    "finding_statuses": [
                        {
                            "finding_id": finding.finding_id,
                            "status": finding.status,
                            "human_review_status": finding.human_review_status,
                            "support_level": finding.support_level,
                            "evidence_coverage": finding.evidence_coverage,
                        }
                        for finding in findings
                    ],
                    "gate_statuses": [
                        {
                            "gate_id": _clean_text(gate.get("gate_id")),
                            "status": _clean_text(gate.get("status")) or "pending",
                        }
                        for gate in gates
                    ],
                    "note": _clean_text(note),
                },
                "status": "approved",
            }
            self._upsert_artifact_row(
                connection,
                tenant_id=tenant_id,
                user_id=user_id,
                matter_id=matter_id,
                source_task_id=source_task_id or matter_id,
                artifact=artifact,
                now=now,
            )
            connection.execute(
                sql.UPDATE_MATTER_UPDATED_AT,
                (_datetime_to_db(now), matter_id, tenant_id, user_id),
            )
            self._emit_event(
                connection,
                matter_id=matter_id,
                tenant_id=tenant_id,
                user_id=user_id,
                event_type="formal_report_created",
                entity_type="artifact",
                entity_id="formal_report",
                old_value=None,
                new_value=artifact,
                actor=generated_by,
                created_at=now,
            )

        return self.get(
            matter_id,
            tenant_id,
            user_id,
            include_artifacts=True,
            include_findings=True,
        )

    # ── private persistence helpers ───────────────────────────

    def _upsert_artifact_row(
        self,
        connection: sqlite3.Connection,
        *,
        tenant_id: str,
        user_id: str,
        matter_id: str,
        source_task_id: str,
        artifact: dict[str, Any],
        now: datetime,
    ) -> None:
        artifact_id = _clean_text(artifact.get("artifact_id")) or _clean_text(
            artifact.get("artifact_type")
        )
        if not artifact_id:
            return
        existing = connection.execute(
            sql.SELECT_ARTIFACT_EXISTING,
            (tenant_id, user_id, matter_id, artifact_id),
        ).fetchone()
        version = int(existing["version"]) + 1 if existing else 1
        created_at = _datetime_from_db(existing["created_at"]) if existing else now
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
                _datetime_to_db(created_at or now),
                _datetime_to_db(now),
            ),
        )
        self._emit_event(
            connection,
            matter_id=matter_id,
            tenant_id=tenant_id,
            user_id=user_id,
            event_type="artifact_upserted",
            entity_type="artifact",
            entity_id=artifact_id,
            old_value=dict(existing) if existing else None,
            new_value={
                "artifact_id": artifact_id,
                "artifact_type": _clean_text(artifact.get("artifact_type")) or "custom",
                "title": _clean_text(artifact.get("title")) or artifact_id,
                "summary": _clean_text(artifact.get("summary")),
                "version": version,
            },
            actor=user_id,
            created_at=now,
        )

    def _upsert_finding_row(
        self,
        connection: sqlite3.Connection,
        *,
        tenant_id: str,
        user_id: str,
        matter_id: str,
        source_task_id: str,
        finding: dict[str, Any],
        now: datetime,
    ) -> None:
        finding_id = _clean_text(finding.get("finding_id"))
        if not finding_id:
            return
        existing = connection.execute(
            sql.SELECT_FINDING_EXISTING,
            (tenant_id, user_id, matter_id, finding_id),
        ).fetchone()
        created_at = _datetime_from_db(existing["created_at"]) if existing else now
        existing_human_status = _clean_text(existing["human_review_status"]) if existing else ""
        incoming_human_status = _clean_text(finding.get("human_review_status")) or "pending"
        human_review_status = (
            existing_human_status
            if existing_human_status in {"approved", "waived", "resolved"}
            else incoming_human_status
        )
        needs_human_review = _as_bool(finding.get("needs_human_review"), default=True)
        status_value = _finding_status(
            needs_human_review=needs_human_review,
            human_review_status=human_review_status,
            evidence_coverage=_clean_text(finding.get("evidence_coverage")) or "missing",
        )
        metadata = _as_dict(finding.get("metadata")).copy()
        if finding.get("evidence"):
            metadata["evidence"] = finding.get("evidence")
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
                _clean_text(finding.get("evidence_coverage")) or "missing",
                _clean_text(finding.get("support_level")) or "missing",
                _clean_text(finding.get("unsupported_reason")),
                _clean_text(finding.get("source_quote")),
                _clean_text(finding.get("location_label")),
                1 if needs_human_review else 0,
                human_review_status,
                status_value,
                json.dumps(metadata, ensure_ascii=False),
                source_task_id,
                _datetime_to_db(created_at or now),
                _datetime_to_db(now),
            ),
        )
        self._emit_event(
            connection,
            matter_id=matter_id,
            tenant_id=tenant_id,
            user_id=user_id,
            event_type="finding_upserted",
            entity_type="finding",
            entity_id=finding_id,
            old_value=dict(existing) if existing else None,
            new_value={
                "finding_id": finding_id,
                "category": _clean_text(finding.get("category")) or "Finding",
                "severity": _clean_text(finding.get("severity")) or "Needs human review",
                "status": status_value,
                "human_review_status": human_review_status,
            },
            actor=user_id,
            created_at=now,
        )

    def _update_finding_review_rows(
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
    ) -> None:
        for finding_id in finding_ids:
            row = connection.execute(
                sql.SELECT_FINDING_ROW_BY_ID,
                (tenant_id, user_id, matter_id, finding_id),
            ).fetchone()
            if row is None:
                continue
            metadata = json.loads(row["metadata_json"] or "{}")
            decisions = _as_dict_list(metadata.get("decisions"))
            metadata["decisions"] = [*decisions[-19:], decision]
            metadata["last_decision"] = decision
            status_value = _finding_status(
                needs_human_review=bool(row["needs_human_review"]),
                human_review_status=human_review_status,
                evidence_coverage=row["evidence_coverage"],
            )
            connection.execute(
                sql.UPDATE_FINDING_REVIEW,
                (
                    human_review_status,
                    status_value,
                    json.dumps(metadata, ensure_ascii=False),
                    _datetime_to_db(now),
                    tenant_id,
                    user_id,
                    matter_id,
                    finding_id,
                ),
            )

    def _list_artifacts_with_connection(
        self,
        connection: sqlite3.Connection,
        *,
        matter_id: str,
        tenant_id: str,
        user_id: str,
    ) -> list[MatterArtifactRecord]:
        rows = connection.execute(
            sql.SELECT_ARTIFACTS_BY_MATTER,
            (matter_id, tenant_id, user_id),
        ).fetchall()
        return [_row_to_artifact(row) for row in rows]

    def _list_findings_with_connection(
        self,
        connection: sqlite3.Connection,
        *,
        matter_id: str,
        tenant_id: str,
        user_id: str,
    ) -> list[MatterFindingRecord]:
        rows = connection.execute(
            sql.SELECT_FINDINGS_BY_MATTER,
            (matter_id, tenant_id, user_id),
        ).fetchall()
        return [_row_to_finding(row) for row in rows]

    def _emit_event(
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
                _datetime_to_db(created_at),
            ),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(sql.CREATE_TABLE_MATTERS)
            connection.execute(sql.CREATE_TABLE_MATTER_ARTIFACTS)
            connection.execute(sql.CREATE_TABLE_REVIEW_FINDINGS)
            connection.execute(sql.CREATE_INDEX_MATTERS_USER_UPDATED)
            connection.execute(sql.CREATE_INDEX_MATTER_ARTIFACTS_MATTER)
            connection.execute(sql.CREATE_INDEX_REVIEW_FINDINGS_MATTER)
            connection.execute(sql.CREATE_TABLE_MATTER_EVENTS)
            connection.execute(sql.CREATE_INDEX_MATTER_EVENTS_MATTER)



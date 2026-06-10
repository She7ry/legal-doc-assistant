from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Any

from doc_assistant.config.settings import settings


@dataclass(frozen=True)
class MatterArtifactRecord:
    artifact_id: str
    matter_id: str
    tenant_id: str
    user_id: str
    artifact_type: str
    title: str
    summary: str
    items: list[dict[str, Any]]
    source_finding_ids: list[str]
    citations: list[str]
    metadata: dict[str, Any]
    source_task_id: str
    version: int
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class MatterFindingRecord:
    finding_id: str
    matter_id: str
    tenant_id: str
    user_id: str
    category: str
    severity: str
    summary: str
    recommended_action: str
    citations: list[str]
    source_step_id: str
    clause_reference: str
    evidence_coverage: str
    support_level: str
    unsupported_reason: str
    source_quote: str
    location_label: str
    needs_human_review: bool
    human_review_status: str
    status: str
    metadata: dict[str, Any]
    source_task_id: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class MatterRecord:
    matter_id: str
    tenant_id: str
    user_id: str
    title: str
    status: str
    matter_profile: dict[str, Any]
    source_task_id: str
    latest_task_id: str
    created_at: datetime
    updated_at: datetime
    artifacts: list[MatterArtifactRecord] | None = None
    findings: list[MatterFindingRecord] | None = None


class MatterStore:
    """SQLite-backed repository for matter profiles and generated artifacts."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or settings.matter_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def upsert_from_agent_result(
        self,
        *,
        tenant_id: str,
        user_id: str,
        matter_id: str,
        result: dict[str, Any],
    ) -> MatterRecord:
        profile = result.get("matter_profile")
        if not isinstance(profile, dict):
            profile = {"matter_id": matter_id, "open_questions": ["Matter profile was not produced."]}
        profile = {**profile, "matter_id": matter_id}
        task_id = _clean_text(result.get("task_id")) or matter_id
        now = _utc_now()
        title = _matter_title(result, profile)

        with self._connect() as connection, self._lock:
            existing = connection.execute(
                """
                SELECT created_at FROM matters
                WHERE tenant_id = ? AND user_id = ? AND matter_id = ?
                """,
                (tenant_id, user_id, matter_id),
            ).fetchone()
            created_at = _datetime_from_db(existing["created_at"]) if existing else now
            connection.execute(
                """
                INSERT INTO matters (
                    matter_id, tenant_id, user_id, title, status, matter_profile_json,
                    source_task_id, latest_task_id, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(matter_id, tenant_id, user_id)
                DO UPDATE SET
                    title = excluded.title,
                    status = excluded.status,
                    matter_profile_json = excluded.matter_profile_json,
                    latest_task_id = excluded.latest_task_id,
                    updated_at = excluded.updated_at
                """,
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
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM matters
                WHERE matter_id = ? AND tenant_id = ? AND user_id = ?
                """,
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
                """
                SELECT * FROM matters
                WHERE tenant_id = ? AND user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
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
                """
                SELECT * FROM matters
                WHERE matter_id = ? AND tenant_id = ? AND user_id = ?
                """,
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
                """
                UPDATE matters
                SET status = ?, matter_profile_json = ?, updated_at = ?
                WHERE matter_id = ? AND tenant_id = ? AND user_id = ?
                """,
                (
                    _matter_status(profile),
                    json.dumps(profile, ensure_ascii=False),
                    _datetime_to_db(now),
                    matter_id,
                    tenant_id,
                    user_id,
                ),
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
                """
                SELECT finding_id FROM review_findings
                WHERE matter_id = ? AND tenant_id = ? AND user_id = ? AND finding_id = ?
                """,
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
                """
                UPDATE matters
                SET updated_at = ?
                WHERE matter_id = ? AND tenant_id = ? AND user_id = ?
                """,
                (_datetime_to_db(now), matter_id, tenant_id, user_id),
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
                """
                SELECT * FROM matters
                WHERE matter_id = ? AND tenant_id = ? AND user_id = ?
                """,
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
                """
                UPDATE matters
                SET updated_at = ?
                WHERE matter_id = ? AND tenant_id = ? AND user_id = ?
                """,
                (_datetime_to_db(now), matter_id, tenant_id, user_id),
            )

        return self.get(
            matter_id,
            tenant_id,
            user_id,
            include_artifacts=True,
            include_findings=True,
        )

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
            """
            SELECT version, created_at FROM matter_artifacts
            WHERE tenant_id = ? AND user_id = ? AND matter_id = ? AND artifact_id = ?
            """,
            (tenant_id, user_id, matter_id, artifact_id),
        ).fetchone()
        version = int(existing["version"]) + 1 if existing else 1
        created_at = _datetime_from_db(existing["created_at"]) if existing else now
        connection.execute(
            """
            INSERT INTO matter_artifacts (
                artifact_id, matter_id, tenant_id, user_id, artifact_type, title, summary,
                items_json, source_finding_ids_json, citations_json, metadata_json,
                source_task_id, version, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(artifact_id, matter_id, tenant_id, user_id)
            DO UPDATE SET
                artifact_type = excluded.artifact_type,
                title = excluded.title,
                summary = excluded.summary,
                items_json = excluded.items_json,
                source_finding_ids_json = excluded.source_finding_ids_json,
                citations_json = excluded.citations_json,
                metadata_json = excluded.metadata_json,
                source_task_id = excluded.source_task_id,
                version = excluded.version,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
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
            """
            SELECT created_at, human_review_status, status, metadata_json FROM review_findings
            WHERE tenant_id = ? AND user_id = ? AND matter_id = ? AND finding_id = ?
            """,
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
            """
            INSERT INTO review_findings (
                finding_id, matter_id, tenant_id, user_id, category, severity, summary,
                recommended_action, citations_json, source_step_id, clause_reference,
                evidence_coverage, support_level, unsupported_reason, source_quote,
                location_label, needs_human_review, human_review_status, status,
                metadata_json, source_task_id, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(finding_id, matter_id, tenant_id, user_id)
            DO UPDATE SET
                category = excluded.category,
                severity = excluded.severity,
                summary = excluded.summary,
                recommended_action = excluded.recommended_action,
                citations_json = excluded.citations_json,
                source_step_id = excluded.source_step_id,
                clause_reference = excluded.clause_reference,
                evidence_coverage = excluded.evidence_coverage,
                support_level = excluded.support_level,
                unsupported_reason = excluded.unsupported_reason,
                source_quote = excluded.source_quote,
                location_label = excluded.location_label,
                needs_human_review = excluded.needs_human_review,
                human_review_status = excluded.human_review_status,
                status = excluded.status,
                metadata_json = excluded.metadata_json,
                source_task_id = excluded.source_task_id,
                updated_at = excluded.updated_at
            """,
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
                """
                SELECT * FROM review_findings
                WHERE tenant_id = ? AND user_id = ? AND matter_id = ? AND finding_id = ?
                """,
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
                """
                UPDATE review_findings
                SET human_review_status = ?, status = ?, metadata_json = ?, updated_at = ?
                WHERE tenant_id = ? AND user_id = ? AND matter_id = ? AND finding_id = ?
                """,
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
            """
            SELECT * FROM matter_artifacts
            WHERE matter_id = ? AND tenant_id = ? AND user_id = ?
            ORDER BY artifact_type ASC
            """,
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
            """
            SELECT * FROM review_findings
            WHERE matter_id = ? AND tenant_id = ? AND user_id = ?
            ORDER BY finding_id ASC
            """,
            (matter_id, tenant_id, user_id),
        ).fetchall()
        return [_row_to_finding(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS matters (
                    matter_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    matter_profile_json TEXT NOT NULL,
                    source_task_id TEXT NOT NULL,
                    latest_task_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(matter_id, tenant_id, user_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS matter_artifacts (
                    artifact_id TEXT NOT NULL,
                    matter_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    items_json TEXT NOT NULL,
                    source_finding_ids_json TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    source_task_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(artifact_id, matter_id, tenant_id, user_id),
                    FOREIGN KEY(matter_id, tenant_id, user_id)
                        REFERENCES matters(matter_id, tenant_id, user_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS review_findings (
                    finding_id TEXT NOT NULL,
                    matter_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    source_step_id TEXT NOT NULL,
                    clause_reference TEXT NOT NULL,
                    evidence_coverage TEXT NOT NULL,
                    support_level TEXT NOT NULL,
                    unsupported_reason TEXT NOT NULL,
                    source_quote TEXT NOT NULL,
                    location_label TEXT NOT NULL,
                    needs_human_review INTEGER NOT NULL,
                    human_review_status TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    source_task_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(finding_id, matter_id, tenant_id, user_id),
                    FOREIGN KEY(matter_id, tenant_id, user_id)
                        REFERENCES matters(matter_id, tenant_id, user_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_matters_user_updated
                ON matters(tenant_id, user_id, updated_at)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_matter_artifacts_matter
                ON matter_artifacts(tenant_id, user_id, matter_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_review_findings_matter
                ON review_findings(tenant_id, user_id, matter_id)
                """
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


def _matter_title(result: dict[str, Any], profile: dict[str, Any]) -> str:
    document_type = _clean_text(profile.get("document_type"))
    objective = _clean_text(result.get("objective"))
    if document_type and document_type != "Unknown":
        return document_type
    return objective[:120] or _clean_text(profile.get("matter_id")) or "Untitled matter"


def _matter_status(profile: dict[str, Any]) -> str:
    open_questions = _as_text_list(profile.get("open_questions"))
    if open_questions or _unresolved_required_gate_ids(profile):
        return "needs_input"
    return "active"


def _formal_report_blockers(
    profile: dict[str, Any],
    findings: list[MatterFindingRecord] | None = None,
) -> list[str]:
    blockers = []
    unresolved_gate_ids = _unresolved_required_gate_ids(profile)
    if unresolved_gate_ids:
        blockers.append(
            "Matter still has unresolved required confirmation gates: "
            + ", ".join(unresolved_gate_ids)
        )
    for finding in findings or []:
        missing = _formal_finding_missing_fields(finding)
        if missing:
            blockers.append(
                f"Finding {finding.finding_id} is not formal-report ready: "
                + ", ".join(missing)
            )
    return blockers


def _formal_finding_missing_fields(finding: MatterFindingRecord) -> list[str]:
    missing = []
    if not finding.citations:
        missing.append("source citation")
    if not finding.source_quote:
        missing.append("exact quote")
    if not finding.location_label:
        missing.append("location")
    if not finding.support_level:
        missing.append("support level")
    if finding.support_level != "direct" and not finding.unsupported_reason:
        missing.append("unsupported reason")
    if finding.needs_human_review and finding.human_review_status not in {
        "approved",
        "waived",
        "resolved",
        "not_required",
    }:
        missing.append("human review status")
    return missing


def _unresolved_required_gate_ids(profile: dict[str, Any]) -> list[str]:
    unresolved_gate_ids = []
    for gate in _as_dict_list(profile.get("confirmation_gates")):
        if not gate.get("required", True):
            continue
        if _clean_text(gate.get("status")) not in {"approved", "waived"}:
            unresolved_gate_ids.append(_clean_text(gate.get("gate_id")) or "unknown_gate")
    return unresolved_gate_ids


def _apply_gate_profile_decision(
    profile: dict[str, Any],
    *,
    gate: dict[str, Any],
    status: str,
    confirmed_value: str | None,
    decision: dict[str, Any],
) -> None:
    if status != "approved":
        return
    metadata = _as_dict(gate.get("metadata"))
    profile_field = _clean_text(metadata.get("profile_field"))
    value = _clean_text(confirmed_value)
    if not profile_field or not value:
        return
    if profile_field not in {"user_side", "governing_law", "jurisdiction", "document_type"}:
        return
    profile[profile_field] = value
    confirmed_facts = _as_dict_list(profile.get("confirmed_facts"))
    confirmed_facts.append(
        {
            "field": profile_field,
            "value": value,
            "source": "confirmation_gate",
            "gate_id": _clean_text(gate.get("gate_id")),
            "decided_by": decision.get("decided_by", ""),
            "decided_at": decision.get("decided_at", ""),
        }
    )
    profile["confirmed_facts"] = confirmed_facts[-50:]


def _human_review_status_for_gate_status(status: str) -> str:
    if status == "approved":
        return "approved"
    if status == "waived":
        return "waived"
    if status == "needs_info":
        return "needs_info"
    return "pending"


def _finding_status(
    *,
    needs_human_review: bool,
    human_review_status: str,
    evidence_coverage: str,
) -> str:
    if human_review_status == "needs_info":
        return "needs_info"
    if needs_human_review and human_review_status not in {
        "approved",
        "waived",
        "resolved",
        "not_required",
    }:
        return "needs_human_review"
    if evidence_coverage == "direct":
        return "resolved"
    if human_review_status in {"approved", "waived", "resolved"}:
        return "resolved_with_evidence_gap"
    return "open"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_clean_text(item) for item in value if _clean_text(item)]


def _as_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return default


def _dedupe_texts(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _datetime_to_db(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _datetime_from_db(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

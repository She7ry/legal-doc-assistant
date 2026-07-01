"""案件（matter）数据模型：MatterRecord 及其关联的 Artifact / Finding / Event 记录。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class MatterArtifactRecord:
    """持久化到 MatterStore 的单条交付物（风险矩阵、谈判清单等）。"""

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
    """持久化到 MatterStore 的单条风险 finding，可跟踪人工复核状态。"""

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
class MatterEventRecord:
    """案件审计日志：finding/artifact 状态变更、人工确认等操作的 before/after 快照。"""

    event_id: str
    matter_id: str
    tenant_id: str
    user_id: str
    event_type: str
    entity_type: str
    entity_id: str
    old_value: dict[str, Any] | list[Any] | str | None
    new_value: dict[str, Any] | list[Any] | str | None
    actor: str
    created_at: datetime


@dataclass(frozen=True)
class MatterRecord:
    """一个法律「案件」的主记录：关联 profile、最新 task、可选嵌套 findings/artifacts。"""

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

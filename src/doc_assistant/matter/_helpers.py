"""MatterStore 业务辅助函数 —— 与 MatterStore 领域强耦合的纯函数。"""

from __future__ import annotations

from typing import Any

from doc_assistant.matter._utils import _as_dict, _as_dict_list, _as_text_list, _clean_text
from doc_assistant.matter.schemas import MatterFindingRecord


def _matter_title(result: dict[str, Any], profile: dict[str, Any]) -> str:
    """从 Agent 结果推断 matter 标题。"""
    document_type = _clean_text(profile.get("document_type"))
    objective = _clean_text(result.get("objective"))
    if document_type and document_type != "Unknown":
        return document_type
    return objective[:120] or _clean_text(profile.get("matter_id")) or "Untitled matter"


def _matter_status(profile: dict[str, Any]) -> str:
    """根据 open_questions 和 confirmation gates 状态决定 matter 状态。"""
    open_questions = _as_text_list(profile.get("open_questions"))
    if open_questions or _unresolved_required_gate_ids(profile):
        return "needs_input"
    return "active"


def _formal_report_blockers(
    profile: dict[str, Any],
    findings: list[MatterFindingRecord] | None = None,
) -> list[str]:
    """检查创建 formal report 的阻塞条件。"""
    blockers: list[str] = []
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
    """检查单个 finding 是否满足 formal report 所需字段。"""
    missing: list[str] = []
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
    """获取所有 required 但尚未 approved/waived 的 gate ID。"""
    unresolved: list[str] = []
    for gate in _as_dict_list(profile.get("confirmation_gates")):
        if not gate.get("required", True):
            continue
        if _clean_text(gate.get("status")) not in {"approved", "waived"}:
            unresolved.append(_clean_text(gate.get("gate_id")) or "unknown_gate")
    return unresolved


def _apply_gate_profile_decision(
    profile: dict[str, Any],
    *,
    gate: dict[str, Any],
    status: str,
    confirmed_value: str | None,
    decision: dict[str, Any],
) -> None:
    """gate 审批通过时将确认值写回 matter profile。"""
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
    """将 confirmation gate 状态映射为 human review 状态。"""
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
    """综合 needs_human_review / human_review_status / evidence_coverage 计算 finding 状态。"""
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

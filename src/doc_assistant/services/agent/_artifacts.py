"""结构化交付物：风险矩阵、律师问题清单、谈判清单、义务日历。"""

from __future__ import annotations

from typing import Any

from doc_assistant.services.agent._helpers import (
    _as_text_list,
    _clean_text,
    _dedupe_texts,
    _source_id_list,
)
from doc_assistant.services.agent._matter_profile import _extract_key_dates
from doc_assistant.services.agent.schemas import (
    AgentArtifact,
    AgentFinding,
    AgentStepResult,
    MatterProfile,
)


def _build_agent_artifacts(
    *,
    matter_profile: MatterProfile,
    findings: list[AgentFinding],
    steps: list[AgentStepResult],
    missing_information: list[str],
    user_role: str,
) -> list[AgentArtifact]:
    return [
        _risk_matrix_artifact(findings),
        _lawyer_questions_artifact(findings, steps, missing_information, user_role),
        _negotiation_checklist_artifact(findings, matter_profile),
        _obligation_calendar_artifact(matter_profile, findings),
    ]


def _risk_matrix_artifact(findings: list[AgentFinding]) -> AgentArtifact:
    items = [
        {
            "item_id": f"risk-{index}",
            "finding_id": finding.finding_id,
            "category": finding.category,
            "severity": finding.severity,
            "issue": finding.summary,
            "recommended_action": finding.recommended_action,
            "citations": finding.citations,
            "needs_human_review": finding.needs_human_review,
            "status": finding.status,
            "human_review_status": finding.human_review_status,
            "evidence_coverage": finding.evidence_coverage,
            "support_level": finding.support_level,
            "unsupported_reason": finding.unsupported_reason,
            "source_quote": finding.source_quote,
            "location_label": finding.location_label,
            "clause_reference": finding.clause_reference,
        }
        for index, finding in enumerate(findings, start=1)
    ]
    return AgentArtifact(
        artifact_id="risk_matrix",
        artifact_type="risk_matrix",
        title="Risk matrix",
        summary="Structured risk rows derived from evidence-backed review findings.",
        items=items,
        source_finding_ids=[finding.finding_id for finding in findings],
        citations=_artifact_citations(items),
    )


def _lawyer_questions_artifact(
    findings: list[AgentFinding],
    steps: list[AgentStepResult],
    missing_information: list[str],
    user_role: str,
) -> AgentArtifact:
    items: list[dict[str, Any]] = []
    for step in steps:
        metadata = step.output.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        for question in _as_text_list(metadata.get("questions_for_lawyer")):
            items.append(
                {
                    "item_id": f"question-{len(items) + 1}",
                    "question": question,
                    "reason": f"Raised during {step.title}.",
                    "priority": "legal_review",
                    "source_step_id": step.step_id,
                    "citations": [citation.source_id for citation in step.citations[:2]],
                }
            )

    for finding in findings:
        if finding.recommended_action and "?" in finding.recommended_action:
            question = finding.recommended_action
        else:
            question = f"What position should we take on {finding.category}?"
        items.append(
            {
                "item_id": f"question-{len(items) + 1}",
                "question": question,
                "reason": finding.summary,
                "priority": "high" if finding.needs_human_review else "normal",
                "source_finding_id": finding.finding_id,
                "citations": finding.citations,
            }
        )

    for item in missing_information:
        items.append(
            {
                "item_id": f"question-{len(items) + 1}",
                "question": f"Please confirm: {item}",
                "reason": "The workflow marked this information as missing.",
                "priority": "blocking",
                "citations": [],
            }
        )

    items = _dedupe_artifact_items(items, "question")
    title = "Lawyer questions" if user_role == "lawyer" else "Review questions"
    return AgentArtifact(
        artifact_id="lawyer_questions",
        artifact_type="lawyer_questions",
        title=title,
        summary="Questions to resolve before relying on the review output.",
        items=items,
        source_finding_ids=_artifact_finding_ids(items),
        citations=_artifact_citations(items),
    )


def _negotiation_checklist_artifact(
    findings: list[AgentFinding],
    matter_profile: MatterProfile,
) -> AgentArtifact:
    items = []
    owner = matter_profile.user_side or "User side"
    for index, finding in enumerate(findings, start=1):
        items.append(
            {
                "item_id": f"negotiation-{index}",
                "issue": finding.category,
                "ask": finding.recommended_action
                or f"Clarify or revise the {finding.category} language.",
                "fallback": "Escalate for legal review before accepting the current language.",
                "owner": owner,
                "priority": _negotiation_priority(finding.severity),
                "source_finding_id": finding.finding_id,
                "citations": finding.citations,
            }
        )
    return AgentArtifact(
        artifact_id="negotiation_checklist",
        artifact_type="negotiation_checklist",
        title="Negotiation checklist",
        summary="Negotiation asks and fallback positions generated from current findings.",
        items=items,
        source_finding_ids=[finding.finding_id for finding in findings],
        citations=_artifact_citations(items),
    )


def _obligation_calendar_artifact(
    matter_profile: MatterProfile,
    findings: list[AgentFinding],
) -> AgentArtifact:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for date_item in matter_profile.key_dates:
        key = _clean_text(date_item.get("value")).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "item_id": f"obligation-{len(items) + 1}",
                "trigger": date_item.get("description") or date_item.get("label"),
                "deadline": date_item.get("value"),
                "owner": matter_profile.user_side or "To confirm",
                "status": "needs_confirmation",
                "citations": date_item.get("citations") or [],
            }
        )

    for finding in findings:
        for date_item in _extract_key_dates(finding.summary, finding.citations):
            key = _clean_text(date_item.get("value")).casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "item_id": f"obligation-{len(items) + 1}",
                    "trigger": finding.category,
                    "deadline": date_item.get("value"),
                    "owner": matter_profile.user_side or "To confirm",
                    "status": "needs_confirmation",
                    "source_finding_id": finding.finding_id,
                    "citations": finding.citations,
                }
            )
    return AgentArtifact(
        artifact_id="obligation_calendar",
        artifact_type="obligation_calendar",
        title="Obligation calendar",
        summary="Dates and relative deadlines extracted from the reviewed evidence.",
        items=items,
        source_finding_ids=_artifact_finding_ids(items),
        citations=_artifact_citations(items),
    )


# ── 辅助函数 ──────────────────────────────────────────────────────────────────


def _negotiation_priority(severity: str) -> str:
    normalized = severity.casefold()
    if "high" in normalized or "human" in normalized:
        return "must_address"
    if "medium" in normalized:
        return "negotiate"
    return "consider"


def _dedupe_artifact_items(
    items: list[dict[str, Any]],
    key_name: str,
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        text = _clean_text(item.get(key_name))
        key = text.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append({**item, "item_id": f"{key_name}-{len(deduped) + 1}"})
    return deduped


def _artifact_citations(items: list[dict[str, Any]]) -> list[str]:
    citations: list[str] = []
    for item in items:
        for source_id in _source_id_list(item.get("citations")):
            if source_id not in citations:
                citations.append(source_id)
    return citations


def _artifact_finding_ids(items: list[dict[str, Any]]) -> list[str]:
    finding_ids: list[str] = []
    for item in items:
        value = _clean_text(item.get("source_finding_id"))
        if value and value not in finding_ids:
            finding_ids.append(value)
    return finding_ids

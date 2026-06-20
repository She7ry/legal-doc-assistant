"""人工确认闸门：缺失信息、高风险 finding、guard 告警时阻断自动完成。"""

from __future__ import annotations

from typing import Any

from doc_assistant.services.agent._constants import CURRENT_LAW_KEYWORDS
from doc_assistant.services.agent._helpers import _dedupe_texts, _mentions_any
from doc_assistant.services.agent.schemas import (
    AgentArtifact,
    AgentConfirmationGate,
    AgentFinding,
    MatterProfile,
)


def _build_confirmation_gates(
    *,
    objective: str,
    matter_profile: MatterProfile,
    findings: list[AgentFinding],
    missing_information: list[str],
    guard_warnings: list[str],
    artifacts: list[AgentArtifact],
    user_role: str,
) -> list[AgentConfirmationGate]:
    """构建人工确认闸门，在关键事实缺失或证据不足时阻断自动交付。

    gate_type 含义：
    - matter_fact：案件基础事实（管辖法、代表方等）未确认
    - missing_information：工作流识别到未回答的问题或缺失文档
    - legal_review / evidence：finding 或报告证据不达标
    - permission：需用户授权外部检索（如现行法规）
    - delivery：正式交付前的最终审批
    """
    gates: list[AgentConfirmationGate] = []

    if not matter_profile.governing_law and not matter_profile.jurisdiction:
        gates.append(
            AgentConfirmationGate(
                gate_id="confirm_jurisdiction",
                gate_type="matter_fact",
                title="Confirm governing law",
                question=(
                    "Confirm the governing law or jurisdiction before relying on "
                    "legal/compliance conclusions."
                ),
                priority="high",
                reason="The Matter Profile does not contain a confirmed law or jurisdiction.",
                citations=matter_profile.citations,
                metadata={"profile_field": "governing_law"},
            )
        )

    if not matter_profile.user_side:
        gates.append(
            AgentConfirmationGate(
                gate_id="confirm_user_side",
                gate_type="matter_fact",
                title="Confirm represented side",
                question="Confirm which side the user represents or wants optimized.",
                priority="high",
                reason="Negotiation advice depends on the represented party or business side.",
                citations=matter_profile.citations,
                metadata={"profile_field": "user_side"},
            )
        )

    if missing_information:
        gates.append(
            AgentConfirmationGate(
                gate_id="resolve_missing_information",
                gate_type="missing_information",
                title="Resolve missing information",
                question=(
                    f"Resolve {len(missing_information)} missing information item(s) "
                    "before treating the report as complete."
                ),
                priority="high",
                reason="The workflow identified unanswered facts or missing documents.",
                metadata={"missing_information": missing_information[:12]},
            )
        )

    review_findings = [finding for finding in findings if finding.needs_human_review]
    if review_findings:
        gates.append(
            AgentConfirmationGate(
                gate_id="review_high_risk_findings",
                gate_type="legal_review",
                title="Review flagged findings",
                question=(
                    f"Have counsel reviewed {len(review_findings)} finding(s) marked "
                    "as needing human review?"
                ),
                priority="high",
                reason="At least one finding was not safe to rely on without legal review.",
                related_finding_ids=[finding.finding_id for finding in review_findings],
                citations=_dedupe_texts(
                    [source_id for finding in review_findings for source_id in finding.citations]
                ),
            )
        )

    weak_evidence_findings = [
        finding
        for finding in findings
        if (
            not finding.citations
            or not finding.source_quote
            or not finding.location_label
            or finding.support_level != "direct"
        )
    ]
    if weak_evidence_findings:
        gates.append(
            AgentConfirmationGate(
                gate_id="resolve_finding_evidence",
                gate_type="evidence",
                title="Resolve finding evidence",
                question=(
                    f"Resolve evidence gaps for {len(weak_evidence_findings)} finding(s) "
                    "before they can enter a formal report."
                ),
                priority="high",
                reason=(
                    "Every formal finding needs a source citation, exact quote/location, "
                    "support level, unsupported reason when applicable, and human review status."
                ),
                related_finding_ids=[
                    finding.finding_id for finding in weak_evidence_findings
                ],
                citations=_dedupe_texts(
                    [
                        source_id
                        for finding in weak_evidence_findings
                        for source_id in finding.citations
                    ]
                ),
            )
        )

    if guard_warnings:
        gates.append(
            AgentConfirmationGate(
                gate_id="resolve_evidence_guard",
                gate_type="evidence",
                title="Resolve evidence warnings",
                question="Resolve evidence guard warnings before using the output externally.",
                priority="high",
                reason="The verifier found unsupported or weakly supported report content.",
                metadata={"guard_warnings": guard_warnings[:12]},
            )
        )

    if _mentions_any(objective.casefold(), CURRENT_LAW_KEYWORDS):
        gates.append(
            AgentConfirmationGate(
                gate_id="authorize_external_research",
                gate_type="permission",
                title="Authorize external research",
                question=(
                    "Confirm whether the Agent may search current public legal sources "
                    "before making up-to-date legal statements."
                ),
                priority="normal",
                reason="The objective asks for current law, regulation, or compliance context.",
                required=True,
                metadata={"requested_capability": "web_search"},
            )
        )

    if findings or artifacts:
        gates.append(
            AgentConfirmationGate(
                gate_id="approve_report_use",
                gate_type="delivery",
                title="Approve report use",
                question=(
                    "Confirm the evidence is sufficient before treating this as a formal "
                    "deliverable or negotiation position."
                ),
                priority="normal" if user_role == "lawyer" else "high",
                reason="Legal deliverables should be approved before external reliance.",
                related_artifact_ids=[artifact.artifact_id for artifact in artifacts],
                citations=_dedupe_texts(
                    [source_id for artifact in artifacts for source_id in artifact.citations]
                ),
                metadata={"user_role": user_role},
            )
        )

    return _dedupe_confirmation_gates(gates)


def _dedupe_confirmation_gates(
    gates: list[AgentConfirmationGate],
) -> list[AgentConfirmationGate]:
    deduped: list[AgentConfirmationGate] = []
    seen: set[str] = set()
    for gate in gates:
        key = gate.gate_id.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(gate)
    return deduped[:12]


def _confirmation_gate_payload(gate: AgentConfirmationGate) -> dict[str, Any]:
    return {
        "gate_id": gate.gate_id,
        "gate_type": gate.gate_type,
        "title": gate.title,
        "question": gate.question,
        "status": gate.status,
        "priority": gate.priority,
        "required": gate.required,
        "reason": gate.reason,
        "related_finding_ids": gate.related_finding_ids,
        "related_artifact_ids": gate.related_artifact_ids,
        "citations": gate.citations,
        "metadata": gate.metadata,
    }

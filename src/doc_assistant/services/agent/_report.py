"""Agent 最终报告渲染：将 findings、artifacts、gates 等组装为 Markdown。"""

from __future__ import annotations

from doc_assistant.services.agent._helpers import _format_refs
from doc_assistant.services.agent.schemas import (
    AgentArtifact,
    AgentConfirmationGate,
    AgentFinding,
    AgentStepResult,
    MatterProfile,
)


def render_agent_report(
    *,
    objective: str,
    user_role: str,
    steps: list[AgentStepResult],
    findings: list[AgentFinding],
    missing_information: list[str],
    matter_profile: MatterProfile | None,
    artifacts: list[AgentArtifact],
    confirmation_gates: list[AgentConfirmationGate],
) -> str:
    """将 findings、artifacts、gates 等组装为 Markdown 格式的最终 Agent 报告。"""
    lines = [
        "## Agent task report", f"Objective: {objective}",
        f"User mode: {user_role}", "", "## Matter profile",
    ]
    if matter_profile:
        parties = ", ".join(matter_profile.parties) or "Unknown"
        lines.extend([
            f"- Matter ID: {matter_profile.matter_id}",
            f"- Document type: {matter_profile.document_type}",
            f"- Parties: {parties}",
            f"- User side: {matter_profile.user_side or 'Unspecified'}",
            f"- Governing law: {matter_profile.governing_law or 'Unspecified'}",
            f"- Jurisdiction: {matter_profile.jurisdiction or 'Unspecified'}",
            f"- Review scope: {', '.join(matter_profile.review_scope) or 'Unspecified'}",
        ])
    else:
        lines.append("- No structured matter profile was produced.")

    lines.extend(["", "## Work performed"])
    for step in steps:
        if step.tool == "synthesize_report":
            continue
        refs = _format_refs([c.source_id for c in step.citations[:2]])
        lines.append(f"- {step.title}: {step.status}.{refs}")

    lines.extend(["", "## Key findings"])
    if findings:
        for f in findings:
            refs = _format_refs(f.citations)
            action = f" Recommended action: {f.recommended_action}" if f.recommended_action else ""
            support = f" Support: {f.support_level}" if f.support_level else ""
            lines.append(f"- {f.category} ({f.severity}): {f.summary}{refs}{support}{action}")
    else:
        lines.append("- No structured risk findings were produced from the cited excerpts.")

    lines.extend(["", "## Missing information"])
    if missing_information:
        for item in missing_information:
            lines.append(f"- {item}")
    else:
        lines.append("- No additional missing information was identified by this workflow.")

    lines.extend(["", "## Artifacts"])
    if artifacts:
        for a in artifacts:
            lines.append(f"- {a.title}: {len(a.items)} item(s). {a.summary}")
    else:
        lines.append("- No structured artifacts were generated.")

    lines.extend(["", "## Confirmation gates"])
    if confirmation_gates:
        for gate in confirmation_gates:
            lines.append(f"- {gate.title} ({gate.priority}): {gate.question}")
    else:
        lines.append("- No blocking confirmation gates were generated.")

    lines.extend([
        "", "## Human review gate",
        "A qualified legal professional should review this output before it is used "
        "for legal decisions, negotiation positions, filings, or formal advice.",
    ])
    return "\n".join(lines).strip()

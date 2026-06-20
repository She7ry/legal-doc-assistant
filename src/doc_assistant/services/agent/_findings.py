"""Finding 提取与审计：从步骤结果中抽取 Finding，再做证据评估。

- ``findings_from_step``     从单步结果中提取结构化 Finding
- ``_audit_findings``        对每条 Finding 做证据审计，补齐支持度
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from doc_assistant.schemas.citation import Citation
from doc_assistant.services.agent._helpers import (
    _as_text_list,
    _clean_text,
    _first_text,
    _format_refs,
    _renumber_findings,
    _source_id_list,
)
from doc_assistant.services.agent.schemas import AgentFinding, AgentStepResult
from doc_assistant.services.evidence import build_evidence_profile


# ── Finding 提取 ─────────────────────────────────────────────────────────


def findings_from_step(step: AgentStepResult) -> list[AgentFinding]:
    """从单步执行结果中提取结构化 Finding 列表。"""
    metadata = step.output.get("metadata", {})
    if not isinstance(metadata, dict):
        return []
    if step.tool == "review_clause":
        return _clause_findings(step, metadata)
    if step.tool == "check_conflict":
        return _conflict_findings(step, metadata)
    if step.tool in {
        "compare_document_versions", "build_evidence_profile", "suggest_clause_revision",
    }:
        return _generic_findings(step)
    return []


def _clause_findings(
    step: AgentStepResult, metadata: dict[str, Any],
) -> list[AgentFinding]:
    category = _clean_text(metadata.get("clause_type")) or step.title
    severity = _clean_text(metadata.get("risk_level")) or "Needs human review"
    needs_review = bool(metadata.get("needs_human_review", True))
    recommendations = _as_text_list(metadata.get("questions_for_lawyer"))
    default_citations = [c.source_id for c in step.citations[:1]]
    reasons = metadata.get("risk_reasons")
    findings: list[AgentFinding] = []

    if isinstance(reasons, list):
        for reason in reasons:
            if not isinstance(reason, dict):
                continue
            summary = _clean_text(reason.get("reason"))
            if not summary:
                continue
            citations = _source_id_list(reason.get("citation")) or default_citations
            findings.append(AgentFinding(
                finding_id=f"f{len(findings) + 1}", category=category,
                severity=severity, summary=summary, citations=citations,
                recommended_action=_first_text(recommendations),
                needs_human_review=needs_review, source_step_id=step.step_id,
            ))

    if findings:
        return _renumber_findings(findings)

    summary = _clean_text(metadata.get("summary"))
    if not summary:
        return []
    return [AgentFinding(
        finding_id="f1", category=category, severity=severity,
        summary=summary, citations=default_citations,
        recommended_action=_first_text(recommendations),
        needs_human_review=needs_review, source_step_id=step.step_id,
    )]


def _conflict_findings(
    step: AgentStepResult, metadata: dict[str, Any],
) -> list[AgentFinding]:
    conflicts = metadata.get("conflicts")
    if not isinstance(conflicts, list):
        return []
    findings: list[AgentFinding] = []
    for conflict in conflicts:
        if not isinstance(conflict, dict):
            continue
        topic = _clean_text(conflict.get("topic")) or "Potential conflict"
        why_conflict = _clean_text(conflict.get("why_conflict"))
        if not why_conflict:
            continue
        citations = _source_id_list(conflict.get("contract_citations"))
        citations.extend(
            sid for sid in _source_id_list(conflict.get("policy_citations"))
            if sid not in citations
        )
        findings.append(AgentFinding(
            finding_id=f"f{len(findings) + 1}", category=topic,
            severity=_clean_text(conflict.get("severity")) or "Needs human review",
            summary=why_conflict, citations=citations,
            recommended_action=_clean_text(conflict.get("recommended_action")),
            needs_human_review=bool(conflict.get("needs_human_review", True)),
            source_step_id=step.step_id,
        ))
    return _renumber_findings(findings)


def _generic_findings(step: AgentStepResult) -> list[AgentFinding]:
    findings: list[AgentFinding] = []
    evidence = step.evidence if isinstance(step.evidence, dict) else {}
    claims = evidence.get("claims")
    if isinstance(claims, list):
        for claim in claims[:6]:
            if not isinstance(claim, dict):
                continue
            text = _clean_text(claim.get("text"))
            if not text:
                continue
            citations = _source_id_list(claim.get("citations"))
            if not citations:
                citations = [c.source_id for c in step.citations[:1]]
            support_level = _clean_text(claim.get("support_level")) or "partial"
            findings.append(AgentFinding(
                finding_id=f"f{len(findings) + 1}", category=step.title,
                severity="Medium" if support_level == "direct" else "Needs human review",
                summary=text, citations=citations,
                recommended_action="Confirm the business position and source support.",
                needs_human_review=support_level != "direct",
                source_step_id=step.step_id,
            ))
    if findings:
        return _renumber_findings(findings)
    summary = _clean_text(step.summary)
    if not summary:
        return []
    return [AgentFinding(
        finding_id="f1", category=step.title, severity="Needs human review",
        summary=summary[:500],
        citations=[c.source_id for c in step.citations[:2]],
        recommended_action="Review and confirm before relying on this output.",
        needs_human_review=True, source_step_id=step.step_id,
    )]


# ── Finding 审计 ─────────────────────────────────────────────────────────


def _audit_findings(
    findings: list[AgentFinding],
    citations: list[Citation],
) -> list[AgentFinding]:
    """对每条 finding 做证据审计，补齐引用、原文摘录、支持度与人工复核状态。"""
    if not findings:
        return []

    citations_by_id = {citation.source_id.upper(): citation for citation in citations}
    audited: list[AgentFinding] = []
    for finding in findings:
        normalized_ids = [
            source_id
            for source_id in _source_id_list(finding.citations)
            if source_id in citations_by_id
        ]
        claim_text = f"{finding.summary}{_format_refs(normalized_ids)}"
        profile = build_evidence_profile(claim_text, citations)
        claim = _first_evidence_claim(profile)
        evidence_items = claim.get("evidence", []) if claim else []
        if not isinstance(evidence_items, list):
            evidence_items = []

        support_level = _clean_text(claim.get("support_level")) if claim else ""
        if not support_level:
            support_level = "missing" if not normalized_ids else "partial"

        unsupported_reason = _finding_unsupported_reason(
            claim=claim,
            has_valid_citations=bool(normalized_ids),
        )
        source_quote = _first_evidence_text(evidence_items, "quote")
        location_label = _first_evidence_text(evidence_items, "location_label")
        if not source_quote and normalized_ids:
            citation = citations_by_id[normalized_ids[0]]
            source_quote = citation.exact_quote or citation.preview
            location_label = citation.location_label()

        evidence_coverage = _finding_evidence_coverage(
            support_level=support_level,
            has_quote=bool(source_quote),
            has_location=bool(location_label),
            citation_count=len(normalized_ids),
        )
        needs_human_review = (
            finding.needs_human_review
            or support_level != "direct"
            or evidence_coverage != "direct"
        )
        human_review_status = "pending" if needs_human_review else "not_required"
        status = "needs_human_review" if needs_human_review else "evidence_backed"
        if support_level != "direct" and not unsupported_reason:
            unsupported_reason = "The finding is not directly supported by the cited excerpt."

        audited.append(
            replace(
                finding,
                citations=normalized_ids,
                evidence_coverage=evidence_coverage,
                support_level=support_level,
                unsupported_reason=unsupported_reason,
                source_quote=source_quote[:1200],
                location_label=location_label,
                human_review_status=human_review_status,
                status=status,
                evidence=[
                    item
                    for item in evidence_items
                    if isinstance(item, dict)
                ],
            )
        )
    return audited


def _first_evidence_claim(profile: dict[str, Any]) -> dict[str, Any] | None:
    claims = profile.get("claims")
    if not isinstance(claims, list):
        return None
    for claim in claims:
        if isinstance(claim, dict):
            return claim
    return None


def _first_evidence_text(evidence_items: list[Any], key: str) -> str:
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        text = _clean_text(item.get(key))
        if text:
            return text
    return ""


def _finding_unsupported_reason(
    *,
    claim: dict[str, Any] | None,
    has_valid_citations: bool,
) -> str:
    if not has_valid_citations:
        return "Missing source citation."
    if not claim:
        return "Evidence support could not be evaluated."
    unsupported_facts = claim.get("unsupported_facts")
    if isinstance(unsupported_facts, list) and unsupported_facts:
        facts = [_clean_text(item) for item in unsupported_facts if _clean_text(item)]
        if facts:
            return "Unsupported facts: " + ", ".join(facts)
    return _clean_text(claim.get("uncertainty"))


def _finding_evidence_coverage(
    *,
    support_level: str,
    has_quote: bool,
    has_location: bool,
    citation_count: int,
) -> str:
    if not citation_count:
        return "missing"
    if support_level == "direct" and has_quote and has_location:
        return "direct"
    if has_quote or has_location:
        return "partial"
    return "missing"

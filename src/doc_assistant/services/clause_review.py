"""条款审查逻辑：元数据解析 + Markdown 渲染。

从 ``DocumentQAService`` 中拆出；``review_clause`` 仍保留在主类作为入口，
本模块提供元数据构建与渲染。
"""

from __future__ import annotations

from typing import Any

from doc_assistant.schemas.citation import Citation
from doc_assistant.services.review_taxonomy import ClauseProfile
from doc_assistant.utils.coercion import (
    as_list_str,
    as_str,
    citation_suffix,
    coerce_bool,
    coerce_risk_level,
    extract_json_object,
    format_source_refs,
    optional_str,
    risk_reason_list,
)


def empty_clause_review_metadata(
    clause_type: str,
    profile: ClauseProfile,
) -> dict[str, Any]:
    return {
        "structured": True,
        "clause_type": clause_type,
        "normalized_clause_type": profile.key,
        "found": False,
        "summary": "No relevant content found in indexed documents for the requested clause type.",
        "risk_level": "Needs human review",
        "risk_reasons": [],
        "affected_party": None,
        "plain_language_explanation": "The system did not retrieve enough cited text to review this clause.",
        "questions_for_lawyer": [],
        "missing_information": ["Relevant clause text or a more specific clause query."],
        "needs_human_review": True,
    }


def clause_review_metadata(
    clause_type: str,
    profile: ClauseProfile,
    raw_content: str,
    citations: list[Citation],
) -> dict[str, Any]:
    data = extract_json_object(raw_content)
    if not isinstance(data, dict):
        return {
            **empty_clause_review_metadata(clause_type, profile),
            "structured": False,
            "summary": raw_content.strip(),
            "found": None,
        }

    found = coerce_bool(data.get("found"))
    risk_level = coerce_risk_level(data.get("risk_level"))
    risk_reasons = risk_reason_list(data.get("risk_reasons"), citations)
    needs_human_review = coerce_bool(data.get("needs_human_review"))
    if needs_human_review is None:
        needs_human_review = found is not True or risk_level == "Needs human review"

    summary = as_str(data.get("summary"))
    plain_language = as_str(
        data.get("plain_language_explanation")
        or data.get("plain_language")
        or data.get("explanation")
        or summary
    )

    return {
        "structured": True,
        "clause_type": as_str(data.get("clause_type"), clause_type),
        "normalized_clause_type": as_str(
            data.get("normalized_clause_type") or data.get("clause_key"),
            profile.key,
        ),
        "found": found,
        "summary": summary,
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "affected_party": optional_str(data.get("affected_party")),
        "plain_language_explanation": plain_language,
        "questions_for_lawyer": as_list_str(
            data.get("questions_for_lawyer")
            or data.get("negotiation_or_review_points")
            or data.get("review_points")
        ),
        "missing_information": as_list_str(data.get("missing_information")),
        "needs_human_review": needs_human_review,
    }


def render_clause_review(metadata: dict[str, Any], citations: list[Citation]) -> str:
    cite_suffix = citation_suffix(
        [
            reason.get("citation")
            for reason in metadata.get("risk_reasons", [])
            if isinstance(reason, dict)
        ],
        citations,
    )
    found = metadata.get("found")
    found_label = "Yes" if found is True else "No" if found is False else "Unclear"
    lines = [
        "## Clause review",
        f"Clause type: {metadata.get('clause_type') or 'Unspecified'}",
        f"Normalized type: {metadata.get('normalized_clause_type') or 'custom'}",
        f"Found: {found_label}",
        f"Risk level: {metadata.get('risk_level') or 'Needs human review'}",
    ]

    summary = as_str(metadata.get("summary"))
    if summary:
        lines.append(f"Summary: {summary}{cite_suffix}")

    affected_party = optional_str(metadata.get("affected_party"))
    if affected_party:
        lines.append(f"Affected party: {affected_party}{cite_suffix}")

    explanation = as_str(metadata.get("plain_language_explanation"))
    if explanation and explanation != summary:
        lines.append(f"Plain-language explanation: {explanation}{cite_suffix}")

    risk_reasons = [
        reason
        for reason in metadata.get("risk_reasons", [])
        if isinstance(reason, dict) and reason.get("reason")
    ]
    if risk_reasons:
        lines.append("\n## Risk reasons")
        for reason in risk_reasons:
            reason_suffix = citation_suffix([reason.get("citation")], citations)
            lines.append(f"- {reason['reason']}{reason_suffix}")

    questions = as_list_str(metadata.get("questions_for_lawyer"))
    if questions:
        lines.append("\n## Questions for lawyer")
        for question in questions:
            lines.append(f"- {question}{cite_suffix}")

    missing_information = as_list_str(metadata.get("missing_information"))
    if missing_information:
        lines.append("\n## Missing information")
        for item in missing_information:
            lines.append(f"- {item}")

    if metadata.get("needs_human_review"):
        lines.append("\nNeeds human review: Yes")

    return "\n".join(lines).strip()

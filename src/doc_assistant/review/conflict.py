"""合同-政策冲突比对逻辑：元数据解析 + Markdown 渲染。

从 ``DocumentQAService`` 中拆出；``check_conflict`` 仍保留在主类作为入口，
本模块提供元数据构建与渲染。
"""

from __future__ import annotations

from typing import Any

from doc_assistant.schemas.citation import Citation
from doc_assistant.utils.coercion import (
    as_str,
    coerce_bool,
    coerce_conflict_status,
    coerce_conflict_type,
    coerce_risk_level,
    extract_json_object,
    format_source_refs,
    optional_str,
    source_id_list,
)


def empty_conflict_metadata() -> dict[str, Any]:
    return {
        "structured": True,
        "overall_status": "Insufficient information",
        "conflicts": [],
        "needs_human_review": True,
        "supporting_citations": [],
    }


def conflict_metadata(raw_content: str, citations: list[Citation]) -> dict[str, Any]:
    data = extract_json_object(raw_content)
    if not isinstance(data, dict):
        return {
            **empty_conflict_metadata(),
            "structured": False,
            "overall_status": coerce_conflict_status(raw_content),
        }

    raw_conflicts = data.get("conflicts")
    conflicts: list[dict[str, Any]] = []
    if isinstance(raw_conflicts, list):
        for raw_conflict in raw_conflicts:
            if not isinstance(raw_conflict, dict):
                continue
            contract_citations = source_id_list(
                raw_conflict.get("contract_citations")
                or raw_conflict.get("contract_citation"),
                citations,
                prefix="C",
            )
            policy_citations = source_id_list(
                raw_conflict.get("policy_citations")
                or raw_conflict.get("policy_citation"),
                citations,
                prefix="P",
            )
            severity = coerce_risk_level(raw_conflict.get("severity"))
            needs_human_review = coerce_bool(raw_conflict.get("needs_human_review"))
            if needs_human_review is None:
                needs_human_review = severity == "Needs human review"
            conflicts.append(
                {
                    "topic": as_str(raw_conflict.get("topic"), "Unspecified topic"),
                    "conflict_type": coerce_conflict_type(
                        raw_conflict.get("conflict_type")
                    ),
                    "severity": severity,
                    "contract_position": as_str(
                        raw_conflict.get("contract_position")
                    ),
                    "policy_position": as_str(raw_conflict.get("policy_position")),
                    "why_conflict": as_str(
                        raw_conflict.get("why_conflict")
                        or raw_conflict.get("explanation")
                        or raw_conflict.get("reason")
                    ),
                    "recommended_action": as_str(
                        raw_conflict.get("recommended_action")
                        or raw_conflict.get("next_step")
                    ),
                    "contract_citations": contract_citations,
                    "policy_citations": policy_citations,
                    "needs_human_review": needs_human_review,
                    "confidence": optional_str(raw_conflict.get("confidence")),
                }
            )

    overall_status = coerce_conflict_status(data.get("overall_status"))
    if overall_status == "Insufficient information" and conflicts:
        overall_status = "Potential conflict"
    needs_human_review = coerce_bool(data.get("needs_human_review"))
    if needs_human_review is None:
        needs_human_review = overall_status == "Insufficient information" or any(
            conflict.get("needs_human_review") for conflict in conflicts
        )

    return {
        "structured": True,
        "overall_status": overall_status,
        "conflicts": conflicts,
        "needs_human_review": needs_human_review,
        "supporting_citations": source_id_list(
            data.get("supporting_citations"),
            citations,
        ),
    }


def render_conflict_check(metadata: dict[str, Any]) -> str:
    lines = [
        "## Conflict check",
        f"Status: {metadata.get('overall_status') or 'Insufficient information'}",
    ]
    conflicts = [
        conflict
        for conflict in metadata.get("conflicts", [])
        if isinstance(conflict, dict)
    ]
    if not conflicts:
        supporting_suffix = format_source_refs(metadata.get("supporting_citations", []))
        if metadata.get("overall_status") == "No conflict found":
            lines.append(f"No conflict found based on the provided excerpts.{supporting_suffix}")
        else:
            lines.append(
                "Insufficient cited information was found to produce a structured conflict item."
            )
        if metadata.get("needs_human_review"):
            lines.append("Needs human review: Yes")
        return "\n".join(lines).strip()

    for index, conflict in enumerate(conflicts, start=1):
        contract_refs = conflict.get("contract_citations", [])
        policy_refs = conflict.get("policy_citations", [])
        evidence_suffix = format_source_refs([*contract_refs, *policy_refs])
        lines.extend(
            [
                f"\n## Conflict {index}: {conflict.get('topic') or 'Unspecified topic'}",
                f"Type: {conflict.get('conflict_type')}",
                f"Severity: {conflict.get('severity')}",
            ]
        )
        contract_position = as_str(conflict.get("contract_position"))
        if contract_position:
            lines.append(
                f"Contract position: {contract_position}"
                f"{format_source_refs(contract_refs)}"
            )
        policy_position = as_str(conflict.get("policy_position"))
        if policy_position:
            lines.append(
                f"Policy position: {policy_position}{format_source_refs(policy_refs)}"
            )
        why_conflict = as_str(conflict.get("why_conflict"))
        if why_conflict:
            lines.append(f"Why this may conflict: {why_conflict}{evidence_suffix}")
        recommended_action = as_str(conflict.get("recommended_action"))
        if recommended_action:
            lines.append(f"Recommended next step: {recommended_action}{evidence_suffix}")
        if conflict.get("needs_human_review"):
            lines.append("Needs human review: Yes")

    return "\n".join(lines).strip()

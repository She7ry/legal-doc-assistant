"""合同-政策冲突比对逻辑：元数据解析 + Markdown 渲染。

从 ``DocumentQAService`` 中拆出；``check_conflict`` 仍保留在主类作为入口，
本模块提供元数据构建与渲染。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ai.rag.schemas import Citation
from ai.review.coercion import (
    as_str,
    coerce_conflict_type,
    format_source_refs,
    source_id_list,
)
from ai.utils.text import optional_text


class ConflictItemOutput(BaseModel):
    topic: str
    conflict_type: str
    severity: Literal["High", "Medium", "Low", "Needs human review"]
    contract_position: str = ""
    policy_position: str = ""
    why_conflict: str = ""
    recommended_action: str = ""
    contract_citations: list[str] = Field(default_factory=list)
    policy_citations: list[str] = Field(default_factory=list)
    needs_human_review: bool = True
    confidence: str | None = None


class ConflictCheckOutput(BaseModel):
    overall_status: Literal[
        "No conflict found", "Potential conflict", "Insufficient information"
    ]
    conflicts: list[ConflictItemOutput] = Field(default_factory=list)
    needs_human_review: bool
    supporting_citations: list[str] = Field(default_factory=list)


def empty_conflict_metadata() -> dict[str, Any]:
    return {
        "structured": True,
        "overall_status": "Insufficient information",
        "conflicts": [],
        "needs_human_review": True,
        "supporting_citations": [],
    }


def conflict_metadata(output: ConflictCheckOutput, citations: list[Citation]) -> dict[str, Any]:
    conflicts: list[dict[str, Any]] = []
    for conflict in output.conflicts:
        conflicts.append(
            {
                "topic": as_str(conflict.topic, "未指定主题"),
                "conflict_type": coerce_conflict_type(conflict.conflict_type),
                "severity": conflict.severity,
                "contract_position": as_str(conflict.contract_position),
                "policy_position": as_str(conflict.policy_position),
                "why_conflict": as_str(conflict.why_conflict),
                "recommended_action": as_str(conflict.recommended_action),
                "contract_citations": source_id_list(
                    conflict.contract_citations, citations, prefix="C"
                ),
                "policy_citations": source_id_list(
                    conflict.policy_citations, citations, prefix="P"
                ),
                "needs_human_review": conflict.needs_human_review,
                "confidence": optional_text(conflict.confidence),
            }
        )

    overall_status = output.overall_status
    if overall_status == "Insufficient information" and conflicts:
        overall_status = "Potential conflict"

    return {
        "structured": True,
        "overall_status": overall_status,
        "conflicts": conflicts,
        "needs_human_review": output.needs_human_review,
        "supporting_citations": source_id_list(output.supporting_citations, citations),
    }


def render_conflict_check(metadata: dict[str, Any]) -> str:
    status = {
        "No conflict found": "未发现冲突",
        "Potential conflict": "可能存在冲突",
        "Insufficient information": "信息不足",
    }.get(metadata.get("overall_status"), "信息不足")
    lines = [
        "## 冲突检查",
        f"状态：{status}",
    ]
    conflicts = [
        conflict
        for conflict in metadata.get("conflicts", [])
        if isinstance(conflict, dict)
    ]
    if not conflicts:
        supporting_suffix = format_source_refs(metadata.get("supporting_citations", []))
        if metadata.get("overall_status") == "No conflict found":
            lines.append(f"根据所提供的摘录，未发现冲突。{supporting_suffix}")
        else:
            lines.append("引用信息不足，无法生成结构化冲突项。")
        if metadata.get("needs_human_review"):
            lines.append("需要人工审阅：是")
        return "\n".join(lines).strip()

    for index, conflict in enumerate(conflicts, start=1):
        contract_refs = conflict.get("contract_citations", [])
        policy_refs = conflict.get("policy_citations", [])
        evidence_suffix = format_source_refs([*contract_refs, *policy_refs])
        lines.extend(
            [
                f"\n## 冲突 {index}：{conflict.get('topic') or '未指定主题'}",
                f"类型：{conflict.get('conflict_type')}",
                "严重程度："
                + {
                    "High": "高",
                    "Medium": "中",
                    "Low": "低",
                    "Needs human review": "需要人工审阅",
                }.get(conflict.get("severity"), "需要人工审阅"),
            ]
        )
        contract_position = as_str(conflict.get("contract_position"))
        if contract_position:
            lines.append(
                f"合同立场：{contract_position}"
                f"{format_source_refs(contract_refs)}"
            )
        policy_position = as_str(conflict.get("policy_position"))
        if policy_position:
            lines.append(f"政策立场：{policy_position}{format_source_refs(policy_refs)}")
        why_conflict = as_str(conflict.get("why_conflict"))
        if why_conflict:
            lines.append(f"可能冲突的原因：{why_conflict}{evidence_suffix}")
        recommended_action = as_str(conflict.get("recommended_action"))
        if recommended_action:
            lines.append(f"建议下一步：{recommended_action}{evidence_suffix}")
        if conflict.get("needs_human_review"):
            lines.append("需要人工审阅：是")

    return "\n".join(lines).strip()

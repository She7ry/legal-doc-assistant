"""条款审查逻辑：元数据解析 + Markdown 渲染。

从 ``DocumentQAService`` 中拆出；``review_clause`` 仍保留在主类作为入口，
本模块提供元数据构建与渲染。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ai.rag.schemas import Citation
from ai.review.coercion import (
    as_list_str,
    as_str,
    citation_suffix,
    risk_reason_list,
)
from ai.review.taxonomy import ClauseProfile
from ai.utils.text import optional_text


class ClauseRiskReason(BaseModel):
    reason: str
    citation: str | None = None


class ClauseReviewOutput(BaseModel):
    clause_type: str
    normalized_clause_type: str
    found: bool
    summary: str
    risk_level: Literal["High", "Medium", "Low", "Needs human review"]
    risk_reasons: list[ClauseRiskReason] = Field(default_factory=list)
    affected_party: str | None = None
    plain_language_explanation: str = ""
    questions_for_lawyer: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    needs_human_review: bool


def empty_clause_review_metadata(
    clause_type: str,
    profile: ClauseProfile,
) -> dict[str, Any]:
    return {
        "structured": True,
        "clause_type": clause_type,
        "normalized_clause_type": profile.key,
        "found": False,
        "summary": "索引文档中未找到与所请求条款类型相关的内容。",
        "risk_level": "Needs human review",
        "risk_reasons": [],
        "affected_party": None,
        "plain_language_explanation": "当前没有检索到足以审阅该条款的引用文本。",
        "questions_for_lawyer": [],
        "missing_information": ["相关条款原文，或更具体的条款检索条件。"],
        "needs_human_review": True,
    }


def clause_review_metadata(
    clause_type: str,
    profile: ClauseProfile,
    output: ClauseReviewOutput,
    citations: list[Citation],
) -> dict[str, Any]:
    data = output.model_dump()
    summary = as_str(output.summary)

    return {
        "structured": True,
        "clause_type": as_str(output.clause_type, clause_type),
        "normalized_clause_type": as_str(output.normalized_clause_type, profile.key),
        "found": output.found,
        "summary": summary,
        "risk_level": output.risk_level,
        "risk_reasons": risk_reason_list(data["risk_reasons"], citations),
        "affected_party": optional_text(output.affected_party),
        "plain_language_explanation": as_str(output.plain_language_explanation or summary),
        "questions_for_lawyer": as_list_str(output.questions_for_lawyer),
        "missing_information": as_list_str(output.missing_information),
        "needs_human_review": output.needs_human_review,
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
    found_label = "是" if found is True else "否" if found is False else "不明确"
    risk_level = {
        "High": "高",
        "Medium": "中",
        "Low": "低",
        "Needs human review": "需要人工审阅",
    }.get(metadata.get("risk_level"), "需要人工审阅")
    lines = [
        "## 条款审阅",
        f"条款类型：{metadata.get('clause_type') or '未指定'}",
        f"归一化类型：{metadata.get('normalized_clause_type') or '自定义'}",
        f"是否找到：{found_label}",
        f"风险等级：{risk_level}",
    ]

    summary = as_str(metadata.get("summary"))
    if summary:
        lines.append(f"摘要：{summary}{cite_suffix}")

    affected_party = optional_text(metadata.get("affected_party"))
    if affected_party:
        lines.append(f"受影响方：{affected_party}{cite_suffix}")

    explanation = as_str(metadata.get("plain_language_explanation"))
    if explanation and explanation != summary:
        lines.append(f"通俗解释：{explanation}{cite_suffix}")

    risk_reasons = [
        reason
        for reason in metadata.get("risk_reasons", [])
        if isinstance(reason, dict) and reason.get("reason")
    ]
    if risk_reasons:
        lines.append("\n## 风险理由")
        for reason in risk_reasons:
            reason_suffix = citation_suffix([reason.get("citation")], citations)
            lines.append(f"- {reason['reason']}{reason_suffix}")

    questions = as_list_str(metadata.get("questions_for_lawyer"))
    if questions:
        lines.append("\n## 可向律师咨询的问题")
        for question in questions:
            lines.append(f"- {question}{cite_suffix}")

    missing_information = as_list_str(metadata.get("missing_information"))
    if missing_information:
        lines.append("\n## 缺失信息")
        for item in missing_information:
            lines.append(f"- {item}")

    if metadata.get("needs_human_review"):
        lines.append("\n需要人工审阅：是")

    return "\n".join(lines).strip()

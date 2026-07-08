"""证据画像：把答案拆成主张（claims）并评估每条是否有引用支持。

供 QA、ToolCalling、Agent 在 metadata / finding 审计中使用；
support_level 分为 direct（直接支持）、partial（部分支持）、missing（缺失）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from doc_assistant.schemas.citation import Citation

SOURCE_REF_PATTERN = re.compile(r"\[([SCDPW]\d+)\]", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|\d+(?:\.\d+)?%?|[\u4e00-\u9fff]")
FACT_PATTERN = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    r"\b\d{4}年\d{1,2}月\d{1,2}日\b|"
    r"\$\s?\d[\d,]*(?:\.\d+)?|"
    r"\b(?:USD|EUR|RMB|CNY)\s?\d[\d,]*(?:\.\d+)?\b|"
    r"\b\d+(?:\.\d+)?%\b|"
    r"\b\d+\s+(?:days?|business days?|calendar days?|months?|years?)\b|"
    r"\d+\s*(?:个工作日|工作日|日|天|个月|月|年)",
    re.IGNORECASE,
)
STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "based",
    "before",
    "between",
    "but",
    "can",
    "cannot",
    "confidence",
    "could",
    "does",
    "for",
    "from",
    "has",
    "have",
    "high",
    "into",
    "low",
    "may",
    "medium",
    "must",
    "not",
    "only",
    "or",
    "should",
    "source",
    "that",
    "the",
    "this",
    "under",
    "with",
    "within",
    "would",
}
# 主张与引用片段的 token 重叠率阈值：≥0.45 为直接支持，≥0.20 为部分支持。
DIRECT_SUPPORT_THRESHOLD = 0.45
PARTIAL_SUPPORT_THRESHOLD = 0.20


def build_evidence_profile(
    answer: str,
    citations: Sequence[Citation],
    guard_issues: Sequence[str] | None = None,
) -> dict[str, Any]:
    """将答案拆分为可审计主张（claims），逐条评估引用支持度。

    返回 support_level（direct / partial / missing）、证据摘录、
    不支持的主张列表，供 Agent finding 审计与报告闸门使用。
    """
    citations_by_id = {citation.source_id.upper(): citation for citation in citations}
    claims = []
    unsupported_claims = []

    for text in _candidate_claims(answer):
        cited_ids = _source_ids(text)
        valid_ids = [source_id for source_id in cited_ids if source_id in citations_by_id]
        evidence = [_evidence_item(citations_by_id[source_id]) for source_id in valid_ids]
        cited_text = "\n".join(str(item["quote"] or "") for item in evidence)
        support_score = _support_score(text, cited_text)
        unsupported_facts = _unsupported_facts(text, cited_text) if valid_ids else []
        support_level = _support_level(
            cited_ids,
            valid_ids,
            support_score=support_score,
            unsupported_facts=unsupported_facts,
        )
        needs_human_review = support_level != "direct"
        uncertainty = _uncertainty_for_claim(support_level, unsupported_facts)

        if support_level == "missing":
            unsupported_claims.append(text)

        claims.append(
            {
                "claim_id": f"c{len(claims) + 1}",
                "text": text,
                "citations": valid_ids,
                "support_level": support_level,
                "support_score": support_score,
                "unsupported_facts": unsupported_facts,
                "evidence": evidence,
                "uncertainty": uncertainty,
                "needs_human_review": needs_human_review,
            }
        )

    missing_evidence = _missing_evidence(guard_issues or [], unsupported_claims)
    return {
        "claims": claims,
        "unsupported_claims": unsupported_claims,
        "missing_evidence": missing_evidence,
        "possibly_conflicting_clauses": [],
    }


# ---------------------------------------------------------------------------
# 内部辅助：主张拆分、token 重叠打分、事实术语匹配
# ---------------------------------------------------------------------------


def _candidate_claims(answer: str) -> list[str]:
    """从答案逐行提取可审计主张，过滤标题、元信息和过短行。"""
    claims = []
    for raw_line in (answer or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        if not _looks_material(line):
            continue
        claims.append(line)
    return claims


def _looks_material(text: str) -> bool:
    lowered = text.casefold()
    if len(text) < 28:
        return False
    if lowered.startswith(("confidence:", "answer:", "assistant:", "note:")):
        return False
    return True


def _source_ids(text: str) -> list[str]:
    result = []
    for match in SOURCE_REF_PATTERN.finditer(text or ""):
        source_id = match.group(1).upper()
        if source_id not in result:
            result.append(source_id)
    return result


def _support_level(
    cited_ids: list[str],
    valid_ids: list[str],
    *,
    support_score: float,
    unsupported_facts: list[str],
) -> str:
    """判定主张的支持等级：missing（无引用）/ partial（部分匹配）/ direct（充分支持）。"""
    if not cited_ids:
        return "missing"
    if not valid_ids:
        return "missing"
    if len(valid_ids) != len(cited_ids):
        return "partial"
    if unsupported_facts:
        return "partial"
    if support_score >= DIRECT_SUPPORT_THRESHOLD:
        return "direct"
    if support_score >= PARTIAL_SUPPORT_THRESHOLD:
        return "partial"
    return "partial"


def _support_score(claim: str, cited_text: str) -> float:
    """用主张与引用片段的 token 重叠率估算支持度（Jaccard 式比率）。"""
    claim_tokens = _material_tokens(_strip_source_refs(claim))
    if not claim_tokens:
        return 1.0 if cited_text.strip() else 0.0
    cited_tokens = _material_tokens(cited_text)
    if not cited_tokens:
        return 0.0

    overlap = claim_tokens & cited_tokens
    return len(overlap) / len(claim_tokens)


def _unsupported_facts(claim: str, cited_text: str) -> list[str]:
    normalized_evidence = _normalize_fact_text(cited_text)
    unsupported = []
    for fact in _fact_terms(_strip_source_refs(claim)):
        if _normalize_fact_text(fact) not in normalized_evidence:
            unsupported.append(fact)
    return unsupported


def _fact_terms(text: str) -> list[str]:
    facts = []
    for match in FACT_PATTERN.finditer(text or ""):
        fact = " ".join(match.group(0).split())
        if fact not in facts:
            facts.append(fact)
    return facts


def _material_tokens(text: str) -> set[str]:
    tokens = set()
    for match in TOKEN_PATTERN.finditer(text or ""):
        token = match.group(0).casefold()
        if token in STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _strip_source_refs(text: str) -> str:
    return SOURCE_REF_PATTERN.sub("", text or "")


def _normalize_fact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").casefold()


def _uncertainty_for_claim(support_level: str, unsupported_facts: Sequence[str]) -> str:
    if support_level == "direct":
        return "Supported by the cited excerpt, subject to full-document context."
    if support_level == "partial":
        if unsupported_facts:
            return "The cited excerpt does not contain these specific facts: " + ", ".join(
                unsupported_facts
            )
        return "The citation is available, but the claim only partially matches the cited excerpt."
    return "No source citation was attached to this material claim."


def _missing_evidence(guard_issues: Sequence[str], unsupported_claims: Sequence[str]) -> list[str]:
    missing = []
    for issue in guard_issues:
        lowered = issue.casefold()
        if (
            "lacks a source citation" in lowered
            or "without a nearby citation" in lowered
            or "without retrieved documents" in lowered
            or "does not include any source citations" in lowered
        ):
            missing.append(issue)
    if unsupported_claims and not missing:
        missing.append("One or more material claims were not tied to a source citation.")
    return missing


def _evidence_item(citation: Citation) -> dict[str, Any]:
    return {
        "source_id": citation.source_id,
        "source_type": citation.source_type,
        "file_name": citation.file_name,
        "file_id": citation.file_id,
        "document_key": citation.document_key,
        "document_version": citation.document_version,
        "page": citation.page,
        "page_label": citation.page_label,
        "chunk_id": citation.chunk_id,
        "section_heading": citation.section_heading,
        "quote": citation.exact_quote or citation.preview,
        "location_label": citation.location_label(),
        "char_start": citation.char_start,
        "char_end": citation.char_end,
        "retrieval_score": citation.retrieval_score,
        "retrieval_relevance": citation.retrieval_relevance,
    }

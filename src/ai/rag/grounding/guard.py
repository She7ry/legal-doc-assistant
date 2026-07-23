"""答案合规校验（Answer Guard）。

在 LLM 输出返回用户前，用规则检测：
- 是否包含合法格式的引用标记 [Sx]
- 是否出现未标注来源的强法律结论、法条、具体事实
- 无检索文档时是否明确说明证据不足

``validate_answer`` 返回 ``AnswerGuardResult``；``needs_repair=True`` 时
qa_service 会尝试用 answer_repair prompt 自动修复一次。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai.rag.grounding.citation_support import (
    CitationSupportResult,
    verify_citation_support,
)
from ai.rag.schemas import Citation

# 答案中合法的引用标记格式，如 [S1]、[D2]、[C3] 等。
CITATION_PATTERN = re.compile(r"\[(?:S|D|C|P|W)\d+\]")
# 需软化的强法律结论表述，避免模型给出确定性胜诉/无效等承诺。
_STRONG_LEGAL_CONCLUSION_PATTERNS = (
    r"必然胜诉",
    r"一定能赢",
    r"一定会赢",
    r"该条款无效",
    r"条款无效",
    r"必然",
    r"一定会",
    r"必定",
    r"保证胜诉",
)
# 法条、判例、法规缩写等权威引用，附近必须有 [Sx] 标注来源。
_UNSOURCED_AUTHORITY_PATTERNS = (
    r"(?:民法典|刑法|劳动法|合同法|公司法)",
    r"第[一二三四五六七八九十百千\d]+条",
)
# 具体事实（日期、金额、期限等），附近必须有引用或能在检索片段中找到。
_UNSOURCED_FACT_PATTERNS = (
    r"\d+(?:\.\d+)?%",
    r"\d{4}年\d{1,2}月(?:\d{1,2}日)?",
    r"(?:人民币|[￥¥])\s*\d[\d,]*(?:\.\d+)?(?:万|亿)?元?",
    r"\d[\d,]*(?:\.\d+)?(?:万|亿)?元",
    r"\d+\s*(?:个)?(?:工作日|自然日|日|天|个月|月|年)",
)
STRONG_LEGAL_CONCLUSION_PATTERNS = tuple(
    re.compile(pattern) for pattern in _STRONG_LEGAL_CONCLUSION_PATTERNS
)
UNSOURCED_AUTHORITY_PATTERNS = tuple(
    re.compile(pattern) for pattern in _UNSOURCED_AUTHORITY_PATTERNS
)
UNSOURCED_FACT_PATTERNS = tuple(
    re.compile(pattern) for pattern in _UNSOURCED_FACT_PATTERNS
)
# 无检索文档时，答案应包含这些措辞以表明证据不足，而非凭空作答。
REFUSAL_TERMS = (
    "未找到",
    "未检索到",
    "未提供",
    "无法确定",
    "无法判断",
    "无法完整回答",
    "没有足够信息",
    "信息不足",
    "文档未说明",
    "文档中未说明",
)


@dataclass(frozen=True)
class AnswerGuardResult:
    """answer_guard 对 LLM 答案的规则校验结果。

    - passed: 无 blocking issues
    - confidence: High / Medium / Low，供前端展示可信度
    - needs_repair: True 时 qa_service 会用 repair prompt 尝试自动改写一次
    """

    passed: bool
    confidence: str
    issues: list[str] = field(default_factory=list)
    needs_repair: bool = False
    citation_support: CitationSupportResult | None = None


def validate_answer(
    answer: str,
    citations: list[Citation],
    *,
    has_retrieved_documents: bool = True,
    verify_citation_semantics: bool = True,
) -> AnswerGuardResult:
    """校验 LLM 答案的引用合规性与表述风险。

    有检索文档时：要求引用存在、ID 合法、实质性段落带引用、具体事实有旁证。
    无检索文档时：要求明确承认证据缺失。
    无论何种情况：拦截强法律结论与无来源的法条引用。
    """
    text = (answer or "").strip()
    issues: list[str] = []

    if not text:
        return AnswerGuardResult(
            passed=False,
            confidence="Low",
            issues=["Answer is empty."],
            needs_repair=True,
        )

    valid_source_ids = {citation.source_id for citation in citations if citation.source_id}
    cited_ids = set(CITATION_PATTERN.findall(text))
    normalised_cited_ids = {citation_id.strip("[]") for citation_id in cited_ids}
    citation_support: CitationSupportResult | None = None

    if has_retrieved_documents:
        if not normalised_cited_ids:
            issues.append("Answer with retrieved documents does not include any source citations.")
        else:
            unknown_ids = sorted(normalised_cited_ids - valid_source_ids)
            if unknown_ids:
                issues.append(
                    "Answer cites source IDs that were not returned in retrieval: "
                    + ", ".join(f"[{source_id}]" for source_id in unknown_ids)
                    + "."
                )

        # 实质性段落（≥40 字符）必须带 [Sx] 引用，防止大段无来源论述。
        for paragraph in _non_empty_paragraphs(text):
            if _looks_like_material_paragraph(paragraph) and not CITATION_PATTERN.search(paragraph):
                issues.append("A material paragraph lacks a source citation.")
                break

        # 日期、金额等具体事实须在事实前后 120 字符窗口内有引用，或在检索片段中可找到。
        for pattern in UNSOURCED_FACT_PATTERNS:
            for match in pattern.finditer(text):
                if not _fact_supported_nearby(text, match.group(0), normalised_cited_ids, citations):
                    issues.append(
                        f"Answer includes a specific fact '{match.group(0)}' without a nearby citation."
                    )
                    break
            if issues and issues[-1].startswith("Answer includes a specific fact"):
                break

        if verify_citation_semantics:
            citation_support = verify_citation_support(text, citations)
            issues.extend(citation_support.issues)
    elif not _contains_refusal(text):
        issues.append("Answer was generated without retrieved documents but does not acknowledge missing evidence.")

    # 拦截「必然胜诉」「条款无效」等过度确定的法律结论。
    for pattern in STRONG_LEGAL_CONCLUSION_PATTERNS:
        match = pattern.search(text)
        if match:
            issues.append(
                f"Answer contains a strong legal conclusion ('{match.group(0)}') that should be softened."
            )
            break

    # 法条/判例引用须在前后 80~120 字符窗口内有 [Sx] 旁证。
    for pattern in UNSOURCED_AUTHORITY_PATTERNS:
        match = pattern.search(text)
        if match and not _authority_supported_nearby(text, match.start(), normalised_cited_ids):
            issues.append(
                f"Answer references legal authority or statute-like text ('{match.group(0)}') without citation."
            )
            break

    confidence = _confidence_from_issues(issues, has_retrieved_documents=has_retrieved_documents)
    needs_repair = bool(issues) and confidence == "Low"
    return AnswerGuardResult(
        passed=not issues,
        confidence=confidence,
        issues=issues,
        needs_repair=needs_repair,
        citation_support=citation_support,
    )


def _confidence_from_issues(issues: list[str], *, has_retrieved_documents: bool) -> str:
    if not issues:
        return "High"
    if not has_retrieved_documents:
        return "Medium" if len(issues) == 1 else "Low"

    severe_markers = (
        "does not include any source citations",
        "source ids that were not returned",
        "without retrieved documents",
        "strong legal conclusion",
    )
    if any(any(marker in issue.casefold() for marker in severe_markers) for issue in issues):
        return "Low"
    return "Medium"


def _non_empty_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        for line in block.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                paragraphs.append(stripped)
    return paragraphs


def _looks_like_material_paragraph(paragraph: str) -> bool:
    """判断段落是否为需要强制引用的实质性论述（非元信息、足够长）。"""
    if paragraph.startswith(("回答：", "助手：", "备注：", "置信度：")):
        return False
    return len(paragraph) >= 12


def _fact_supported_nearby(
    text: str,
    fact: str,
    cited_ids: set[str],
    citations: list[Citation],
) -> bool:
    """检查具体事实是否在前后 120 字符内有引用，或出现在检索片段正文中。"""
    index = text.find(fact)
    if index < 0:
        return True

    line_start = text.rfind("\n", 0, index) + 1
    line_end = text.find("\n", index)
    line = text[line_start : line_end if line_end >= 0 else len(text)]
    if CITATION_PATTERN.search(line):
        return True

    context = "\n".join(citation.preview for citation in citations)
    return fact in context


def _authority_supported_nearby(text: str, start: int, cited_ids: set[str]) -> bool:
    window = text[max(0, start - 80) : start + 120]
    return bool(CITATION_PATTERN.search(window))


def _contains_refusal(text: str) -> bool:
    return any(term in text for term in REFUSAL_TERMS)

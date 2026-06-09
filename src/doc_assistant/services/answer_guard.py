from __future__ import annotations

import re
from dataclasses import dataclass, field

from doc_assistant.schemas.citation import Citation

CITATION_PATTERN = re.compile(r"\[(?:S|D|C|P|W)\d+\]")
_STRONG_LEGAL_CONCLUSION_PATTERNS = (
    r"\bguarantee(?:d|s)?\b",
    r"\bwill definitely\b",
    r"\bwill certainly\b",
    r"\bmust win\b",
    r"\bwill win\b",
    r"\bis invalid\b",
    r"\b必然胜诉\b",
    r"\b一定能赢\b",
    r"\b一定会赢\b",
    r"\b该条款无效\b",
    r"\b条款无效\b",
    r"\b必然\b",
    r"\b一定会\b",
    r"\b必定\b",
    r"\b保证胜诉\b",
)
_UNSOURCED_AUTHORITY_PATTERNS = (
    r"\b(?:U\.?S\.?C\.?|C\.?F\.?R\.?)\s*\S+",
    r"\b\d+\s+U\.?S\.?C\.?\s*\S+",
    r"\bv\.\s+[A-Z][A-Za-z'.-]+",
    r"\b(?:GDPR|CCPA|HIPAA|SOX)\b",
    r"\b(?:民法典|刑法|劳动法|合同法|公司法)\b",
    r"\b第[一二三四五六七八九十百千\d]+条\b",
)
_UNSOURCED_FACT_PATTERNS = (
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    r"\$\s?\d[\d,]*(?:\.\d+)?",
    r"\b\d+(?:\.\d+)?%\b",
    r"\b\d+\s+(?:days?|business days?|calendar days?|months?|years?)\b",
    r"\b\d{4}年\d{1,2}月\d{1,2}日\b",
)
STRONG_LEGAL_CONCLUSION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in _STRONG_LEGAL_CONCLUSION_PATTERNS
)
UNSOURCED_AUTHORITY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in _UNSOURCED_AUTHORITY_PATTERNS
)
UNSOURCED_FACT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in _UNSOURCED_FACT_PATTERNS
)
REFUSAL_TERMS = (
    "not found",
    "not provided",
    "cannot determine",
    "not enough information",
    "relevant text was not found",
    "did not find enough relevant text",
    "do not contain",
    "does not contain",
    "未发现明显缺失信息",
    "信息不足",
)


@dataclass(frozen=True)
class AnswerGuardResult:
    passed: bool
    confidence: str
    issues: list[str] = field(default_factory=list)
    needs_repair: bool = False


def validate_answer(
    answer: str,
    citations: list[Citation],
    *,
    has_retrieved_documents: bool = True,
) -> AnswerGuardResult:
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

        for paragraph in _non_empty_paragraphs(text):
            if _looks_like_material_paragraph(paragraph) and not CITATION_PATTERN.search(paragraph):
                issues.append("A material paragraph lacks a source citation.")
                break

        for pattern in UNSOURCED_FACT_PATTERNS:
            for match in pattern.finditer(text):
                if not _fact_supported_nearby(text, match.group(0), normalised_cited_ids, citations):
                    issues.append(
                        f"Answer includes a specific fact '{match.group(0)}' without a nearby citation."
                    )
                    break
            if issues and issues[-1].startswith("Answer includes a specific fact"):
                break
    elif not _contains_refusal(text):
        issues.append("Answer was generated without retrieved documents but does not acknowledge missing evidence.")

    for pattern in STRONG_LEGAL_CONCLUSION_PATTERNS:
        match = pattern.search(text)
        if match:
            issues.append(
                f"Answer contains a strong legal conclusion ('{match.group(0)}') that should be softened."
            )
            break

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
    )


def _confidence_from_issues(issues: list[str], *, has_retrieved_documents: bool) -> str:
    if not issues:
        return "High"
    if not has_retrieved_documents:
        return "Medium" if len(issues) == 1 else "Low"

    severe_markers = (
        "does not include any source citations",
        "source IDs that were not returned",
        "without retrieved documents",
        "strong legal conclusion",
    )
    if any(any(marker in issue for marker in severe_markers) for issue in issues):
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
    lowered = paragraph.casefold()
    if lowered.startswith(("answer:", "assistant:", "note:")):
        return False
    if len(paragraph) < 40:
        return False
    return True


def _fact_supported_nearby(
    text: str,
    fact: str,
    cited_ids: set[str],
    citations: list[Citation],
) -> bool:
    index = text.find(fact)
    if index < 0:
        return True

    window = text[max(0, index - 120) : index + len(fact) + 120]
    if CITATION_PATTERN.search(window):
        return True

    context = "\n".join(citation.preview for citation in citations)
    return fact.casefold() in context.casefold()


def _authority_supported_nearby(text: str, start: int, cited_ids: set[str]) -> bool:
    window = text[max(0, start - 80) : start + 120]
    return bool(CITATION_PATTERN.search(window))


def _contains_refusal(text: str) -> bool:
    lowered = text.casefold()
    return any(term.casefold() in lowered for term in REFUSAL_TERMS)

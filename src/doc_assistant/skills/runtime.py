"""Deterministic retrieval and evidence controls activated by runtime skills."""

from __future__ import annotations

import re
from collections import defaultdict

from langchain_core.documents import Document

from doc_assistant.retrieval.document_identity import document_identity as _document_identity
from doc_assistant.skills.models import EvidenceSufficiencyResult

_COMPLEX_QUERY_PATTERN = re.compile(
    r"[;；\n]|\b(?:and|versus|vs\.?|compare|comparison|both)\b|"
    r"(?:以及|并且|同时|比较|对比|分别|两者|多个|跨文档)",
    re.IGNORECASE,
)
_SPLIT_QUERY_PATTERN = re.compile(
    r"\s+(?:and|versus|vs\.?)\s+|[;；\n]+|(?:以及|并且|同时)",
    re.IGNORECASE,
)
_NEGATIVE_QUERY_PATTERN = re.compile(
    r"\b(?:does|do|is|are)\b.{0,40}\b(?:not|no)\b|"
    r"\b(?:without|absent|missing)\b|"
    r"(?:是否没有|是否未|有没有|不存在|未规定|缺少)",
    re.IGNORECASE,
)


def is_complex_retrieval_query(question: str) -> bool:
    return len(question) >= 160 or bool(_COMPLEX_QUERY_PATTERN.search(question))


def decompose_retrieval_query(question: str, *, max_queries: int = 4) -> list[str]:
    """Produce bounded subqueries without invoking or trusting skill-provided code."""
    normalized = " ".join(question.split()).strip()
    if not normalized:
        return []
    if not is_complex_retrieval_query(normalized) or max_queries <= 1:
        return [normalized]

    candidates = [normalized]
    for part in _SPLIT_QUERY_PATTERN.split(normalized):
        part = part.strip(" ,，。?？:")
        if len(part) >= 8 and part.casefold() != normalized.casefold():
            candidates.append(part)
    return list(dict.fromkeys(candidates))[:max_queries]


def fuse_retrieval_results(
    result_sets: list[list[Document]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[Document]:
    """Fuse multi-query results with RRF and stable passage deduplication."""
    scores: defaultdict[str, float] = defaultdict(float)
    documents: dict[str, Document] = {}
    query_hits: defaultdict[str, list[int]] = defaultdict(list)
    for query_index, result_set in enumerate(result_sets):
        for rank, document in enumerate(result_set, start=1):
            key = _document_identity(document)
            scores[key] += 1.0 / (rrf_k + rank)
            query_hits[key].append(query_index)
            documents.setdefault(key, document)

    ranked_keys = sorted(scores, key=lambda key: (-scores[key], key))[: max(1, top_k)]
    fused: list[Document] = []
    for key in ranked_keys:
        document = documents[key]
        metadata = dict(document.metadata or {})
        metadata["multi_query_rrf_score"] = scores[key]
        metadata["retrieval_query_hits"] = query_hits[key]
        fused.append(Document(page_content=document.page_content, metadata=metadata))
    return fused


def assess_evidence_sufficiency(
    question: str,
    documents: list[Document],
) -> EvidenceSufficiencyResult:
    """Run a conservative pre-generation evidence audit."""
    if not documents:
        return EvidenceSufficiencyResult(
            sufficient=False,
            status="insufficient",
            reasons=("No document evidence was retrieved.",),
            missing_information=("Relevant source text is required before making a document-grounded claim.",),
        )

    reasons = [f"Retrieved {len(documents)} distinct evidence passage(s)."]
    missing: list[str] = []
    conflicts = _version_conflicts(documents)
    relevances = [
        float(value)
        for document in documents
        if isinstance((value := (document.metadata or {}).get("retrieval_relevance")), int | float)
    ]
    if relevances and max(relevances) <= 0:
        missing.append("Retrieved passages have no positive relevance signal.")
    if _NEGATIVE_QUERY_PATTERN.search(question):
        missing.append(
            "A negative or absence claim requires adequate corpus coverage; phrase the result as "
            "what was not found in the retrieved material."
        )
    if conflicts:
        reasons.append("Potentially conflicting document versions were retrieved.")

    if conflicts or missing:
        return EvidenceSufficiencyResult(
            sufficient=False,
            status="partial",
            reasons=tuple(reasons),
            missing_information=tuple(missing),
            conflicts=tuple(conflicts),
        )
    return EvidenceSufficiencyResult(
        sufficient=True,
        status="sufficient",
        reasons=tuple(reasons),
    )


def render_evidence_assessment(result: EvidenceSufficiencyResult) -> str:
    lines = [f'<evidence_sufficiency status="{result.status}">']
    lines.extend(f"- {reason}" for reason in result.reasons)
    lines.extend(f"- Missing: {item}" for item in result.missing_information)
    lines.extend(f"- Conflict: {item}" for item in result.conflicts)
    if result.status != "sufficient":
        lines.append("Answer only supported portions and explicitly state the limitations.")
    lines.append("</evidence_sufficiency>")
    return "\n".join(lines)


def _version_conflicts(documents: list[Document]) -> list[str]:
    versions: defaultdict[str, set[int]] = defaultdict(set)
    for document in documents:
        metadata = document.metadata or {}
        key = metadata.get("document_key")
        version = metadata.get("document_version")
        if key and isinstance(version, int):
            versions[str(key)].add(version)
    return [
        f"Document {key!r} appears in versions {sorted(found_versions)}."
        for key, found_versions in versions.items()
        if len(found_versions) > 1
    ]

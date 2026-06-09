from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from doc_assistant.schemas.citation import Citation

SOURCE_REF_PATTERN = re.compile(r"\[([SCDPW]\d+)\]", re.IGNORECASE)


def build_evidence_profile(
    answer: str,
    citations: Sequence[Citation],
    guard_issues: Sequence[str] | None = None,
) -> dict[str, Any]:
    citations_by_id = {citation.source_id.upper(): citation for citation in citations}
    claims = []
    unsupported_claims = []

    for text in _candidate_claims(answer):
        cited_ids = _source_ids(text)
        valid_ids = [source_id for source_id in cited_ids if source_id in citations_by_id]
        evidence = [_evidence_item(citations_by_id[source_id]) for source_id in valid_ids]
        support_level = _support_level(cited_ids, valid_ids)
        needs_human_review = support_level != "direct"
        uncertainty = _uncertainty_for_claim(support_level)

        if support_level == "missing":
            unsupported_claims.append(text)

        claims.append(
            {
                "claim_id": f"c{len(claims) + 1}",
                "text": text,
                "citations": valid_ids,
                "support_level": support_level,
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


def _candidate_claims(answer: str) -> list[str]:
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


def _support_level(cited_ids: list[str], valid_ids: list[str]) -> str:
    if not cited_ids:
        return "missing"
    if len(valid_ids) == len(cited_ids):
        return "direct"
    if valid_ids:
        return "partial"
    return "missing"


def _uncertainty_for_claim(support_level: str) -> str:
    if support_level == "direct":
        return "Supported by the cited excerpt, subject to full-document context."
    if support_level == "partial":
        return "Some cited source IDs were not available in retrieved evidence."
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

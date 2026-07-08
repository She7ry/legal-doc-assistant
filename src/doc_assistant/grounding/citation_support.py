"""Claim-level semantic checks that complement citation format validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from doc_assistant.schemas.citation import Citation

_CITATION = re.compile(r"\[([SCDPW]\d+)\]", re.IGNORECASE)
_NUMBER = re.compile(
    r"\$?\d[\d,]*(?:\.\d+)?%?|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{4}年\d{1,2}月\d{1,2}日"
)
_TOKEN = re.compile(r"[a-z][a-z0-9'-]{2,}|[\u4e00-\u9fff]+", re.IGNORECASE)
_STOP = {
    "and", "are", "for", "from", "has", "have", "into", "not", "that", "the", "their",
    "this", "was", "were", "with", "within", "source", "section",
}


@dataclass(frozen=True)
class CitationSupportCheck:
    claim: str
    citation_ids: tuple[str, ...]
    status: str
    reason: str = ""


@dataclass(frozen=True)
class CitationSupportResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    checks: list[CitationSupportCheck] = field(default_factory=list)


def verify_citation_support(answer: str, citations: list[Citation]) -> CitationSupportResult:
    """Check whether cited source text has concrete lexical and numeric support."""
    by_id = {citation.source_id.upper(): citation for citation in citations}
    checks: list[CitationSupportCheck] = []
    issues: list[str] = []
    for claim in _material_claims(answer):
        ids = tuple(dict.fromkeys(match.upper() for match in _CITATION.findall(claim)))
        if not ids:
            continue
        valid = [by_id[source_id] for source_id in ids if source_id in by_id]
        if not valid:
            continue  # Unknown IDs are reported by AnswerGuard's structural pass.
        claim_text = _CITATION.sub("", claim).strip()
        sources = "\n".join((citation.exact_quote or citation.preview) for citation in valid)
        status, reason = _support_status(claim_text, sources)
        checks.append(
            CitationSupportCheck(
                claim=claim_text,
                citation_ids=ids,
                status=status,
                reason=reason,
            )
        )
        if status == "unsupported":
            clipped = " ".join(claim_text.split())[:120]
            issues.append(
                f"Citation does not support the complete claim '{clipped}' "
                f"({', '.join(f'[{source_id}]' for source_id in ids)}): {reason}"
            )
    return CitationSupportResult(passed=not issues, issues=issues, checks=checks)


def _material_claims(text: str) -> list[str]:
    claims = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("-*0123456789. ")
        if not stripped or stripped.startswith("#"):
            continue
        claims.extend(part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", stripped) if part.strip())
    return claims


def _support_status(claim: str, source: str) -> tuple[str, str]:
    claim_numbers = {
        _normalize_number(match.group(0))
        for match in _NUMBER.finditer(claim)
        if not _is_locator_number(claim, match.start())
    }
    source_numbers = {_normalize_number(value) for value in _NUMBER.findall(source)}
    missing_numbers = sorted(claim_numbers - source_numbers)
    if missing_numbers:
        return "unsupported", "source text does not contain claimed value(s): " + ", ".join(missing_numbers)

    claim_tokens = _semantic_tokens(claim)
    source_tokens = _semantic_tokens(source)
    if not claim_tokens:
        return "supported", "No material lexical terms require comparison."
    overlap = claim_tokens & source_tokens
    coverage = len(overlap) / len(claim_tokens)
    if not overlap or (len(claim_tokens) >= 5 and coverage < 0.12):
        return "unsupported", "source text is only topically adjacent or lexically unrelated"
    return "supported", f"lexical coverage={coverage:.2f}"


def _semantic_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _TOKEN.findall(text.casefold()):
        if token in _STOP:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 2:
            tokens.update(token[index : index + 2] for index in range(len(token) - 1))
        else:
            tokens.add(token)
    return tokens


def _normalize_number(value: str) -> str:
    return value.replace(",", "").replace(" ", "").casefold()


def _is_locator_number(text: str, start: int) -> bool:
    prefix = text[max(0, start - 12) : start].casefold()
    return bool(re.search(r"(?:section|article|clause|§|第)\s*$", prefix))

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from math import log2
from statistics import mean
from typing import Any

from doc_assistant.evaluation.constants import DEFAULT_REFUSAL_TERMS


@dataclass(frozen=True)
class SourceCandidate:
    file_name: str | None = None
    page: int | None = None
    chunk_id: int | None = None
    text: str = ""
    source_id: str | None = None


def source_candidate_from_document(document: Any) -> SourceCandidate:
    metadata = getattr(document, "metadata", {}) or {}
    return SourceCandidate(
        file_name=_optional_str(metadata.get("file_name") or metadata.get("source")),
        page=_int_mapping_value(metadata, "page"),
        chunk_id=_int_mapping_value(metadata, "chunk_id"),
        text=str(getattr(document, "page_content", "") or ""),
    )


def source_candidate_from_citation(citation: Any) -> SourceCandidate:
    if isinstance(citation, dict):
        return SourceCandidate(
            file_name=_optional_str(citation.get("file_name")),
            page=_int_mapping_value(citation, "page"),
            chunk_id=_int_mapping_value(citation, "chunk_id"),
            text=str(citation.get("exact_quote") or citation.get("preview") or ""),
            source_id=_optional_str(citation.get("source_id")),
        )

    return SourceCandidate(
        file_name=_optional_str(getattr(citation, "file_name", None)),
        page=_int_attr(citation, "page"),
        chunk_id=_int_attr(citation, "chunk_id"),
        text=str(getattr(citation, "exact_quote", None) or getattr(citation, "preview", "") or ""),
        source_id=_optional_str(getattr(citation, "source_id", None)),
    )


def score_retrieval_case(
    gold_sources: list[dict[str, Any]],
    retrieved: list[SourceCandidate],
    k: int,
) -> dict[str, float | None]:
    if not gold_sources:
        return {"recall": None, "hit": None, "precision": None, "mrr": None, "ndcg": None}

    if k <= 0:
        return {"recall": 0.0, "hit": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}

    top_k = retrieved[:k]
    matched_gold_indexes: set[int] = set()
    matched_candidate_count = 0
    first_match_rank: int | None = None
    dcg = 0.0

    for rank, candidate in enumerate(top_k, start=1):
        candidate_matched = False
        candidate_gold_indexes = set()
        for gold_index, gold_source in enumerate(gold_sources):
            if source_matches(gold_source, candidate):
                candidate_gold_indexes.add(gold_index)
                candidate_matched = True
        new_gold_indexes = candidate_gold_indexes - matched_gold_indexes
        if new_gold_indexes:
            dcg += 1.0 / log2(rank + 1)
            matched_gold_indexes.update(new_gold_indexes)
        if candidate_matched:
            matched_candidate_count += 1
            if first_match_rank is None:
                first_match_rank = rank

    return {
        "recall": len(matched_gold_indexes) / len(gold_sources),
        "hit": 1.0 if matched_gold_indexes else 0.0,
        "precision": matched_candidate_count / k,
        "mrr": 1.0 / first_match_rank if first_match_rank else 0.0,
        "ndcg": dcg / _ideal_dcg(min(len(gold_sources), k)),
    }


def score_generation_case(
    case: dict[str, Any],
    answer: str,
    citations: list[SourceCandidate],
) -> dict[str, float | None]:
    answer_type = case.get("answer_type", "answerable")
    gold_sources = list(case.get("gold_sources") or [])
    answer_text = answer or ""

    if answer_type == "unanswerable":
        refusal = 1.0 if _contains_refusal(answer_text, case) else 0.0
        return {
            "answer_correctness": refusal,
            "faithfulness": refusal,
            "citation_accuracy": None,
            "refusal_accuracy": refusal,
        }

    answer_correct = _contains_all(answer_text, case.get("required_answer_terms") or [])
    answer_correct = answer_correct and not _contains_any(
        answer_text,
        case.get("forbidden_answer_terms") or [],
    )

    return {
        "answer_correctness": 1.0 if answer_correct else 0.0,
        "faithfulness": (
            1.0
            if _is_faithful(
                answer_text,
                citations,
                case.get("required_answer_terms") or [],
            )
            else 0.0
        ),
        "citation_accuracy": _citation_accuracy(answer_text, citations, gold_sources),
        "refusal_accuracy": None,
    }


def aggregate_scores(
    case_scores: list[dict[str, float | None]],
    keys: Iterable[str] | None = None,
) -> dict[str, float | None]:
    if not case_scores:
        return {key: None for key in sorted(keys or [])}

    metric_keys = sorted(set(keys or []) | {key for scores in case_scores for key in scores})
    aggregate: dict[str, float | None] = {}
    for key in metric_keys:
        values = [scores[key] for scores in case_scores if scores.get(key) is not None]
        aggregate[key] = mean(values) if values else None
    return aggregate


def source_matches(gold_source: dict[str, Any], candidate: SourceCandidate) -> bool:
    marker = gold_source.get("marker")
    if marker and _contains_marker(candidate.text, str(marker)):
        return True

    gold_file = gold_source.get("file_name")
    if gold_file and str(gold_file) != candidate.file_name:
        return False

    if gold_source.get("page") is not None and gold_source.get("page") != candidate.page:
        return False

    if gold_source.get("chunk_id") is not None and gold_source.get("chunk_id") != candidate.chunk_id:
        return False

    return bool(gold_file or gold_source.get("page") is not None or gold_source.get("chunk_id") is not None)


def _citation_accuracy(
    answer: str,
    citations: list[SourceCandidate],
    gold_sources: list[dict[str, Any]],
) -> float:
    cited_ids = set(re.findall(r"\[(S\d+)\]", answer or ""))
    if not cited_ids:
        return 0.0

    citations_by_id = {citation.source_id: citation for citation in citations if citation.source_id}
    correct = 0
    for source_id in cited_ids:
        citation = citations_by_id.get(source_id)
        if citation and any(source_matches(gold_source, citation) for gold_source in gold_sources):
            correct += 1

    return correct / len(cited_ids)


def _is_faithful(
    answer: str,
    citations: list[SourceCandidate],
    required_answer_terms: list[str],
) -> bool:
    context = "\n".join(candidate.text for candidate in citations)
    return _is_faithful_by_numbers(answer, context) and _contains_all(context, required_answer_terms)


def _is_faithful_by_numbers(answer: str, context: str) -> bool:
    answer_numbers = set(_number_like_terms(answer))
    if not answer_numbers:
        return True

    context_numbers = set(_number_like_terms(context))
    return answer_numbers.issubset(context_numbers)


def _number_like_terms(text: str) -> list[str]:
    return re.findall(
        r"\b\d+(?:\.\d+)?(?:%|\s*(?:calendar\s+days?|business\s+days?|days?))\b"
        r"|\b\d+(?:\.\d+)?%?",
        text,
    )


def _contains_all(text: str, terms: list[str]) -> bool:
    normalized_text = text.casefold()
    return all(str(term).casefold() in normalized_text for term in terms)


def _contains_any(text: str, terms: list[str]) -> bool:
    normalized_text = text.casefold()
    return any(str(term).casefold() in normalized_text for term in terms)


def _contains_refusal(text: str, case: dict[str, Any]) -> bool:
    refusal_terms = case.get("required_refusal_terms") or DEFAULT_REFUSAL_TERMS
    return _contains_any(text, list(refusal_terms))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _int_mapping_value(values: dict[str, Any], key: str) -> int | None:
    value = values.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _int_attr(obj: Any, name: str) -> int | None:
    value = getattr(obj, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _contains_marker(text: str, marker: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9]){re.escape(marker)}(?![A-Za-z0-9])",
        text,
    ) is not None


def _ideal_dcg(relevant_count: int) -> float:
    if relevant_count <= 0:
        return 0.0
    return sum(1.0 / log2(rank + 1) for rank in range(1, relevant_count + 1))

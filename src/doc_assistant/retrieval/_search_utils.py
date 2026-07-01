"""搜索工具：分词、词法重排、MMR 多样性选择。

独立于 DocumentVectorStore，供检索流水线与 BM25 模块复用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document

_SEARCH_TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:[-_][A-Za-z0-9]+)*|\d+(?:\.\d+)*%?|[一-鿿]"
)


@dataclass
class _SearchCandidate:
    """混合检索流水线中的单个候选 chunk，携带 dense/BM25/RRF 各阶段分数。"""

    identity: str
    document: Document
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    bm25_relevance: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    rank_score: float = 0.0
    relevance: float = 0.0


def _tokenize_for_search(text: str) -> list[str]:
    return [token.casefold() for token in _SEARCH_TOKEN_PATTERN.findall(text or "")]


def _lexical_rerank_score(query: str, document: Document) -> float:
    query_tokens = set(_tokenize_for_search(query))
    if not query_tokens:
        return 0.0

    metadata = document.metadata or {}
    document_text = "\n".join(
        part
        for part in [
            str(metadata.get("section_heading") or ""),
            document.page_content or "",
        ]
        if part
    )
    document_tokens = set(_tokenize_for_search(document_text))
    if not document_tokens:
        return 0.0

    token_overlap = len(query_tokens & document_tokens) / len(query_tokens)
    query_numbers = {token for token in query_tokens if any(char.isdigit() for char in token)}
    if query_numbers:
        number_overlap = len(query_numbers & document_tokens) / len(query_numbers)
    else:
        number_overlap = 0.0
    phrase_bonus = 1.0 if query.strip().casefold() in document_text.casefold() else 0.0
    return _clamp_float(
        0.75 * token_overlap + 0.20 * number_overlap + 0.05 * phrase_bonus,
        minimum=0.0,
        maximum=1.0,
    )


def _select_diverse_candidates(
    candidates: list[_SearchCandidate],
    *,
    top_k: int,
    lambda_mult: float,
) -> list[_SearchCandidate]:
    """MMR（Maximal Marginal Relevance）多样性选择，平衡相关性与结果差异性。"""
    if lambda_mult >= 1.0:
        return candidates[:top_k]

    selected: list[_SearchCandidate] = []
    remaining = list(candidates)
    selected_token_sets: list[set[str]] = []
    token_sets = {
        candidate.identity: set(_tokenize_for_search(candidate.document.page_content))
        for candidate in candidates
    }

    while remaining and len(selected) < top_k:
        best_candidate = max(
            remaining,
            key=lambda candidate: (
                _mmr_score(
                    candidate,
                    token_sets.get(candidate.identity, set()),
                    selected_token_sets,
                    lambda_mult=lambda_mult,
                ),
                candidate.rank_score,
            ),
        )
        selected.append(best_candidate)
        selected_token_sets.append(token_sets.get(best_candidate.identity, set()))
        remaining.remove(best_candidate)

    return selected


def _mmr_score(
    candidate: _SearchCandidate,
    candidate_tokens: set[str],
    selected_token_sets: list[set[str]],
    *,
    lambda_mult: float,
) -> float:
    if not selected_token_sets:
        return candidate.relevance
    max_similarity = max(
        (_jaccard_similarity(candidate_tokens, selected_tokens) for selected_tokens in selected_token_sets),
        default=0.0,
    )
    return lambda_mult * candidate.relevance - (1 - lambda_mult) * max_similarity


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _clamp_float(value: float | None, *, minimum: float, maximum: float) -> float:
    if value is None:
        return minimum
    return max(minimum, min(maximum, float(value)))

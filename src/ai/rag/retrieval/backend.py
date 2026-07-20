"""Shared Qdrant client, collection, payload, and sparse-vector utilities."""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from typing import Any

from qdrant_client import models

from ai.config.settings import settings
from ai.rag.ingestion.loader import INGEST_WARNINGS_METADATA_KEY
from ai.rag.retrieval.search import _tokenize_for_search

DOCUMENT_PAYLOAD_KEY = "_document"
RECORD_ID_PAYLOAD_KEY = "_record_id"

logger = logging.getLogger(__name__)


def clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if key == INGEST_WARNINGS_METADATA_KEY or value is None:
            continue
        if isinstance(value, str | int | float | bool):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def metadata_is_active(metadata: dict[str, Any]) -> bool:
    return metadata.get("active", True) is not False


def active_filter() -> models.Filter:
    return models.Filter(
        should=[
            models.FieldCondition(
                key="active",
                match=models.MatchValue(value=True),
            ),
            models.IsEmptyCondition(is_empty=models.PayloadField(key="active")),
        ]
    )


def field_filter(**values: str | int | float | bool | None) -> models.Filter | None:
    conditions = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in values.items()
        if value is not None
    ]
    return models.Filter(must=conditions) if conditions else None


def sparse_document_vector(text: str, metadata: dict[str, Any]) -> models.SparseVector:
    search_text = _sparse_search_text(text, metadata)
    tokens = _tokenize_for_search(search_text)
    if not tokens:
        return models.SparseVector(indices=[], values=[])

    token_counts = Counter(tokens)
    document_length = len(tokens)
    k1 = float(settings.retrieval_bm25_k1)
    b = float(settings.retrieval_bm25_b)
    average_length = float(settings.retrieval_bm25_average_length)
    normalizer = k1 * (1 - b + b * document_length / max(average_length, 1.0))

    weights: dict[int, float] = {}
    for token, frequency in token_counts.items():
        index = _sparse_token_index(token)
        value = frequency * (k1 + 1) / (frequency + normalizer)
        weights[index] = weights.get(index, 0.0) + value
    indices = sorted(weights)
    return models.SparseVector(indices=indices, values=[weights[index] for index in indices])


def sparse_query_vector(text: str) -> models.SparseVector:
    token_counts = Counter(_tokenize_for_search(text))
    weights: dict[int, float] = {}
    for token, frequency in token_counts.items():
        index = _sparse_token_index(token)
        weights[index] = weights.get(index, 0.0) + float(frequency)
    indices = sorted(weights)
    return models.SparseVector(indices=indices, values=[weights[index] for index in indices])


def _sparse_search_text(text: str, metadata: dict[str, Any]) -> str:
    file_name = str(metadata.get("file_name") or metadata.get("source") or "")
    heading = str(metadata.get("section_heading") or "")
    return "\n".join(part for part in (file_name, text, heading, heading) if part)


def _sparse_token_index(token: str) -> int:
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)

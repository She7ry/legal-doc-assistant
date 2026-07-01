"""BM25 索引集成：文档转换、搜索、评分。

作为 PersistentBM25Index 与 DocumentVectorStore 之间的胶水层，
将 Chroma 记录格式转换为 BM25Document，并提供纯 Python BM25 回退搜索。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from doc_assistant.ingestion.document_loader import INGEST_WARNINGS_METADATA_KEY
from doc_assistant.retrieval._search_utils import _tokenize_for_search
from doc_assistant.retrieval.bm25_index import BM25Document

_COLLECTION_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_COLLECTION_NAME_LENGTH = 63


# ── metadata helpers (also imported by vector_store to avoid circular imports) ──


def _metadata_is_active(metadata: dict[str, Any]) -> bool:
    return metadata.get("active", True) is not False


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if key == INGEST_WARNINGS_METADATA_KEY or value is None:
            continue
        if isinstance(value, str | int | float | bool):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


# ── collection naming ──


def _sanitize_collection_component(value: str) -> str:
    sanitized = _COLLECTION_COMPONENT_PATTERN.sub("_", value.strip()).strip("_-")
    if len(sanitized) < 3:
        sanitized = f"{sanitized or 'col'}_collection"
    return sanitized[:_MAX_COLLECTION_NAME_LENGTH].strip("_-") or "col_collection"


def _bm25_index_path(persist_directory: Path, collection_name: str) -> Path:
    return persist_directory / f"{_sanitize_collection_component(collection_name)}_bm25.sqlite3"


# ── document conversion ──


def _bm25_documents_for_chunks(
    chunks: list[Document],
    ids: list[str],
) -> list[BM25Document]:
    documents = []
    for chunk, doc_id in zip(chunks, ids, strict=True):
        record = {
            "id": doc_id,
            "metadata": chunk.metadata or {},
            "document": chunk.page_content or "",
        }
        indexed = _bm25_document_for_record(record)
        if indexed is not None:
            documents.append(indexed)
    return documents


def _bm25_documents_for_records(records: list[dict[str, Any]]) -> list[BM25Document]:
    documents = []
    for record in records:
        indexed = _bm25_document_for_record(record)
        if indexed is not None:
            documents.append(indexed)
    return documents


def _bm25_document_for_record(record: dict[str, Any]) -> BM25Document | None:
    record_id = str(record.get("id") or "")
    if not record_id:
        return None

    metadata = dict(record.get("metadata") or {})
    search_text = _record_search_text(record)
    tokens = _tokenize_for_search(search_text)
    if not tokens:
        return None

    return BM25Document(
        doc_id=record_id,
        tokens=tokens,
        document=str(record.get("document") or ""),
        metadata=_clean_metadata(metadata),
        active=_metadata_is_active(metadata),
    )


# ── BM25 fallback search ──


def _bm25_rank(
    query: str,
    records: list[dict[str, Any]],
    k: int,
) -> list[tuple[Document, float, str]]:
    query_tokens = _tokenize_for_search(query)
    if not query_tokens:
        return []

    indexed_records = []
    document_frequency: Counter[str] = Counter()
    total_length = 0
    for record in records:
        text = _record_search_text(record)
        tokens = _tokenize_for_search(text)
        if not tokens:
            continue
        counts = Counter(tokens)
        indexed_records.append((record, counts, len(tokens)))
        document_frequency.update(counts.keys())
        total_length += len(tokens)

    document_count = len(indexed_records)
    if document_count == 0:
        return []

    average_length = total_length / document_count
    query_counts = Counter(query_tokens)
    scored_documents = []
    for record, token_counts, document_length in indexed_records:
        score = _bm25_score(
            query_counts,
            token_counts,
            document_frequency,
            document_count=document_count,
            document_length=document_length,
            average_length=average_length,
        )
        if score <= 0:
            continue
        scored_documents.append(
            (
                Document(
                    page_content=str(record.get("document") or ""),
                    metadata=dict(record.get("metadata") or {}),
                ),
                score,
                str(record.get("id") or ""),
            )
        )

    return sorted(scored_documents, key=lambda item: item[1], reverse=True)[:k]


def _bm25_score(
    query_counts: Counter[str],
    token_counts: Counter[str],
    document_frequency: Counter[str],
    *,
    document_count: int,
    document_length: int,
    average_length: float,
) -> float:
    k1 = 1.5
    b = 0.75
    score = 0.0
    normalizer = k1 * (1 - b + b * (document_length / max(average_length, 1.0)))
    for token, query_frequency in query_counts.items():
        term_frequency = token_counts.get(token, 0)
        if term_frequency == 0:
            continue
        term_document_frequency = document_frequency.get(token, 0)
        idf = math.log(1 + (document_count - term_document_frequency + 0.5) / (term_document_frequency + 0.5))
        score += query_frequency * idf * (
            term_frequency * (k1 + 1) / (term_frequency + normalizer)
        )
    return score


def _record_search_text(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    parts = [
        str(metadata.get("file_name") or metadata.get("source") or ""),
        str(record.get("document") or ""),
    ]
    section_heading = metadata.get("section_heading")
    if section_heading:
        heading = str(section_heading)
        parts.extend([heading, heading])
    return "\n".join(part for part in parts if part)

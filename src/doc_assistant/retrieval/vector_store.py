from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from doc_assistant.config.settings import settings
from doc_assistant.ingestion.document_loader import (
    INGEST_WARNINGS_METADATA_KEY,
    file_sha256,
    load_documents,
)
from doc_assistant.models.language_model import build_embedding_model
from doc_assistant.schemas.citation import IngestResult

logger = logging.getLogger(__name__)

_COLLECTION_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_COLLECTION_NAME_LENGTH = 63
_SEARCH_TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:[-_][A-Za-z0-9]+)*|\d+(?:\.\d+)*%?|[\u4e00-\u9fff]"
)
_LEGAL_SECTION_PATTERN = re.compile(
    r"^\s*("
    r"第[一二三四五六七八九十百千万\d]+[章节条款项]|"
    r"\d+(?:\.\d+)*[\.)、]?|"
    r"(?:Section|Article|Clause|Schedule|Exhibit|Appendix)\s+[\w\dIVXLC]+"
    r")\s*[:：.-]?\s*(.*)$",
    re.IGNORECASE,
)
ProgressCallback = Callable[[str, int, str | None], None]


@dataclass
class _SearchCandidate:
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


class DocumentVectorStore:
    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: Path | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self.tenant_id = tenant_id or settings.default_tenant_id
        effective_collection_name = collection_name or collection_name_for_tenant(
            settings.collection_name,
            self.tenant_id,
        )
        self.vector_store = Chroma(
            collection_name=effective_collection_name,
            embedding_function=build_embedding_model(),
            persist_directory=str(persist_directory or settings.vector_store_dir),
        )
        self._write_lock = threading.Lock()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=[
                "\n第",
                "\nSection ",
                "\nArticle ",
                "\nClause ",
                "\nSchedule ",
                "\nExhibit ",
                "\n\n",
                "\n",
                "。 ",
                "；",
                ". ",
                "; ",
                ", ",
                " ",
                "",
            ],
        )

    def ingest_file(
        self,
        file_path: Path,
        file_name: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IngestResult:
        file_path = Path(file_path)
        display_name = file_name or file_path.name
        file_id = file_sha256(file_path)
        document_key = document_key_for_file_name(display_name)
        existing_records = self._records_for_document_key(document_key)
        active_records = [
            record for record in existing_records if _metadata_is_active(record["metadata"])
        ]
        active_same_content = [
            record for record in active_records if record["metadata"].get("file_id") == file_id
        ]

        if active_same_content:
            version = max(_metadata_int(record["metadata"], "document_version", 1) for record in active_same_content)
            warning = "Document content is already indexed as the active version; existing chunks were reused."
            _report_progress(progress_callback, "completed", 100, warning)
            first_metadata = active_same_content[0]["metadata"]
            return IngestResult(
                file_id=file_id,
                file_name=display_name,
                document_count=_metadata_int(first_metadata, "document_count", 1),
                chunk_count=len(active_same_content),
                document_key=document_key,
                document_version=version,
                file_extension=file_path.suffix.lower(),
                page_count=_optional_metadata_int(first_metadata, "page_count"),
                skipped=True,
                warnings=[warning],
            )

        version = max(
            [_metadata_int(record["metadata"], "document_version", 0) for record in existing_records]
            or [0]
        ) + 1

        _report_progress(progress_callback, "parsing", 20)
        documents = load_documents(file_path)
        warnings = _collect_warnings(documents)
        if not any((document.page_content or "").strip() for document in documents):
            raise ValueError(
                "No extractable text was found in the uploaded document. "
                "For scanned PDFs, enable OCR and install OCR dependencies."
            )

        page_count = _page_count(documents)
        indexed_at = datetime.now(timezone.utc).isoformat()
        replaced_file_ids = sorted(
            {
                str(record["metadata"].get("file_id"))
                for record in active_records
                if record["metadata"].get("file_id")
            }
        )

        for document in documents:
            document.metadata["file_id"] = file_id
            document.metadata["file_name"] = display_name
            document.metadata["tenant_id"] = self.tenant_id
            document.metadata["document_key"] = document_key
            document.metadata["document_version"] = version
            document.metadata["file_extension"] = file_path.suffix.lower()

        _report_progress(progress_callback, "chunking", 40)
        section_documents = _split_legal_sections(documents)
        chunks = self.splitter.split_documents(section_documents)
        ids = []
        for index, chunk in enumerate(chunks):
            chunk.page_content = _chunk_text_with_heading(
                chunk.page_content,
                chunk.metadata.get("section_heading"),
            )
            chunk.metadata["file_id"] = file_id
            chunk.metadata["file_name"] = display_name
            chunk.metadata["tenant_id"] = self.tenant_id
            chunk.metadata["chunk_id"] = index
            chunk.metadata["document_key"] = document_key
            chunk.metadata["document_version"] = version
            chunk.metadata["active"] = True
            chunk.metadata["indexed_at"] = indexed_at
            chunk.metadata["file_extension"] = file_path.suffix.lower()
            chunk.metadata["document_count"] = len(documents)
            if page_count is not None:
                chunk.metadata["page_count"] = page_count
            if replaced_file_ids:
                chunk.metadata["replaces_file_ids"] = ",".join(replaced_file_ids)
            if warnings:
                chunk.metadata["warning_count"] = len(warnings)
            chunk.metadata = _clean_metadata(chunk.metadata)
            ids.append(f"{document_key}:{file_id}:v{version}:{index}")

        if ids:
            _report_progress(progress_callback, "embedding", 70)
            with self._write_lock:
                try:
                    self.vector_store.delete(ids=ids)
                except Exception as exc:
                    logger.warning(
                        "Failed to clear stale chunks for this ingest version",
                        extra={"file_id": file_id, "tenant_id": self.tenant_id},
                        exc_info=True,
                    )
                    raise RuntimeError("Failed to prepare vector store for document ingest.") from exc

                _report_progress(progress_callback, "indexing", 85)
                self.vector_store.add_documents(chunks, ids=ids)
                try:
                    self._deactivate_records(active_records, superseded_by_file_id=file_id)
                except Exception as exc:
                    warning = (
                        "New document version was indexed, but older versions could not be "
                        f"marked inactive: {exc}"
                    )
                    logger.warning(
                        "Failed to deactivate older document versions",
                        extra={"file_id": file_id, "tenant_id": self.tenant_id},
                        exc_info=True,
                    )
                    warnings.append(warning)
                    _report_progress(progress_callback, "indexing", 92, warning)

        _report_progress(progress_callback, "completed", 100)
        return IngestResult(
            file_id=file_id,
            file_name=display_name,
            document_count=len(documents),
            chunk_count=len(chunks),
            document_key=document_key,
            document_version=version,
            file_extension=file_path.suffix.lower(),
            page_count=page_count,
            skipped=False,
            warnings=warnings,
        )

    def search(self, query: str, k: int | None = None) -> list[Document]:
        top_k = max(1, int(k or settings.top_k))
        fetch_k = max(top_k, int(settings.retrieval_fetch_k), top_k * 5)
        candidates = self._rank_candidates(query, fetch_k=fetch_k)
        selected = _select_diverse_candidates(
            candidates,
            top_k=top_k,
            lambda_mult=_clamp_float(settings.retrieval_mmr_lambda, minimum=0.0, maximum=1.0),
        )
        return [_document_with_retrieval_metadata(candidate) for candidate in selected]

    def list_documents(self) -> list[dict[str, Any]]:
        records = self._all_records(include_documents=False)
        grouped: dict[str, dict[str, Any]] = {}
        for record in records:
            metadata = record["metadata"]
            if not _metadata_is_active(metadata):
                continue

            key = str(metadata.get("document_key") or metadata.get("file_id") or record["id"])
            version = _metadata_int(metadata, "document_version", 1)
            current = grouped.get(key)
            if current is None or version > int(current["document_version"]):
                grouped[key] = {
                    "file_id": str(metadata.get("file_id") or ""),
                    "file_name": str(metadata.get("file_name") or metadata.get("source") or "unknown"),
                    "document_key": key,
                    "document_version": version,
                    "file_extension": str(metadata.get("file_extension") or ""),
                    "indexed_at": metadata.get("indexed_at"),
                    "document_count": _metadata_int(metadata, "document_count", 1),
                    "page_count": _optional_metadata_int(metadata, "page_count"),
                    "chunk_count": 0,
                    "warning_count": _metadata_int(metadata, "warning_count", 0),
                }

            if grouped[key]["document_version"] == version:
                grouped[key]["chunk_count"] += 1

        return sorted(
            grouped.values(),
            key=lambda item: str(item.get("indexed_at") or ""),
            reverse=True,
        )

    def _rank_candidates(self, query: str, *, fetch_k: int) -> list[_SearchCandidate]:
        mode = str(settings.retrieval_mode or "hybrid").strip().lower()
        if mode not in {"hybrid", "dense", "vector", "bm25", "sparse"}:
            logger.warning("Unknown retrieval mode %r; falling back to hybrid.", mode)
            mode = "hybrid"

        use_dense = mode in {"hybrid", "dense", "vector"}
        use_bm25 = mode in {"hybrid", "bm25", "sparse"}
        candidates: dict[str, _SearchCandidate] = {}

        if use_dense:
            for rank, (document, dense_score) in enumerate(
                self._dense_candidates(query, fetch_k=fetch_k),
                start=1,
            ):
                identity = _document_identity(document, fallback_id=None)
                candidate = candidates.setdefault(
                    identity,
                    _SearchCandidate(identity=identity, document=document),
                )
                candidate.dense_rank = rank
                candidate.dense_score = dense_score

        max_bm25_score = 0.0
        if use_bm25:
            bm25_candidates = self._bm25_candidates(query, fetch_k=fetch_k)
            max_bm25_score = max((score for _, score, _ in bm25_candidates), default=0.0)
            for rank, (document, bm25_score, record_id) in enumerate(bm25_candidates, start=1):
                identity = _document_identity(document, fallback_id=record_id)
                candidate = candidates.setdefault(
                    identity,
                    _SearchCandidate(identity=identity, document=document),
                )
                candidate.bm25_rank = rank
                candidate.bm25_score = bm25_score
                if max_bm25_score > 0:
                    candidate.bm25_relevance = bm25_score / max_bm25_score

        min_relevance = max(0.0, float(settings.retrieval_min_relevance))
        rerank_mode = str(getattr(settings, "retrieval_rerank_mode", "lexical")).strip().lower()
        rerank_weight = max(0.0, float(getattr(settings, "retrieval_rerank_weight", 0.25)))
        ranked_candidates = []
        for candidate in candidates.values():
            if candidate.dense_rank is not None:
                candidate.rrf_score += (
                    float(settings.retrieval_dense_weight)
                    / (float(settings.retrieval_rrf_k) + candidate.dense_rank)
                )
            if candidate.bm25_rank is not None:
                candidate.rrf_score += (
                    float(settings.retrieval_bm25_weight)
                    / (float(settings.retrieval_rrf_k) + candidate.bm25_rank)
                )

            dense_relevance = (
                _clamp_float(candidate.dense_score, minimum=0.0, maximum=1.0)
                if candidate.dense_score is not None
                else 0.0
            )
            candidate.relevance = max(dense_relevance, candidate.bm25_relevance)
            if candidate.relevance < min_relevance:
                continue
            if rerank_mode in {"lexical", "local"} and rerank_weight > 0:
                candidate.rerank_score = _lexical_rerank_score(query, candidate.document)
            elif rerank_mode not in {"", "none", "off", "disabled"}:
                logger.warning("Unknown retrieval rerank mode %r; skipping rerank.", rerank_mode)
            candidate.rank_score = candidate.rrf_score * (1 + rerank_weight * candidate.rerank_score)
            ranked_candidates.append(candidate)

        return sorted(
            ranked_candidates,
            key=lambda candidate: (candidate.rank_score, candidate.relevance),
            reverse=True,
        )

    def _dense_candidates(self, query: str, *, fetch_k: int) -> list[tuple[Document, float]]:
        docs_and_scores = self._similarity_search_with_active_filter(query, fetch_k=fetch_k)
        return [
            (document, float(score))
            for document, score in docs_and_scores
            if _metadata_is_active(document.metadata or {})
        ]

    def _similarity_search_with_active_filter(
        self,
        query: str,
        *,
        fetch_k: int,
    ) -> list[tuple[Document, float]]:
        try:
            docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(
                query,
                k=fetch_k,
                filter={"active": True},
            )
            if docs_and_scores:
                return docs_and_scores
        except Exception:
            logger.debug("Vector search with active metadata filter failed; retrying without filter.", exc_info=True)

        return self.vector_store.similarity_search_with_relevance_scores(query, k=fetch_k)

    def _bm25_candidates(
        self,
        query: str,
        *,
        fetch_k: int,
    ) -> list[tuple[Document, float, str]]:
        return _bm25_rank(query, self._active_records_for_search(), fetch_k)

    def _active_records_for_search(self) -> list[dict[str, Any]]:
        try:
            collection = self.vector_store.get(
                where={"active": True},
                include=["metadatas", "documents"],
            )
            records = _records_from_collection(collection)
            if records:
                return records
        except Exception:
            logger.debug("Chroma active-record filter failed; falling back to client-side filtering.", exc_info=True)

        return [
            record
            for record in self._all_records(include_documents=True)
            if _metadata_is_active(record["metadata"])
        ]

    def _all_records(self, *, include_documents: bool = True) -> list[dict[str, Any]]:
        include = ["metadatas"]
        if include_documents:
            include.append("documents")
        collection = self.vector_store.get(include=include)
        return _records_from_collection(collection)

    def _records_for_document_key(self, document_key: str) -> list[dict[str, Any]]:
        return [
            record
            for record in self._all_records(include_documents=False)
            if record["metadata"].get("document_key") == document_key
        ]

    def _latest_active_versions(self) -> dict[str, int]:
        latest: dict[str, int] = {}
        for record in self._all_records(include_documents=False):
            metadata = record["metadata"]
            if not _metadata_is_active(metadata):
                continue
            document_key = metadata.get("document_key")
            if not document_key:
                continue
            latest[str(document_key)] = max(
                latest.get(str(document_key), 0),
                _metadata_int(metadata, "document_version", 1),
            )
        return latest

    def _deactivate_records(
        self,
        records: list[dict[str, Any]],
        *,
        superseded_by_file_id: str,
    ) -> None:
        if not records:
            return

        ids = []
        metadatas = []
        for record in records:
            metadata = dict(record["metadata"])
            metadata["active"] = False
            metadata["superseded_by_file_id"] = superseded_by_file_id
            ids.append(record["id"])
            metadatas.append(_clean_metadata(metadata))

        self.vector_store._collection.update(ids=ids, metadatas=metadatas)


def _records_from_collection(collection: dict[str, Any]) -> list[dict[str, Any]]:
    ids = collection.get("ids") or []
    metadatas = collection.get("metadatas") or []
    documents = collection.get("documents") or []
    records = []
    for index, record_id in enumerate(ids):
        records.append(
            {
                "id": record_id,
                "metadata": metadatas[index] or {},
                "document": documents[index] if index < len(documents) else "",
            }
        )
    return records


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


def _tokenize_for_search(text: str) -> list[str]:
    return [token.casefold() for token in _SEARCH_TOKEN_PATTERN.findall(text or "")]


def _chunk_text_with_heading(text: str, heading: Any) -> str:
    heading_text = str(heading or "").strip()
    content = text or ""
    if not heading_text:
        return content
    if content.lstrip().startswith(heading_text):
        return content
    return f"{heading_text}\n{content}".strip()


def _document_identity(document: Document, fallback_id: str | None) -> str:
    metadata = document.metadata or {}
    identity_parts = [
        metadata.get("document_key"),
        metadata.get("file_id"),
        metadata.get("document_version"),
        metadata.get("chunk_id"),
    ]
    if any(part is not None for part in identity_parts):
        return "|".join(str(part) for part in identity_parts)

    fallback = fallback_id or "|".join(
        [
            str(metadata.get("file_name") or metadata.get("source") or ""),
            str(metadata.get("page") or ""),
            document.page_content or "",
        ]
    )
    return hashlib.sha1(fallback.encode("utf-8")).hexdigest()


def _select_diverse_candidates(
    candidates: list[_SearchCandidate],
    *,
    top_k: int,
    lambda_mult: float,
) -> list[_SearchCandidate]:
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


def _document_with_retrieval_metadata(candidate: _SearchCandidate) -> Document:
    metadata = dict(candidate.document.metadata or {})
    metadata.update(
        {
            "retrieval_score": candidate.rank_score,
            "retrieval_relevance": candidate.relevance,
            "rrf_score": candidate.rrf_score,
            "rerank_score": candidate.rerank_score,
        }
    )
    if candidate.dense_score is not None:
        metadata["dense_score"] = candidate.dense_score
    if candidate.dense_rank is not None:
        metadata["dense_rank"] = candidate.dense_rank
    if candidate.bm25_score is not None:
        metadata["bm25_score"] = candidate.bm25_score
    if candidate.bm25_rank is not None:
        metadata["bm25_rank"] = candidate.bm25_rank
    return Document(page_content=candidate.document.page_content, metadata=_clean_metadata(metadata))


def _clamp_float(value: float | None, *, minimum: float, maximum: float) -> float:
    if value is None:
        return minimum
    return max(minimum, min(maximum, float(value)))


def collection_name_for_tenant(base_name: str, tenant_id: str | None) -> str:
    base = _sanitize_collection_component(base_name)
    if not tenant_id or tenant_id == settings.default_tenant_id:
        return base

    tenant = _sanitize_collection_component(tenant_id)
    collection_name = f"{base}_{tenant}"
    if len(collection_name) <= _MAX_COLLECTION_NAME_LENGTH:
        return collection_name

    digest = hashlib.sha1(tenant_id.encode("utf-8")).hexdigest()[:12]
    available_base_length = _MAX_COLLECTION_NAME_LENGTH - len(digest) - 1
    return f"{base[:available_base_length].rstrip('_-')}_{digest}"


def _sanitize_collection_component(value: str) -> str:
    sanitized = _COLLECTION_COMPONENT_PATTERN.sub("_", value.strip()).strip("_-")
    if len(sanitized) < 3:
        sanitized = f"{sanitized or 'col'}_collection"
    return sanitized[:_MAX_COLLECTION_NAME_LENGTH].strip("_-") or "col_collection"


def document_key_for_file_name(file_name: str) -> str:
    normalized = " ".join(file_name.casefold().strip().split()) or "uploaded_document"
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _report_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    progress: int,
    warning: str | None = None,
) -> None:
    if progress_callback is None:
        return
    progress_callback(stage, max(0, min(progress, 100)), warning)


def _collect_warnings(documents: list[Document]) -> list[str]:
    warnings = []
    seen = set()
    for document in documents:
        raw = document.metadata.get(INGEST_WARNINGS_METADATA_KEY)
        if not isinstance(raw, list):
            continue
        for warning in raw:
            text = str(warning).strip()
            if text and text not in seen:
                warnings.append(text)
                seen.add(text)
    return warnings


def _page_count(documents: list[Document]) -> int | None:
    pages = {
        document.metadata.get("page")
        for document in documents
        if isinstance(document.metadata.get("page"), int)
    }
    if pages:
        return len(pages)
    return len(documents) if documents else None


def _split_legal_sections(documents: list[Document]) -> list[Document]:
    section_documents = []
    for document in documents:
        text = document.page_content or ""
        blocks = _section_blocks(text)
        if len(blocks) <= 1:
            section_documents.append(document)
            continue

        for section_index, (heading, content) in enumerate(blocks):
            metadata = dict(document.metadata)
            metadata["section_index"] = section_index
            if heading:
                metadata["section_heading"] = heading[:180]
            section_documents.append(Document(page_content=content, metadata=metadata))

    return section_documents


def _section_blocks(text: str) -> list[tuple[str | None, str]]:
    blocks: list[tuple[str | None, str]] = []
    current_lines: list[str] = []
    current_heading: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current_lines:
                current_lines.append("")
            continue

        heading = _legal_section_heading(line)
        if heading and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                blocks.append((current_heading, content))
            current_lines = [line]
            current_heading = heading
            continue

        if heading and not current_lines:
            current_heading = heading
        current_lines.append(line)

    content = "\n".join(current_lines).strip()
    if content:
        blocks.append((current_heading, content))

    return blocks or [(None, text)]


def _legal_section_heading(line: str) -> str | None:
    if len(line) > 220:
        return None
    match = _LEGAL_SECTION_PATTERN.match(line)
    if not match:
        return None
    return " ".join(line.split())


def _metadata_is_active(metadata: dict[str, Any]) -> bool:
    return metadata.get("active", True) is not False


def _metadata_int(metadata: dict[str, Any], key: str, default: int) -> int:
    value = metadata.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default


def _optional_metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = _metadata_int(metadata, key, -1)
    return value if value >= 0 else None


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

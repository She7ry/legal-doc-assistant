"""BM25 稀疏检索索引：SQLite 持久化 + 内存倒排，与 Chroma 向量检索互补。

``PersistentBM25Index`` 与 ``DocumentVectorStore`` 共用 tenant/collection 维度；
hybrid 模式下通过 RRF 融合 dense 与 bm25 排名。
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterable


@dataclass(frozen=True)
class BM25Document:
    """索引中的一篇文档：doc_id、分词 tokens、原文与 metadata。"""

    doc_id: str
    tokens: list[str]
    document: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    active: bool = True


@dataclass(frozen=True)
class BM25Hit:
    """BM25 检索命中项：doc_id、相关性分数及原文 metadata。"""

    doc_id: str
    score: float
    document: str
    metadata: dict[str, Any]


class PersistentBM25Index:
    """SQLite 持久化的 BM25 倒排索引，支持增量 upsert 与按 collection 查询。"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS bm25_docs (
                    doc_id TEXT PRIMARY KEY,
                    token_count INTEGER NOT NULL,
                    tokens_json TEXT NOT NULL,
                    document TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bm25_terms (
                    token TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    term_freq INTEGER NOT NULL,
                    PRIMARY KEY (token, doc_id),
                    FOREIGN KEY (doc_id) REFERENCES bm25_docs(doc_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS bm25_df (
                    token TEXT PRIMARY KEY,
                    doc_freq INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS bm25_stats (
                    key TEXT PRIMARY KEY,
                    value REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_bm25_terms_doc_id
                    ON bm25_terms(doc_id);
                CREATE INDEX IF NOT EXISTS idx_bm25_docs_active
                    ON bm25_docs(active);
                """
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO bm25_stats(key, value) VALUES (?, ?)",
                ("document_count", 0.0),
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO bm25_stats(key, value) VALUES (?, ?)",
                ("total_token_count", 0.0),
            )

    def close(self) -> None:
        """关闭底层 SQLite 连接。"""
        self._conn.close()

    def __enter__(self) -> PersistentBM25Index:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def add_document(self, document: BM25Document) -> None:
        self.add_documents([document])

    def add_documents(self, documents: Iterable[BM25Document]) -> None:
        with self._lock:
            with self._conn:
                for document in documents:
                    self._add_document_locked(document)

    def replace_all(self, documents: Iterable[BM25Document]) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM bm25_terms")
                self._conn.execute("DELETE FROM bm25_docs")
                self._conn.execute("DELETE FROM bm25_df")
                self._set_stat_locked("document_count", 0.0)
                self._set_stat_locked("total_token_count", 0.0)
                for document in documents:
                    self._add_document_locked(document)

    def delete_documents(self, doc_ids: Iterable[str]) -> None:
        with self._lock:
            with self._conn:
                for doc_id in doc_ids:
                    self._remove_document_locked(str(doc_id))

    def mark_inactive(self, doc_ids: Iterable[str]) -> None:
        with self._lock:
            with self._conn:
                for doc_id in doc_ids:
                    self._mark_inactive_locked(str(doc_id))

    def active_document_count(self) -> int:
        with self._lock:
            return int(self._get_stat_locked("document_count"))

    def search(self, query_tokens: list[str], k: int) -> list[BM25Hit]:
        query_counts = Counter(token for token in query_tokens if token)
        if not query_counts:
            return []

        with self._lock:
            document_count = int(self._get_stat_locked("document_count"))
            if document_count <= 0:
                return []

            token_list = list(query_counts)
            placeholders = ",".join("?" for _ in token_list)
            term_rows = self._conn.execute(
                f"""
                SELECT t.token, t.doc_id, t.term_freq, d.token_count
                FROM bm25_terms AS t
                JOIN bm25_docs AS d ON d.doc_id = t.doc_id
                WHERE d.active = 1 AND t.token IN ({placeholders})
                """,
                token_list,
            ).fetchall()
            if not term_rows:
                return []

            doc_terms: dict[str, Counter[str]] = defaultdict(Counter)
            doc_lengths: dict[str, int] = {}
            for row in term_rows:
                doc_id = str(row["doc_id"])
                doc_terms[doc_id][str(row["token"])] = int(row["term_freq"])
                doc_lengths[doc_id] = int(row["token_count"])

            df_rows = self._conn.execute(
                f"SELECT token, doc_freq FROM bm25_df WHERE token IN ({placeholders})",
                token_list,
            ).fetchall()
            document_frequency = {
                str(row["token"]): int(row["doc_freq"])
                for row in df_rows
            }
            average_length = self._get_stat_locked("total_token_count") / max(
                document_count,
                1,
            )

            scored = []
            for doc_id, token_counts in doc_terms.items():
                score = _bm25_score(
                    query_counts,
                    token_counts,
                    document_frequency,
                    document_count=document_count,
                    document_length=doc_lengths.get(doc_id, 0),
                    average_length=average_length,
                )
                if score > 0:
                    scored.append((doc_id, score))

            top_scored = sorted(scored, key=lambda item: item[1], reverse=True)[:k]
            if not top_scored:
                return []

            top_doc_ids = [doc_id for doc_id, _ in top_scored]
            doc_placeholders = ",".join("?" for _ in top_doc_ids)
            doc_rows = self._conn.execute(
                f"""
                SELECT doc_id, document, metadata_json
                FROM bm25_docs
                WHERE active = 1 AND doc_id IN ({doc_placeholders})
                """,
                top_doc_ids,
            ).fetchall()
            docs_by_id = {
                str(row["doc_id"]): (
                    str(row["document"] or ""),
                    _json_dict(row["metadata_json"]),
                )
                for row in doc_rows
            }

        hits = []
        for doc_id, score in top_scored:
            document_text, metadata = docs_by_id.get(doc_id, ("", {}))
            hits.append(
                BM25Hit(
                    doc_id=doc_id,
                    score=score,
                    document=document_text,
                    metadata=metadata,
                )
            )
        return hits

    def _add_document_locked(self, document: BM25Document) -> None:
        if not document.doc_id:
            raise ValueError("BM25 document id cannot be empty.")

        tokens = [token for token in document.tokens if token]
        self._remove_document_locked(document.doc_id)
        token_counts = Counter(tokens)
        active = 1 if document.active else 0
        updated_at = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO bm25_docs(
                doc_id, token_count, tokens_json, document,
                metadata_json, active, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.doc_id,
                len(tokens),
                json.dumps(token_counts, ensure_ascii=False, sort_keys=True),
                document.document,
                json.dumps(document.metadata, ensure_ascii=False, sort_keys=True, default=str),
                active,
                updated_at,
            ),
        )

        self._conn.executemany(
            "INSERT INTO bm25_terms(token, doc_id, term_freq) VALUES (?, ?, ?)",
            [(token, document.doc_id, int(freq)) for token, freq in token_counts.items()],
        )

        if not document.active:
            return

        self._increment_stat_locked("document_count", 1.0)
        self._increment_stat_locked("total_token_count", float(len(tokens)))
        self._conn.executemany(
            """
            INSERT INTO bm25_df(token, doc_freq)
            VALUES (?, 1)
            ON CONFLICT(token) DO UPDATE SET doc_freq = doc_freq + 1
            """,
            [(token,) for token in token_counts],
        )

    def _remove_document_locked(self, doc_id: str) -> None:
        row = self._conn.execute(
            "SELECT active, token_count FROM bm25_docs WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if row is None:
            return

        if int(row["active"]):
            terms = self._conn.execute(
                "SELECT token FROM bm25_terms WHERE doc_id = ?",
                (doc_id,),
            ).fetchall()
            for term in terms:
                self._decrement_document_frequency_locked(str(term["token"]))
            self._increment_stat_locked("document_count", -1.0)
            self._increment_stat_locked("total_token_count", -float(row["token_count"]))

        self._conn.execute("DELETE FROM bm25_terms WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM bm25_docs WHERE doc_id = ?", (doc_id,))

    def _mark_inactive_locked(self, doc_id: str) -> None:
        row = self._conn.execute(
            "SELECT active, token_count FROM bm25_docs WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if row is None or not int(row["active"]):
            return

        terms = self._conn.execute(
            "SELECT token FROM bm25_terms WHERE doc_id = ?",
            (doc_id,),
        ).fetchall()
        for term in terms:
            self._decrement_document_frequency_locked(str(term["token"]))
        self._increment_stat_locked("document_count", -1.0)
        self._increment_stat_locked("total_token_count", -float(row["token_count"]))
        self._conn.execute(
            "UPDATE bm25_docs SET active = 0, updated_at = ? WHERE doc_id = ?",
            (datetime.now(timezone.utc).isoformat(), doc_id),
        )

    def _decrement_document_frequency_locked(self, token: str) -> None:
        row = self._conn.execute(
            "SELECT doc_freq FROM bm25_df WHERE token = ?",
            (token,),
        ).fetchone()
        if row is None:
            return
        doc_frequency = int(row["doc_freq"])
        if doc_frequency <= 1:
            self._conn.execute("DELETE FROM bm25_df WHERE token = ?", (token,))
            return
        self._conn.execute(
            "UPDATE bm25_df SET doc_freq = ? WHERE token = ?",
            (doc_frequency - 1, token),
        )

    def _get_stat_locked(self, key: str) -> float:
        row = self._conn.execute(
            "SELECT value FROM bm25_stats WHERE key = ?",
            (key,),
        ).fetchone()
        return float(row["value"]) if row is not None else 0.0

    def _set_stat_locked(self, key: str, value: float) -> None:
        self._conn.execute(
            """
            INSERT INTO bm25_stats(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, max(0.0, value)),
        )

    def _increment_stat_locked(self, key: str, delta: float) -> None:
        self._conn.execute(
            """
            INSERT INTO bm25_stats(key, value) VALUES (?, MAX(0.0, ?))
            ON CONFLICT(key) DO UPDATE SET value = MAX(0.0, bm25_stats.value + ?)
            """,
            (key, delta, delta),
        )


def _bm25_score(
    query_counts: Counter[str],
    token_counts: Counter[str],
    document_frequency: dict[str, int],
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
        idf = math.log(
            1
            + (document_count - term_document_frequency + 0.5)
            / (term_document_frequency + 0.5)
        )
        score += query_frequency * idf * (
            term_frequency * (k1 + 1) / (term_frequency + normalizer)
        )
    return score


def _json_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}

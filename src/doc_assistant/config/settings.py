from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str, default: str = "") -> tuple[str, ...]:
    value = os.getenv(name, default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _json_object_env(name: str) -> dict[str, Any]:
    value = os.getenv(name)
    if not value:
        return {}

    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a valid JSON object.") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a valid JSON object.")
    return data


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"
    ingest_jobs_db_path: Path = _path_env(
        "DOC_ASSISTANT_INGEST_JOBS_DB_PATH",
        PROJECT_ROOT / "data" / "ingest_jobs.sqlite3",
    )
    agent_tasks_db_path: Path = _path_env(
        "DOC_ASSISTANT_AGENT_TASKS_DB_PATH",
        PROJECT_ROOT / "data" / "agent_tasks.sqlite3",
    )
    matter_db_path: Path = _path_env(
        "DOC_ASSISTANT_MATTER_DB_PATH",
        PROJECT_ROOT / "data" / "matters.sqlite3",
    )
    memory_db_path: Path = _path_env(
        "DOC_ASSISTANT_MEMORY_DB_PATH",
        PROJECT_ROOT / "data" / "memory.sqlite3",
    )
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    collection_name: str = os.getenv("DOC_ASSISTANT_COLLECTION", "legal_documents")
    memory_collection_name: str = os.getenv("DOC_ASSISTANT_MEMORY_COLLECTION", "user_memories")
    chat_provider: str = os.getenv("DOC_ASSISTANT_CHAT_PROVIDER", "dashscope")
    chat_model_name: str = os.getenv("DOC_ASSISTANT_CHAT_MODEL", "qwen3.5-flash")
    chat_api: str = os.getenv("DOC_ASSISTANT_CHAT_API", "compatible")
    chat_api_key: str = _first_env("DOC_ASSISTANT_CHAT_API_KEY")
    chat_base_url: str = os.getenv("DOC_ASSISTANT_CHAT_BASE_URL", "")
    chat_extra_body: dict[str, Any] = field(
        default_factory=lambda: _json_object_env("DOC_ASSISTANT_CHAT_EXTRA_BODY")
    )
    embedding_provider: str = os.getenv("DOC_ASSISTANT_EMBEDDING_PROVIDER", "dashscope")
    embedding_api_key: str = _first_env("DOC_ASSISTANT_EMBEDDING_API_KEY", "DASHSCOPE_API_KEY")
    embedding_model_name: str = os.getenv("DOC_ASSISTANT_EMBEDDING_MODEL", "text-embedding-v3")
    enable_thinking: bool = _bool_env("DOC_ASSISTANT_ENABLE_THINKING", False)
    temperature: float = _float_env("DOC_ASSISTANT_TEMPERATURE", 0.0)
    top_k: int = _int_env("DOC_ASSISTANT_TOP_K", 5)
    retrieval_mode: str = os.getenv("DOC_ASSISTANT_RETRIEVAL_MODE", "hybrid")
    retrieval_fetch_k: int = _int_env("DOC_ASSISTANT_RETRIEVAL_FETCH_K", 40)
    retrieval_min_relevance: float = _float_env("DOC_ASSISTANT_RETRIEVAL_MIN_RELEVANCE", 0.0)
    retrieval_rrf_k: int = _int_env("DOC_ASSISTANT_RETRIEVAL_RRF_K", 60)
    retrieval_dense_weight: float = _float_env("DOC_ASSISTANT_RETRIEVAL_DENSE_WEIGHT", 1.0)
    retrieval_bm25_weight: float = _float_env("DOC_ASSISTANT_RETRIEVAL_BM25_WEIGHT", 1.0)
    retrieval_rerank_mode: str = os.getenv("DOC_ASSISTANT_RETRIEVAL_RERANK_MODE", "lexical")
    retrieval_rerank_weight: float = _float_env("DOC_ASSISTANT_RETRIEVAL_RERANK_WEIGHT", 0.25)
    retrieval_mmr_lambda: float = _float_env("DOC_ASSISTANT_RETRIEVAL_MMR_LAMBDA", 0.85)
    memory_top_k: int = _int_env("DOC_ASSISTANT_MEMORY_TOP_K", 5)
    memory_min_confidence: float = _float_env("DOC_ASSISTANT_MEMORY_MIN_CONFIDENCE", 0.55)
    chunk_size: int = _int_env("DOC_ASSISTANT_CHUNK_SIZE", 900)
    chunk_overlap: int = _int_env("DOC_ASSISTANT_CHUNK_OVERLAP", 120)
    tool_call_max_iterations: int = _int_env("DOC_ASSISTANT_TOOL_CALL_MAX_ITERATIONS", 6)
    web_search_enabled: bool = _bool_env("DOC_ASSISTANT_WEB_SEARCH_ENABLED", False)
    web_search_provider: str = os.getenv("DOC_ASSISTANT_WEB_SEARCH_PROVIDER", "duckduckgo")
    web_search_api_key: str = _first_env(
        "DOC_ASSISTANT_WEB_SEARCH_API_KEY",
        "BRAVE_SEARCH_API_KEY",
        "BING_SEARCH_API_KEY",
    )
    web_search_base_url: str = os.getenv("DOC_ASSISTANT_WEB_SEARCH_BASE_URL", "")
    web_search_max_results: int = _int_env("DOC_ASSISTANT_WEB_SEARCH_MAX_RESULTS", 5)
    web_search_timeout_seconds: int = _int_env("DOC_ASSISTANT_WEB_SEARCH_TIMEOUT_SECONDS", 10)
    api_keys: tuple[str, ...] = _csv_env("DOC_ASSISTANT_API_KEYS")
    cors_origins: tuple[str, ...] = _csv_env(
        "DOC_ASSISTANT_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,"
        "http://127.0.0.1:3000,http://127.0.0.1:5173",
    )
    cors_allow_credentials: bool = _bool_env("DOC_ASSISTANT_CORS_ALLOW_CREDENTIALS", False)
    default_tenant_id: str = os.getenv("DOC_ASSISTANT_DEFAULT_TENANT_ID", "default")
    max_upload_bytes: int = _int_env("DOC_ASSISTANT_MAX_UPLOAD_BYTES", 20 * 1024 * 1024)
    pdf_ocr_enabled: bool = _bool_env("DOC_ASSISTANT_PDF_OCR_ENABLED", False)
    pdf_ocr_lang: str = os.getenv("DOC_ASSISTANT_PDF_OCR_LANG", "eng")


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
settings.ingest_jobs_db_path.parent.mkdir(parents=True, exist_ok=True)
settings.agent_tasks_db_path.parent.mkdir(parents=True, exist_ok=True)
settings.matter_db_path.parent.mkdir(parents=True, exist_ok=True)
settings.memory_db_path.parent.mkdir(parents=True, exist_ok=True)

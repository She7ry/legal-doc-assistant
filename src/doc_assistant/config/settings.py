from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

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


def _float_csv_env(name: str, default: str = "") -> tuple[float, ...]:
    values = []
    for part in _csv_env(name, default):
        try:
            values.append(float(part))
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated list of numbers.") from exc
    return tuple(values)


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
    memory_vector_store_dir: Path = field(
        default_factory=lambda: _path_env(
            "DOC_ASSISTANT_MEMORY_VECTOR_STORE_DIR",
            PROJECT_ROOT / "data" / "memory_vector_store",
        )
    )
    ingest_jobs_db_path: Path = field(
        default_factory=lambda: _path_env(
            "DOC_ASSISTANT_INGEST_JOBS_DB_PATH",
            PROJECT_ROOT / "data" / "ingest_jobs.sqlite3",
        )
    )
    agent_tasks_db_path: Path = field(
        default_factory=lambda: _path_env(
            "DOC_ASSISTANT_AGENT_TASKS_DB_PATH",
            PROJECT_ROOT / "data" / "agent_tasks.sqlite3",
        )
    )
    matter_db_path: Path = field(
        default_factory=lambda: _path_env(
            "DOC_ASSISTANT_MATTER_DB_PATH",
            PROJECT_ROOT / "data" / "matters.sqlite3",
        )
    )
    memory_db_path: Path = field(
        default_factory=lambda: _path_env(
            "DOC_ASSISTANT_MEMORY_DB_PATH",
            PROJECT_ROOT / "data" / "memory.sqlite3",
        )
    )
    dashscope_api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    collection_name: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_COLLECTION", "legal_documents"))
    memory_collection_name: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_MEMORY_COLLECTION", "user_memories"))
    chat_provider: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_CHAT_PROVIDER", "dashscope"))
    chat_model_name: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_CHAT_MODEL", "qwen3.5-flash"))
    chat_api: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_CHAT_API", "compatible"))
    chat_api_key: str = field(default_factory=lambda: _first_env("DOC_ASSISTANT_CHAT_API_KEY"))
    chat_base_url: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_CHAT_BASE_URL", ""))
    chat_extra_body: dict[str, Any] = field(
        default_factory=lambda: _json_object_env("DOC_ASSISTANT_CHAT_EXTRA_BODY")
    )
    llm_max_retries: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_LLM_MAX_RETRIES", 3))
    llm_circuit_breaker_threshold: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_LLM_CIRCUIT_BREAKER_THRESHOLD", 5))
    llm_circuit_breaker_cooldown_seconds: int = field(
        default_factory=lambda: _int_env("DOC_ASSISTANT_LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS", 30)
    )
    embedding_provider: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_EMBEDDING_PROVIDER", "dashscope"))
    embedding_api_key: str = field(default_factory=lambda: _first_env("DOC_ASSISTANT_EMBEDDING_API_KEY", "DASHSCOPE_API_KEY"))
    embedding_base_url: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_EMBEDDING_BASE_URL", ""))
    embedding_model_name: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_EMBEDDING_MODEL", "text-embedding-v3"))
    embedding_device: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_EMBEDDING_DEVICE", "cpu"))
    enable_thinking: bool = field(default_factory=lambda: _bool_env("DOC_ASSISTANT_ENABLE_THINKING", False))
    temperature: float = field(default_factory=lambda: _float_env("DOC_ASSISTANT_TEMPERATURE", 0.0))
    top_k: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_TOP_K", 5))
    retrieval_mode: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_RETRIEVAL_MODE", "hybrid"))
    retrieval_fetch_k: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_RETRIEVAL_FETCH_K", 40))
    retrieval_min_relevance: float = field(default_factory=lambda: _float_env("DOC_ASSISTANT_RETRIEVAL_MIN_RELEVANCE", 0.0))
    retrieval_rrf_k: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_RETRIEVAL_RRF_K", 60))
    retrieval_dense_weight: float = field(default_factory=lambda: _float_env("DOC_ASSISTANT_RETRIEVAL_DENSE_WEIGHT", 1.0))
    retrieval_bm25_weight: float = field(default_factory=lambda: _float_env("DOC_ASSISTANT_RETRIEVAL_BM25_WEIGHT", 1.0))
    retrieval_rerank_mode: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_RETRIEVAL_RERANK_MODE", "lexical"))
    retrieval_rerank_weight: float = field(default_factory=lambda: _float_env("DOC_ASSISTANT_RETRIEVAL_RERANK_WEIGHT", 0.25))
    retrieval_mmr_lambda: float = field(default_factory=lambda: _float_env("DOC_ASSISTANT_RETRIEVAL_MMR_LAMBDA", 0.85))
    retrieval_cache_ttl_seconds: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_RETRIEVAL_CACHE_TTL_SECONDS", 300))
    retrieval_cache_max_size: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_RETRIEVAL_CACHE_MAX_SIZE", 128))
    query_rewrite_enabled: bool = field(default_factory=lambda: _bool_env("DOC_ASSISTANT_QUERY_REWRITE_ENABLED", True))
    embedding_batch_size: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_EMBED_BATCH_SIZE", 20))
    embedding_max_workers: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_EMBED_MAX_WORKERS", 4))
    agent_max_parallel_steps: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_AGENT_MAX_PARALLEL_STEPS", 3))
    agent_step_max_retries: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_AGENT_STEP_MAX_RETRIES", 2))
    agent_step_retry_backoff_seconds: tuple[float, ...] = field(
        default_factory=lambda: _float_csv_env(
            "DOC_ASSISTANT_AGENT_STEP_RETRY_BACKOFF_SECONDS",
            "2,5",
        )
    )
    agent_llm_planner_enabled: bool = field(default_factory=lambda: _bool_env("DOC_ASSISTANT_AGENT_LLM_PLANNER_ENABLED", True))
    agent_react_enabled: bool = field(default_factory=lambda: _bool_env("DOC_ASSISTANT_AGENT_REACT_ENABLED", True))
    agent_react_max_iterations: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_AGENT_REACT_MAX_ITERATIONS", 2))
    chat_history_window: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_CHAT_HISTORY_WINDOW", 12))
    memory_top_k: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_MEMORY_TOP_K", 5))
    memory_min_confidence: float = field(default_factory=lambda: _float_env("DOC_ASSISTANT_MEMORY_MIN_CONFIDENCE", 0.55))
    memory_semantic_dedup_min_score: float = field(
        default_factory=lambda: _float_env("DOC_ASSISTANT_MEMORY_SEMANTIC_DEDUP_MIN_SCORE", 0.88)
    )
    memory_session_ttl_hours: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_MEMORY_SESSION_TTL_HOURS", 24))
    memory_task_ttl_hours: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_MEMORY_TASK_TTL_HOURS", 168))
    memory_max_active_per_user: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_MEMORY_MAX_ACTIVE_PER_USER", 500))
    memory_decay_half_life_days: float = field(default_factory=lambda: _float_env("DOC_ASSISTANT_MEMORY_DECAY_HALF_LIFE_DAYS", 90.0))
    memory_maintenance_enabled: bool = field(default_factory=lambda: _bool_env("DOC_ASSISTANT_MEMORY_MAINTENANCE_ENABLED", True))
    memory_maintenance_cooldown_seconds: int = field(
        default_factory=lambda: _int_env("DOC_ASSISTANT_MEMORY_MAINTENANCE_COOLDOWN_SECONDS", 300)
    )
    memory_auto_summary_threshold: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_MEMORY_AUTO_SUMMARY_THRESHOLD", 12))
    memory_auto_summary_interval: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_MEMORY_AUTO_SUMMARY_INTERVAL", 5))
    memory_auto_summary_window: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_MEMORY_AUTO_SUMMARY_WINDOW", 40))
    memory_prompt_max_tokens: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_MEMORY_PROMPT_MAX_TOKENS", 800))
    memory_llm_extraction_enabled: bool = field(default_factory=lambda: _bool_env("DOC_ASSISTANT_MEMORY_LLM_EXTRACTION_ENABLED", True))
    memory_llm_extraction_max_items: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_MEMORY_LLM_EXTRACTION_MAX_ITEMS", 3))
    memory_llm_extraction_min_confidence: float = field(
        default_factory=lambda: _float_env("DOC_ASSISTANT_MEMORY_LLM_EXTRACTION_MIN_CONFIDENCE", 0.6)
    )
    chunk_size: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_CHUNK_OVERLAP", 120))
    tool_call_max_iterations: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_TOOL_CALL_MAX_ITERATIONS", 6))
    tool_call_history_window: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_TOOL_CALL_HISTORY_WINDOW", 12))
    tool_call_timeout_seconds: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_TOOL_CALL_TIMEOUT_SECONDS", 30))
    web_search_enabled: bool = field(default_factory=lambda: _bool_env("DOC_ASSISTANT_WEB_SEARCH_ENABLED", False))
    web_search_provider: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_WEB_SEARCH_PROVIDER", "duckduckgo"))
    web_search_api_key: str = field(
        default_factory=lambda: _first_env(
            "DOC_ASSISTANT_WEB_SEARCH_API_KEY",
            "BRAVE_SEARCH_API_KEY",
            "BING_SEARCH_API_KEY",
        )
    )
    web_search_base_url: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_WEB_SEARCH_BASE_URL", ""))
    web_search_max_results: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_WEB_SEARCH_MAX_RESULTS", 5))
    web_search_timeout_seconds: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_WEB_SEARCH_TIMEOUT_SECONDS", 10))
    web_search_max_retries: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_WEB_SEARCH_MAX_RETRIES", 3))
    api_keys: tuple[str, ...] = field(default_factory=lambda: _csv_env("DOC_ASSISTANT_API_KEYS"))
    rate_limit_enabled: bool = field(default_factory=lambda: _bool_env("DOC_ASSISTANT_RATE_LIMIT_ENABLED", True))
    rate_limit_max_requests: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_RATE_LIMIT_MAX_REQUESTS", 120))
    rate_limit_window_seconds: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_RATE_LIMIT_WINDOW_SECONDS", 60))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: _csv_env(
            "DOC_ASSISTANT_CORS_ORIGINS",
            "http://localhost:3000,http://localhost:5173,"
            "http://127.0.0.1:3000,http://127.0.0.1:5173",
        )
    )
    cors_allow_credentials: bool = field(default_factory=lambda: _bool_env("DOC_ASSISTANT_CORS_ALLOW_CREDENTIALS", False))
    default_tenant_id: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_DEFAULT_TENANT_ID", "default"))
    max_upload_bytes: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
    background_max_workers: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_BACKGROUND_MAX_WORKERS", 4))
    pdf_ocr_enabled: bool = field(default_factory=lambda: _bool_env("DOC_ASSISTANT_PDF_OCR_ENABLED", False))
    pdf_ocr_lang: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_PDF_OCR_LANG", "eng"))

    def __post_init__(self) -> None:
        _validate_positive("chunk_size", self.chunk_size)
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if self.temperature < 0:
            raise ValueError("temperature must be greater than or equal to 0.")
        if self.retrieval_mode.strip().lower() not in {"hybrid", "dense", "vector", "bm25", "sparse"}:
            raise ValueError("retrieval_mode must be one of: hybrid, dense, vector, bm25, sparse.")
        _validate_positive("top_k", self.top_k)
        _validate_positive("retrieval_fetch_k", self.retrieval_fetch_k)
        _validate_positive("retrieval_rrf_k", self.retrieval_rrf_k)
        if not 0 <= self.retrieval_mmr_lambda <= 1:
            raise ValueError("retrieval_mmr_lambda must be between 0 and 1.")
        if self.retrieval_min_relevance < 0:
            raise ValueError("retrieval_min_relevance must be greater than or equal to 0.")
        if self.agent_step_max_retries < 0:
            raise ValueError("agent_step_max_retries must be greater than or equal to 0.")
        if any(value < 0 for value in self.agent_step_retry_backoff_seconds):
            raise ValueError("agent_step_retry_backoff_seconds values must be non-negative.")
        if self.agent_react_max_iterations < 0:
            raise ValueError("agent_react_max_iterations must be greater than or equal to 0.")
        if self.memory_auto_summary_threshold < 0:
            raise ValueError("memory_auto_summary_threshold must be greater than or equal to 0.")
        if self.memory_auto_summary_interval <= 0:
            raise ValueError("memory_auto_summary_interval must be greater than 0.")
        if self.memory_auto_summary_window <= 0:
            raise ValueError("memory_auto_summary_window must be greater than 0.")
        if self.memory_prompt_max_tokens <= 0:
            raise ValueError("memory_prompt_max_tokens must be greater than 0.")
        if not 0 <= self.memory_semantic_dedup_min_score <= 1:
            raise ValueError("memory_semantic_dedup_min_score must be between 0 and 1.")
        if self.memory_maintenance_cooldown_seconds < 0:
            raise ValueError("memory_maintenance_cooldown_seconds must be greater than or equal to 0.")
        if self.memory_llm_extraction_max_items <= 0:
            raise ValueError("memory_llm_extraction_max_items must be greater than 0.")
        if not 0 <= self.memory_llm_extraction_min_confidence <= 1:
            raise ValueError("memory_llm_extraction_min_confidence must be between 0 and 1.")

    def with_overrides(self, **kwargs: Any) -> Settings:
        return replace(self, **kwargs)

    def ensure_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.memory_vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.ingest_jobs_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent_tasks_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.matter_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)

def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")


settings = Settings()

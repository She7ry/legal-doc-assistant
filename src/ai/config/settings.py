"""Validated application settings loaded from environment variables and ``.env``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Flat settings model; environment parsing and type validation are framework-managed."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="DOC_ASSISTANT_",
        enable_decoding=False,
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    project_root: Path = PROJECT_ROOT
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"
    memory_vector_store_dir: Path = PROJECT_ROOT / "data" / "memory_vector_store"
    ingest_jobs_db_path: Path = PROJECT_ROOT / "data" / "personal_ingest_jobs.sqlite3"
    agent_tasks_db_path: Path = PROJECT_ROOT / "data" / "personal_agent_tasks.sqlite3"
    memory_db_path: Path = PROJECT_ROOT / "data" / "personal_memory.sqlite3"
    auth_db_path: Path = PROJECT_ROOT / "data" / "auth.sqlite3"
    qdrant_url: str = ""
    qdrant_api_key: str = Field(default="", repr=False)
    qdrant_prefer_grpc: bool = False

    dashscope_api_key: str = Field(
        default="",
        validation_alias="DASHSCOPE_API_KEY",
        repr=False,
    )
    deepseek_api_key: str = Field(
        default="",
        validation_alias="DEEPSEEK_API_KEY",
        repr=False,
    )
    collection_name: str = Field(
        default="legal_documents",
        validation_alias="DOC_ASSISTANT_COLLECTION",
    )
    memory_collection_name: str = Field(
        default="user_memories",
        validation_alias="DOC_ASSISTANT_MEMORY_COLLECTION",
    )
    chat_provider: str = "deepseek"
    chat_model_name: str = Field(
        default="deepseek-v4-pro",
        validation_alias="DOC_ASSISTANT_CHAT_MODEL",
    )
    chat_api: str = "compatible"
    chat_api_key: str = Field(default="", repr=False)
    chat_base_url: str = ""
    chat_extra_body: dict[str, Any] = Field(default_factory=dict)
    llm_max_retries: int = 3
    temperature: float = 0.0

    embedding_provider: str = "dashscope"
    embedding_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "DOC_ASSISTANT_EMBEDDING_API_KEY",
            "DASHSCOPE_API_KEY",
        ),
        repr=False,
    )
    embedding_base_url: str = ""
    embedding_model_name: str = Field(
        default="text-embedding-v3",
        validation_alias="DOC_ASSISTANT_EMBEDDING_MODEL",
    )
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(
        default=20,
        validation_alias="DOC_ASSISTANT_EMBED_BATCH_SIZE",
    )
    embedding_max_workers: int = Field(
        default=4,
        validation_alias="DOC_ASSISTANT_EMBED_MAX_WORKERS",
    )

    top_k: int = 5
    retrieval_mode: str = "hybrid"
    retrieval_fetch_k: int = 40
    retrieval_min_relevance: float = 0.0
    retrieval_mmr_lambda: float = 0.85
    retrieval_bm25_k1: float = 1.5
    retrieval_bm25_b: float = 0.75
    retrieval_bm25_average_length: float = 256.0
    retrieval_cache_ttl_seconds: int = 300
    retrieval_cache_max_size: int = 128
    query_rewrite_enabled: bool = True
    chunk_size: int = 900
    chunk_overlap: int = 120

    chat_history_window: int = 12
    tool_call_max_iterations: int = 6
    tool_call_history_window: int = 12
    tool_call_context_max_chars: int = 24_000
    memory_top_k: int = 5
    memory_min_confidence: float = 0.55
    memory_prompt_max_tokens: int = 800

    web_search_enabled: bool = False
    web_search_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "DOC_ASSISTANT_WEB_SEARCH_API_KEY",
            "BRAVE_SEARCH_API_KEY",
        ),
        repr=False,
    )
    web_search_max_results: int = 5
    web_search_timeout_seconds: int = 10
    web_search_max_retries: int = 3
    docusign_mcp_enabled: bool = False
    docusign_client_id: str = ""
    docusign_client_secret: str = Field(default="", repr=False)
    docusign_token_path: Path = PROJECT_ROOT / "data" / "docusign_mcp_oauth.json"

    auth_session_ttl_hours: int = 24 * 7
    auth_cookie_secure: bool = False
    rate_limit_enabled: bool = True
    rate_limit_max_requests: int = 120
    rate_limit_window_seconds: int = 60
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    )
    cors_allow_credentials: bool = True
    max_upload_bytes: int = 20 * 1024 * 1024
    background_max_workers: int = 4
    pdf_ocr_enabled: bool = False
    pdf_ocr_lang: str = "eng"

    @field_validator("chat_extra_body", mode="before")
    @classmethod
    def _parse_json_object(cls, value: Any) -> dict[str, Any]:
        if value in (None, ""):
            return {}
        if isinstance(value, dict):
            return value
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("chat_extra_body must be a valid JSON object") from exc
        if not isinstance(parsed, dict):
            raise ValueError("chat_extra_body must be a valid JSON object")
        return parsed

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_csv(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return tuple(value)

    @model_validator(mode="after")
    def _validate_ranges(self) -> Settings:
        for name in (
            "chunk_size",
            "top_k",
            "retrieval_fetch_k",
            "embedding_batch_size",
            "embedding_max_workers",
            "tool_call_max_iterations",
            "tool_call_context_max_chars",
            "memory_top_k",
            "memory_prompt_max_tokens",
            "web_search_max_results",
            "web_search_timeout_seconds",
            "auth_session_ttl_hours",
            "rate_limit_max_requests",
            "rate_limit_window_seconds",
            "max_upload_bytes",
            "background_max_workers",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than 0")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        if self.temperature < 0:
            raise ValueError("temperature must be greater than or equal to 0")
        if self.retrieval_mode.strip().lower() not in {
            "hybrid",
            "dense",
            "vector",
            "bm25",
            "sparse",
        }:
            raise ValueError("unsupported retrieval_mode")
        if not 0 <= self.retrieval_mmr_lambda <= 1:
            raise ValueError("retrieval_mmr_lambda must be between 0 and 1")
        if self.retrieval_bm25_k1 <= 0:
            raise ValueError("retrieval_bm25_k1 must be greater than 0")
        if not 0 <= self.retrieval_bm25_b <= 1:
            raise ValueError("retrieval_bm25_b must be between 0 and 1")
        if self.retrieval_bm25_average_length <= 0:
            raise ValueError("retrieval_bm25_average_length must be greater than 0")
        if self.retrieval_min_relevance < 0:
            raise ValueError("retrieval_min_relevance must be greater than or equal to 0")
        if self.docusign_mcp_enabled and not (
            self.docusign_client_id.strip() and self.docusign_client_secret.strip()
        ):
            raise ValueError("Docusign MCP requires a client ID and client secret")
        return self

    def with_overrides(self, **kwargs: Any) -> Settings:
        unknown = kwargs.keys() - type(self).model_fields.keys()
        if unknown:
            raise TypeError(f"Settings has no field '{next(iter(unknown))}'")
        return type(self)(**(self.model_dump() | kwargs))

    def ensure_directories(self) -> None:
        for path in (
            self.upload_dir,
            self.vector_store_dir,
            self.memory_vector_store_dir,
            self.ingest_jobs_db_path.parent,
            self.agent_tasks_db_path.parent,
            self.memory_db_path.parent,
            self.auth_db_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()

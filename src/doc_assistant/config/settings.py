from __future__ import annotations

import os
from dataclasses import dataclass
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


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"
    memory_db_path: Path = _path_env(
        "DOC_ASSISTANT_MEMORY_DB_PATH",
        PROJECT_ROOT / "data" / "memory.sqlite3",
    )
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    collection_name: str = os.getenv("DOC_ASSISTANT_COLLECTION", "legal_documents")
    memory_collection_name: str = os.getenv("DOC_ASSISTANT_MEMORY_COLLECTION", "user_memories")
    chat_model_name: str = os.getenv("DOC_ASSISTANT_CHAT_MODEL", "qwen3.5-flash")
    chat_api: str = os.getenv("DOC_ASSISTANT_CHAT_API", "compatible")
    chat_base_url: str = os.getenv(
        "DOC_ASSISTANT_CHAT_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    embedding_model_name: str = os.getenv("DOC_ASSISTANT_EMBEDDING_MODEL", "text-embedding-v3")
    enable_thinking: bool = _bool_env("DOC_ASSISTANT_ENABLE_THINKING", False)
    temperature: float = _float_env("DOC_ASSISTANT_TEMPERATURE", 0.0)
    top_k: int = _int_env("DOC_ASSISTANT_TOP_K", 5)
    memory_top_k: int = _int_env("DOC_ASSISTANT_MEMORY_TOP_K", 5)
    memory_min_confidence: float = _float_env("DOC_ASSISTANT_MEMORY_MIN_CONFIDENCE", 0.55)
    chunk_size: int = _int_env("DOC_ASSISTANT_CHUNK_SIZE", 900)
    chunk_overlap: int = _int_env("DOC_ASSISTANT_CHUNK_OVERLAP", 120)
    api_keys: tuple[str, ...] = _csv_env("DOC_ASSISTANT_API_KEYS")
    cors_origins: tuple[str, ...] = _csv_env(
        "DOC_ASSISTANT_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,"
        "http://127.0.0.1:3000,http://127.0.0.1:5173",
    )
    cors_allow_credentials: bool = _bool_env("DOC_ASSISTANT_CORS_ALLOW_CREDENTIALS", False)
    default_tenant_id: str = os.getenv("DOC_ASSISTANT_DEFAULT_TENANT_ID", "default")
    max_upload_bytes: int = _int_env("DOC_ASSISTANT_MAX_UPLOAD_BYTES", 20 * 1024 * 1024)


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
settings.memory_db_path.parent.mkdir(parents=True, exist_ok=True)

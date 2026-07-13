"""全局配置：从环境变量 / .env 读取并校验 Settings 单例。

所有可调参数（模型、检索、Agent、记忆、API 安全等）集中在此；
业务代码应通过 ``from doc_assistant.config.settings import settings`` 访问，
避免直接读 ``os.getenv``。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


# ── 环境变量读取辅助函数 ──────────────────────────────────────────────────


def _int_env(name: str, default: int) -> int:
    """从环境变量读整数，未设置则返回默认值。"""
    value = os.getenv(name)
    return int(value) if value else default


def _float_env(name: str, default: float) -> float:
    """从环境变量读浮点数，未设置则返回默认值。"""
    value = os.getenv(name)
    return float(value) if value else default


def _bool_env(name: str, default: bool) -> bool:
    """从环境变量读布尔值，支持 1/true/yes/on（大小写不敏感）。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str, default: str = "") -> tuple[str, ...]:
    """从环境变量读逗号分隔列表，返回去空白去空项的元组。"""
    value = os.getenv(name, default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _json_object_env(name: str) -> dict[str, Any]:
    """从环境变量读 JSON 对象，解析失败或非 dict 时抛出 ValueError。"""
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
    """从环境变量读文件路径，未设置则返回默认 Path。"""
    value = os.getenv(name)
    return Path(value) if value else default


def _first_env(*names: str, default: str = "") -> str:
    """从多个环境变量中返回第一个非空值。"""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


class _SecretStr(str):
    """敏感字符串：repr 时只显示首尾各两位加星号，防止日志泄露 API Key。"""

    def __repr__(self) -> str:
        if len(self) <= 4:
            return "'***'"
        return f"'{self[:2]}***{self[-2:]}'"

    def __str__(self) -> str:
        return super().__str__()


def _secret_env(name: str, default: str = "") -> _SecretStr:
    """从环境变量读敏感值，包裹为 _SecretStr。"""
    value = os.getenv(name, default)
    return _SecretStr(value.strip() if value else default)


def _secret_first_env(*names: str, default: str = "") -> _SecretStr:
    """从多个环境变量中返回第一个非空的敏感值。"""
    return _SecretStr(_first_env(*names, default=default))


# ── 分组子配置 ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StorageSettings:
    """文件路径与数据库位置。"""

    project_root: Path = PROJECT_ROOT
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    vector_store_dir: Path = field(
        default_factory=lambda: _path_env(
            "DOC_ASSISTANT_VECTOR_STORE_DIR",
            PROJECT_ROOT / "data" / "vector_store",
        )
    )
    memory_vector_store_dir: Path = field(
        default_factory=lambda: _path_env(
            "DOC_ASSISTANT_MEMORY_VECTOR_STORE_DIR",
            PROJECT_ROOT / "data" / "memory_vector_store",
        )
    )
    qdrant_url: str = field(
        default_factory=lambda: os.getenv("DOC_ASSISTANT_QDRANT_URL", "").strip()
    )
    qdrant_api_key: _SecretStr = field(
        default_factory=lambda: _secret_env("DOC_ASSISTANT_QDRANT_API_KEY")
    )
    qdrant_prefer_grpc: bool = field(
        default_factory=lambda: _bool_env("DOC_ASSISTANT_QDRANT_PREFER_GRPC", False)
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


@dataclass(frozen=True)
class LLMSettings:
    """LLM / Chat model 配置。"""

    dashscope_api_key: _SecretStr = field(default_factory=lambda: _secret_env("DASHSCOPE_API_KEY"))
    deepseek_api_key: _SecretStr = field(default_factory=lambda: _secret_env("DEEPSEEK_API_KEY"))
    collection_name: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_COLLECTION", "legal_documents"))
    memory_collection_name: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_MEMORY_COLLECTION", "user_memories"))
    chat_provider: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_CHAT_PROVIDER", "deepseek"))
    chat_model_name: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_CHAT_MODEL", "deepseek-v4-pro"))
    chat_api: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_CHAT_API", "compatible"))
    chat_api_key: _SecretStr = field(default_factory=lambda: _secret_first_env("DOC_ASSISTANT_CHAT_API_KEY"))
    chat_base_url: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_CHAT_BASE_URL", ""))
    chat_extra_body: dict[str, Any] = field(
        default_factory=lambda: _json_object_env("DOC_ASSISTANT_CHAT_EXTRA_BODY")
    )
    llm_max_retries: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_LLM_MAX_RETRIES", 3))
    temperature: float = field(default_factory=lambda: _float_env("DOC_ASSISTANT_TEMPERATURE", 0.0))


@dataclass(frozen=True)
class EmbeddingSettings:
    """Embedding model 配置。"""

    embedding_provider: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_EMBEDDING_PROVIDER", "dashscope"))
    embedding_api_key: _SecretStr = field(default_factory=lambda: _secret_first_env("DOC_ASSISTANT_EMBEDDING_API_KEY", "DASHSCOPE_API_KEY"))
    embedding_base_url: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_EMBEDDING_BASE_URL", ""))
    embedding_model_name: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_EMBEDDING_MODEL", "text-embedding-v3"))
    embedding_device: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_EMBEDDING_DEVICE", "cpu"))
    embedding_batch_size: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_EMBED_BATCH_SIZE", 20))
    embedding_max_workers: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_EMBED_MAX_WORKERS", 4))


@dataclass(frozen=True)
class RetrievalSettings:
    """检索 / RAG 配置。"""

    top_k: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_TOP_K", 5))
    retrieval_mode: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_RETRIEVAL_MODE", "hybrid"))
    retrieval_fetch_k: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_RETRIEVAL_FETCH_K", 40))
    retrieval_min_relevance: float = field(default_factory=lambda: _float_env("DOC_ASSISTANT_RETRIEVAL_MIN_RELEVANCE", 0.0))
    retrieval_mmr_lambda: float = field(default_factory=lambda: _float_env("DOC_ASSISTANT_RETRIEVAL_MMR_LAMBDA", 0.85))
    retrieval_bm25_k1: float = field(default_factory=lambda: _float_env("DOC_ASSISTANT_RETRIEVAL_BM25_K1", 1.5))
    retrieval_bm25_b: float = field(default_factory=lambda: _float_env("DOC_ASSISTANT_RETRIEVAL_BM25_B", 0.75))
    retrieval_bm25_average_length: float = field(
        default_factory=lambda: _float_env("DOC_ASSISTANT_RETRIEVAL_BM25_AVERAGE_LENGTH", 256.0)
    )
    retrieval_cache_ttl_seconds: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_RETRIEVAL_CACHE_TTL_SECONDS", 300))
    retrieval_cache_max_size: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_RETRIEVAL_CACHE_MAX_SIZE", 128))
    query_rewrite_enabled: bool = field(default_factory=lambda: _bool_env("DOC_ASSISTANT_QUERY_REWRITE_ENABLED", True))
    chunk_size: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_CHUNK_OVERLAP", 120))


@dataclass(frozen=True)
class SkillSettings:
    """Skill 启用白名单与资源限制。"""

    skills_enabled: bool = field(
        default_factory=lambda: _bool_env("DOC_ASSISTANT_SKILLS_ENABLED", True)
    )
    skills_root: Path = field(
        default_factory=lambda: _path_env("DOC_ASSISTANT_SKILLS_ROOT", PROJECT_ROOT / "skills")
    )
    skills_allowlist: tuple[str, ...] = field(
        default_factory=lambda: _csv_env(
            "DOC_ASSISTANT_SKILLS_ALLOWLIST",
            "grounded-rag-answer,verify-citation-support,"
            "decompose-retrieval-query,assess-evidence-sufficiency",
        )
    )
    skill_max_catalog_size: int = field(
        default_factory=lambda: _int_env("DOC_ASSISTANT_SKILL_MAX_CATALOG_SIZE", 32)
    )
    skill_max_file_bytes: int = field(
        default_factory=lambda: _int_env("DOC_ASSISTANT_SKILL_MAX_FILE_BYTES", 65_536)
    )
    skill_max_reference_files: int = field(
        default_factory=lambda: _int_env("DOC_ASSISTANT_SKILL_MAX_REFERENCE_FILES", 16)
    )
    skill_max_reference_bytes: int = field(
        default_factory=lambda: _int_env("DOC_ASSISTANT_SKILL_MAX_REFERENCE_BYTES", 131_072)
    )
    skill_max_loaded_tokens: int = field(
        default_factory=lambda: _int_env("DOC_ASSISTANT_SKILL_MAX_LOADED_TOKENS", 4_000)
    )
    skill_max_selected: int = field(
        default_factory=lambda: _int_env("DOC_ASSISTANT_SKILL_MAX_SELECTED", 4)
    )
    skill_query_decomposition_enabled: bool = field(
        default_factory=lambda: _bool_env("DOC_ASSISTANT_SKILL_QUERY_DECOMPOSITION_ENABLED", True)
    )
    skill_max_retrieval_queries: int = field(
        default_factory=lambda: _int_env("DOC_ASSISTANT_SKILL_MAX_RETRIEVAL_QUERIES", 4)
    )


@dataclass(frozen=True)
class AgentSettings:
    """Agent 对话与工具调用运行时限制。"""

    chat_history_window: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_CHAT_HISTORY_WINDOW", 12))
    tool_call_max_iterations: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_TOOL_CALL_MAX_ITERATIONS", 6))
    tool_call_history_window: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_TOOL_CALL_HISTORY_WINDOW", 12))


@dataclass(frozen=True)
class MemorySettings:
    """记忆子系统配置。"""

    memory_top_k: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_MEMORY_TOP_K", 5))
    memory_min_confidence: float = field(default_factory=lambda: _float_env("DOC_ASSISTANT_MEMORY_MIN_CONFIDENCE", 0.55))
    memory_prompt_max_tokens: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_MEMORY_PROMPT_MAX_TOKENS", 800))


@dataclass(frozen=True)
class WebSearchSettings:
    """网页搜索配置（RAG 不足时的外部搜索回退）。"""

    web_search_enabled: bool = field(default_factory=lambda: _bool_env("DOC_ASSISTANT_WEB_SEARCH_ENABLED", False))
    web_search_provider: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_WEB_SEARCH_PROVIDER", "duckduckgo"))
    web_search_api_key: _SecretStr = field(
        default_factory=lambda: _secret_first_env(
            "DOC_ASSISTANT_WEB_SEARCH_API_KEY",
            "BRAVE_SEARCH_API_KEY",
            "BING_SEARCH_API_KEY",
        )
    )
    web_search_base_url: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_WEB_SEARCH_BASE_URL", ""))
    web_search_max_results: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_WEB_SEARCH_MAX_RESULTS", 5))
    web_search_timeout_seconds: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_WEB_SEARCH_TIMEOUT_SECONDS", 10))
    web_search_max_retries: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_WEB_SEARCH_MAX_RETRIES", 3))


@dataclass(frozen=True)
class SecuritySettings:
    """API 鉴权、频率限制与 CORS 配置。"""

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


# ── 主 Settings 类 ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Settings:
    """应用全局配置单例。

    分组子配置通过 ``settings.storage``、``settings.llm`` 等访问；
    为向后兼容，字段也可直接 ``settings.top_k`` 访问（通过 ``__getattr__`` 代理）。
    """

    storage: StorageSettings = field(default_factory=StorageSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    skill: SkillSettings = field(default_factory=SkillSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    web_search: WebSearchSettings = field(default_factory=WebSearchSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    # ── 顶层运行时配置 ────────────────────────────────────────────────────
    default_tenant_id: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_DEFAULT_TENANT_ID", "default"))
    max_upload_bytes: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
    background_max_workers: int = field(default_factory=lambda: _int_env("DOC_ASSISTANT_BACKGROUND_MAX_WORKERS", 4))
    pdf_ocr_enabled: bool = field(default_factory=lambda: _bool_env("DOC_ASSISTANT_PDF_OCR_ENABLED", False))
    pdf_ocr_lang: str = field(default_factory=lambda: os.getenv("DOC_ASSISTANT_PDF_OCR_LANG", "eng"))

    def __getattr__(self, name: str) -> Any:
        """向后兼容：将未找到的属性委托给子配置对象查找。"""
        for sub in (
            "storage", "llm", "embedding", "retrieval", "skill",
            "agent", "memory", "web_search", "security",
        ):
            sub_obj = object.__getattribute__(self, sub)
            if hasattr(sub_obj, name):
                return getattr(sub_obj, name)
        raise AttributeError(f"Settings has no attribute '{name}'")

    def __post_init__(self) -> None:
        """构造后校验：确保各配置值在合法范围内。"""
        r = self.retrieval
        _validate_positive("chunk_size", r.chunk_size)
        if r.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0.")
        if r.chunk_overlap >= r.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if self.llm.temperature < 0:
            raise ValueError("temperature must be greater than or equal to 0.")
        if r.retrieval_mode.strip().lower() not in {"hybrid", "dense", "vector", "bm25", "sparse"}:
            raise ValueError("retrieval_mode must be one of: hybrid, dense, vector, bm25, sparse.")
        _validate_positive("top_k", r.top_k)
        _validate_positive("retrieval_fetch_k", r.retrieval_fetch_k)
        if not 0 <= r.retrieval_mmr_lambda <= 1:
            raise ValueError("retrieval_mmr_lambda must be between 0 and 1.")
        if r.retrieval_bm25_k1 <= 0:
            raise ValueError("retrieval_bm25_k1 must be greater than 0.")
        if not 0 <= r.retrieval_bm25_b <= 1:
            raise ValueError("retrieval_bm25_b must be between 0 and 1.")
        if r.retrieval_bm25_average_length <= 0:
            raise ValueError("retrieval_bm25_average_length must be greater than 0.")
        if r.retrieval_min_relevance < 0:
            raise ValueError("retrieval_min_relevance must be greater than or equal to 0.")
        skill = self.skill
        for name in (
            "skill_max_catalog_size",
            "skill_max_file_bytes",
            "skill_max_reference_files",
            "skill_max_reference_bytes",
            "skill_max_loaded_tokens",
            "skill_max_selected",
            "skill_max_retrieval_queries",
        ):
            _validate_positive(name, getattr(skill, name))
        m = self.memory
        if m.memory_prompt_max_tokens <= 0:
            raise ValueError("memory_prompt_max_tokens must be greater than 0.")

    def with_overrides(self, **kwargs: Any) -> Settings:
        """返回应用临时覆盖项的新 Settings 副本，支持子配置字段名（如 top_k=3）。"""
        sub_mapping: dict[str, str] = {}
        sub_fields: dict[str, dict[str, Any]] = {}
        for sub_name in (
            "storage", "llm", "embedding", "retrieval", "skill",
            "agent", "memory", "web_search", "security",
        ):
            sub_obj = getattr(self, sub_name)
            for f_name in sub_obj.__dataclass_fields__:
                sub_mapping[f_name] = sub_name

        top_level_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in self.__dataclass_fields__:
                top_level_kwargs[key] = value
            elif key in sub_mapping:
                group = sub_mapping[key]
                sub_fields.setdefault(group, {})[key] = value
            else:
                raise TypeError(f"Settings has no field '{key}'")

        for group, overrides in sub_fields.items():
            top_level_kwargs[group] = replace(getattr(self, group), **overrides)

        return replace(self, **top_level_kwargs)

    def ensure_directories(self) -> None:
        """创建 data/ 下所有必需子目录（应用启动时调用）。"""
        s = self.storage
        s.upload_dir.mkdir(parents=True, exist_ok=True)
        s.vector_store_dir.mkdir(parents=True, exist_ok=True)
        s.memory_vector_store_dir.mkdir(parents=True, exist_ok=True)
        s.ingest_jobs_db_path.parent.mkdir(parents=True, exist_ok=True)
        s.agent_tasks_db_path.parent.mkdir(parents=True, exist_ok=True)
        s.matter_db_path.parent.mkdir(parents=True, exist_ok=True)
        s.memory_db_path.parent.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """安全 repr，只展示非敏感字段。"""
        return (
            f"Settings(default_tenant_id={self.default_tenant_id!r}, "
            f"llm=LLMSettings(chat_provider={self.llm.chat_provider!r}, "
            f"chat_model_name={self.llm.chat_model_name!r}, ...))"
        )


def _validate_positive(name: str, value: int) -> None:
    """校验值为正数，否则抛出 ValueError。"""
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")


# ── 全局单例 ──────────────────────────────────────────────────────────
settings = Settings()

"""LangChain chat and embedding model factories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from doc_assistant.config.settings import settings

DEEPSEEK_COMPATIBLE_BASE_URL = "https://api.deepseek.com"


@dataclass(frozen=True)
class CompatibleProviderDefaults:
    label: str
    base_url: str
    api_key_setting: str
    api_key_env_var: str


COMPATIBLE_PROVIDER_DEFAULTS: dict[str, CompatibleProviderDefaults] = {
    "deepseek": CompatibleProviderDefaults(
        label="DeepSeek",
        base_url=DEEPSEEK_COMPATIBLE_BASE_URL,
        api_key_setting="deepseek_api_key",
        api_key_env_var="DEEPSEEK_API_KEY",
    ),
    "openai-compatible": CompatibleProviderDefaults(
        label="OpenAI-compatible",
        base_url="",
        api_key_setting="chat_api_key",
        api_key_env_var="DOC_ASSISTANT_CHAT_API_KEY",
    ),
}

PROVIDER_ALIASES = {"compatible": "openai-compatible", "openai": "openai-compatible"}


def _normalise_provider(provider: str | None) -> str:
    normalized = (provider or "deepseek").strip().lower().replace("_", "-")
    return PROVIDER_ALIASES.get(normalized, normalized)


def _normalise_chat_api(chat_api: str | None) -> str:
    return (chat_api or "compatible").strip().lower().replace("_", "-")


def _resolve_provider_defaults(provider: str) -> CompatibleProviderDefaults:
    return COMPATIBLE_PROVIDER_DEFAULTS.get(
        provider,
        CompatibleProviderDefaults(
            label=provider,
            base_url="",
            api_key_setting="chat_api_key",
            api_key_env_var="DOC_ASSISTANT_CHAT_API_KEY",
        ),
    )


def _resolve_chat_api_key(defaults: CompatibleProviderDefaults) -> str:
    return settings.chat_api_key or getattr(settings, defaults.api_key_setting, "")


def _resolve_chat_base_url(provider: str, defaults: CompatibleProviderDefaults) -> str:
    base_url = settings.chat_base_url or defaults.base_url
    if base_url:
        return base_url
    raise ValueError(f"DOC_ASSISTANT_CHAT_BASE_URL is required for chat provider '{provider}'.")


def _chat_model_cache_key() -> tuple[object, ...]:
    return (
        _normalise_provider(settings.chat_provider),
        _normalise_chat_api(settings.chat_api),
        settings.chat_model_name,
        settings.chat_api_key,
        settings.deepseek_api_key,
        settings.chat_base_url,
        json.dumps(settings.chat_extra_body or {}, sort_keys=True),
        settings.temperature,
        settings.llm_max_retries,
    )


@lru_cache(maxsize=16)
def _build_chat_model_cached(_cache_key: tuple[object, ...]) -> BaseChatModel:
    provider = _normalise_provider(settings.chat_provider)
    chat_api = _normalise_chat_api(settings.chat_api)
    if chat_api not in {"compatible", "openai-compatible", "chat-completions"}:
        raise ValueError(
            "Unsupported chat configuration: "
            f"DOC_ASSISTANT_CHAT_PROVIDER={provider}, DOC_ASSISTANT_CHAT_API={chat_api}."
        )

    defaults = _resolve_provider_defaults(provider)
    common: dict[str, Any] = {
        "model": settings.chat_model_name,
        "api_key": _resolve_chat_api_key(defaults),
        "base_url": _resolve_chat_base_url(provider, defaults),
        "temperature": settings.temperature,
        "max_retries": max(0, settings.llm_max_retries),
        "timeout": 120,
    }
    if settings.chat_extra_body:
        common["extra_body"] = settings.chat_extra_body
    return ChatDeepSeek(**common) if provider == "deepseek" else ChatOpenAI(**common)


def build_chat_model() -> BaseChatModel:
    return _build_chat_model_cached(_chat_model_cache_key())


def _embedding_model_cache_key() -> tuple[object, ...]:
    return (
        _normalise_provider(settings.embedding_provider),
        settings.embedding_model_name,
        settings.embedding_api_key,
        settings.dashscope_api_key,
        settings.embedding_base_url,
        settings.chat_base_url,
        settings.embedding_device,
    )


@lru_cache(maxsize=16)
def _build_embedding_model_cached(_cache_key: tuple[object, ...]):
    provider = _normalise_provider(settings.embedding_provider)
    if provider == "dashscope":
        return DashScopeEmbeddings(
            model=settings.embedding_model_name,
            dashscope_api_key=settings.embedding_api_key or settings.dashscope_api_key,
        )
    if provider == "openai-compatible":
        return OpenAIEmbeddings(
            model=settings.embedding_model_name,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url or settings.chat_base_url,
        )
    if provider == "local":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFaceEmbeddings is required for DOC_ASSISTANT_EMBEDDING_PROVIDER=local."
            ) from exc
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": settings.embedding_device},
        )
    raise ValueError(
        f"Unsupported embedding provider '{provider}'. Configure "
        "DOC_ASSISTANT_EMBEDDING_PROVIDER as openai-compatible, dashscope, or local."
    )


def build_embedding_model():
    return _build_embedding_model_cached(_embedding_model_cache_key())

"""LLM 与 Embedding 模型工厂。

``build_chat_model`` / ``build_embedding_model`` 根据 settings 选择：
- DeepSeek 或其他 OpenAI 兼容 HTTP 客户端（带重试与熔断）
- 或本地 HuggingFace Embedding 模型

业务层通常只调用 factory，不直接实例化 ``OpenAICompatibleChatModel``。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from time import monotonic, sleep
from typing import Any, Protocol, runtime_checkable

import httpx
import requests
from langchain_community.embeddings import DashScopeEmbeddings

from doc_assistant.config.settings import settings

DEEPSEEK_COMPATIBLE_BASE_URL = "https://api.deepseek.com"
_SSE_DONE = object()
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_ORIGINAL_REQUESTS_POST = requests.post


@runtime_checkable
class MessageChatModelProtocol(Protocol):
    """Model capability for OpenAI-style messages and optional tool calling."""

    def invoke_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
    ) -> dict[str, Any]: ...


@runtime_checkable
class InvokableChatModelProtocol(Protocol):
    """Model capability for a generic synchronous invocation."""

    def invoke(
        self,
        prompt: str | list[dict[str, Any]] | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
    ) -> str: ...


@runtime_checkable
class StreamingChatModelProtocol(Protocol):
    """Model capability for synchronous streaming."""

    def stream(self, messages: list[dict[str, Any]]) -> Iterator[str]: ...


@runtime_checkable
class MessageStreamingChatModelProtocol(
    MessageChatModelProtocol,
    StreamingChatModelProtocol,
    Protocol,
):
    """Model capability for streaming the project's message format."""


@runtime_checkable
class ChatModelProtocol(
    MessageStreamingChatModelProtocol,
    InvokableChatModelProtocol,
    Protocol,
):
    """Combined protocol retained for fully capable synchronous clients.

    Services should depend on the smallest capability protocol they use.
    ``OpenAICompatibleChatModel`` satisfies this combined protocol.
    """


@runtime_checkable
class AsyncMessageChatModelProtocol(Protocol):
    """Model capability for asynchronous OpenAI-style message invocation."""

    async def ainvoke_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
    ) -> dict[str, Any]: ...


@runtime_checkable
class AsyncInvokableChatModelProtocol(Protocol):
    """Model capability for a generic asynchronous invocation."""

    async def ainvoke(
        self,
        prompt: str | list[dict[str, Any]] | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
    ) -> str: ...


@runtime_checkable
class AsyncChatModelProtocol(
    AsyncMessageChatModelProtocol,
    AsyncInvokableChatModelProtocol,
    Protocol,
):
    """Combined protocol retained for fully capable asynchronous clients."""


@dataclass(frozen=True)
class CompatibleProviderDefaults:
    """某个 LLM 提供商的默认 base_url 与 API Key 环境变量名。"""

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

PROVIDER_ALIASES = {
    "compatible": "openai-compatible",
    "openai": "openai-compatible",
}


class CircuitBreaker:
    """线程安全的熔断器：连续失败超过阈值后拒绝新请求，冷却后自动恢复。"""

    def __init__(
        self,
        threshold: int | None = None,
        cooldown_seconds: int | None = None,
    ) -> None:
        if threshold is None:
            threshold = settings.llm_circuit_breaker_threshold
        if cooldown_seconds is None:
            cooldown_seconds = settings.llm_circuit_breaker_cooldown_seconds
        self._threshold = threshold
        self._cooldown_seconds = cooldown_seconds
        self._lock = __import__("threading").Lock()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def ensure_allows_request(self) -> None:
        with self._lock:
            if self._circuit_open_until > monotonic():
                raise RuntimeError("Circuit breaker is open after repeated failures.")

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._threshold:
                self._circuit_open_until = monotonic() + self._cooldown_seconds


@dataclass(frozen=True)
class OpenAICompatibleChatModel:
    """直连 OpenAI Chat Completions API 的 HTTP 客户端（项目默认 LLM 实现）。

    能力：同步/异步 invoke、SSE 流式、tool calling（invoke_messages）、
    失败重试与熔断（settings.llm_circuit_breaker_*）。
    DeepSeek 等 OpenAI-compatible 提供商均走本客户端。
    """

    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float
    api_key_env_var: str = "DOC_ASSISTANT_CHAT_API_KEY"
    extra_body: dict[str, Any] | None = None
    timeout: int = 120
    enable_thinking: bool | None = None
    _session: requests.Session = field(init=False, repr=False, compare=False)
    _async_client: httpx.AsyncClient | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _circuit_breaker: CircuitBreaker = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_session", requests.Session())
        object.__setattr__(self, "_circuit_breaker", CircuitBreaker())

    def invoke(self, prompt: str | list[dict[str, Any]] | None = None, *, messages: list[dict[str, Any]] | None = None) -> str:
        if messages is not None:
            return str(self.invoke_messages(messages).get("content") or "")
        if isinstance(prompt, list):
            return str(self.invoke_messages(prompt).get("content") or "")
        return "".join(self.stream(prompt or ""))

    def invoke_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError(f"{self.api_key_env_var} is not set for {self.provider}.")

        payload = self._chat_payload(
            messages,
            stream=False,
            tools=tools,
            tool_choice=tool_choice,
        )
        response = self._post_json(payload)
        if response.status_code >= 400:
            raise RuntimeError(self._format_error(response))

        data = response.json()
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected {self.provider} chat response: {data}") from exc
        if not isinstance(message, dict):
            raise RuntimeError(f"Unexpected {self.provider} chat response: {data}")
        return message

    async def ainvoke_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError(f"{self.api_key_env_var} is not set for {self.provider}.")

        payload = self._chat_payload(messages, stream=False, tools=tools, tool_choice=tool_choice)
        response = await self._apost_json(payload)
        if response.status_code >= 400:
            raise RuntimeError(self._format_error(response))

        data = response.json()
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected {self.provider} chat response: {data}") from exc
        if not isinstance(message, dict):
            raise RuntimeError(f"Unexpected {self.provider} chat response: {data}")
        return message

    def stream(self, prompt: str | list[dict[str, Any]] | None = None, *, messages: list[dict[str, Any]] | None = None) -> Iterator[str]:
        if not self.api_key:
            raise ValueError(f"{self.api_key_env_var} is not set for {self.provider}.")

        resolved_messages = messages or (
            prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt or ""}]
        )
        payload = self._chat_payload(resolved_messages, stream=True)

        with self._post_stream(payload) as response:
            if response.status_code >= 400:
                raise RuntimeError(self._format_error(response))

            for line in response.iter_lines(decode_unicode=True):
                parsed = self._parse_sse_line(line, self.provider)
                if parsed is _SSE_DONE:
                    break
                if parsed:
                    yield str(parsed)

    async def astream(
        self,
        prompt: str | list[dict[str, Any]] | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
    ):
        if not self.api_key:
            raise ValueError(f"{self.api_key_env_var} is not set for {self.provider}.")

        resolved_messages = messages or (
            prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt or ""}]
        )
        payload = self._chat_payload(resolved_messages, stream=True)
        async with self._astream_response(payload) as response:
            if response.status_code >= 400:
                body = await response.aread()
                response._content = body
                raise RuntimeError(self._format_error(response))

            async for line in response.aiter_lines():
                parsed = self._parse_sse_line(line, self.provider)
                if parsed is _SSE_DONE:
                    break
                if parsed:
                    yield str(parsed)

    def _chat_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if self.extra_body:
            payload.update(self.extra_body)
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _sync_post(self, *args, **kwargs) -> requests.Response:
        if requests.post is not _ORIGINAL_REQUESTS_POST:
            return requests.post(*args, **kwargs)
        return self._session.post(*args, **kwargs)

    def _post_json(self, payload: dict[str, object]) -> requests.Response:
        self._ensure_circuit_allows_request()
        attempts = max(1, settings.llm_max_retries)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._sync_post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    self._record_request_success()
                    return response
                last_error = RuntimeError(self._format_error(response))
            except requests.RequestException as exc:
                last_error = exc
            if attempt < attempts - 1:
                sleep(min(2**attempt, 8))
        self._record_request_failure()
        if last_error:
            raise RuntimeError(str(last_error)) from last_error
        raise RuntimeError(f"{self.provider} request failed without a response.")

    def _post_stream(self, payload: dict[str, object]) -> requests.Response:
        self._ensure_circuit_allows_request()
        attempts = max(1, settings.llm_max_retries)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._sync_post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                    stream=True,
                )
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    self._record_request_success()
                    return response
                last_error = RuntimeError(self._format_error(response))
                response.close()
            except requests.RequestException as exc:
                last_error = exc
            if attempt < attempts - 1:
                sleep(min(2**attempt, 8))
        self._record_request_failure()
        if last_error:
            raise RuntimeError(str(last_error)) from last_error
        raise RuntimeError(f"{self.provider} stream request failed without a response.")

    async def _apost_json(self, payload: dict[str, object]) -> httpx.Response:
        self._ensure_circuit_allows_request()
        attempts = max(1, settings.llm_max_retries)
        last_error: Exception | None = None
        client = await self._get_async_client()
        for attempt in range(attempts):
            try:
                response = await client.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    self._record_request_success()
                    return response
                last_error = RuntimeError(self._format_error(response))
            except httpx.HTTPError as exc:
                last_error = exc
            if attempt < attempts - 1:
                await self._async_sleep(min(2**attempt, 8))
        self._record_request_failure()
        if last_error:
            raise RuntimeError(str(last_error)) from last_error
        raise RuntimeError(f"{self.provider} async request failed without a response.")

    @asynccontextmanager
    async def _astream_response(self, payload: dict[str, object]):
        self._ensure_circuit_allows_request()
        attempts = max(1, settings.llm_max_retries)
        last_error: Exception | None = None
        client = await self._get_async_client()
        for attempt in range(attempts):
            stream = None
            try:
                stream = client.stream(
                    "POST",
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                response = await stream.__aenter__()
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    self._record_request_success()
                    try:
                        yield response
                    finally:
                        await stream.__aexit__(None, None, None)
                    return
                body = await response.aread()
                response._content = body
                last_error = RuntimeError(self._format_error(response))
                await stream.__aexit__(None, None, None)
            except httpx.HTTPError as exc:
                last_error = exc
                if stream is not None:
                    await stream.__aexit__(type(exc), exc, exc.__traceback__)
            if attempt < attempts - 1:
                await self._async_sleep(min(2**attempt, 8))
        self._record_request_failure()
        if last_error:
            raise RuntimeError(str(last_error)) from last_error
        raise RuntimeError(f"{self.provider} async stream request failed without a response.")

    async def _get_async_client(self) -> httpx.AsyncClient:
        client = self._async_client
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=self.timeout)
            object.__setattr__(self, "_async_client", client)
        return client

    @staticmethod
    async def _async_sleep(seconds: int) -> None:
        import asyncio

        await asyncio.sleep(seconds)

    def _ensure_circuit_allows_request(self) -> None:
        try:
            self._circuit_breaker.ensure_allows_request()
        except RuntimeError:
            raise RuntimeError(f"{self.provider} circuit breaker is open after repeated failures.")

    def _record_request_success(self) -> None:
        self._circuit_breaker.record_success()

    def _record_request_failure(self) -> None:
        self._circuit_breaker.record_failure()

    def close(self) -> None:
        self._session.close()

    async def aclose(self) -> None:
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()

    @staticmethod
    def _extract_content(chunk: dict[str, Any]) -> str | None:
        choice = chunk["choices"][0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content is None:
            message = choice.get("message") or {}
            content = message.get("content")
        return content

    @staticmethod
    def _parse_sse_line(line: str, provider: str) -> str | None | object:
        if not line or line.startswith(":") or not line.startswith("data:"):
            return None
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            return _SSE_DONE
        try:
            chunk = json.loads(data)
            return OpenAICompatibleChatModel._extract_content(chunk)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected {provider} stream response: {data}") from exc

    def _format_error(self, response: requests.Response | httpx.Response) -> str:
        provider_label = self.provider
        try:
            data = response.json()
        except ValueError:
            return f"{provider_label} request failed: {response.status_code} {response.text}"

        code = data.get("code") or data.get("error", {}).get("code")
        message = data.get("message") or data.get("error", {}).get("message")
        details = f"{response.status_code}"
        if code:
            details += f" {code}"
        if message:
            details += f": {message}"
        return f"{provider_label} request failed: {details}"


AsyncOpenAICompatibleChatModel = OpenAICompatibleChatModel


def _normalise_provider(provider: str | None) -> str:
    normalised = (provider or "deepseek").strip().lower().replace("_", "-")
    return PROVIDER_ALIASES.get(normalised, normalised)


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


def _resolve_chat_api_key(provider: str, defaults: CompatibleProviderDefaults) -> tuple[str, str]:
    explicit_key = settings.chat_api_key
    if explicit_key:
        return explicit_key, "DOC_ASSISTANT_CHAT_API_KEY"

    provider_key = getattr(settings, defaults.api_key_setting, "")
    if provider_key:
        return provider_key, defaults.api_key_env_var

    return "", defaults.api_key_env_var


def _resolve_chat_base_url(provider: str, defaults: CompatibleProviderDefaults) -> str:
    base_url = settings.chat_base_url
    if base_url:
        return base_url
    if defaults.base_url:
        return defaults.base_url
    raise ValueError(
        f"DOC_ASSISTANT_CHAT_BASE_URL is required for chat provider '{provider}'."
    )


def _provider_extra_body(provider: str, model: str) -> dict[str, Any]:
    extra_body: dict[str, Any] = {}
    extra_body.update(settings.chat_extra_body or {})
    return extra_body


def _build_openai_compatible_chat_model(provider: str) -> OpenAICompatibleChatModel:
    defaults = _resolve_provider_defaults(provider)
    api_key, api_key_env_var = _resolve_chat_api_key(provider, defaults)
    model = settings.chat_model_name
    extra_body = _provider_extra_body(provider, model)

    return AsyncOpenAICompatibleChatModel(
        provider=defaults.label,
        model=model,
        api_key=api_key,
        api_key_env_var=api_key_env_var,
        base_url=_resolve_chat_base_url(provider, defaults),
        temperature=settings.temperature,
        extra_body=extra_body,
    )


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
        settings.enable_thinking,
    )


@lru_cache(maxsize=16)
def _build_chat_model_cached(_cache_key: tuple[object, ...]):
    provider = _normalise_provider(settings.chat_provider)
    chat_api = _normalise_chat_api(settings.chat_api)
    if chat_api in {"compatible", "openai-compatible", "chat-completions"}:
        return _build_openai_compatible_chat_model(provider)

    raise ValueError(
        "Unsupported chat configuration: "
        f"DOC_ASSISTANT_CHAT_PROVIDER={provider}, DOC_ASSISTANT_CHAT_API={chat_api}."
    )


def build_chat_model():
    """按 settings 构建并缓存聊天模型（LRU cache，配置变更需重启进程）。"""
    return _build_chat_model_cached(_chat_model_cache_key())


def build_embedding_model():
    """按 settings 构建并缓存 embedding 模型。"""
    return _build_embedding_model_cached(_embedding_model_cache_key())


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
        try:
            from langchain_community.embeddings import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "langchain-community OpenAIEmbeddings is required for "
                "DOC_ASSISTANT_EMBEDDING_PROVIDER=openai-compatible."
            ) from exc

        return OpenAIEmbeddings(
            model=settings.embedding_model_name,
            openai_api_key=settings.embedding_api_key,
            openai_api_base=settings.embedding_base_url
            or settings.chat_base_url,
        )

    if provider == "local":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFaceEmbeddings is required for "
                "DOC_ASSISTANT_EMBEDDING_PROVIDER=local."
            ) from exc

        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": settings.embedding_device},
        )

    raise ValueError(
        "Unsupported embedding provider "
        f"'{provider}'. Configure DOC_ASSISTANT_EMBEDDING_PROVIDER as "
        "openai-compatible, dashscope, or local."
    )


from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Any
from time import monotonic, sleep

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
import httpx
import requests

from doc_assistant.config.settings import settings

DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_COMPATIBLE_BASE_URL = "https://api.deepseek.com"
_SSE_DONE = object()
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_ORIGINAL_REQUESTS_POST = requests.post


@dataclass(frozen=True)
class CompatibleProviderDefaults:
    label: str
    base_url: str
    api_key_setting: str
    api_key_env_var: str


COMPATIBLE_PROVIDER_DEFAULTS: dict[str, CompatibleProviderDefaults] = {
    "dashscope": CompatibleProviderDefaults(
        label="DashScope",
        base_url=DASHSCOPE_COMPATIBLE_BASE_URL,
        api_key_setting="dashscope_api_key",
        api_key_env_var="DASHSCOPE_API_KEY",
    ),
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
    "qwen": "dashscope",
    "tongyi": "dashscope",
}


@dataclass(frozen=True)
class OpenAICompatibleChatModel:
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
    _consecutive_failures: int = field(default=0, init=False, repr=False, compare=False)
    _circuit_open_until: float = field(default=0.0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_session", requests.Session())

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
        if self._circuit_open_until > monotonic():
            raise RuntimeError(f"{self.provider} circuit breaker is open after repeated failures.")

    def _record_request_success(self) -> None:
        object.__setattr__(self, "_consecutive_failures", 0)
        object.__setattr__(self, "_circuit_open_until", 0.0)

    def _record_request_failure(self) -> None:
        failures = self._consecutive_failures + 1
        object.__setattr__(self, "_consecutive_failures", failures)
        if failures >= settings.llm_circuit_breaker_threshold:
            object.__setattr__(
                self,
                "_circuit_open_until",
                monotonic() + settings.llm_circuit_breaker_cooldown_seconds,
            )

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


class DashScopeCompatibleChatModel(OpenAICompatibleChatModel):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float,
        enable_thinking: bool,
        api_key_env_var: str = "DASHSCOPE_API_KEY",
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        body = dict(extra_body or {})
        if model.startswith("qwen3.5"):
            body.setdefault("enable_thinking", enable_thinking)

        super().__init__(
            provider="DashScope",
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            api_key_env_var=api_key_env_var,
            extra_body=body,
            enable_thinking=enable_thinking,
        )


def _normalise_provider(provider: str | None) -> str:
    normalised = (provider or "dashscope").strip().lower().replace("_", "-")
    return PROVIDER_ALIASES.get(normalised, normalised)


def _normalise_chat_api(chat_api: str | None) -> str:
    return (chat_api or "compatible").strip().lower().replace("_", "-")


def _setting(name: str, default: Any) -> Any:
    return getattr(settings, name, default)


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
    explicit_key = _setting("chat_api_key", "")
    if explicit_key:
        return explicit_key, "DOC_ASSISTANT_CHAT_API_KEY"

    provider_key = _setting(defaults.api_key_setting, "")
    if provider_key:
        return provider_key, defaults.api_key_env_var

    return "", defaults.api_key_env_var


def _resolve_chat_base_url(provider: str, defaults: CompatibleProviderDefaults) -> str:
    base_url = _setting("chat_base_url", "")
    if base_url:
        return base_url
    if defaults.base_url:
        return defaults.base_url
    raise ValueError(
        f"DOC_ASSISTANT_CHAT_BASE_URL is required for chat provider '{provider}'."
    )


def _provider_extra_body(provider: str, model: str) -> dict[str, Any]:
    extra_body: dict[str, Any] = {}
    if provider == "dashscope" and model.startswith("qwen3.5"):
        extra_body["enable_thinking"] = bool(_setting("enable_thinking", False))
    extra_body.update(_setting("chat_extra_body", {}) or {})
    return extra_body


def _build_openai_compatible_chat_model(provider: str) -> OpenAICompatibleChatModel:
    defaults = _resolve_provider_defaults(provider)
    api_key, api_key_env_var = _resolve_chat_api_key(provider, defaults)
    model = _setting("chat_model_name", "qwen3.5-flash")
    extra_body = _provider_extra_body(provider, model)
    if provider == "dashscope":
        return DashScopeCompatibleChatModel(
            model=model,
            api_key=api_key,
            base_url=_resolve_chat_base_url(provider, defaults),
            temperature=_setting("temperature", 0.0),
            enable_thinking=bool(extra_body.get("enable_thinking", False)),
            api_key_env_var=api_key_env_var,
            extra_body=extra_body,
        )

    return AsyncOpenAICompatibleChatModel(
        provider=defaults.label,
        model=model,
        api_key=api_key,
        api_key_env_var=api_key_env_var,
        base_url=_resolve_chat_base_url(provider, defaults),
        temperature=_setting("temperature", 0.0),
        extra_body=extra_body,
    )


def _build_tongyi_chat_model():
    model = _setting("chat_model_name", "qwen3.5-flash")
    if model.startswith("qwen3.5"):
        raise ValueError(
            "Qwen3.5 models must use DOC_ASSISTANT_CHAT_API=compatible because "
            "DashScope's text-generation endpoint returns url error for this family."
        )

    return ChatTongyi(
        model=model,
        temperature=_setting("temperature", 0.0),
        api_key=_setting("dashscope_api_key", "") or _setting("chat_api_key", ""),
    )


def _chat_model_cache_key() -> tuple[object, ...]:
    return (
        _normalise_provider(_setting("chat_provider", "dashscope")),
        _normalise_chat_api(_setting("chat_api", "compatible")),
        _setting("chat_model_name", "qwen3.5-flash"),
        _setting("chat_api_key", ""),
        _setting("dashscope_api_key", ""),
        _setting("deepseek_api_key", ""),
        _setting("chat_base_url", ""),
        json.dumps(_setting("chat_extra_body", {}) or {}, sort_keys=True),
        _setting("temperature", 0.0),
        _setting("enable_thinking", False),
    )


@lru_cache(maxsize=16)
def _build_chat_model_cached(_cache_key: tuple[object, ...]):
    provider = _normalise_provider(_setting("chat_provider", "dashscope"))
    chat_api = _normalise_chat_api(_setting("chat_api", "compatible"))
    if chat_api in {"compatible", "openai-compatible", "chat-completions"}:
        return _build_openai_compatible_chat_model(provider)
    if provider == "dashscope" and chat_api in {"tongyi", "native", "text-generation"}:
        return _build_tongyi_chat_model()

    raise ValueError(
        "Unsupported chat configuration: "
        f"DOC_ASSISTANT_CHAT_PROVIDER={provider}, DOC_ASSISTANT_CHAT_API={chat_api}."
    )


def build_chat_model():
    return _build_chat_model_cached(_chat_model_cache_key())


def _embedding_model_cache_key() -> tuple[object, ...]:
    return (
        _normalise_provider(_setting("embedding_provider", "dashscope")),
        _setting("embedding_model_name", "text-embedding-v3"),
        _setting("embedding_api_key", ""),
        _setting("dashscope_api_key", ""),
        _setting("embedding_base_url", ""),
        _setting("chat_base_url", ""),
        _setting("embedding_device", "cpu"),
    )


@lru_cache(maxsize=16)
def _build_embedding_model_cached(_cache_key: tuple[object, ...]):
    provider = _normalise_provider(_setting("embedding_provider", "dashscope"))
    if provider == "dashscope":
        return DashScopeEmbeddings(
            model=_setting("embedding_model_name", "text-embedding-v3"),
            dashscope_api_key=_setting("embedding_api_key", "") or _setting("dashscope_api_key", ""),
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
            model=_setting("embedding_model_name", "text-embedding-3-small"),
            openai_api_key=_setting("embedding_api_key", ""),
            openai_api_base=_setting("embedding_base_url", "")
            or _setting("chat_base_url", ""),
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
            model_name=_setting("embedding_model_name", "BAAI/bge-m3"),
            model_kwargs={"device": _setting("embedding_device", "cpu")},
        )

    raise ValueError(
        "Unsupported embedding provider "
        f"'{provider}'. Configure DOC_ASSISTANT_EMBEDDING_PROVIDER as "
        "dashscope, openai-compatible, or local."
    )


def build_embedding_model():
    return _build_embedding_model_cached(_embedding_model_cache_key())

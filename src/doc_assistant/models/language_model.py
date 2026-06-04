from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Iterator
from typing import Any

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
import requests

from doc_assistant.config.settings import settings

DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_COMPATIBLE_BASE_URL = "https://api.deepseek.com"


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

    def invoke(self, prompt: str) -> str:
        return "".join(self.stream(prompt))

    def invoke_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError(f"{self.api_key_env_var} is not set for {self.provider}.")

        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if self.extra_body:
            payload.update(self.extra_body)
        payload["stream"] = False

        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
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

    def stream(self, prompt: str) -> Iterator[str]:
        if not self.api_key:
            raise ValueError(f"{self.api_key_env_var} is not set for {self.provider}.")

        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": True,
        }
        if self.extra_body:
            payload.update(self.extra_body)

        with requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
            stream=True,
        ) as response:
            if response.status_code >= 400:
                raise RuntimeError(self._format_error(response))

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue

                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    content = self._extract_content(chunk)
                except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError(f"Unexpected {self.provider} stream response: {data}") from exc

                if content:
                    yield str(content)

    @staticmethod
    def _extract_content(chunk: dict[str, Any]) -> str | None:
        choice = chunk["choices"][0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content is None:
            message = choice.get("message") or {}
            content = message.get("content")
        return content

    def _format_error(self, response: requests.Response) -> str:
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
        )
        object.__setattr__(self, "enable_thinking", enable_thinking)


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

    return OpenAICompatibleChatModel(
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


def build_chat_model():
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


def build_embedding_model() -> DashScopeEmbeddings:
    provider = _normalise_provider(_setting("embedding_provider", "dashscope"))
    if provider != "dashscope":
        raise ValueError(
            "Unsupported embedding provider "
            f"'{provider}'. Configure DOC_ASSISTANT_EMBEDDING_PROVIDER=dashscope "
            "or add a provider implementation in doc_assistant.models.language_model."
        )
    return DashScopeEmbeddings(
        model=_setting("embedding_model_name", "text-embedding-v3"),
        dashscope_api_key=_setting("embedding_api_key", "") or _setting("dashscope_api_key", ""),
    )

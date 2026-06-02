from __future__ import annotations

from dataclasses import dataclass

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
import requests

from doc_assistant.config.settings import settings


@dataclass(frozen=True)
class DashScopeCompatibleChatModel:
    model: str
    api_key: str
    base_url: str
    temperature: float
    enable_thinking: bool

    def invoke(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set.")

        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.model.startswith("qwen3.5"):
            payload["enable_thinking"] = self.enable_thinking

        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )

        if response.status_code >= 400:
            raise RuntimeError(self._format_error(response))

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected DashScope response: {data}") from exc

    @staticmethod
    def _format_error(response: requests.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return f"DashScope request failed: {response.status_code} {response.text}"

        code = data.get("code") or data.get("error", {}).get("code")
        message = data.get("message") or data.get("error", {}).get("message")
        details = f"{response.status_code}"
        if code:
            details += f" {code}"
        if message:
            details += f": {message}"
        return f"DashScope request failed: {details}"


def build_chat_model():
    if settings.chat_api == "compatible":
        return DashScopeCompatibleChatModel(
            model=settings.chat_model_name,
            api_key=settings.dashscope_api_key,
            base_url=settings.chat_base_url,
            temperature=settings.temperature,
            enable_thinking=settings.enable_thinking,
        )

    if settings.chat_model_name.startswith("qwen3.5"):
        raise ValueError(
            "Qwen3.5 models must use DOC_ASSISTANT_CHAT_API=compatible because "
            "DashScope's text-generation endpoint returns url error for this family."
        )

    return ChatTongyi(
        model=settings.chat_model_name,
        temperature=settings.temperature,
    )


def build_embedding_model() -> DashScopeEmbeddings:
    return DashScopeEmbeddings(model=settings.embedding_model_name)

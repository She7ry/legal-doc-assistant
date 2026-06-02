from __future__ import annotations

from types import SimpleNamespace

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from doc_assistant.models import language_model
from doc_assistant.models.language_model import DashScopeCompatibleChatModel
from doc_assistant.services.qa_service import DocumentQAService


class EmptyVectorStore:
    def search(self, query: str, k: int | None = None) -> list:
        return []


def test_ask_uses_general_chat_when_no_documents_are_found() -> None:
    service = DocumentQAService(
        vector_store=EmptyVectorStore(),
        chat_model=FakeListChatModel(responses=["Hello! Nice to see you."]),
    )

    answer = service.ask("hello", chat_history=[{"role": "assistant", "content": "welcome back"}])

    assert answer.content == "Hello! Nice to see you."
    assert answer.citations == []


def test_format_chat_history_keeps_recent_user_and_assistant_messages() -> None:
    history = DocumentQAService._format_chat_history(
        [
            {"role": "system", "content": "hidden"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi, how can I help?"},
            {"role": "assistant", "content": ""},
        ]
    )

    assert history == "User: hello\nAssistant: Hi, how can I help?"


def test_qwen35_model_uses_compatible_chat_api(monkeypatch) -> None:
    monkeypatch.setattr(
        language_model,
        "settings",
        SimpleNamespace(
            chat_model_name="qwen3.5-flash",
            chat_api="compatible",
            dashscope_api_key="test-key",
            chat_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
            enable_thinking=False,
        ),
    )

    model = language_model.build_chat_model()

    assert isinstance(model, DashScopeCompatibleChatModel)
    assert model.model == "qwen3.5-flash"
    assert model.enable_thinking is False

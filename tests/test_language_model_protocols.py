from types import SimpleNamespace

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from ai import llm as language_model


def _settings(provider: str):
    return SimpleNamespace(
        chat_provider=provider,
        chat_model_name="test-model",
        chat_api="compatible",
        chat_api_key="test-key",
        deepseek_api_key="",
        chat_base_url="" if provider == "deepseek" else "https://example.test/v1",
        chat_extra_body={},
        temperature=0,
        llm_max_retries=2,
    )


def test_deepseek_builds_native_langchain_model(monkeypatch) -> None:
    monkeypatch.setattr(language_model, "settings", _settings("deepseek"))
    model = language_model.build_chat_model()
    assert isinstance(model, BaseChatModel)
    assert isinstance(model, ChatDeepSeek)


def test_compatible_endpoint_builds_chat_openai(monkeypatch) -> None:
    monkeypatch.setattr(language_model, "settings", _settings("openai-compatible"))
    model = language_model.build_chat_model()
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base == "https://example.test/v1"

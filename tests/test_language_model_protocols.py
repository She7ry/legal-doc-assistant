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


def test_deepseek_compatibility_omits_tool_choice_and_uses_json_mode(monkeypatch) -> None:
    calls = {}

    def bind_tools(self, tools, **kwargs):
        calls["bind"] = kwargs
        return self

    def with_structured_output(self, schema, **kwargs):
        calls["structured"] = kwargs
        return self

    monkeypatch.setattr(ChatDeepSeek, "bind_tools", bind_tools)
    monkeypatch.setattr(ChatDeepSeek, "with_structured_output", with_structured_output)
    model = ChatDeepSeek(model="test-model", api_key="test-key")

    language_model.bind_chat_tools(model, [], tool_choice="search_documents")
    language_model.structured_chat_output(model, dict)

    assert calls == {"bind": {}, "structured": {"method": "json_mode"}}

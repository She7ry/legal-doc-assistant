from __future__ import annotations

from types import SimpleNamespace

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from api.routers.chat import _stream_answer_events
from doc_assistant.models import language_model
from doc_assistant.models.language_model import OpenAICompatibleChatModel
from doc_assistant.services.qa_service import DocumentQAService


class EmptyVectorStore:
    def search(self, query: str, k: int | None = None) -> list:
        return []


class StreamingChatModel:
    def invoke(self, prompt: str) -> str:
        return "Hello!"

    def stream(self, prompt: str):
        yield "Hel"
        yield "lo!"


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


def test_stream_prepared_answer_yields_chat_chunks() -> None:
    service = DocumentQAService(vector_store=EmptyVectorStore(), chat_model=StreamingChatModel())
    prepared = service.prepare_answer("hello")

    assert list(service.stream_prepared_answer(prepared)) == ["Hel", "lo!"]


def test_stream_answer_events_emit_metadata_delta_and_done() -> None:
    service = DocumentQAService(vector_store=EmptyVectorStore(), chat_model=StreamingChatModel())
    prepared = service.prepare_answer("hello")

    events = "".join(_stream_answer_events(service, prepared))

    assert "event: metadata" in events
    assert 'event: delta\ndata: {"content": "Hel"}' in events
    assert 'event: delta\ndata: {"content": "lo!"}' in events
    assert 'event: done\ndata: {"content": "Hello!"' in events


def test_qwen35_model_uses_compatible_chat_api(monkeypatch) -> None:
    monkeypatch.setattr(
        language_model,
        "settings",
        SimpleNamespace(
            chat_provider="dashscope",
            chat_model_name="qwen3.5-flash",
            chat_api="compatible",
            chat_api_key="",
            dashscope_api_key="test-key",
            chat_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            chat_extra_body={},
            temperature=0,
            enable_thinking=False,
        ),
    )

    model = language_model.build_chat_model()

    assert isinstance(model, OpenAICompatibleChatModel)
    assert model.provider == "DashScope"
    assert model.model == "qwen3.5-flash"
    assert model.extra_body == {"enable_thinking": False}


def test_deepseek_provider_uses_openai_compatible_defaults(monkeypatch) -> None:
    monkeypatch.setattr(
        language_model,
        "settings",
        SimpleNamespace(
            chat_provider="deepseek",
            chat_model_name="deepseek-v4-flash",
            chat_api="compatible",
            chat_api_key="test-key",
            deepseek_api_key="",
            chat_base_url="",
            chat_extra_body={},
            temperature=0,
            enable_thinking=False,
        ),
    )

    model = language_model.build_chat_model()

    assert isinstance(model, OpenAICompatibleChatModel)
    assert model.provider == "DeepSeek"
    assert model.model == "deepseek-v4-flash"
    assert model.api_key == "test-key"
    assert model.base_url == "https://api.deepseek.com"


def test_openai_compatible_model_streams_openai_style_chunks(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def iter_lines(self, decode_unicode: bool):
            return iter(
                [
                    'data: {"choices":[{"delta":{"content":"Hel"}}]}',
                    'data: {"choices":[{"delta":{"content":"lo"}}]}',
                    "data: [DONE]",
                ]
            )

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return FakeResponse()

    monkeypatch.setattr(language_model.requests, "post", fake_post)
    model = OpenAICompatibleChatModel(
        provider="DashScope",
        model="qwen3.5-flash",
        api_key="test-key",
        base_url="https://example.test/v1",
        temperature=0,
        extra_body={"enable_thinking": False},
    )

    assert list(model.stream("prompt")) == ["Hel", "lo"]
    assert calls[0]["json"]["stream"] is True
    assert calls[0]["stream"] is True
    assert calls[0]["json"]["enable_thinking"] is False


def test_openai_compatible_model_invokes_messages_with_tools(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "final"}}]}

    def fake_post(*args, **kwargs):
        calls.append(kwargs)
        return FakeResponse()

    monkeypatch.setattr(language_model.requests, "post", fake_post)
    model = OpenAICompatibleChatModel(
        provider="DeepSeek",
        model="deepseek-v4-flash",
        api_key="test-key",
        base_url="https://example.test",
        temperature=0,
    )
    tools = [{"type": "function", "function": {"name": "search_documents", "parameters": {}}}]

    message = model.invoke_messages(
        [{"role": "user", "content": "question"}],
        tools=tools,
    )

    assert message == {"content": "final"}
    assert calls[0]["json"]["tools"] == tools
    assert calls[0]["json"]["tool_choice"] == "auto"
    assert calls[0]["json"]["stream"] is False

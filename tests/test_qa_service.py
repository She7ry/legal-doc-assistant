from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from api.routers.chat import _stream_answer_events
from doc_assistant.models import language_model
from doc_assistant.models.language_model import OpenAICompatibleChatModel
from doc_assistant.schemas.citation import Citation
from doc_assistant.services.qa_service import DocumentQAService


class EmptyVectorStore:
    def search(self, query: str, k: int | None = None) -> list:
        return []


class StaticVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.queries: list[str] = []

    def search(self, query: str, k: int | None = None) -> list[Document]:
        self.queries.append(query)
        return self.documents[: k or len(self.documents)]


class SequentialVectorStore:
    def __init__(self, results: list[list[Document]]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str, k: int | None = None) -> list[Document]:
        self.queries.append(query)
        documents = self.results.pop(0)
        return documents[: k or len(documents)]


class StreamingChatModel:
    def invoke_messages(self, messages):
        return {"content": "Hello!"}

    def stream(self, prompt=None, *, messages=None):
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
    assert "event: guard_result" in events
    assert 'event: done\ndata: {"content": "Hello!"' in events


def test_query_rewrite_uses_chat_history_for_vague_follow_up() -> None:
    vector_store = StaticVectorStore(
        [
            Document(
                page_content="Termination requires 30 days written notice.",
                metadata={"file_name": "contract.pdf", "page": 0, "chunk_id": 1},
            )
        ]
    )
    service = DocumentQAService(
        vector_store=vector_store,
        chat_model=FakeListChatModel(
            responses=[
                "termination notice period",
                "Termination requires 30 days written notice [S1].",
            ]
        ),
    )

    answer = service.ask(
        "这个期限是多少？",
        chat_history=[
            {"role": "user", "content": "请看 termination 条款。"},
            {"role": "assistant", "content": "我会查看终止条款。"},
        ],
    )

    assert vector_store.queries[0] == "termination notice period"
    assert answer.content == "Termination requires 30 days written notice [S1]."


def test_lightweight_repair_removes_invalid_citation_without_llm_call() -> None:
    class CountingChatModel:
        def __init__(self) -> None:
            self.calls = 0

        def invoke_messages(self, messages):
            self.calls += 1
            return {"content": "initial"}

    chat_model = CountingChatModel()
    service = DocumentQAService(vector_store=EmptyVectorStore(), chat_model=chat_model)
    prepared = service.prepare_answer("hello")
    prepared = type(prepared)(
        messages=prepared.messages,
        citations=[
            Citation(
                source_id="S1",
                file_name="contract.pdf",
                preview="The notice period is 30 days.",
            )
        ],
        memories_used=prepared.memories_used,
        user_id=prepared.user_id,
        conversation_id=prepared.conversation_id,
        user_message_recorded=prepared.user_message_recorded,
        has_retrieved_documents=True,
    )

    answer = service.finalize_prepared_answer(prepared, "The notice period is 30 days [S9].")

    assert "[S9]" not in answer.content
    assert "[S1]" in answer.content
    assert chat_model.calls == 0


def test_review_clause_returns_structured_metadata_and_expands_taxonomy_query() -> None:
    vector_store = StaticVectorStore(
        [
            Document(
                page_content="The customer may terminate this agreement with 30 days written notice.",
                metadata={"file_name": "contract.pdf", "page": 0, "chunk_id": 4},
            )
        ]
    )
    service = DocumentQAService(
        vector_store=vector_store,
        chat_model=FakeListChatModel(
            responses=[
                """
                {
                  "clause_type": "termination",
                  "normalized_clause_type": "termination",
                  "found": true,
                  "summary": "The customer may terminate with 30 days written notice.",
                  "risk_level": "Medium",
                  "risk_reasons": [
                    {
                      "reason": "The clause requires 30 days written notice before termination.",
                      "citation": "S1"
                    }
                  ],
                  "affected_party": "Customer",
                  "plain_language_explanation": "The customer can end the agreement, but must give written notice first.",
                  "questions_for_lawyer": [
                    "Can the notice period be shortened or made mutual?"
                  ],
                  "missing_information": [],
                  "needs_human_review": false
                }
                """
            ]
        ),
    )

    answer = service.review_clause("termination", top_k=1)

    assert "early cancellation" in vector_store.queries[0]
    assert answer.metadata["risk_level"] == "Medium"
    assert answer.metadata["risk_reasons"] == [
        {
            "reason": "The clause requires 30 days written notice before termination.",
            "citation": "S1",
        }
    ]
    assert "Risk level: Medium" in answer.content
    assert "[S1]" in answer.content


def test_check_conflict_returns_structured_conflict_matrix() -> None:
    vector_store = SequentialVectorStore(
        [
            [
                Document(
                    page_content="Payment is due within 10 days of invoice.",
                    metadata={"file_name": "contract.pdf", "page": 1, "chunk_id": 2},
                )
            ],
            [
                Document(
                    page_content="Company policy requires standard supplier payment terms of 30 days.",
                    metadata={"file_name": "policy.pdf", "page": 2, "chunk_id": 8},
                )
            ],
        ]
    )
    service = DocumentQAService(
        vector_store=vector_store,
        chat_model=FakeListChatModel(
            responses=[
                """
                {
                  "overall_status": "Potential conflict",
                  "conflicts": [
                    {
                      "topic": "Payment timeline",
                      "conflict_type": "timeline_conflict",
                      "severity": "High",
                      "contract_position": "Payment is due within 10 days of invoice.",
                      "policy_position": "Supplier payment terms are 30 days.",
                      "why_conflict": "The contract payment deadline is shorter than the policy payment term.",
                      "recommended_action": "Confirm whether a policy exception is needed before signing.",
                      "contract_citations": ["C1"],
                      "policy_citations": ["P1"],
                      "needs_human_review": true,
                      "confidence": "Medium"
                    }
                  ],
                  "needs_human_review": true,
                  "supporting_citations": []
                }
                """
            ]
        ),
    )

    answer = service.check_conflict("payment terms", "payment policy", top_k=1)

    assert answer.metadata["overall_status"] == "Potential conflict"
    assert answer.metadata["conflicts"][0]["conflict_type"] == "deadline_mismatch"
    assert answer.metadata["conflicts"][0]["contract_citations"] == ["C1"]
    assert answer.metadata["conflicts"][0]["policy_citations"] == ["P1"]
    assert "Conflict 1: Payment timeline" in answer.content
    assert "[C1]" in answer.content
    assert "[P1]" in answer.content


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
    assert calls[0]["json"]["messages"] == [{"role": "user", "content": "prompt"}]
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

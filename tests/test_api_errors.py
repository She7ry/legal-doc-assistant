from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend import dependencies
from backend.main import app
from backend.middleware import rate_limit as rate_limit_module
from backend.middleware.rate_limit import SlidingWindowRateLimiter


def test_chat_keyword_hit_uses_tool_and_miss_uses_agent_flow() -> None:
    calls: list[tuple[str, str]] = []

    class FakeToolService:
        def ask(self, question, **_kwargs):
            calls.append(("tool", question))
            return SimpleNamespace(
                content="keyword result",
                citations=[],
                confidence=None,
                guard_warnings=[],
                metadata={},
            )

    def fake_agent_service(**kwargs):
        calls.append(("agent", kwargs["objective"]))
        return SimpleNamespace(
            report="agent result",
            citations=[],
            confidence=None,
            guard_warnings=[],
            evidence=None,
        )

    app.dependency_overrides[dependencies.get_tool_calling_service] = lambda: FakeToolService()
    app.dependency_overrides[dependencies.get_agent_service] = lambda: fake_agent_service
    try:
        client = TestClient(app)
        keyword = client.post("/api/v1/chat/ask", json={"question": "审查终止条款"})
        semantic = client.post("/api/v1/chat/ask", json={"question": "请帮我完成这项工作"})
    finally:
        app.dependency_overrides.pop(dependencies.get_tool_calling_service, None)
        app.dependency_overrides.pop(dependencies.get_agent_service, None)

    assert keyword.json()["content"] == "keyword result"
    assert semantic.json()["content"] == "agent result"
    assert calls == [("tool", "审查终止条款"), ("agent", "请帮我完成这项工作")]


def test_internal_value_error_returns_safe_500_and_validation_stays_4xx() -> None:
    class ExplodingToolService:
        def ask(self, *_args, **_kwargs):
            raise ValueError("internal-secret-value")

    app.dependency_overrides[dependencies.get_tool_calling_service] = lambda: ExplodingToolService()
    app.dependency_overrides[dependencies.get_agent_service] = lambda: lambda **_kwargs: None
    try:
        client = TestClient(app, raise_server_exceptions=False)
        failed = client.post("/api/v1/chat/ask", json={"question": "审查这个条款。"})
        invalid = client.post("/api/v1/chat/ask", json={"question": ""})
    finally:
        app.dependency_overrides.pop(dependencies.get_tool_calling_service, None)
        app.dependency_overrides.pop(dependencies.get_agent_service, None)

    assert failed.status_code == 500
    assert failed.json()["code"] == "internal_error"
    assert "internal-secret-value" not in failed.text
    assert invalid.status_code == 422


def test_stream_error_does_not_expose_internal_exception() -> None:
    class ExplodingQAService:
        def prepare_answer(self, *_args, **_kwargs):
            return SimpleNamespace(citations=[], task_id=None)

        def stream_prepared_answer(self, _prepared):
            raise RuntimeError("stream-secret-value")

    app.dependency_overrides[dependencies.get_qa_service] = lambda: ExplodingQAService()
    try:
        client = TestClient(app)
        response = client.post("/api/v1/chat/ask/stream", json={"question": "Review this."})
    finally:
        app.dependency_overrides.pop(dependencies.get_qa_service, None)

    assert response.status_code == 200
    assert "Answer stream failed." in response.text
    assert "stream-secret-value" not in response.text


def test_rate_limiter_discards_expired_keys(monkeypatch) -> None:
    now = [0.0]
    monkeypatch.setattr(rate_limit_module.time, "monotonic", lambda: now[0])
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=10)

    assert limiter.is_allowed("expired")
    assert not limiter.is_allowed("expired")
    now[0] = 11.0
    assert limiter.is_allowed("current")

    assert set(limiter._requests) == {"current"}

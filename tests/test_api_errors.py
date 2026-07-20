from __future__ import annotations

from fastapi.testclient import TestClient

from ai.rag.qa_service import PreparedQAAnswer
from backend import dependencies
from backend.main import app


def test_internal_value_error_returns_safe_500_and_validation_stays_4xx() -> None:
    class ExplodingQAService:
        async def aask(self, *_args, **_kwargs):
            raise ValueError("internal-secret-value")

    app.dependency_overrides[dependencies.get_qa_service] = lambda: ExplodingQAService()
    try:
        client = TestClient(app, raise_server_exceptions=False)
        failed = client.post("/api/v1/chat/ask", json={"question": "Review this clause."})
        invalid = client.post("/api/v1/chat/ask", json={"question": ""})
    finally:
        app.dependency_overrides.pop(dependencies.get_qa_service, None)

    assert failed.status_code == 500
    assert failed.json()["code"] == "internal_error"
    assert "internal-secret-value" not in failed.text
    assert invalid.status_code == 422


def test_stream_error_does_not_expose_internal_exception() -> None:
    class ExplodingStreamQAService:
        def prepare_answer(self, *_args, **_kwargs) -> PreparedQAAnswer:
            return PreparedQAAnswer(
                messages=[],
                citations=[],
                memories_used=[],
                user_id="user-a",
                conversation_id=None,
                user_message_recorded=False,
                task_id="task-a",
            )

        def stream_prepared_answer(self, _prepared):
            raise RuntimeError("stream-secret-value")
            yield

    app.dependency_overrides[dependencies.get_qa_service] = lambda: ExplodingStreamQAService()
    try:
        client = TestClient(app)
        response = client.post("/api/v1/chat/ask/stream", json={"question": "Review this."})
    finally:
        app.dependency_overrides.pop(dependencies.get_qa_service, None)

    assert response.status_code == 200
    assert "Answer stream failed." in response.text
    assert "stream-secret-value" not in response.text

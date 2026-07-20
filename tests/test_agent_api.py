from __future__ import annotations

import time

from fastapi.testclient import TestClient

from ai.agent.schemas import AgentTaskResult
from backend import dependencies
from backend.agent_tasks import AgentTaskStore
from backend.main import app


class FastAgentService:
    def run_task(self, **kwargs) -> AgentTaskResult:
        progress_callback = kwargs.get("progress_callback")
        if progress_callback:
            progress_callback(
                event_type="react_started",
                stage="answering",
                progress=10,
                message="Running ReAct tool-calling workflow.",
            )
        return AgentTaskResult(
            task_id=kwargs["task_id"],
            status="completed",
            objective=kwargs["objective"],
            steps=[],
            human_review_required=False,
            report="Test report.",
            citations=[],
            confidence="High",
        )


class ExplodingAgentService:
    def run_task(self, **_kwargs) -> AgentTaskResult:
        raise AssertionError("Agent service should not run when clarification is required.")


def _wait_for_agent_task(client: TestClient, task_id: str, *, user_id: str = "api-test-user") -> dict:
    deadline = time.monotonic() + 3
    last_data = None
    while time.monotonic() < deadline:
        loaded = client.get(
            f"/api/v1/agent/tasks/{task_id}",
            headers={"X-User-Id": user_id},
        )
        assert loaded.status_code == 200
        last_data = loaded.json()
        if last_data["status"] in {"succeeded", "failed", "needs_input"}:
            return last_data
        time.sleep(0.02)
    raise AssertionError(f"Agent task did not finish in time: {last_data}")


def test_agent_task_api_creates_gets_and_streams_events(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "agent_tasks.sqlite3")
    app.dependency_overrides[dependencies.get_agent_task_store] = lambda: store
    app.dependency_overrides[dependencies.get_agent_service] = lambda: FastAgentService().run_task
    client = TestClient(app)

    try:
        created = client.post(
            "/api/v1/agent/tasks",
            json={
                "objective": "Review payment terms.",
                "focus_areas": ["payment"],
                "user_role": "ordinary",
                "max_steps": 3,
            },
            headers={"X-User-Id": "api-test-user"},
        )
        assert created.status_code == 202
        task_id = created.json()["task_id"]

        data = _wait_for_agent_task(client, task_id)
        assert data["status"] == "succeeded"
        assert data["result"]["report"] == "Test report."

        events = client.get(
            f"/api/v1/agent/tasks/{task_id}/events",
            headers={"X-User-Id": "api-test-user"},
        )
        assert events.status_code == 200
        assert "event: queued" in events.text
        assert "event: succeeded" in events.text
    finally:
        app.dependency_overrides.clear()


def test_agent_task_api_marks_underspecified_tasks_as_needing_input(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "agent_tasks.sqlite3")
    app.dependency_overrides[dependencies.get_agent_task_store] = lambda: store
    app.dependency_overrides[dependencies.get_agent_service] = lambda: ExplodingAgentService().run_task
    client = TestClient(app)

    try:
        created = client.post(
            "/api/v1/agent/tasks",
            json={
                "objective": "帮我看看",
                "focus_areas": [],
                "user_role": "ordinary",
                "max_steps": 3,
            },
            headers={"X-User-Id": "api-test-user"},
        )
        assert created.status_code == 202
        data = created.json()
        assert data["status"] == "needs_input"
        assert data["stage"] == "needs_input"
        assert data["result"] is None
        assert data["events"][-1]["event_type"] == "needs_input"
        assert data["events"][-1]["payload"]["questions"]

        events = client.get(
            f"/api/v1/agent/tasks/{data['task_id']}/events",
            headers={"X-User-Id": "api-test-user"},
        )
        assert events.status_code == 200
        assert "event: needs_input" in events.text
    finally:
        app.dependency_overrides.clear()


def test_agent_task_api_resumes_task_after_supplemental_input(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "agent_tasks.sqlite3")
    app.dependency_overrides[dependencies.get_agent_task_store] = lambda: store
    app.dependency_overrides[dependencies.get_agent_service] = lambda: FastAgentService().run_task
    client = TestClient(app)

    try:
        created = client.post(
            "/api/v1/agent/tasks",
            json={
                "objective": "review this",
                "focus_areas": [],
                "user_role": "ordinary",
                "max_steps": 3,
            },
            headers={"X-User-Id": "api-test-user"},
        )
        assert created.status_code == 202
        task_id = created.json()["task_id"]
        assert created.json()["status"] == "needs_input"

        resumed = client.post(
            f"/api/v1/agent/tasks/{task_id}/resume",
            json={
                "clarification_answers": [
                    (
                        "Review payment and termination risk. "
                        "I represent the customer."
                    )
                ],
                "focus_areas": ["payment", "termination"],
                "user_role": "lawyer",
                "max_steps": 4,
            },
            headers={"X-User-Id": "api-test-user"},
        )
        assert resumed.status_code == 202
        assert resumed.json()["status"] == "queued"
        assert resumed.json()["events"][-2]["event_type"] == "input_received"

        data = _wait_for_agent_task(client, task_id)
        assert data["status"] == "succeeded"
        assert data["user_role"] == "lawyer"
        assert data["focus_areas"] == ["payment", "termination"]
        assert data["result"]["objective"].startswith("review this")
        assert "Review payment and termination risk" in data["result"]["objective"]
        event_types = [event["event_type"] for event in data["events"]]
        assert "input_received" in event_types
        assert event_types[-4:] == ["queued", "running", "react_started", "succeeded"]
    finally:
        app.dependency_overrides.clear()


def test_agent_task_api_rejects_resume_for_non_needs_input_task(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "agent_tasks.sqlite3")
    app.dependency_overrides[dependencies.get_agent_task_store] = lambda: store
    app.dependency_overrides[dependencies.get_agent_service] = lambda: FastAgentService().run_task
    client = TestClient(app)

    try:
        created = client.post(
            "/api/v1/agent/tasks",
            json={
                "objective": "Review payment terms.",
                "focus_areas": ["payment"],
                "user_role": "ordinary",
                "max_steps": 3,
            },
            headers={"X-User-Id": "api-test-user"},
        )
        task_id = created.json()["task_id"]

        rejected = client.post(
            f"/api/v1/agent/tasks/{task_id}/resume",
            json={"clarification_answers": ["I represent the customer."]},
            headers={"X-User-Id": "api-test-user"},
        )

        assert rejected.status_code == 409
    finally:
        app.dependency_overrides.clear()

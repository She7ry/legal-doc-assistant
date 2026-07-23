from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from backend.agent_tasks import AgentTaskStatus, AgentTaskStore


def test_agent_task_store_hides_tasks_from_other_users(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "agent_tasks.sqlite3")
    task = store.create(
        user_id="user-a",
        objective="Review payment terms.",
        focus_areas=["payment"],
        user_role="ordinary",
        max_steps=4,
        conversation_id=None,
    )

    assert store.get(task.task_id, "user-a") is not None
    assert store.get(task.task_id, "user-b") is None


def test_agent_task_store_tracks_progress_events_and_result(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "agent_tasks.sqlite3")
    task = store.create(
        user_id="user-a",
        objective="Review termination.",
        focus_areas=["termination"],
        user_role="lawyer",
        max_steps=4,
        conversation_id="conversation-1",
    )

    assert store.claim(task.task_id)
    store.update_progress(
        task.task_id,
        event_type="step_completed",
        stage="review_1",
        progress=60,
        message="Reviewed termination.",
        step_id="review_1",
        payload={"citation_count": 2},
    )
    store.mark_succeeded(task.task_id, {"task_id": task.task_id, "status": "completed"})

    finished = store.get(task.task_id, "user-a")
    assert finished is not None
    assert finished.status == AgentTaskStatus.SUCCEEDED
    assert finished.stage == "completed"
    assert finished.progress == 100
    assert finished.result == {"task_id": task.task_id, "status": "completed"}
    assert [event.event_type for event in finished.events or []] == [
        "queued",
        "running",
        "step_completed",
        "succeeded",
    ]


def test_agent_task_store_marks_task_as_needing_input(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "agent_tasks.sqlite3")
    task = store.create(
        user_id="user-a",
        objective="帮我看看",
        focus_areas=[],
        user_role="ordinary",
        max_steps=4,
        conversation_id=None,
    )

    store.mark_needs_input(task.task_id, ["请说明具体审查目标。"])

    loaded = store.get(task.task_id, "user-a")
    assert loaded is not None
    assert loaded.status == AgentTaskStatus.NEEDS_INPUT
    assert loaded.stage == "needs_input"
    assert loaded.progress == 0
    assert loaded.result is None
    assert [event.event_type for event in loaded.events or []] == ["queued", "needs_input"]
    assert loaded.events
    assert loaded.events[-1].payload == {"questions": ["请说明具体审查目标。"]}


def test_agent_task_store_resumes_task_with_supplemental_input(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "agent_tasks.sqlite3")
    task = store.create(
        user_id="user-a",
        objective="review this",
        focus_areas=[],
        user_role="ordinary",
        max_steps=4,
        conversation_id=None,
    )
    store.mark_needs_input(task.task_id, ["请说明具体审查目标。"])

    resumed = store.resume_with_input(
        task.task_id,
        objective="Review payment risk.\n\nSupplemental user input:\n- I represent the buyer.",
        focus_areas=["payment"],
        user_role="lawyer",
        max_steps=5,
        conversation_id="conversation-2",
        clarification_answers=["I represent the buyer."],
    )

    assert resumed.status == AgentTaskStatus.QUEUED
    assert resumed.stage == "queued"
    assert resumed.focus_areas == ["payment"]
    assert resumed.user_role == "lawyer"
    assert resumed.max_steps == 5
    assert resumed.conversation_id == "conversation-2"
    assert resumed.result is None
    assert [event.event_type for event in resumed.events or []] == [
        "queued",
        "needs_input",
        "input_received",
        "queued",
    ]
    assert resumed.events
    assert resumed.events[-2].payload == {"answers": ["I represent the buyer."]}


def test_agent_task_store_persists_tasks_to_sqlite(tmp_path) -> None:
    db_path = tmp_path / "agent_tasks.sqlite3"
    first_store = AgentTaskStore(db_path)
    task = first_store.create(
        user_id="user-a",
        objective="Review liability.",
        focus_areas=["liability limitation"],
        user_role="ordinary",
        max_steps=5,
        conversation_id=None,
    )

    assert first_store.claim(task.task_id)
    first_store.mark_failed(task.task_id, "model unavailable")

    second_store = AgentTaskStore(db_path)
    loaded = second_store.get(task.task_id, "user-a")

    assert loaded is not None
    assert loaded.status == AgentTaskStatus.FAILED
    assert loaded.error == "model unavailable"
    assert loaded.events
    assert loaded.events[-1].event_type == "failed"


def test_agent_task_store_claims_sqlite_task_once_and_does_not_restart_running(tmp_path) -> None:
    db_path = tmp_path / "agent_tasks.sqlite3"
    first_store = AgentTaskStore(db_path)
    second_store = AgentTaskStore(db_path)
    task = first_store.create(
        user_id="user-a",
        objective="Review liability.",
        focus_areas=["liability limitation"],
        user_role="ordinary",
        max_steps=5,
        conversation_id=None,
    )
    barrier = Barrier(2)

    def claim(store: AgentTaskStore) -> bool:
        barrier.wait()
        return store.claim(task.task_id)

    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed = list(executor.map(claim, (first_store, second_store)))

    assert sum(claimed) == 1
    assert first_store.list_restartable() == []
    loaded = second_store.get(task.task_id, "user-a")
    assert loaded is not None
    assert loaded.status == AgentTaskStatus.RUNNING
    assert [event.event_type for event in loaded.events or []].count("running") == 1


def test_agent_task_store_requeues_interrupted_work(tmp_path) -> None:
    store = AgentTaskStore(tmp_path / "agent_tasks.sqlite3")
    task = store.create(
        user_id="user-a",
        objective="Review liability.",
        focus_areas=["liability"],
        user_role="ordinary",
        max_steps=4,
        conversation_id=None,
    )
    assert store.claim(task.task_id)

    assert store.requeue_interrupted() == 1

    recovered = store.get(task.task_id, "user-a")
    assert recovered is not None
    assert recovered.status == AgentTaskStatus.QUEUED
    assert recovered.started_at is None
    assert recovered.events
    assert recovered.events[-1].payload == {"recovered": True}


def test_agent_task_status_and_terminal_event_commit_together(tmp_path, monkeypatch) -> None:
    store = AgentTaskStore(tmp_path / "agent_tasks.sqlite3")
    task = store.create(
        user_id="user-a",
        objective="Review liability.",
        focus_areas=["liability"],
        user_role="ordinary",
        max_steps=4,
        conversation_id=None,
    )
    assert store.claim(task.task_id)

    def fail_event(*_args, **_kwargs):
        raise RuntimeError("event write failed")

    monkeypatch.setattr(store, "_insert_event_with_connection", fail_event)
    with pytest.raises(RuntimeError, match="event write failed"):
        store.mark_succeeded(task.task_id, {"status": "completed"})

    loaded = store.get(task.task_id, "user-a")
    assert loaded is not None
    assert loaded.status == AgentTaskStatus.RUNNING

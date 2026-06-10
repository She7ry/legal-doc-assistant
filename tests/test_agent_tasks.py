from __future__ import annotations

from api.agent_tasks import AgentTaskStatus, AgentTaskStore


def test_agent_task_store_hides_tasks_from_other_tenants_and_users() -> None:
    store = AgentTaskStore()
    task = store.create(
        tenant_id="tenant-a",
        user_id="user-a",
        objective="Review payment terms.",
        focus_areas=["payment"],
        user_role="ordinary",
        max_steps=4,
        conversation_id=None,
    )

    assert store.get(task.task_id, "tenant-a", "user-a") is not None
    assert store.get(task.task_id, "tenant-b", "user-a") is None
    assert store.get(task.task_id, "tenant-a", "user-b") is None


def test_agent_task_store_tracks_progress_events_and_result() -> None:
    store = AgentTaskStore()
    task = store.create(
        tenant_id="tenant-a",
        user_id="user-a",
        objective="Review termination.",
        focus_areas=["termination"],
        user_role="lawyer",
        max_steps=4,
        conversation_id="conversation-1",
    )

    store.mark_running(task.task_id)
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

    finished = store.get(task.task_id, "tenant-a", "user-a")
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


def test_agent_task_store_marks_task_as_needing_input() -> None:
    store = AgentTaskStore()
    task = store.create(
        tenant_id="tenant-a",
        user_id="user-a",
        objective="帮我看看",
        focus_areas=[],
        user_role="ordinary",
        max_steps=4,
        conversation_id=None,
    )

    store.mark_needs_input(task.task_id, ["请说明具体审查目标。"])

    loaded = store.get(task.task_id, "tenant-a", "user-a")
    assert loaded is not None
    assert loaded.status == AgentTaskStatus.NEEDS_INPUT
    assert loaded.stage == "needs_input"
    assert loaded.progress == 0
    assert loaded.result is None
    assert [event.event_type for event in loaded.events or []] == ["queued", "needs_input"]
    assert loaded.events
    assert loaded.events[-1].payload == {"questions": ["请说明具体审查目标。"]}


def test_agent_task_store_resumes_task_with_supplemental_input() -> None:
    store = AgentTaskStore()
    task = store.create(
        tenant_id="tenant-a",
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
        tenant_id="tenant-a",
        user_id="user-a",
        objective="Review liability.",
        focus_areas=["liability limitation"],
        user_role="ordinary",
        max_steps=5,
        conversation_id=None,
    )

    first_store.mark_running(task.task_id)
    first_store.mark_failed(task.task_id, "model unavailable")

    second_store = AgentTaskStore(db_path)
    loaded = second_store.get(task.task_id, "tenant-a", "user-a")

    assert loaded is not None
    assert loaded.status == AgentTaskStatus.FAILED
    assert loaded.error == "model unavailable"
    assert loaded.events
    assert loaded.events[-1].event_type == "failed"

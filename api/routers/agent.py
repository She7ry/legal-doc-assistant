from __future__ import annotations

from collections.abc import Iterator
import logging
from time import sleep

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from api.agent_tasks import AgentTaskRecord, AgentTaskStatus, AgentTaskStore
from api.dependencies import (
    AgentServiceDep,
    AgentTaskStoreDep,
    MatterStoreDep,
    TenantIdDep,
    UserIdDep,
    require_api_key,
)
from api.schemas.requests import AgentTaskRequest, AgentTaskResumeRequest
from api.schemas.responses import AgentTaskRecordResponse, AgentTaskResponse
from api.sse import SSE_HEADERS, format_sse
from api.task_queue import submit_background_task
from doc_assistant.services.agent._constants import clarification_questions_for_task
from doc_assistant.services.agent_service import LegalAgentService
from doc_assistant.matter.store import MatterStore
from langgraph.errors import GraphInterrupt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"], dependencies=[Depends(require_api_key)])


@router.post(
    "/tasks",
    response_model=AgentTaskRecordResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create and run a task-oriented legal document agent workflow",
)
def create_agent_task(
    body: AgentTaskRequest,
    agent_service: AgentServiceDep,
    task_store: AgentTaskStoreDep,
    matter_store: MatterStoreDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> AgentTaskRecordResponse:
    record = task_store.create(
        tenant_id=tenant_id,
        user_id=user_id,
        objective=body.objective,
        focus_areas=body.focus_areas,
        user_role=body.user_role,
        max_steps=body.max_steps,
        conversation_id=body.conversation_id,
        matter_id=body.matter_id,
    )
    return _check_clarification_and_enqueue(
        record, agent_service, task_store, matter_store, tenant_id, user_id,
    )


@router.post(
    "/tasks/{task_id}/resume",
    response_model=AgentTaskRecordResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Resume an Agent task after required supplemental input is provided",
)
def resume_agent_task(
    task_id: str,
    body: AgentTaskResumeRequest,
    agent_service: AgentServiceDep,
    task_store: AgentTaskStoreDep,
    matter_store: MatterStoreDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> AgentTaskRecordResponse:
    record = task_store.get(task_id, tenant_id, user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent task not found.")
    if record.status != AgentTaskStatus.NEEDS_INPUT:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only Agent tasks in needs_input status can be resumed.",
        )

    clarification_answers = _clean_text_list(body.clarification_answers)
    objective_override = _clean_text(body.objective)
    if not objective_override and not clarification_answers:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide an updated objective or at least one clarification answer.",
        )

    resumed = task_store.resume_with_input(
        task_id,
        objective=_compose_resumed_objective(
            original_objective=record.objective,
            objective_override=objective_override,
            clarification_answers=clarification_answers,
        ),
        focus_areas=body.focus_areas if body.focus_areas is not None else record.focus_areas,
        user_role=body.user_role or record.user_role,
        max_steps=body.max_steps or record.max_steps,
        conversation_id=body.conversation_id or record.conversation_id,
        matter_id=body.matter_id or record.matter_id,
        clarification_answers=clarification_answers,
    )
    return _check_clarification_and_enqueue(
        resumed, agent_service, task_store, matter_store, tenant_id, user_id,
    )


@router.get(
    "/tasks/{task_id}",
    response_model=AgentTaskRecordResponse,
    summary="Get an Agent task status and result",
)
def get_agent_task(
    task_id: str,
    task_store: AgentTaskStoreDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> AgentTaskRecordResponse:
    record = task_store.get(task_id, tenant_id, user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent task not found.")
    return AgentTaskRecordResponse.from_record(record)


@router.get(
    "/tasks/{task_id}/events",
    summary="Stream Agent task progress events",
)
def stream_agent_task_events(
    task_id: str,
    task_store: AgentTaskStoreDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
    after_event_id: int = Query(default=0, ge=0),
) -> StreamingResponse:
    record = task_store.get(task_id, tenant_id, user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent task not found.")

    return StreamingResponse(
        _agent_task_event_stream(task_store, task_id, tenant_id, user_id, after_event_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# ------------------------------------------------------------------
# Background execution
# ------------------------------------------------------------------


def _run_agent_task(
    record: AgentTaskRecord,
    agent_service: LegalAgentService,
    task_store: AgentTaskStore,
    matter_store: MatterStore,
) -> None:
    task_id = record.task_id
    task_store.mark_running(task_id)

    def progress_callback(**event) -> None:
        task_store.update_progress(task_id, **event)

    try:
        result = agent_service.run_task(
            objective=record.objective,
            focus_areas=record.focus_areas,
            user_role=record.user_role,
            max_steps=record.max_steps,
            user_id=record.user_id,
            conversation_id=record.conversation_id,
            task_id=task_id,
            matter_id=record.matter_id,
            progress_callback=progress_callback,
            thread_id=task_id,  # P1-1: checkpoint thread
        )
        response = AgentTaskResponse.from_result(result)
        encoded_response = jsonable_encoder(response)
    except GraphInterrupt as interrupt_exc:
        # P1-1: 图在 finalize_result 因确认闸门而中断
        interrupt_data = interrupt_exc.args[0] if interrupt_exc.args else {}
        task_store.mark_interrupted(task_id, interrupt_data)
        return
    except Exception as exc:
        logger.exception("Agent task failed", extra={"task_id": task_id})
        task_store.mark_failed(task_id, f"Failed to run Agent task: {exc}")
        return

    try:
        matter_store.upsert_from_agent_result(
            tenant_id=record.tenant_id,
            user_id=record.user_id,
            matter_id=record.matter_id or task_id,
            result=encoded_response,
        )
    except Exception as exc:
        logger.exception("Agent task completed but matter persistence failed", extra={"task_id": task_id})
        metadata = dict(encoded_response.get("metadata") or {})
        metadata["matter_persist_error"] = str(exc)
        encoded_response["metadata"] = metadata

    task_store.mark_succeeded(task_id, encoded_response)


def enqueue_agent_task(
    record: AgentTaskRecord,
    agent_service: LegalAgentService,
    task_store: AgentTaskStore,
    matter_store: MatterStore,
) -> bool:
    return submit_background_task(
        f"agent:{record.task_id}",
        _run_agent_task,
        record,
        agent_service,
        task_store,
        matter_store,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _check_clarification_and_enqueue(
    record: AgentTaskRecord,
    agent_service: LegalAgentService,
    task_store: AgentTaskStore,
    matter_store: MatterStore,
    tenant_id: str,
    user_id: str,
) -> AgentTaskRecordResponse:
    questions = clarification_questions_for_task(record.objective, record.focus_areas)
    if questions:
        task_store.mark_needs_input(record.task_id, questions)
        updated = task_store.get(record.task_id, tenant_id, user_id) or record
        return AgentTaskRecordResponse.from_record(updated)
    enqueue_agent_task(record, agent_service, task_store, matter_store)
    return AgentTaskRecordResponse.from_record(record)


def _agent_task_event_stream(
    task_store: AgentTaskStore,
    task_id: str,
    tenant_id: str,
    user_id: str,
    after_event_id: int,
) -> Iterator[str]:
    last_event_id = after_event_id
    idle_ticks = 0

    while True:
        events = task_store.events_after(task_id, tenant_id, user_id, last_event_id)
        if events is None:
            yield format_sse("error", {"detail": "Agent task not found."})
            return

        if events:
            idle_ticks = 0
        for event in events:
            last_event_id = event.event_id
            yield format_sse(
                event.event_type,
                {
                    "event_id": event.event_id,
                    "task_id": event.task_id,
                    "event_type": event.event_type,
                    "stage": event.stage,
                    "progress": event.progress,
                    "message": event.message,
                    "step_id": event.step_id,
                    "payload": event.payload or {},
                    "created_at": event.created_at,
                },
                event_id=event.event_id,
            )

        record = task_store.get(task_id, tenant_id, user_id)
        if record is None or record.status in {
            AgentTaskStatus.NEEDS_INPUT,
            AgentTaskStatus.SUCCEEDED,
            AgentTaskStatus.FAILED,
        }:
            return

        idle_ticks += 1
        if idle_ticks % 15 == 0:
            yield format_sse(
                "heartbeat",
                {
                    "task_id": task_id,
                    "stage": record.stage,
                    "progress": record.progress,
                    "message": "Agent task is still running.",
                },
            )
        sleep(0.8)


def _compose_resumed_objective(
    *,
    original_objective: str,
    objective_override: str,
    clarification_answers: list[str],
) -> str:
    base_objective = objective_override or _clean_text(original_objective)
    if not clarification_answers:
        return base_objective

    answer_lines = "\n".join(f"- {answer}" for answer in clarification_answers)
    return f"{base_objective}\n\nSupplemental user input:\n{answer_lines}"


def _clean_text_list(values: list[str]) -> list[str]:
    result = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _clean_text(value: str | None) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""

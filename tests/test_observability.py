from __future__ import annotations

from unittest.mock import Mock

import ai.observability as observability
from ai.agent.schemas import AgentStepResult, AgentTaskResult
from ai.observability import (
    agent_trace_inputs,
    agent_trace_outputs,
    hash_trace_identifier,
    langsmith_agent_context,
)
from ai.rag.schemas import Citation


def test_agent_trace_inputs_removes_runtime_objects_and_hashes_identity() -> None:
    callback = object()
    service = object()

    traced = agent_trace_inputs(
        {
            "qa_service": service,
            "objective": "Review contract",
            "focus_areas": ["payment"],
            "user_role": "ordinary",
            "max_steps": 4,
            "user_id": "user@example.com",
            "conversation_id": "conversation-1",
            "task_id": "task-1",
            "progress_callback": callback,
        }
    )

    assert traced["objective"] == "Review contract"
    assert traced["user_id_hash"] == hash_trace_identifier("user@example.com")
    assert traced["user_id_hash"] != "user@example.com"
    assert "qa_service" not in traced
    assert "progress_callback" not in traced


def test_agent_trace_outputs_keeps_judge_evidence_without_raw_tool_output() -> None:
    citation = Citation(
        source_id="S1",
        file_name="contract.pdf",
        preview="Payment is due in 30 days.",
        exact_quote="Payment is due in 30 days.",
    )
    result = AgentTaskResult(
        task_id="task-1",
        status="completed",
        objective="Review payment terms",
        steps=[
            AgentStepResult(
                step_id="step-1",
                title="Find payment clause",
                tool="document_qa",
                status="completed",
                summary="Payment is due in 30 days [S1].",
                citations=[citation],
                output={"raw_tool_payload": "must-not-be-traced"},
            )
        ],
        human_review_required=False,
        report="Payment is due in 30 days [S1].",
        citations=[citation],
        metadata={"runtime": "test", "tool_calls": ["search_documents"]},
    )

    traced = agent_trace_outputs(result)

    assert traced["report"] == result.report
    assert traced["citations"][0]["exact_quote"] == citation.exact_quote
    assert "output" not in traced["steps"][0]
    assert "must-not-be-traced" not in str(traced)
    assert traced["tool_call_count"] == 1
    assert traced["runtime"] == "test"


def test_disabled_langsmith_context_does_not_create_client(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    client_factory = Mock()
    monkeypatch.setattr(observability, "get_langsmith_client", client_factory)

    with langsmith_agent_context(
        task_id="task-1",
        user_id="user-1",
        conversation_id="conversation-1",
    ):
        pass

    client_factory.assert_not_called()

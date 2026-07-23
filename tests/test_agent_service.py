from __future__ import annotations

import sqlite3
from dataclasses import replace
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import Field

from ai.agent import clarification_questions_for_task, react_task
from ai.agent import graph as _planned_task
from ai.agent.react_task import run_react_agent_task
from ai.agent.schemas import AgentTaskPause
from ai.agent.tool_calling import ToolCallingAnswer, ToolCallTrace
from ai.rag.qa_service import DocumentQAService
from ai.rag.schemas import Citation, QAAnswer


def _sqlite_checkpointer(path):
    connection = sqlite3.connect(path, check_same_thread=False)
    return connection, SqliteSaver(
        connection,
        serde=JsonPlusSerializer(
            allowed_msgpack_modules=[
                ("ai.agent.schemas", "AgentStepResult"),
                ("ai.agent.schemas", "AgentTaskResult"),
                ("ai.rag.grounding.guard", "AnswerGuardResult"),
                ("ai.rag.schemas", "Citation"),
            ]
        ),
    )


class StaticVectorStore:
    user_id = "default"

    def __init__(self) -> None:
        self.queries: list[tuple[str, int | None]] = []

    def search(self, query: str, k: int | None = None) -> list[Document]:
        self.queries.append((query, k))
        return [
            Document(
                page_content=(
                    "This SaaS agreement is between VendorCo and CustomerCo. "
                    "New York law governs the agreement. "
                    "The customer may terminate with 30 days written notice."
                ),
                metadata={"file_name": "saas-msa.pdf", "page": 0, "chunk_id": 1},
            )
        ]


class AgentToolModel(BaseChatModel):
    calls: int = 0
    bound_tool_names: list[str] = Field(default_factory=list)
    tool_name: str = "review_clause"
    tool_args: dict = Field(default_factory=lambda: {"clause_type": "termination", "top_k": 2})
    final_content: str = "Termination requires 30 days written notice [D1]."
    plan_result: dict[str, Any] | None = None
    structured_error: bool = False
    structured_invocations: int = 0
    structured_inputs: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "agent-react-test"

    def bind_tools(self, tools, **kwargs):
        del kwargs
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def with_structured_output(self, schema, **kwargs):
        del kwargs

        def validate(messages):
            self.structured_invocations += 1
            self.structured_inputs.append(messages)
            if self.structured_error:
                raise ValueError("invalid structured response")
            return schema.model_validate(self.plan_result or {})

        return RunnableLambda(validate)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del stop, run_manager, kwargs
        self.calls += 1
        if self.calls == 1:
            assert self.tool_name in self.bound_tool_names
            message = AIMessage(
                content="",
                tool_calls=[{"name": self.tool_name, "args": self.tool_args, "id": "call_tool"}],
            )
        else:
            assert any(isinstance(message, ToolMessage) for message in messages)
            message = AIMessage(content=self.final_content)
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_legal_agent_runs_react_clause_review_with_citation_trace() -> None:
    model = AgentToolModel()
    qa_service = DocumentQAService(vector_store=StaticVectorStore(), chat_model=model)

    def fake_review_clause(clause_type: str, top_k: int | None = None) -> QAAnswer:
        assert clause_type == "termination"
        assert top_k == 2
        return QAAnswer(
            content="Termination requires 30 days written notice [S1].",
            citations=[
                Citation(
                    source_id="S1",
                    file_name="saas-msa.pdf",
                    preview="The customer may terminate with 30 days written notice.",
                    page=0,
                    chunk_id=1,
                    exact_quote="The customer may terminate with 30 days written notice.",
                )
            ],
            confidence="High",
            metadata={"risk_reasons": [{"reason": "Notice is required.", "citation": "S1"}]},
        )

    qa_service.review_clause = fake_review_clause  # type: ignore[method-assign]
    events: list[dict] = []
    result = run_react_agent_task(
        qa_service,
        objective="Review termination risk in the SaaS agreement.",
        focus_areas=["termination"],
        user_role="lawyer",
        max_steps=4,
        progress_callback=lambda **event: events.append(event),
    )

    assert result.status == "completed"
    assert [step.step_id for step in result.steps] == ["react"]
    assert result.report == "Termination requires 30 days written notice [D1]."
    assert result.steps[0].output["tool_calls"][0]["name"] == "review_clause"
    assert (
        result.steps[0].output["tool_calls"][0]["result"]["metadata"]["risk_reasons"][0]["citation"]
        == "D1"
    )
    assert result.citations[0].source_id == "D1"
    assert qa_service.vector_store.queries == []
    assert result.metadata["runtime"] == "react_langgraph_v1"
    assert [event["event_type"] for event in events] == ["react_started", "react_completed"]


def test_complex_focus_steps_are_isolated_retried_and_globally_renumbered(monkeypatch) -> None:
    qa_service = DocumentQAService(vector_store=StaticVectorStore(), chat_model=AgentToolModel())
    calls: list[tuple[str, dict[str, Any]]] = []
    attempts: dict[str, int] = {}
    synthesis_inputs: list[Any] = []
    repair_calls: list[str] = []
    events: list[dict[str, Any]] = []

    payment = Citation(
        source_id="D1",
        file_name="msa.pdf",
        page=0,
        chunk_id=1,
        preview="Payment is due within 30 days.",
        exact_quote="Payment is due within 30 days.",
    )
    liability = Citation(
        source_id="D1",
        file_name="msa.pdf",
        page=1,
        chunk_id=2,
        preview="Liability is capped at fees paid.",
        exact_quote="Liability is capped at fees paid.",
    )

    class FakeToolCallingChatService:
        def __init__(self, _qa_service) -> None:
            pass

        def ask(self, question: str, **kwargs) -> ToolCallingAnswer:
            step_id = str(kwargs["task_id"]).rsplit(":", 1)[-1]
            attempts[step_id] = attempts.get(step_id, 0) + 1
            calls.append((question, kwargs))
            if step_id == "step-1" and attempts[step_id] == 1:
                raise RuntimeError("transient model failure")
            if step_id == "step-3":
                raise RuntimeError("persistent model failure")

            trace = ToolCallTrace(
                tool_call_id=f"call-{step_id}",
                name="search_documents",
                arguments={"query": step_id},
                result={"citation": "D1", "internal": "RAW_TRACE_MARKER"},
            )
            if step_id == "step-1":
                return ToolCallingAnswer(
                    content="Payment is due within 30 days [D1].",
                    citations=[payment],
                    tool_calls=[trace],
                    metadata={"evidence": {"claims": [{"citations": ["D1"]}]}},
                )
            return ToolCallingAnswer(
                content=(
                    "Liability is capped at fees paid [D1]. "
                    "Payment remains due within 30 days [D2]."
                ),
                citations=[liability, replace(payment, source_id="D2")],
                tool_calls=[trace],
                metadata={"evidence": {"claims": [{"citations": ["D1", "D2"]}]}},
            )

    def synthesize(messages) -> str:
        synthesis_inputs.append(messages)
        return "Payment is due within 30 days [D99]."

    def repair(answer, _guard_result, _citations) -> str:
        repair_calls.append(answer)
        return (
            "Payment is due within 30 days [D1].\n\n"
            "Liability is capped at fees paid [D2].\n\n"
            "The data privacy step failed and requires human review."
        )

    monkeypatch.setattr(_planned_task, "ToolCallingChatService", FakeToolCallingChatService)
    monkeypatch.setattr(qa_service, "_invoke_chat_messages", synthesize)
    monkeypatch.setattr(qa_service, "repair_content", repair)

    result = run_react_agent_task(
        qa_service,
        objective="Review payment, liability, and data privacy risk.",
        focus_areas=["payment", "liability", "data privacy"],
        user_role="lawyer",
        user_id="user-1",
        conversation_id="conversation-1",
        task_id="task-1",
        progress_callback=lambda **event: events.append(event),
    )

    assert attempts == {"step-1": 2, "step-2": 1, "step-3": 2}
    assert [step.status for step in result.steps] == ["completed", "completed", "failed"]
    assert [citation.source_id for citation in result.citations] == ["D1", "D2"]
    assert [citation.source_id for citation in result.steps[1].citations] == ["D2", "D1"]
    assert "fees paid [D2]" in result.steps[1].summary
    assert "30 days [D1]" in result.steps[1].summary
    assert result.steps[1].output["tool_calls"][0]["result"]["citation"] == "D2"
    assert result.steps[1].evidence["claims"][0]["citations"] == ["D2", "D1"]
    assert "[D1]" in result.report and "[D2]" in result.report and "[D99]" not in result.report
    assert len(repair_calls) == 1
    assert result.metadata["runtime"] == "react_langgraph_bounded_v1"
    assert "RAW_TRACE_MARKER" not in str(synthesis_inputs[0])
    assert "Payment is due within 30 days" not in calls[2][0]
    assert {call[1]["conversation_id"] for call in calls} == {
        "conversation-1:step-1",
        "conversation-1:step-2",
        "conversation-1:step-3",
    }
    step_events = [event for event in events if event["event_type"].startswith("step_")]
    assert [(event["event_type"], event["step_id"]) for event in step_events] == [
        ("step_started", "step-1"),
        ("step_completed", "step-1"),
        ("step_started", "step-2"),
        ("step_completed", "step-2"),
        ("step_started", "step-3"),
        ("step_failed", "step-3"),
    ]
    assert result.status == "needs_human_review"


def test_complex_objective_uses_bounded_structured_planner(monkeypatch) -> None:
    model = AgentToolModel(
        plan_result={
            "steps": [
                {"title": "付款", "instruction": "审查付款义务。"},
                {"title": "终止", "instruction": "审查终止权。"},
            ]
        }
    )
    qa_service = DocumentQAService(vector_store=StaticVectorStore(), chat_model=model)

    class FakeToolCallingChatService:
        def __init__(self, _qa_service) -> None:
            pass

        def ask(self, _question: str, **_kwargs) -> ToolCallingAnswer:
            return ToolCallingAnswer(content="Relevant source text was not found for this step.")

    monkeypatch.setattr(_planned_task, "ToolCallingChatService", FakeToolCallingChatService)
    monkeypatch.setattr(
        qa_service,
        "_invoke_chat_messages",
        lambda _messages: "Relevant source text was not found for the planned review.",
    )

    result = run_react_agent_task(
        qa_service,
        objective="审查合同中的付款义务以及终止权风险。",
        task_id="task-planned",
    )

    assert [step.step_id for step in result.steps] == ["step-1", "step-2"]
    assert [step.title for step in result.steps] == ["付款", "终止"]
    assert result.metadata["planning_mode"] == "planner"
    assert model.structured_invocations == 1


def test_l2_focus_plan_is_bounded_to_five_steps() -> None:
    steps, planning_mode = react_task._plan_task(
        DocumentQAService(vector_store=StaticVectorStore(), chat_model=AgentToolModel()),
        objective="Review all requested areas.",
        focus_areas=[f"area-{index}" for index in range(7)],
        user_role="lawyer",
    )

    assert planning_mode == "focus_areas"
    assert len(steps) == 5


def test_planner_failure_falls_back_to_single_react() -> None:
    model = AgentToolModel(
        tool_name="search_documents",
        tool_args={"query": "付款 终止", "top_k": 1},
        structured_error=True,
    )
    result = run_react_agent_task(
        DocumentQAService(vector_store=StaticVectorStore(), chat_model=model),
        objective="审查合同中的付款义务以及终止权风险。",
    )

    assert [step.step_id for step in result.steps] == ["react"]
    assert result.metadata["runtime"] == "react_langgraph_v1"
    assert model.structured_invocations == 1


def test_agent_checkpoint_persists_interrupt_and_resumes_after_reopen(tmp_path) -> None:
    checkpoint_path = tmp_path / "agent_checkpoints.sqlite3"
    qa_service = DocumentQAService(vector_store=StaticVectorStore(), chat_model=AgentToolModel())

    connection, checkpointer = _sqlite_checkpointer(checkpoint_path)
    try:
        paused = run_react_agent_task(
            qa_service,
            objective="帮我看看",
            task_id="checkpoint-task",
            checkpointer=checkpointer,
        )
    finally:
        connection.close()

    assert isinstance(paused, AgentTaskPause)
    assert paused.questions

    connection, checkpointer = _sqlite_checkpointer(checkpoint_path)
    try:
        result = run_react_agent_task(
            qa_service,
            objective="Review termination risk in the SaaS agreement.",
            focus_areas=["termination"],
            task_id="checkpoint-task",
            checkpointer=checkpointer,
        )
    finally:
        connection.close()

    assert result.status == "completed"
    assert result.metadata["checkpointing"] is True
    with sqlite3.connect(checkpoint_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0


def test_planned_checkpoint_skips_completed_steps_after_worker_restart(
    tmp_path, monkeypatch
) -> None:
    calls: list[str] = []
    interrupted = False

    class InterruptingToolCallingService:
        def __init__(self, _qa_service) -> None:
            pass

        def ask(self, _question: str, **kwargs) -> ToolCallingAnswer:
            nonlocal interrupted
            step_id = str(kwargs["task_id"]).rsplit(":", 1)[-1]
            calls.append(step_id)
            if step_id == "step-2" and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            return ToolCallingAnswer(content="No material issue found for this step.")

    qa_service = DocumentQAService(vector_store=StaticVectorStore(), chat_model=AgentToolModel())
    monkeypatch.setattr(_planned_task, "ToolCallingChatService", InterruptingToolCallingService)
    monkeypatch.setattr(
        qa_service,
        "_invoke_chat_messages",
        lambda _messages: "No material issue was found in the planned review.",
    )
    checkpoint_path = tmp_path / "planned_checkpoints.sqlite3"

    connection, checkpointer = _sqlite_checkpointer(checkpoint_path)
    try:
        with pytest.raises(KeyboardInterrupt):
            run_react_agent_task(
                qa_service,
                objective="Review payment and termination risk.",
                focus_areas=["payment", "termination"],
                task_id="restart-task",
                checkpointer=checkpointer,
            )
    finally:
        connection.close()

    connection, checkpointer = _sqlite_checkpointer(checkpoint_path)
    try:
        result = run_react_agent_task(
            qa_service,
            objective="Review payment and termination risk.",
            focus_areas=["payment", "termination"],
            task_id="restart-task",
            checkpointer=checkpointer,
        )
    finally:
        connection.close()

    assert result.status in {"completed", "needs_human_review"}
    assert calls == ["step-1", "step-2", "step-2"]


def test_clarification_questions_detect_underspecified_task() -> None:
    questions = clarification_questions_for_task("帮我看看", [])

    assert questions
    assert len(questions) <= 3
    assert "具体任务" in questions[0]


def test_clarification_questions_allow_specific_contract_review() -> None:
    questions = clarification_questions_for_task(
        "审查这份 SaaS MSA 的终止、付款和责任限制风险，并给出律师问题清单。",
        ["termination", "payment", "liability limitation"],
    )

    assert questions == []

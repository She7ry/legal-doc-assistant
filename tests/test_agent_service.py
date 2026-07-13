from __future__ import annotations

from dataclasses import replace
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from doc_assistant.agent import LegalAgentService, clarification_questions_for_task, react_task
from doc_assistant.schemas.citation import Citation, QAAnswer
from doc_assistant.services.qa_service import DocumentQAService
from doc_assistant.services.tool_calling_service import ToolCallingAnswer, ToolCallTrace


class StaticVectorStore:
    tenant_id = "default"

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
    structured_result: dict[str, Any] = Field(default_factory=dict)
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
            result = (
                self.plan_result
                if schema.__name__ == "AgentPlanOutput" and self.plan_result is not None
                else self.structured_result
            )
            return schema.model_validate(result)

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
    model = AgentToolModel(
        structured_result={
            "matter_profile": {
                "document_type": "SaaS agreement",
                "parties": ["VendorCo", "CustomerCo"],
                "governing_law": "New York",
                "review_scope": ["termination"],
                "open_questions": [],
            },
            "findings": [
                {
                    "finding_id": "f1",
                    "category": "termination",
                    "severity": "Medium",
                    "summary": "Termination requires advance written notice.",
                    "recommended_action": "Confirm the notice procedure.",
                    "citations": ["D1", "invented", "d1"],
                    "needs_human_review": True,
                }
            ],
            "artifacts": [
                {
                    "artifact_id": "risk_matrix",
                    "artifact_type": "risk_matrix",
                    "title": "Termination risk matrix",
                    "summary": "Structured termination risk.",
                    "items": [{"finding_id": "f1", "risk": "Notice required"}],
                    "source_finding_ids": ["f1"],
                    "citations": ["invented", "D1"],
                }
            ],
        }
    )
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
    agent = LegalAgentService(qa_service)

    result = agent.run_task(
        objective="Review termination risk in the SaaS agreement.",
        focus_areas=["termination"],
        user_role="lawyer",
        max_steps=4,
        matter_id="matter-saas-1",
        progress_callback=lambda **event: events.append(event),
    )

    assert result.status == "completed"
    assert [step.step_id for step in result.steps] == ["react"]
    assert result.report == "Termination requires 30 days written notice [D1]."
    assert result.steps[0].output["tool_calls"][0]["name"] == "review_clause"
    assert result.steps[0].output["tool_calls"][0]["result"]["metadata"]["risk_reasons"][0][
        "citation"
    ] == "D1"
    assert result.citations[0].source_id == "D1"
    assert result.matter_profile == {
        "matter_id": "matter-saas-1",
        "document_type": "SaaS agreement",
        "parties": ["VendorCo", "CustomerCo"],
        "governing_law": "New York",
        "review_scope": ["termination"],
    }
    assert result.findings[0]["citations"] == ["D1"]
    assert result.findings[0]["source_quote"] == (
        "The customer may terminate with 30 days written notice."
    )
    assert result.findings[0]["location_label"] == " (page 1, chunk 1)"
    assert result.artifacts[0]["citations"] == ["D1"]
    assert model.structured_invocations == 1
    assert "Review termination risk in the SaaS agreement." in str(model.structured_inputs[0])
    assert "saas-msa.pdf (page 1, chunk 1)" in str(model.structured_inputs[0])
    assert qa_service.vector_store.queries == []
    assert result.metadata["runtime"] == "react_tool_calling_v1"
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

    monkeypatch.setattr(react_task, "ToolCallingChatService", FakeToolCallingChatService)
    monkeypatch.setattr(react_task, "_extract_matter", lambda *_args, **_kwargs: (None, [], []))
    monkeypatch.setattr(qa_service, "_invoke_chat_messages", synthesize)
    monkeypatch.setattr(qa_service, "repair_content", repair)

    result = LegalAgentService(qa_service).run_task(
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
                {"title": "Payment", "instruction": "Review payment obligations."},
                {"title": "Termination", "instruction": "Review termination rights."},
            ]
        }
    )
    qa_service = DocumentQAService(vector_store=StaticVectorStore(), chat_model=model)

    class FakeToolCallingChatService:
        def __init__(self, _qa_service) -> None:
            pass

        def ask(self, _question: str, **_kwargs) -> ToolCallingAnswer:
            return ToolCallingAnswer(content="Relevant source text was not found for this step.")

    monkeypatch.setattr(react_task, "ToolCallingChatService", FakeToolCallingChatService)
    monkeypatch.setattr(react_task, "_extract_matter", lambda *_args, **_kwargs: (None, [], []))
    monkeypatch.setattr(
        qa_service,
        "_invoke_chat_messages",
        lambda _messages: "Relevant source text was not found for the planned review.",
    )

    result = LegalAgentService(qa_service).run_task(
        objective="Review payment and termination risk.",
        task_id="task-planned",
    )

    assert [step.step_id for step in result.steps] == ["step-1", "step-2"]
    assert [step.title for step in result.steps] == ["Payment", "Termination"]
    assert result.metadata["planning_mode"] == "planner"
    assert model.structured_invocations == 1


def test_planner_failure_falls_back_to_single_react() -> None:
    model = AgentToolModel(
        tool_name="search_documents",
        tool_args={"query": "payment termination", "top_k": 1},
        structured_error=True,
    )
    result = LegalAgentService(
        DocumentQAService(vector_store=StaticVectorStore(), chat_model=model)
    ).run_task(objective="Review payment and termination risk.")

    assert [step.step_id for step in result.steps] == ["react"]
    assert result.metadata["runtime"] == "react_tool_calling_v1"
    assert model.structured_invocations == 2


def test_legal_agent_keeps_report_when_matter_structured_output_fails(caplog) -> None:
    model = AgentToolModel(
        tool_name="search_documents",
        tool_args={"query": "termination notice", "top_k": 1},
        structured_error=True,
    )
    vector_store = StaticVectorStore()
    agent = LegalAgentService(DocumentQAService(vector_store=vector_store, chat_model=model))

    with caplog.at_level("WARNING", logger="doc_assistant.agent.react_task"):
        result = agent.run_task(
            objective="Review termination notice requirements.",
            max_steps=3,
        )

    assert result.status == "completed"
    assert result.report == "Termination requires 30 days written notice [D1]."
    assert result.matter_profile is None
    assert result.findings == []
    assert result.artifacts == []
    assert model.structured_invocations == 1
    assert vector_store.queries == [("termination notice", 1)]
    assert "Matter structured output failed; returning no Matter" in caplog.text


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

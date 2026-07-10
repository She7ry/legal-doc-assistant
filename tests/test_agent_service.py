from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from doc_assistant.agent import LegalAgentService, clarification_questions_for_task
from doc_assistant.schemas.citation import Citation, QAAnswer
from doc_assistant.services.qa_service import DocumentQAService


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

    @property
    def _llm_type(self) -> str:
        return "agent-react-test"

    def bind_tools(self, tools, **kwargs):
        del kwargs
        self.bound_tool_names = [tool.name for tool in tools]
        return self

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
        progress_callback=lambda **event: events.append(event),
    )

    assert result.status == "completed"
    assert result.report == "Termination requires 30 days written notice [D1]."
    assert result.steps[0].output["tool_calls"][0]["name"] == "review_clause"
    assert result.steps[0].output["tool_calls"][0]["result"]["metadata"]["risk_reasons"][0][
        "citation"
    ] == "D1"
    assert result.citations[0].source_id == "D1"
    assert result.findings == []
    assert result.artifacts == []
    assert result.metadata["runtime"] == "react_tool_calling_v1"
    assert [event["event_type"] for event in events] == ["react_started", "react_completed"]


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

from __future__ import annotations

import threading
from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from doc_assistant.schemas.citation import Citation, QAAnswer
from doc_assistant.services.agent._constants import clarification_questions_for_task
from doc_assistant.services.agent._helpers import _CitationRegistry
from doc_assistant.services.agent.schemas import AgentPlanStep
from doc_assistant.services.agent_service import LegalAgentService
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


class ConcurrentReviewQAService:
    tenant_id = "default"

    def __init__(self) -> None:
        self._barrier = threading.Barrier(2)
        self._lock = threading.Lock()
        self._active_review_calls = 0
        self.max_active_review_calls = 0

    def ask(self, *_args, **_kwargs) -> QAAnswer:
        return QAAnswer(
            content="The document is a SaaS agreement between VendorCo and CustomerCo [S1].",
            citations=[
                Citation(
                    source_id="S1",
                    file_name="saas-msa.pdf",
                    preview="SaaS agreement between VendorCo and CustomerCo.",
                    exact_quote="SaaS agreement between VendorCo and CustomerCo.",
                )
            ],
        )

    def review_clause(self, clause_type: str, top_k: int | None = None) -> QAAnswer:
        del top_k
        with self._lock:
            self._active_review_calls += 1
            self.max_active_review_calls = max(
                self.max_active_review_calls,
                self._active_review_calls,
            )
        try:
            self._barrier.wait(timeout=2)
        finally:
            with self._lock:
                self._active_review_calls -= 1

        return QAAnswer(
            content=f"{clause_type} needs review [S1].",
            citations=[
                Citation(
                    source_id="S1",
                    file_name=f"{clause_type}.pdf",
                    preview=f"{clause_type} clause.",
                    exact_quote=f"{clause_type} clause.",
                )
            ],
            metadata={
                "clause_type": clause_type,
                "risk_level": "Medium",
                "risk_reasons": [
                    {
                        "reason": f"{clause_type} risk requires review.",
                        "citation": "S1",
                    }
                ],
                "questions_for_lawyer": [],
                "missing_information": [],
                "needs_human_review": True,
            },
        )


class RecordingAgentQAService:
    tenant_id = "default"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def ask(self, question: str, **kwargs) -> QAAnswer:
        self.calls.append(
            {
                "question": question,
                "chat_history": kwargs.get("chat_history") or [],
                "merge_persisted_history": kwargs.get("merge_persisted_history"),
            }
        )
        return QAAnswer(content=f"Answer for {question}.")


class ReactEvidenceRepairQAService:
    tenant_id = "default"

    def __init__(self) -> None:
        self.ask_calls: list[str] = []

    def ask(self, question: str, **_kwargs) -> QAAnswer:
        self.ask_calls.append(question)
        if question.startswith("Identify the document type"):
            return QAAnswer(
                content=(
                    "The document is a SaaS agreement between VendorCo and CustomerCo. "
                    "New York law governs it [S1]."
                ),
                citations=[
                    Citation(
                        source_id="S1",
                        file_name="saas-msa.pdf",
                        preview=(
                            "This SaaS agreement is between VendorCo and CustomerCo. "
                            "New York law governs it."
                        ),
                        exact_quote=(
                            "This SaaS agreement is between VendorCo and CustomerCo. "
                            "New York law governs it."
                        ),
                    )
                ],
            )
        return QAAnswer(
            content="The customer must give 30 days written notice before termination [S1].",
            citations=[
                Citation(
                    source_id="S1",
                    file_name="saas-msa.pdf",
                    preview="The customer must give 30 days written notice before termination.",
                    exact_quote="The customer must give 30 days written notice before termination.",
                )
            ],
        )

    def review_clause(self, clause_type: str, top_k: int | None = None) -> QAAnswer:
        del clause_type, top_k
        return QAAnswer(
            content="The customer must give 30 days written notice before termination.",
            metadata={
                "clause_type": "termination",
                "risk_level": "Medium",
                "risk_reasons": [
                    {
                        "reason": "The customer must give 30 days written notice before termination.",
                        "citation": None,
                    }
                ],
                "questions_for_lawyer": [],
                "missing_information": [],
                "needs_human_review": False,
            },
        )


def test_legal_agent_runs_planned_clause_review_with_citation_trace() -> None:
    vector_store = StaticVectorStore()
    qa_service = DocumentQAService(
        vector_store=vector_store,
        chat_model=FakeListChatModel(
            responses=[
                (
                    "The document is a SaaS agreement between VendorCo and CustomerCo, "
                    "and New York law governs it [S1]."
                ),
                """
                {
                  "clause_type": "termination",
                  "normalized_clause_type": "termination",
                  "found": true,
                  "summary": "The customer may terminate with 30 days written notice.",
                  "risk_level": "Medium",
                  "risk_reasons": [
                    {
                      "reason": "The customer must provide 30 days written notice before termination.",
                      "citation": "S1"
                    }
                  ],
                  "affected_party": "Customer",
                  "plain_language_explanation": "The customer can end the agreement after notice.",
                  "questions_for_lawyer": [
                    "Should termination rights be mutual?"
                  ],
                  "missing_information": [],
                  "needs_human_review": true
                }
                """,
            ]
        ),
    )
    agent = LegalAgentService(qa_service)

    result = agent.run_task(
        objective="Review termination risk in the SaaS agreement.",
        focus_areas=["termination"],
        user_role="lawyer",
        max_steps=4,
    )

    assert result.status == "needs_human_review"
    assert [step.tool for step in result.plan] == [
        "document_qa",
        "review_clause",
        "synthesize_report",
    ]
    assert result.findings[0].category == "termination"
    assert result.findings[0].citations == ["S2"]
    assert result.findings[0].support_level == "direct"
    assert result.findings[0].evidence_coverage == "direct"
    assert result.findings[0].source_quote
    assert result.findings[0].location_label
    assert result.findings[0].human_review_status == "pending"
    assert result.citations[0].source_id == "S1"
    assert result.citations[1].source_id == "S2"
    assert result.matter_profile is not None
    assert result.matter_profile.document_type == "SaaS agreement"
    assert result.matter_profile.parties == ["VendorCo", "CustomerCo"]
    assert result.matter_profile.governing_law == "New York"
    assert result.matter_profile.review_scope == ["termination"]
    assert [artifact.artifact_type for artifact in result.artifacts] == [
        "risk_matrix",
        "lawyer_questions",
        "negotiation_checklist",
        "obligation_calendar",
    ]
    assert result.artifacts[0].items[0]["finding_id"] == "f1"
    assert result.artifacts[1].items
    assert result.artifacts[3].items[0]["deadline"] == "30 days"
    assert result.confirmation_gates
    assert [gate.gate_id for gate in result.confirmation_gates] == [
        "confirm_user_side",
        "resolve_missing_information",
        "review_high_risk_findings",
        "resolve_evidence_guard",
        "approve_report_use",
    ]
    assert result.matter_profile.confirmation_gates[0]["gate_id"] == "confirm_user_side"
    assert "Review termination" in result.report
    assert "## Matter profile" in result.report
    assert "## Artifacts" in result.report
    assert "## Confirmation gates" in result.report
    assert "[S2]" in result.report
    assert result.metadata["executor"] == "plan_react_v1"
    assert vector_store.queries


def test_legal_agent_react_repairs_uncited_clause_review_with_followup_evidence() -> None:
    qa_service = ReactEvidenceRepairQAService()
    agent = LegalAgentService(qa_service)  # type: ignore[arg-type]

    result = agent.run_task(
        objective="Review termination risk in the SaaS agreement.",
        focus_areas=["termination"],
        user_role="lawyer",
        max_steps=4,
    )

    review_step = next(step for step in result.steps if step.tool == "review_clause")
    react_trace = review_step.output["react_trace"]

    assert len(qa_service.ask_calls) == 2
    assert "Observed evidence gap" in qa_service.ask_calls[1]
    assert react_trace[0]["action"]["tool"] == "document_qa"
    assert review_step.citations[0].source_id == "S2"
    assert not any(
        item.startswith("No cited document evidence was found")
        for item in result.missing_information
    )
    assert result.findings[0].citations == ["S2"]
    assert result.findings[0].support_level == "direct"


def test_legal_agent_runs_independent_clause_reviews_in_parallel(monkeypatch) -> None:
    from doc_assistant.services.agent import _planning

    monkeypatch.setattr(
        _planning,
        "settings",
        SimpleNamespace(agent_max_parallel_steps=2),
    )
    qa_service = ConcurrentReviewQAService()
    agent = LegalAgentService(qa_service)  # type: ignore[arg-type]

    result = agent.run_task(
        objective="Review payment and termination risks in the SaaS agreement.",
        focus_areas=["payment", "termination"],
        user_role="lawyer",
        max_steps=5,
    )

    assert qa_service.max_active_review_calls == 2
    assert [step.step_id for step in result.steps] == [
        "profile",
        "review_1",
        "review_2",
        "report",
    ]
    assert [citation.file_name for citation in result.citations[1:3]] == [
        "payment.pdf",
        "termination.pdf",
    ]


def test_legal_agent_passes_step_history_between_qa_steps(monkeypatch) -> None:
    from doc_assistant.services.agent import _react

    monkeypatch.setattr(
        _react,
        "settings",
        SimpleNamespace(agent_react_enabled=False),
    )
    qa_service = RecordingAgentQAService()
    agent = LegalAgentService(qa_service)  # type: ignore[arg-type]
    plan = [
        AgentPlanStep(
            step_id="profile",
            title="Build profile",
            purpose="Profile the matter.",
            tool="document_qa",
            arguments={"question": "Build the matter profile."},
        ),
        AgentPlanStep(
            step_id="calendar",
            title="Build calendar",
            purpose="Extract obligations.",
            tool="create_obligation_calendar",
            arguments={"query": "key dates"},
        ),
    ]

    agent._execute_plan_steps(
        plan,
        objective="Review key dates.",
        user_id=None,
        conversation_id=None,
        task_id="task-a",
        citation_registry=_CitationRegistry(),
        progress_callback=None,
    )

    assert qa_service.calls[0]["chat_history"][0]["content"] == "Agent objective: Review key dates."
    assert qa_service.calls[0]["merge_persisted_history"] is False
    assert "Completed agent step 'Build profile'" in qa_service.calls[1]["chat_history"][-1]["content"]
    assert qa_service.calls[1]["merge_persisted_history"] is False


def test_legal_agent_plans_conflict_check_for_policy_tasks() -> None:
    qa_service = DocumentQAService(
        vector_store=StaticVectorStore(),
        chat_model=FakeListChatModel(responses=["Profile [S1]."]),
    )
    agent = LegalAgentService(qa_service)

    plan = agent.plan_task(
        objective="Compare the contract with company policy for conflicts.",
        focus_areas=["payment"],
        user_role="ordinary",
        max_steps=5,
    )

    assert [step.tool for step in plan] == [
        "document_qa",
        "review_clause",
        "check_conflict",
        "synthesize_report",
    ]


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

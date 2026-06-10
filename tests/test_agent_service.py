from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from doc_assistant.services.agent_service import (
    LegalAgentService,
    clarification_questions_for_task,
)
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
    assert vector_store.queries


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

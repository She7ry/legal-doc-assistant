from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable
from uuid import uuid4

from doc_assistant.schemas.citation import Citation, QAAnswer
from doc_assistant.services.answer_guard import validate_answer
from doc_assistant.services.evidence import build_evidence_profile
from doc_assistant.services.qa_service import DocumentQAService

SOURCE_REF_PATTERN = re.compile(r"\[([SCDPW]\d+)\]", re.IGNORECASE)
BARE_SOURCE_REF_PATTERN = re.compile(r"(?<![A-Za-z0-9])([SCDPW]\d+)(?![A-Za-z0-9])", re.IGNORECASE)
ProgressCallback = Callable[..., None]

DEFAULT_FOCUS_AREAS = [
    "payment",
    "termination",
    "liability limitation",
    "confidentiality",
    "data privacy",
]

FOCUS_KEYWORDS = {
    "payment": ("payment", "fee", "invoice"),
    "termination": ("termination", "renewal", "cancel"),
    "liability limitation": ("liability", "damages", "cap"),
    "indemnification": ("indemnity", "indemnification", "hold harmless"),
    "confidentiality": ("confidential", "nda", "non-disclosure"),
    "data privacy": ("privacy", "data", "security", "personal information"),
    "governing law": ("governing law", "jurisdiction", "venue"),
    "assignment": ("assignment", "transfer"),
    "audit rights": ("audit", "inspection"),
}

AGENT_TOOL_REGISTRY: dict[str, dict[str, str]] = {
    "document_qa": {
        "label": "Document QA",
        "description": "Answer a focused question against uploaded documents with citations.",
    },
    "review_clause": {
        "label": "Clause review",
        "description": "Assess a clause type and produce structured risk reasons.",
    },
    "check_conflict": {
        "label": "Conflict check",
        "description": "Compare contract and policy excerpts for inconsistent obligations.",
    },
    "extract_parties_dates_jurisdiction": {
        "label": "Matter fact extraction",
        "description": "Extract parties, dates, governing law, jurisdiction, and missing facts.",
    },
    "compare_document_versions": {
        "label": "Version comparison",
        "description": "Compare versions or drafts and summarize changed legal positions.",
    },
    "create_obligation_calendar": {
        "label": "Obligation calendar",
        "description": "Extract obligations, triggers, owners, deadlines, and source citations.",
    },
    "suggest_clause_revision": {
        "label": "Clause revision",
        "description": "Suggest citation-grounded drafting changes for a reviewed clause.",
    },
    "build_evidence_profile": {
        "label": "Evidence profile",
        "description": "Audit claims for source citations, quotes, support, and gaps.",
    },
    "generate_negotiation_checklist": {
        "label": "Negotiation checklist",
        "description": "Turn findings into negotiation asks, fallback positions, and priorities.",
    },
    "synthesize_report": {
        "label": "Report synthesis",
        "description": "Compile the final task report with gates and artifacts.",
    },
}

VERSION_COMPARE_KEYWORDS = (
    "compare versions",
    "version comparison",
    "redline",
    "draft comparison",
    "changed",
    "changes between",
    "版本",
    "对比",
    "修订",
)
OBLIGATION_CALENDAR_KEYWORDS = (
    "obligation calendar",
    "calendar",
    "deadline",
    "due date",
    "key dates",
    "义务日历",
    "期限",
    "截止",
)
CLAUSE_REVISION_KEYWORDS = (
    "revise",
    "rewrite",
    "redraft",
    "clause language",
    "suggest language",
    "改写",
    "修改条款",
    "修订条款",
)
NEGOTIATION_KEYWORDS = (
    "negotiate",
    "negotiation",
    "fallback",
    "position",
    "谈判",
    "谈判清单",
)

GENERIC_OBJECTIVE_PATTERNS = {
    "帮我看看",
    "帮我看下",
    "看看合同",
    "看一下合同",
    "看下合同",
    "审查合同",
    "审核合同",
    "看看文件",
    "看下文件",
    "reviewthis",
    "reviewcontract",
    "checkthis",
    "checkcontract",
    "analyzethis",
}
DEADLINE_KEYWORDS = (
    "urgent",
    "asap",
    "deadline",
    "due",
    "expire",
    "expires",
    "expiring",
    "紧急",
    "尽快",
    "截止",
    "期限",
    "到期",
    "马上",
)
CURRENT_LAW_KEYWORDS = (
    "current law",
    "latest law",
    "up-to-date law",
    "statute",
    "regulation",
    "compliance",
    "legal requirement",
    "legal authority",
    "is this legal",
    "现行法律",
    "最新法律",
    "法规",
    "监管",
    "合规",
    "法条",
    "法律依据",
    "是否合法",
    "法定",
)
JURISDICTION_INDICATORS = (
    "new york",
    "california",
    "delaware",
    "united states",
    "u.s.",
    "usa",
    "china",
    "prc",
    "hong kong",
    "singapore",
    "england",
    "wales",
    "eu",
    "european union",
    "中国",
    "美国",
    "英国",
    "香港",
    "新加坡",
    "纽约",
    "加州",
    "特拉华",
    "内地",
    "大陆",
    "欧盟",
    "北京",
    "上海",
    "广东",
    "深圳",
    "江苏",
    "浙江",
)
PARTY_SENSITIVE_KEYWORDS = (
    "lawsuit",
    "sued",
    "court case",
    "arbitration demand",
    "claim against",
    "eviction",
    "employment termination",
    "被起诉",
    "起诉",
    "法院案件",
    "劳动仲裁",
    "被辞退",
    "解雇",
    "驱逐",
    "退租纠纷",
    "合同纠纷",
)
PARTY_SIDE_INDICATORS = (
    "plaintiff",
    "defendant",
    "buyer",
    "seller",
    "customer",
    "vendor",
    "employer",
    "employee",
    "landlord",
    "tenant",
    "甲方",
    "乙方",
    "原告",
    "被告",
    "买方",
    "卖方",
    "客户",
    "供应商",
    "雇主",
    "员工",
    "房东",
    "租客",
)
DATE_OR_TIME_PATTERN = re.compile(
    r"(\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|"
    r"\b\d{1,2}[-/.]\d{1,2}\b|"
    r"\d{4}年\d{1,2}月\d{1,2}日|"
    r"\d+\s*(business\s+days?|days?|hours?|weeks?|工作日|小时|天|日|周)|"
    r"today|tomorrow|next\s+week|今天|明天|后天|本周|下周|月底|月末)",
    re.IGNORECASE,
)


def clarification_questions_for_task(
    objective: str,
    focus_areas: list[str] | None = None,
) -> list[str]:
    """Return blocking clarification questions for underspecified Agent tasks."""
    text = _clean_text(objective)
    lowered = text.casefold()
    normalized_focus = [_clean_text(area) for area in focus_areas or [] if _clean_text(area)]
    questions: list[str] = []

    if _looks_underspecified_objective(text, normalized_focus):
        questions.append(
            "请说明你希望 Agent 完成的具体任务，例如风险审查、条款解释、谈判清单或律师问题清单。"
        )

    if _mentions_any(lowered, DEADLINE_KEYWORDS) and not DATE_OR_TIME_PATTERN.search(text):
        questions.append("请补充具体截止时间或希望在什么日期前完成处理。")

    if _mentions_any(lowered, CURRENT_LAW_KEYWORDS) and not _mentions_any(
        lowered,
        JURISDICTION_INDICATORS,
    ):
        questions.append("请补充适用法域或地域，例如国家、州/省、市，或合同中的 governing law。")

    if _mentions_any(lowered, PARTY_SENSITIVE_KEYWORDS) and not _mentions_any(
        lowered,
        PARTY_SIDE_INDICATORS,
    ):
        questions.append(
            "请说明你在事项中的立场或代表方，例如甲方/乙方、买方/卖方、雇主/员工、房东/租客。"
        )

    return questions[:3]


@dataclass(frozen=True)
class AgentPlanStep:
    step_id: str
    title: str
    purpose: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False


@dataclass(frozen=True)
class AgentStepResult:
    step_id: str
    title: str
    tool: str
    status: str
    summary: str
    citations: list[Citation] = field(default_factory=list)
    evidence: dict[str, Any] | None = None
    guard_warnings: list[str] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentFinding:
    finding_id: str
    category: str
    severity: str
    summary: str
    citations: list[str] = field(default_factory=list)
    recommended_action: str = ""
    needs_human_review: bool = True
    source_step_id: str = ""
    clause_reference: str = ""
    evidence_coverage: str = "missing"
    support_level: str = "missing"
    unsupported_reason: str = ""
    source_quote: str = ""
    location_label: str = ""
    human_review_status: str = "pending"
    status: str = "open"
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class MatterProfile:
    matter_id: str
    document_type: str = "Unknown"
    parties: list[str] = field(default_factory=list)
    user_side: str = ""
    governing_law: str = ""
    jurisdiction: str = ""
    key_dates: list[dict[str, Any]] = field(default_factory=list)
    review_scope: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    confidence: str = "Low"
    citations: list[str] = field(default_factory=list)
    source_step_id: str = ""
    confirmation_gates: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AgentArtifact:
    artifact_id: str
    artifact_type: str
    title: str
    summary: str
    items: list[dict[str, Any]] = field(default_factory=list)
    source_finding_ids: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfirmationGate:
    gate_id: str
    gate_type: str
    title: str
    question: str
    status: str = "pending"
    priority: str = "normal"
    required: bool = True
    reason: str = ""
    related_finding_ids: list[str] = field(default_factory=list)
    related_artifact_ids: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentTaskResult:
    task_id: str
    status: str
    objective: str
    plan: list[AgentPlanStep]
    steps: list[AgentStepResult]
    findings: list[AgentFinding]
    missing_information: list[str]
    human_review_required: bool
    report: str
    citations: list[Citation]
    confidence: str | None = None
    guard_warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] | None = None
    matter_profile: MatterProfile | None = None
    artifacts: list[AgentArtifact] = field(default_factory=list)
    confirmation_gates: list[AgentConfirmationGate] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class LegalAgentService:
    """Task-oriented legal assistant workflow built on the citation-first QA service."""

    def __init__(self, qa_service: DocumentQAService) -> None:
        self.qa_service = qa_service

    def run_task(
        self,
        *,
        objective: str,
        focus_areas: list[str] | None = None,
        user_role: str = "ordinary",
        max_steps: int = 6,
        user_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        matter_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentTaskResult:
        resolved_task_id = task_id or uuid4().hex
        resolved_matter_id = matter_id or resolved_task_id
        plan = self.plan_task(
            objective=objective,
            focus_areas=focus_areas or [],
            user_role=user_role,
            max_steps=max_steps,
        )
        _emit_progress(
            progress_callback,
            event_type="plan_created",
            stage="planning",
            progress=10,
            message=f"Created a {len(plan)} step agent plan.",
            payload={"plan": [_plan_step_payload(step) for step in plan]},
        )
        citation_registry = _CitationRegistry()
        steps: list[AgentStepResult] = []
        findings: list[AgentFinding] = []
        missing_information: list[str] = []

        executable_steps = [step for step in plan if step.tool != "synthesize_report"]
        step_count = max(len(executable_steps), 1)
        for step_index, plan_step in enumerate(plan, start=1):
            if plan_step.tool != "synthesize_report":
                _emit_progress(
                    progress_callback,
                    event_type="step_started",
                    stage=plan_step.step_id,
                    progress=10 + int((step_index - 1) / step_count * 70),
                    message=f"Started step: {plan_step.title}",
                    step_id=plan_step.step_id,
                    payload={"step": _plan_step_payload(plan_step)},
                )
            step = self._execute_step(
                plan_step,
                objective=objective,
                user_id=user_id,
                conversation_id=conversation_id,
                task_id=resolved_task_id,
                citation_registry=citation_registry,
            )
            steps.append(step)
            findings.extend(self._findings_from_step(step))
            missing_information.extend(_dedupe_texts(step.output.get("missing_information", [])))
            if plan_step.tool != "synthesize_report":
                _emit_progress(
                    progress_callback,
                    event_type="step_completed",
                    stage=plan_step.step_id,
                    progress=15 + int(step_index / step_count * 70),
                    message=f"Completed step: {plan_step.title}",
                    step_id=plan_step.step_id,
                    payload={"step": _step_result_payload(step)},
                )

        findings = _audit_findings(_renumber_findings(findings), citation_registry.citations)
        missing_information = _dedupe_texts(missing_information)
        matter_profile = _build_matter_profile(
            matter_id=resolved_matter_id,
            objective=objective,
            review_scope=_review_scope_from_plan(plan),
            steps=steps,
            missing_information=missing_information,
        )
        missing_information = _dedupe_texts(
            [*missing_information, *matter_profile.open_questions]
        )
        artifacts = _build_agent_artifacts(
            matter_profile=matter_profile,
            findings=findings,
            steps=steps,
            missing_information=missing_information,
            user_role=user_role,
        )
        confirmation_gates = _build_confirmation_gates(
            objective=objective,
            matter_profile=matter_profile,
            findings=findings,
            missing_information=missing_information,
            guard_warnings=[],
            artifacts=artifacts,
            user_role=user_role,
        )
        matter_profile = replace(
            matter_profile,
            confirmation_gates=[
                _confirmation_gate_payload(gate) for gate in confirmation_gates
            ],
        )
        _emit_progress(
            progress_callback,
            event_type="report_started",
            stage="reporting",
            progress=90,
            message="Compiling the final agent report.",
        )
        report = self._render_report(
            objective=objective,
            user_role=user_role,
            steps=steps,
            findings=findings,
            missing_information=missing_information,
            matter_profile=matter_profile,
            artifacts=artifacts,
            confirmation_gates=confirmation_gates,
        )
        guard_result = validate_answer(
            report,
            citation_registry.citations,
            has_retrieved_documents=bool(citation_registry.citations),
        )
        evidence = build_evidence_profile(report, citation_registry.citations, guard_result.issues)
        if guard_result.issues:
            confirmation_gates = _build_confirmation_gates(
                objective=objective,
                matter_profile=matter_profile,
                findings=findings,
                missing_information=missing_information,
                guard_warnings=guard_result.issues,
                artifacts=artifacts,
                user_role=user_role,
            )
            matter_profile = replace(
                matter_profile,
                confirmation_gates=[
                    _confirmation_gate_payload(gate) for gate in confirmation_gates
                ],
            )
        human_review_required = (
            bool(missing_information)
            or any(finding.needs_human_review for finding in findings)
            or bool(guard_result.issues)
            or any(gate.required for gate in confirmation_gates)
        )
        status = "needs_human_review" if human_review_required else "completed"

        return AgentTaskResult(
            task_id=resolved_task_id,
            status=status,
            objective=objective,
            plan=plan,
            steps=steps,
            findings=findings,
            missing_information=missing_information,
            human_review_required=human_review_required,
            report=report,
            citations=citation_registry.citations,
            confidence=guard_result.confidence,
            guard_warnings=guard_result.issues,
            evidence=evidence,
            matter_profile=matter_profile,
            artifacts=artifacts,
            confirmation_gates=confirmation_gates,
            metadata={
                "user_role": user_role,
                "planner": "heuristic_v2",
                "tenant_id": self.qa_service.tenant_id,
                "workflow_type": _workflow_type(objective),
                "available_tools": sorted(AGENT_TOOL_REGISTRY),
            },
        )

    def plan_task(
        self,
        *,
        objective: str,
        focus_areas: list[str],
        user_role: str,
        max_steps: int,
    ) -> list[AgentPlanStep]:
        del user_role
        normalized_max_steps = max(3, min(max_steps, 10))
        workflow_type = _workflow_type(objective)
        wants_conflict = _looks_like_conflict_task(objective)
        special_tool_count = 0
        if workflow_type in {
            "version_comparison",
            "obligation_calendar",
            "evidence_audit",
        }:
            special_tool_count += 1
        if workflow_type in {"clause_revision", "negotiation_prep"}:
            special_tool_count += 1
        reserved_steps = 2 + (1 if wants_conflict else 0) + special_tool_count
        review_budget = max(1, normalized_max_steps - reserved_steps)
        resolved_focus_areas = _resolve_focus_areas(objective, focus_areas)[:review_budget]
        profile_tool = (
            "extract_parties_dates_jurisdiction"
            if workflow_type in {"version_comparison", "obligation_calendar", "evidence_audit"}
            else "document_qa"
        )

        plan = [
            AgentPlanStep(
                step_id="profile",
                title="Build matter profile",
                purpose=(
                    "Identify document type, parties, governing law, dates, "
                    "and immediate gaps."
                ),
                tool=profile_tool,
                arguments={
                    "question": (
                        "Identify the document type, parties, governing law or jurisdiction, "
                        "important dates, and missing context relevant to this task: "
                        f"{objective}"
                    )
                },
            )
        ]

        if workflow_type == "version_comparison":
            plan.append(
                AgentPlanStep(
                    step_id="version_compare",
                    title="Compare document versions",
                    purpose="Identify changed legal positions across drafts or versions.",
                    tool="compare_document_versions",
                    arguments={"query": objective, "top_k": 8},
                )
            )
        elif workflow_type == "obligation_calendar":
            plan.append(
                AgentPlanStep(
                    step_id="obligation_calendar",
                    title="Create obligation calendar",
                    purpose="Extract deadlines, triggers, owners, and follow-up obligations.",
                    tool="create_obligation_calendar",
                    arguments={"query": objective, "top_k": 8},
                )
            )
        else:
            for index, area in enumerate(resolved_focus_areas, start=1):
                plan.append(
                    AgentPlanStep(
                        step_id=f"review_{index}",
                        title=f"Review {area}",
                        purpose=(
                            f"Assess the {area} clause or issue and produce evidence-backed risks."
                        ),
                        tool="review_clause",
                        arguments={"clause_type": area, "top_k": 5},
                    )
                )

        if workflow_type == "clause_revision":
            target_clause = resolved_focus_areas[0] if resolved_focus_areas else "requested clause"
            plan.append(
                AgentPlanStep(
                    step_id="clause_revision",
                    title="Suggest clause revision",
                    purpose="Draft a safer clause position from the cited review evidence.",
                    tool="suggest_clause_revision",
                    arguments={"clause_type": target_clause, "objective": objective},
                    requires_confirmation=True,
                )
            )

        if workflow_type == "negotiation_prep":
            plan.append(
                AgentPlanStep(
                    step_id="negotiation_checklist",
                    title="Generate negotiation checklist",
                    purpose="Turn reviewed risks into asks, fallbacks, and priorities.",
                    tool="generate_negotiation_checklist",
                    arguments={"objective": objective},
                    requires_confirmation=True,
                )
            )

        if workflow_type == "evidence_audit":
            plan.append(
                AgentPlanStep(
                    step_id="evidence_profile",
                    title="Build evidence profile",
                    purpose="Audit cited claims and identify unsupported statements.",
                    tool="build_evidence_profile",
                    arguments={"query": objective},
                )
            )

        if wants_conflict:
            plan.append(
                AgentPlanStep(
                    step_id="conflict_check",
                    title="Check document-policy conflicts",
                    purpose="Compare contract obligations with policy or compliance excerpts.",
                    tool="check_conflict",
                    arguments={
                        "contract_query": f"contract obligations {objective}",
                        "policy_query": f"policy compliance requirements {objective}",
                        "top_k": 5,
                    },
                )
            )

        plan.append(
            AgentPlanStep(
                step_id="report",
                title="Compile report",
                purpose=(
                    "Synthesize findings, evidence, missing information, "
                    "and human-review gates."
                ),
                tool="synthesize_report",
                arguments={},
            )
        )
        return _trim_plan(plan, normalized_max_steps)

    def _execute_step(
        self,
        plan_step: AgentPlanStep,
        *,
        objective: str,
        user_id: str | None,
        conversation_id: str | None,
        task_id: str,
        citation_registry: "_CitationRegistry",
    ) -> AgentStepResult:
        if plan_step.tool == "document_qa":
            answer = self.qa_service.ask(
                str(plan_step.arguments["question"]),
                user_id=user_id,
                conversation_id=conversation_id,
                task_id=task_id,
            )
            return self._answer_step(plan_step, answer, citation_registry)

        if plan_step.tool == "extract_parties_dates_jurisdiction":
            answer = self.qa_service.ask(
                str(plan_step.arguments["question"]),
                user_id=user_id,
                conversation_id=conversation_id,
                task_id=task_id,
            )
            return self._answer_step(plan_step, answer, citation_registry)

        if plan_step.tool == "review_clause":
            answer = self.qa_service.review_clause(
                clause_type=str(plan_step.arguments["clause_type"]),
                top_k=int(plan_step.arguments.get("top_k") or 5),
            )
            return self._answer_step(plan_step, answer, citation_registry)

        if plan_step.tool == "check_conflict":
            answer = self.qa_service.check_conflict(
                contract_query=str(plan_step.arguments["contract_query"]),
                policy_query=str(plan_step.arguments["policy_query"]),
                top_k=int(plan_step.arguments.get("top_k") or 5),
            )
            return self._answer_step(plan_step, answer, citation_registry)

        if plan_step.tool == "compare_document_versions":
            query = str(plan_step.arguments.get("query") or objective)
            answer = self.qa_service.ask(
                (
                    "Compare the available document versions or drafts relevant to this task. "
                    "Identify changed obligations, risk allocation, dates, parties, governing law, "
                    "and negotiation impact. Cite every changed position: "
                    f"{query}"
                ),
                user_id=user_id,
                conversation_id=conversation_id,
                task_id=task_id,
            )
            return self._answer_step(plan_step, answer, citation_registry)

        if plan_step.tool == "create_obligation_calendar":
            query = str(plan_step.arguments.get("query") or objective)
            answer = self.qa_service.ask(
                (
                    "Extract a structured obligation calendar from the cited documents. "
                    "For each item include obligation, trigger, deadline, owner if stated, "
                    "status, and source citation. If a field is not stated, say it is missing. "
                    f"Task: {query}"
                ),
                user_id=user_id,
                conversation_id=conversation_id,
                task_id=task_id,
            )
            return self._answer_step(plan_step, answer, citation_registry)

        if plan_step.tool == "suggest_clause_revision":
            clause_type = str(plan_step.arguments.get("clause_type") or "requested clause")
            answer = self.qa_service.ask(
                (
                    "Suggest a revised clause position for the requested legal issue. "
                    "Do not invent facts. Tie each drafting suggestion to the current cited clause "
                    "and flag points requiring lawyer approval. "
                    f"Clause type: {clause_type}. Task: {objective}"
                ),
                user_id=user_id,
                conversation_id=conversation_id,
                task_id=task_id,
            )
            return self._answer_step(plan_step, answer, citation_registry)

        if plan_step.tool == "build_evidence_profile":
            answer = self.qa_service.ask(
                (
                    "Build an evidence profile for the task. List material claims, source "
                    "citations, exact quoted support, support level, and unsupported reasons. "
                    f"Task: {objective}"
                ),
                user_id=user_id,
                conversation_id=conversation_id,
                task_id=task_id,
            )
            return self._answer_step(plan_step, answer, citation_registry)

        if plan_step.tool == "generate_negotiation_checklist":
            answer = self.qa_service.ask(
                (
                    "Generate a negotiation checklist from the cited contract excerpts. "
                    "For each issue include the ask, fallback position, priority, owner, and "
                    "source citation. Flag any item requiring lawyer approval. "
                    f"Task: {objective}"
                ),
                user_id=user_id,
                conversation_id=conversation_id,
                task_id=task_id,
            )
            return self._answer_step(plan_step, answer, citation_registry)

        if plan_step.tool == "synthesize_report":
            return AgentStepResult(
                step_id=plan_step.step_id,
                title=plan_step.title,
                tool=plan_step.tool,
                status="completed",
                summary=f"Prepared the final report for: {objective}",
                output={},
            )

        return AgentStepResult(
            step_id=plan_step.step_id,
            title=plan_step.title,
            tool=plan_step.tool,
            status="failed",
            summary=f"Unknown agent tool: {plan_step.tool}",
            output={"error": f"Unknown agent tool: {plan_step.tool}"},
        )

    def _answer_step(
        self,
        plan_step: AgentPlanStep,
        answer: QAAnswer,
        citation_registry: "_CitationRegistry",
    ) -> AgentStepResult:
        citation_map, citations = citation_registry.add_step_citations(
            plan_step.step_id,
            answer.citations,
        )
        content = _remap_source_refs(answer.content, citation_map)
        metadata = _remap_metadata(answer.metadata, citation_map)
        evidence = metadata.get("evidence")
        if isinstance(evidence, dict):
            evidence = _remap_metadata(evidence, citation_map)
        elif answer.citations:
            evidence = build_evidence_profile(content, citations, answer.guard_warnings)

        missing_information = _metadata_missing_information(metadata)
        if not answer.citations and plan_step.tool != "synthesize_report":
            missing_information.append(
                f"No cited document evidence was found for step: {plan_step.title}."
            )

        status = "completed"
        if answer.guard_warnings or missing_information:
            status = "needs_review"

        return AgentStepResult(
            step_id=plan_step.step_id,
            title=plan_step.title,
            tool=plan_step.tool,
            status=status,
            summary=content,
            citations=citations,
            evidence=evidence if isinstance(evidence, dict) else None,
            guard_warnings=answer.guard_warnings,
            output={
                "metadata": metadata,
                "missing_information": _dedupe_texts(missing_information),
            },
        )

    def _findings_from_step(self, step: AgentStepResult) -> list[AgentFinding]:
        metadata = step.output.get("metadata", {})
        if not isinstance(metadata, dict):
            return []

        if step.tool == "review_clause":
            return self._clause_findings(step, metadata)
        if step.tool == "check_conflict":
            return self._conflict_findings(step, metadata)
        if step.tool in {
            "compare_document_versions",
            "build_evidence_profile",
            "suggest_clause_revision",
        }:
            return self._generic_findings(step)
        return []

    def _clause_findings(
        self,
        step: AgentStepResult,
        metadata: dict[str, Any],
    ) -> list[AgentFinding]:
        category = _clean_text(metadata.get("clause_type")) or step.title
        severity = _clean_text(metadata.get("risk_level")) or "Needs human review"
        needs_review = bool(metadata.get("needs_human_review", True))
        recommendations = _as_text_list(metadata.get("questions_for_lawyer"))
        default_citations = [citation.source_id for citation in step.citations[:1]]
        reasons = metadata.get("risk_reasons")
        findings: list[AgentFinding] = []

        if isinstance(reasons, list):
            for reason in reasons:
                if not isinstance(reason, dict):
                    continue
                summary = _clean_text(reason.get("reason"))
                if not summary:
                    continue
                citations = _source_id_list(reason.get("citation")) or default_citations
                findings.append(
                    AgentFinding(
                        finding_id=f"f{len(findings) + 1}",
                        category=category,
                        severity=severity,
                        summary=summary,
                        citations=citations,
                        recommended_action=_first_text(recommendations),
                        needs_human_review=needs_review,
                        source_step_id=step.step_id,
                    )
                )

        if findings:
            return _renumber_findings(findings)

        summary = _clean_text(metadata.get("summary"))
        if not summary:
            return []
        return [
            AgentFinding(
                finding_id="f1",
                category=category,
                severity=severity,
                summary=summary,
                citations=default_citations,
                recommended_action=_first_text(recommendations),
                needs_human_review=needs_review,
                source_step_id=step.step_id,
            )
        ]

    def _conflict_findings(
        self,
        step: AgentStepResult,
        metadata: dict[str, Any],
    ) -> list[AgentFinding]:
        conflicts = metadata.get("conflicts")
        if not isinstance(conflicts, list):
            return []

        findings = []
        for conflict in conflicts:
            if not isinstance(conflict, dict):
                continue
            topic = _clean_text(conflict.get("topic")) or "Potential conflict"
            why_conflict = _clean_text(conflict.get("why_conflict"))
            if not why_conflict:
                continue
            citations = _source_id_list(conflict.get("contract_citations"))
            citations.extend(
                source_id
                for source_id in _source_id_list(conflict.get("policy_citations"))
                if source_id not in citations
            )
            findings.append(
                AgentFinding(
                    finding_id=f"f{len(findings) + 1}",
                    category=topic,
                    severity=_clean_text(conflict.get("severity")) or "Needs human review",
                    summary=why_conflict,
                    citations=citations,
                    recommended_action=_clean_text(conflict.get("recommended_action")),
                    needs_human_review=bool(conflict.get("needs_human_review", True)),
                    source_step_id=step.step_id,
                )
            )
        return _renumber_findings(findings)

    def _generic_findings(self, step: AgentStepResult) -> list[AgentFinding]:
        findings: list[AgentFinding] = []
        evidence = step.evidence if isinstance(step.evidence, dict) else {}
        claims = evidence.get("claims")
        if isinstance(claims, list):
            for claim in claims[:6]:
                if not isinstance(claim, dict):
                    continue
                text = _clean_text(claim.get("text"))
                if not text:
                    continue
                citations = _source_id_list(claim.get("citations"))
                if not citations:
                    citations = [citation.source_id for citation in step.citations[:1]]
                support_level = _clean_text(claim.get("support_level")) or "partial"
                findings.append(
                    AgentFinding(
                        finding_id=f"f{len(findings) + 1}",
                        category=step.title,
                        severity="Medium" if support_level == "direct" else "Needs human review",
                        summary=text,
                        citations=citations,
                        recommended_action="Confirm the business position and source support.",
                        needs_human_review=support_level != "direct",
                        source_step_id=step.step_id,
                    )
                )
        if findings:
            return _renumber_findings(findings)

        summary = _clean_text(step.summary)
        if not summary:
            return []
        return [
            AgentFinding(
                finding_id="f1",
                category=step.title,
                severity="Needs human review",
                summary=summary[:500],
                citations=[citation.source_id for citation in step.citations[:2]],
                recommended_action="Review and confirm before relying on this output.",
                needs_human_review=True,
                source_step_id=step.step_id,
            )
        ]

    def _render_report(
        self,
        *,
        objective: str,
        user_role: str,
        steps: list[AgentStepResult],
        findings: list[AgentFinding],
        missing_information: list[str],
        matter_profile: MatterProfile | None,
        artifacts: list[AgentArtifact],
        confirmation_gates: list[AgentConfirmationGate],
    ) -> str:
        lines = [
            "## Agent task report",
            f"Objective: {objective}",
            f"User mode: {user_role}",
            "",
            "## Matter profile",
        ]
        if matter_profile:
            parties = ", ".join(matter_profile.parties) or "Unknown"
            lines.extend(
                [
                    f"- Matter ID: {matter_profile.matter_id}",
                    f"- Document type: {matter_profile.document_type}",
                    f"- Parties: {parties}",
                    f"- User side: {matter_profile.user_side or 'Unspecified'}",
                    f"- Governing law: {matter_profile.governing_law or 'Unspecified'}",
                    f"- Jurisdiction: {matter_profile.jurisdiction or 'Unspecified'}",
                    f"- Review scope: {', '.join(matter_profile.review_scope) or 'Unspecified'}",
                ]
            )
        else:
            lines.append("- No structured matter profile was produced.")

        lines.extend(
            [
                "",
                "## Work performed",
            ]
        )
        for step in steps:
            if step.tool == "synthesize_report":
                continue
            citation_suffix = _format_refs([citation.source_id for citation in step.citations[:2]])
            lines.append(f"- {step.title}: {step.status}.{citation_suffix}")

        lines.extend(["", "## Key findings"])
        if findings:
            for finding in findings:
                refs = _format_refs(finding.citations)
                action = (
                    f" Recommended action: {finding.recommended_action}"
                    if finding.recommended_action
                    else ""
                )
                support = (
                    f" Support: {finding.support_level}"
                    if finding.support_level
                    else ""
                )
                lines.append(
                    f"- {finding.category} ({finding.severity}): "
                    f"{finding.summary}{refs}{support}{action}"
                )
        else:
            lines.append("- No structured risk findings were produced from the cited excerpts.")

        lines.extend(["", "## Missing information"])
        if missing_information:
            for item in missing_information:
                lines.append(f"- {item}")
        else:
            lines.append("- No additional missing information was identified by this workflow.")

        lines.extend(["", "## Artifacts"])
        if artifacts:
            for artifact in artifacts:
                lines.append(
                    f"- {artifact.title}: {len(artifact.items)} item(s). {artifact.summary}"
                )
        else:
            lines.append("- No structured artifacts were generated.")

        lines.extend(["", "## Confirmation gates"])
        if confirmation_gates:
            for gate in confirmation_gates:
                lines.append(
                    f"- {gate.title} ({gate.priority}): {gate.question}"
                )
        else:
            lines.append("- No blocking confirmation gates were generated.")

        lines.extend(
            [
                "",
                "## Human review gate",
                (
                    "A qualified legal professional should review this output before it is used "
                    "for legal decisions, negotiation positions, filings, or formal advice."
                ),
            ]
        )
        return "\n".join(lines).strip()


def _review_scope_from_plan(plan: list[AgentPlanStep]) -> list[str]:
    scope = []
    for step in plan:
        if step.tool != "review_clause":
            continue
        clause_type = _clean_text(step.arguments.get("clause_type"))
        if clause_type:
            scope.append(clause_type)
    return _dedupe_texts(scope)


def _workflow_type(objective: str) -> str:
    lowered = objective.casefold()
    if _mentions_any(lowered, VERSION_COMPARE_KEYWORDS):
        return "version_comparison"
    if _mentions_any(lowered, CLAUSE_REVISION_KEYWORDS):
        return "clause_revision"
    if _mentions_any(lowered, NEGOTIATION_KEYWORDS):
        return "negotiation_prep"
    if _mentions_any(lowered, OBLIGATION_CALENDAR_KEYWORDS):
        return "obligation_calendar"
    if "evidence" in lowered or "citation audit" in lowered or "support" in lowered:
        return "evidence_audit"
    if _looks_like_conflict_task(objective):
        return "conflict_check"
    return "contract_review"


def _trim_plan(plan: list[AgentPlanStep], max_steps: int) -> list[AgentPlanStep]:
    if len(plan) <= max_steps:
        return plan
    if not plan or plan[-1].tool != "synthesize_report":
        return plan[:max_steps]
    return [*plan[: max_steps - 1], plan[-1]]


def _audit_findings(
    findings: list[AgentFinding],
    citations: list[Citation],
) -> list[AgentFinding]:
    if not findings:
        return []

    citations_by_id = {citation.source_id.upper(): citation for citation in citations}
    audited: list[AgentFinding] = []
    for finding in findings:
        normalized_ids = [
            source_id
            for source_id in _source_id_list(finding.citations)
            if source_id in citations_by_id
        ]
        claim_text = f"{finding.summary}{_format_refs(normalized_ids)}"
        profile = build_evidence_profile(claim_text, citations)
        claim = _first_evidence_claim(profile)
        evidence_items = claim.get("evidence", []) if claim else []
        if not isinstance(evidence_items, list):
            evidence_items = []

        support_level = _clean_text(claim.get("support_level")) if claim else ""
        if not support_level:
            support_level = "missing" if not normalized_ids else "partial"

        unsupported_reason = _finding_unsupported_reason(
            claim=claim,
            has_valid_citations=bool(normalized_ids),
        )
        source_quote = _first_evidence_text(evidence_items, "quote")
        location_label = _first_evidence_text(evidence_items, "location_label")
        if not source_quote and normalized_ids:
            citation = citations_by_id[normalized_ids[0]]
            source_quote = citation.exact_quote or citation.preview
            location_label = citation.location_label()

        evidence_coverage = _finding_evidence_coverage(
            support_level=support_level,
            has_quote=bool(source_quote),
            has_location=bool(location_label),
            citation_count=len(normalized_ids),
        )
        needs_human_review = (
            finding.needs_human_review
            or support_level != "direct"
            or evidence_coverage != "direct"
        )
        human_review_status = "pending" if needs_human_review else "not_required"
        status = "needs_human_review" if needs_human_review else "evidence_backed"
        if support_level != "direct" and not unsupported_reason:
            unsupported_reason = "The finding is not directly supported by the cited excerpt."

        audited.append(
            replace(
                finding,
                citations=normalized_ids,
                evidence_coverage=evidence_coverage,
                support_level=support_level,
                unsupported_reason=unsupported_reason,
                source_quote=source_quote[:1200],
                location_label=location_label,
                human_review_status=human_review_status,
                status=status,
                evidence=[
                    item
                    for item in evidence_items
                    if isinstance(item, dict)
                ],
            )
        )
    return audited


def _first_evidence_claim(profile: dict[str, Any]) -> dict[str, Any] | None:
    claims = profile.get("claims")
    if not isinstance(claims, list):
        return None
    for claim in claims:
        if isinstance(claim, dict):
            return claim
    return None


def _first_evidence_text(evidence_items: list[Any], key: str) -> str:
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        text = _clean_text(item.get(key))
        if text:
            return text
    return ""


def _finding_unsupported_reason(
    *,
    claim: dict[str, Any] | None,
    has_valid_citations: bool,
) -> str:
    if not has_valid_citations:
        return "Missing source citation."
    if not claim:
        return "Evidence support could not be evaluated."
    unsupported_facts = claim.get("unsupported_facts")
    if isinstance(unsupported_facts, list) and unsupported_facts:
        facts = [_clean_text(item) for item in unsupported_facts if _clean_text(item)]
        if facts:
            return "Unsupported facts: " + ", ".join(facts)
    return _clean_text(claim.get("uncertainty"))


def _finding_evidence_coverage(
    *,
    support_level: str,
    has_quote: bool,
    has_location: bool,
    citation_count: int,
) -> str:
    if not citation_count:
        return "missing"
    if support_level == "direct" and has_quote and has_location:
        return "direct"
    if has_quote or has_location:
        return "partial"
    return "missing"


def _build_matter_profile(
    *,
    matter_id: str,
    objective: str,
    review_scope: list[str],
    steps: list[AgentStepResult],
    missing_information: list[str],
) -> MatterProfile:
    profile_step = next((step for step in steps if step.step_id == "profile"), None)
    source_text = _profile_source_text(objective, profile_step)
    parties = _extract_parties(source_text)
    governing_law = _extract_governing_law(source_text)
    jurisdiction = _extract_jurisdiction(source_text, governing_law)
    citations = [citation.source_id for citation in profile_step.citations] if profile_step else []

    profile = MatterProfile(
        matter_id=matter_id,
        document_type=_infer_document_type(source_text),
        parties=parties,
        user_side=_extract_user_side(source_text),
        governing_law=governing_law,
        jurisdiction=jurisdiction,
        key_dates=_extract_key_dates(source_text, citations),
        review_scope=_dedupe_texts(review_scope),
        open_questions=[],
        confidence=_profile_confidence(
            citations=citations,
            document_type=_infer_document_type(source_text),
            parties=parties,
            governing_law=governing_law,
        ),
        citations=citations,
        source_step_id=profile_step.step_id if profile_step else "",
    )
    return replace(
        profile,
        open_questions=_matter_open_questions(profile, missing_information),
    )


def _profile_source_text(objective: str, profile_step: AgentStepResult | None) -> str:
    parts = [objective]
    if profile_step:
        parts.append(profile_step.summary)
        for citation in profile_step.citations:
            parts.append(citation.exact_quote or citation.preview)
    return "\n".join(part for part in parts if part)


def _infer_document_type(text: str) -> str:
    lowered = text.casefold()
    rules = [
        ("SaaS MSA", ("saas", "msa")),
        ("SaaS agreement", ("saas", "agreement")),
        ("Master services agreement", ("master services agreement",)),
        ("Mutual NDA", ("mutual nda", "mutual non-disclosure")),
        ("Non-disclosure agreement", ("non-disclosure agreement", "nda")),
        ("Data processing addendum", ("data processing addendum", "dpa")),
        ("Supply agreement", ("supply agreement", "purchase agreement")),
        ("Employment document", ("employee handbook", "employment agreement")),
        ("Policy document", ("policy", "procedure")),
        ("Agreement", ("agreement", "contract")),
    ]
    for label, keywords in rules:
        if all(keyword in lowered for keyword in keywords):
            return label
    return "Unknown"


def _extract_parties(text: str) -> list[str]:
    patterns = [
        re.compile(
            r"\bby\s+and\s+between\s+(.{2,90}?)\s+and\s+(.{2,90}?)(?=\.|,|;|\n|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bbetween\s+(.{2,90}?)\s+and\s+(.{2,90}?)(?=\.|,|;|\n|$)",
            re.IGNORECASE,
        ),
    ]
    parties: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            for group in match.groups():
                party = _clean_party_name(group)
                if party and party.casefold() not in {item.casefold() for item in parties}:
                    parties.append(party)
            if parties:
                return parties[:6]
    return parties


def _clean_party_name(value: str) -> str:
    text = _clean_text(value).strip(" .,:;()[]")
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.split(
        r"\s+(?:under|pursuant|whereas|whose|which|that)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return text[:120].strip(" .,:;()[]")


def _extract_governing_law(text: str) -> str:
    patterns = [
        r"\b([A-Z][A-Za-z .-]{2,60}?)\s+law\s+governs\b",
        r"\bgoverned\s+by\s+(?:the\s+)?laws?\s+of\s+(?:the\s+State\s+of\s+)?"
        r"([A-Z][A-Za-z .-]{2,60})",
        r"\bgoverning\s+law\s*[:;-]\s*([A-Z][A-Za-z .-]{2,60})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_law_name(match.group(1))
    return ""


def _clean_law_name(value: str) -> str:
    text = _clean_text(value).strip(" .,:;()[]")
    text = re.sub(r"^(?:and|the|state\s+of)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+laws?$", "", text, flags=re.IGNORECASE)
    return text.title() if text.islower() else text


def _extract_jurisdiction(text: str, governing_law: str) -> str:
    if governing_law:
        return governing_law
    patterns = [
        r"\bjurisdiction\s+(?:of|in)\s+([A-Z][A-Za-z .-]{2,60})",
        r"\bvenue\s+(?:is\s+)?(?:in|of)\s+([A-Z][A-Za-z .-]{2,60})",
        r"\bcourts?\s+of\s+([A-Z][A-Za-z .-]{2,60})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_law_name(match.group(1))
    return ""


def _extract_user_side(text: str) -> str:
    patterns = [
        r"\bI\s+represent\s+(?:the\s+)?([A-Za-z][A-Za-z -]{1,40})",
        r"\bwe\s+represent\s+(?:the\s+)?([A-Za-z][A-Za-z -]{1,40})",
        r"\bon\s+behalf\s+of\s+(?:the\s+)?([A-Za-z][A-Za-z -]{1,40})",
        (
            r"\bfor\s+(?:the\s+)?"
            r"(buyer|seller|customer|vendor|supplier|employee|employer|tenant|landlord)\b"
        ),
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            side = _clean_text(match.group(1)).strip(" .,:;")
            return side[:1].upper() + side[1:] if side else ""
    return ""


def _extract_key_dates(text: str, citations: list[str]) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|"
        r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|"
        r"\b\d+\s+(?:business\s+days?|calendar\s+days?|days?|hours?|weeks?|months?)\b",
        re.IGNORECASE,
    )
    dates: list[dict[str, Any]] = []
    seen = set()
    for match in pattern.finditer(text):
        value = match.group(0)
        context = _sentence_around(text, match.start(), match.end())
        key = (value.casefold(), context.casefold())
        if key in seen:
            continue
        seen.add(key)
        dates.append(
            {
                "label": _date_label(context),
                "value": value,
                "description": context,
                "citations": citations[:2],
            }
        )
        if len(dates) >= 8:
            break
    return dates


def _sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start), text.rfind(";", 0, start))
    right_candidates = [
        index
        for index in (text.find(".", end), text.find("\n", end), text.find(";", end))
        if index >= 0
    ]
    right = min(right_candidates) if right_candidates else min(len(text), end + 160)
    return _clean_text(text[left + 1 : right + 1])


def _date_label(context: str) -> str:
    lowered = context.casefold()
    if "notice" in lowered:
        return "Notice period"
    if "renew" in lowered:
        return "Renewal date"
    if "terminat" in lowered:
        return "Termination deadline"
    if "pay" in lowered or "invoice" in lowered:
        return "Payment deadline"
    if "effective" in lowered:
        return "Effective date"
    return "Date or deadline"


def _profile_confidence(
    *,
    citations: list[str],
    document_type: str,
    parties: list[str],
    governing_law: str,
) -> str:
    signals = sum(
        [
            bool(citations),
            document_type != "Unknown",
            bool(parties),
            bool(governing_law),
        ]
    )
    if signals >= 4:
        return "High"
    if signals >= 2:
        return "Medium"
    return "Low"


def _matter_open_questions(
    profile: MatterProfile,
    missing_information: list[str],
) -> list[str]:
    questions = []
    if not profile.parties:
        questions.append("Confirm the exact parties and their roles in this matter.")
    if not profile.governing_law:
        questions.append("Confirm the governing law or jurisdiction for this matter.")
    if not profile.user_side:
        questions.append("Confirm which side the user represents or wants optimized.")
    if not profile.review_scope:
        questions.append("Confirm the intended review scope.")
    questions.extend(missing_information)
    return _dedupe_texts(questions)[:12]


def _build_agent_artifacts(
    *,
    matter_profile: MatterProfile,
    findings: list[AgentFinding],
    steps: list[AgentStepResult],
    missing_information: list[str],
    user_role: str,
) -> list[AgentArtifact]:
    artifacts = [
        _risk_matrix_artifact(findings),
        _lawyer_questions_artifact(findings, steps, missing_information, user_role),
        _negotiation_checklist_artifact(findings, matter_profile),
        _obligation_calendar_artifact(matter_profile, findings),
    ]
    return artifacts


def _risk_matrix_artifact(findings: list[AgentFinding]) -> AgentArtifact:
    items = [
        {
            "item_id": f"risk-{index}",
            "finding_id": finding.finding_id,
            "category": finding.category,
            "severity": finding.severity,
            "issue": finding.summary,
            "recommended_action": finding.recommended_action,
            "citations": finding.citations,
            "needs_human_review": finding.needs_human_review,
            "status": finding.status,
            "human_review_status": finding.human_review_status,
            "evidence_coverage": finding.evidence_coverage,
            "support_level": finding.support_level,
            "unsupported_reason": finding.unsupported_reason,
            "source_quote": finding.source_quote,
            "location_label": finding.location_label,
            "clause_reference": finding.clause_reference,
        }
        for index, finding in enumerate(findings, start=1)
    ]
    return AgentArtifact(
        artifact_id="risk_matrix",
        artifact_type="risk_matrix",
        title="Risk matrix",
        summary="Structured risk rows derived from evidence-backed review findings.",
        items=items,
        source_finding_ids=[finding.finding_id for finding in findings],
        citations=_artifact_citations(items),
    )


def _lawyer_questions_artifact(
    findings: list[AgentFinding],
    steps: list[AgentStepResult],
    missing_information: list[str],
    user_role: str,
) -> AgentArtifact:
    items: list[dict[str, Any]] = []
    for step in steps:
        metadata = step.output.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        for question in _as_text_list(metadata.get("questions_for_lawyer")):
            items.append(
                {
                    "item_id": f"question-{len(items) + 1}",
                    "question": question,
                    "reason": f"Raised during {step.title}.",
                    "priority": "legal_review",
                    "source_step_id": step.step_id,
                    "citations": [citation.source_id for citation in step.citations[:2]],
                }
            )

    for finding in findings:
        if finding.recommended_action and "?" in finding.recommended_action:
            question = finding.recommended_action
        else:
            question = f"What position should we take on {finding.category}?"
        items.append(
            {
                "item_id": f"question-{len(items) + 1}",
                "question": question,
                "reason": finding.summary,
                "priority": "high" if finding.needs_human_review else "normal",
                "source_finding_id": finding.finding_id,
                "citations": finding.citations,
            }
        )

    for item in missing_information:
        items.append(
            {
                "item_id": f"question-{len(items) + 1}",
                "question": f"Please confirm: {item}",
                "reason": "The workflow marked this information as missing.",
                "priority": "blocking",
                "citations": [],
            }
        )

    items = _dedupe_artifact_items(items, "question")
    title = "Lawyer questions" if user_role == "lawyer" else "Review questions"
    return AgentArtifact(
        artifact_id="lawyer_questions",
        artifact_type="lawyer_questions",
        title=title,
        summary="Questions to resolve before relying on the review output.",
        items=items,
        source_finding_ids=_artifact_finding_ids(items),
        citations=_artifact_citations(items),
    )


def _negotiation_checklist_artifact(
    findings: list[AgentFinding],
    matter_profile: MatterProfile,
) -> AgentArtifact:
    items = []
    owner = matter_profile.user_side or "User side"
    for index, finding in enumerate(findings, start=1):
        items.append(
            {
                "item_id": f"negotiation-{index}",
                "issue": finding.category,
                "ask": finding.recommended_action
                or f"Clarify or revise the {finding.category} language.",
                "fallback": "Escalate for legal review before accepting the current language.",
                "owner": owner,
                "priority": _negotiation_priority(finding.severity),
                "source_finding_id": finding.finding_id,
                "citations": finding.citations,
            }
        )
    return AgentArtifact(
        artifact_id="negotiation_checklist",
        artifact_type="negotiation_checklist",
        title="Negotiation checklist",
        summary="Negotiation asks and fallback positions generated from current findings.",
        items=items,
        source_finding_ids=[finding.finding_id for finding in findings],
        citations=_artifact_citations(items),
    )


def _obligation_calendar_artifact(
    matter_profile: MatterProfile,
    findings: list[AgentFinding],
) -> AgentArtifact:
    items = []
    seen = set()
    for date_item in matter_profile.key_dates:
        key = _clean_text(date_item.get("value")).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "item_id": f"obligation-{len(items) + 1}",
                "trigger": date_item.get("description") or date_item.get("label"),
                "deadline": date_item.get("value"),
                "owner": matter_profile.user_side or "To confirm",
                "status": "needs_confirmation",
                "citations": date_item.get("citations") or [],
            }
        )

    for finding in findings:
        for date_item in _extract_key_dates(finding.summary, finding.citations):
            key = _clean_text(date_item.get("value")).casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "item_id": f"obligation-{len(items) + 1}",
                    "trigger": finding.category,
                    "deadline": date_item.get("value"),
                    "owner": matter_profile.user_side or "To confirm",
                    "status": "needs_confirmation",
                    "source_finding_id": finding.finding_id,
                    "citations": finding.citations,
                }
            )
    return AgentArtifact(
        artifact_id="obligation_calendar",
        artifact_type="obligation_calendar",
        title="Obligation calendar",
        summary="Dates and relative deadlines extracted from the reviewed evidence.",
        items=items,
        source_finding_ids=_artifact_finding_ids(items),
        citations=_artifact_citations(items),
    )


def _build_confirmation_gates(
    *,
    objective: str,
    matter_profile: MatterProfile,
    findings: list[AgentFinding],
    missing_information: list[str],
    guard_warnings: list[str],
    artifacts: list[AgentArtifact],
    user_role: str,
) -> list[AgentConfirmationGate]:
    gates: list[AgentConfirmationGate] = []

    if not matter_profile.governing_law and not matter_profile.jurisdiction:
        gates.append(
            AgentConfirmationGate(
                gate_id="confirm_jurisdiction",
                gate_type="matter_fact",
                title="Confirm governing law",
                question=(
                    "Confirm the governing law or jurisdiction before relying on "
                    "legal/compliance conclusions."
                ),
                priority="high",
                reason="The Matter Profile does not contain a confirmed law or jurisdiction.",
                citations=matter_profile.citations,
                metadata={"profile_field": "governing_law"},
            )
        )

    if not matter_profile.user_side:
        gates.append(
            AgentConfirmationGate(
                gate_id="confirm_user_side",
                gate_type="matter_fact",
                title="Confirm represented side",
                question="Confirm which side the user represents or wants optimized.",
                priority="high",
                reason="Negotiation advice depends on the represented party or business side.",
                citations=matter_profile.citations,
                metadata={"profile_field": "user_side"},
            )
        )

    if missing_information:
        gates.append(
            AgentConfirmationGate(
                gate_id="resolve_missing_information",
                gate_type="missing_information",
                title="Resolve missing information",
                question=(
                    f"Resolve {len(missing_information)} missing information item(s) "
                    "before treating the report as complete."
                ),
                priority="high",
                reason="The workflow identified unanswered facts or missing documents.",
                metadata={"missing_information": missing_information[:12]},
            )
        )

    review_findings = [finding for finding in findings if finding.needs_human_review]
    if review_findings:
        gates.append(
            AgentConfirmationGate(
                gate_id="review_high_risk_findings",
                gate_type="legal_review",
                title="Review flagged findings",
                question=(
                    f"Have counsel reviewed {len(review_findings)} finding(s) marked "
                    "as needing human review?"
                ),
                priority="high",
                reason="At least one finding was not safe to rely on without legal review.",
                related_finding_ids=[finding.finding_id for finding in review_findings],
                citations=_dedupe_texts(
                    [source_id for finding in review_findings for source_id in finding.citations]
                ),
            )
        )

    weak_evidence_findings = [
        finding
        for finding in findings
        if (
            not finding.citations
            or not finding.source_quote
            or not finding.location_label
            or finding.support_level != "direct"
        )
    ]
    if weak_evidence_findings:
        gates.append(
            AgentConfirmationGate(
                gate_id="resolve_finding_evidence",
                gate_type="evidence",
                title="Resolve finding evidence",
                question=(
                    f"Resolve evidence gaps for {len(weak_evidence_findings)} finding(s) "
                    "before they can enter a formal report."
                ),
                priority="high",
                reason=(
                    "Every formal finding needs a source citation, exact quote/location, "
                    "support level, unsupported reason when applicable, and human review status."
                ),
                related_finding_ids=[
                    finding.finding_id for finding in weak_evidence_findings
                ],
                citations=_dedupe_texts(
                    [
                        source_id
                        for finding in weak_evidence_findings
                        for source_id in finding.citations
                    ]
                ),
            )
        )

    if guard_warnings:
        gates.append(
            AgentConfirmationGate(
                gate_id="resolve_evidence_guard",
                gate_type="evidence",
                title="Resolve evidence warnings",
                question="Resolve evidence guard warnings before using the output externally.",
                priority="high",
                reason="The verifier found unsupported or weakly supported report content.",
                metadata={"guard_warnings": guard_warnings[:12]},
            )
        )

    if _mentions_any(objective.casefold(), CURRENT_LAW_KEYWORDS):
        gates.append(
            AgentConfirmationGate(
                gate_id="authorize_external_research",
                gate_type="permission",
                title="Authorize external research",
                question=(
                    "Confirm whether the Agent may search current public legal sources "
                    "before making up-to-date legal statements."
                ),
                priority="normal",
                reason="The objective asks for current law, regulation, or compliance context.",
                required=True,
                metadata={"requested_capability": "web_search"},
            )
        )

    if findings or artifacts:
        gates.append(
            AgentConfirmationGate(
                gate_id="approve_report_use",
                gate_type="delivery",
                title="Approve report use",
                question=(
                    "Confirm the evidence is sufficient before treating this as a formal "
                    "deliverable or negotiation position."
                ),
                priority="normal" if user_role == "lawyer" else "high",
                reason="Legal deliverables should be approved before external reliance.",
                related_artifact_ids=[artifact.artifact_id for artifact in artifacts],
                citations=_dedupe_texts(
                    [source_id for artifact in artifacts for source_id in artifact.citations]
                ),
                metadata={"user_role": user_role},
            )
        )

    return _dedupe_confirmation_gates(gates)


def _dedupe_confirmation_gates(
    gates: list[AgentConfirmationGate],
) -> list[AgentConfirmationGate]:
    deduped = []
    seen = set()
    for gate in gates:
        key = gate.gate_id.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(gate)
    return deduped[:12]


def _confirmation_gate_payload(gate: AgentConfirmationGate) -> dict[str, Any]:
    return {
        "gate_id": gate.gate_id,
        "gate_type": gate.gate_type,
        "title": gate.title,
        "question": gate.question,
        "status": gate.status,
        "priority": gate.priority,
        "required": gate.required,
        "reason": gate.reason,
        "related_finding_ids": gate.related_finding_ids,
        "related_artifact_ids": gate.related_artifact_ids,
        "citations": gate.citations,
        "metadata": gate.metadata,
    }


def _negotiation_priority(severity: str) -> str:
    normalized = severity.casefold()
    if "high" in normalized or "human" in normalized:
        return "must_address"
    if "medium" in normalized:
        return "negotiate"
    return "consider"


def _dedupe_artifact_items(
    items: list[dict[str, Any]],
    key_name: str,
) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for item in items:
        text = _clean_text(item.get(key_name))
        key = text.casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append({**item, "item_id": f"{key_name}-{len(deduped) + 1}"})
    return deduped


def _artifact_citations(items: list[dict[str, Any]]) -> list[str]:
    citations = []
    for item in items:
        for source_id in _source_id_list(item.get("citations")):
            if source_id not in citations:
                citations.append(source_id)
    return citations


def _artifact_finding_ids(items: list[dict[str, Any]]) -> list[str]:
    finding_ids = []
    for item in items:
        value = _clean_text(item.get("source_finding_id"))
        if value and value not in finding_ids:
            finding_ids.append(value)
    return finding_ids


class _CitationRegistry:
    def __init__(self) -> None:
        self.citations: list[Citation] = []

    def add_step_citations(
        self,
        step_id: str,
        citations: list[Citation],
    ) -> tuple[dict[str, str], list[Citation]]:
        mapping: dict[str, str] = {}
        registered: list[Citation] = []
        for citation in citations:
            new_source_id = f"S{len(self.citations) + 1}"
            mapping[citation.source_id.upper()] = new_source_id
            mapped = replace(citation, source_id=new_source_id)
            self.citations.append(mapped)
            registered.append(mapped)
        return mapping, registered


def _resolve_focus_areas(objective: str, focus_areas: list[str]) -> list[str]:
    explicit = [_clean_text(area) for area in focus_areas]
    explicit = [area for area in explicit if area]
    if explicit:
        return _dedupe_texts(explicit)

    lowered = objective.casefold()
    inferred = [
        area
        for area, keywords in FOCUS_KEYWORDS.items()
        if any(keyword.casefold() in lowered for keyword in keywords)
    ]
    if inferred:
        return _dedupe_texts(inferred)
    return DEFAULT_FOCUS_AREAS.copy()


def _looks_underspecified_objective(text: str, focus_areas: list[str]) -> bool:
    normalized = re.sub(r"[\s，。,.!?！？、：:；;]+", "", text.casefold())
    if not normalized:
        return True
    if normalized in GENERIC_OBJECTIVE_PATTERNS:
        return True
    if normalized.startswith(("帮我看看这份", "帮我看看这个", "帮我看下这份")) and len(
        normalized
    ) <= 10:
        return True
    return not focus_areas and len(text) < 12


def _mentions_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.casefold() in text for keyword in keywords)


def _looks_like_conflict_task(objective: str) -> bool:
    lowered = objective.casefold()
    return any(
        keyword in lowered
        for keyword in (
            "conflict",
            "compare",
            "inconsistent",
            "policy",
        )
    )


def _remap_source_refs(text: str, mapping: dict[str, str]) -> str:
    def replace_match(match: re.Match[str]) -> str:
        source_id = match.group(1).upper()
        return f"[{mapping.get(source_id, source_id)}]"

    remapped = SOURCE_REF_PATTERN.sub(replace_match, text or "")

    def replace_bare_match(match: re.Match[str]) -> str:
        source_id = match.group(1).upper()
        return mapping.get(source_id, source_id)

    return BARE_SOURCE_REF_PATTERN.sub(replace_bare_match, remapped)


def _remap_metadata(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _remap_source_refs(value, mapping)
    if isinstance(value, list):
        return [_remap_metadata(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _remap_metadata(item, mapping) for key, item in value.items()}
    return value


def _metadata_missing_information(metadata: dict[str, Any]) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    missing = _as_text_list(metadata.get("missing_information"))
    evidence = metadata.get("evidence")
    if isinstance(evidence, dict):
        missing.extend(_as_text_list(evidence.get("missing_evidence")))
    return _dedupe_texts(missing)


def _source_id_list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    source_ids = []
    for item in values:
        if not isinstance(item, str):
            continue
        for match in SOURCE_REF_PATTERN.finditer(item):
            source_id = match.group(1).upper()
            if source_id not in source_ids:
                source_ids.append(source_id)
        for match in BARE_SOURCE_REF_PATTERN.finditer(item):
            source_id = match.group(1).upper()
            if source_id not in source_ids:
                source_ids.append(source_id)
    return source_ids


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        result = []
        for item in value:
            text = _clean_text(item)
            if text:
                result.append(text)
        return result
    return []


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _dedupe_texts(values: list[str]) -> list[str]:
    result = []
    seen = set()
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


def _first_text(values: list[str]) -> str:
    return values[0] if values else ""


def _format_refs(source_ids: list[str]) -> str:
    refs = []
    for source_id in source_ids:
        if not isinstance(source_id, str):
            continue
        normalized = source_id.strip().strip("[]").upper()
        if re.fullmatch(r"S\d+", normalized) and normalized not in refs:
            refs.append(normalized)
    return " " + " ".join(f"[{source_id}]" for source_id in refs) if refs else ""


def _renumber_findings(findings: list[AgentFinding]) -> list[AgentFinding]:
    return [
        replace(finding, finding_id=f"f{index}")
        for index, finding in enumerate(findings, start=1)
    ]


def _emit_progress(
    callback: ProgressCallback | None,
    *,
    event_type: str,
    stage: str,
    progress: int,
    message: str,
    step_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    if callback is None:
        return
    callback(
        event_type=event_type,
        stage=stage,
        progress=progress,
        message=message,
        step_id=step_id,
        payload=payload or {},
    )


def _plan_step_payload(step: AgentPlanStep) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "title": step.title,
        "purpose": step.purpose,
        "tool": step.tool,
        "arguments": step.arguments,
        "requires_confirmation": step.requires_confirmation,
    }


def _step_result_payload(step: AgentStepResult) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "title": step.title,
        "tool": step.tool,
        "status": step.status,
        "citation_count": len(step.citations),
        "guard_warning_count": len(step.guard_warnings),
    }

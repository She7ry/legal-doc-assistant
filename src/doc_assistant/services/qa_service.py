"""文档问答（RAG）服务：检索 → 组 prompt → 调 LLM → guard 校验。

主要能力：
- ``ask`` / ``aask``           通用问答（有文档则 RAG，无文档则 general chat）
- ``review_clause``            按条款类型做结构化风险审查
- ``check_conflict``           合同与政策文本冲突比对
- ``prepare_answer``           可单独调用：只做检索与 prompt 组装，便于流式或 Agent 复用

引用格式：[S1]、[S2]… 对应 ``Citation.source_id``，由 answer_guard 校验是否滥用。
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from doc_assistant.config.settings import settings
from doc_assistant.grounding.answer_repair import try_lightweight_repair
from doc_assistant.grounding.document_context import format_document_context
from doc_assistant.grounding.evidence import build_evidence_profile
from doc_assistant.grounding.guard import AnswerGuardResult, validate_answer
from doc_assistant.memory.history import format_chat_history, merge_chat_history
from doc_assistant.memory.schemas import MemoryCandidate, MemoryUsage
from doc_assistant.memory.service import MemoryService
from doc_assistant.models.language_model import build_chat_model
from doc_assistant.retrieval.vector_store import DocumentVectorStore
from doc_assistant.schemas.citation import Citation, QAAnswer
from doc_assistant.skills import SkillEngine, build_skill_engine_from_settings
from doc_assistant.skills.models import EvidenceSufficiencyResult, SkillRuntimeContext
from doc_assistant.skills.runtime import (
    assess_evidence_sufficiency,
    decompose_retrieval_query,
    fuse_retrieval_results,
    is_complex_retrieval_query,
    render_evidence_assessment,
)
from doc_assistant.utils.prompt_loader import load_base_legal_prompt, load_prompt

logger = logging.getLogger(__name__)

QUERY_REWRITE_PROMPT = """Given the user's legal question and chat history, rewrite it as a precise document retrieval query.
Keep legal terms, party names, dates, and clause names intact.
Output only the rewritten query.

Chat history:
{chat_history}

Original question:
{question}

Rewritten query:"""

_VAGUE_QUERY_TERMS = (
    "这个",
    "那个",
    "它",
    "上面",
    "前面",
    "刚才",
    "this",
    "that",
    "it",
    "above",
    "previous",
)


@dataclass(frozen=True)
class PreparedQAAnswer:
    """``prepare_answer()`` 的中间结果，尚未调用 LLM。

    用途：把检索、记忆、prompt 组装与 ``finalize_prepared_answer()`` 拆开，
    便于流式输出或 Agent 复用同一套检索逻辑；``messages`` 可直接发给 chat_model。
    """

    messages: list[dict[str, str]]
    citations: list[Citation]
    memories_used: list[MemoryUsage]
    user_id: str | None
    conversation_id: str | None
    user_message_recorded: bool
    task_id: str | None = None
    has_retrieved_documents: bool = True
    skill_context: SkillRuntimeContext | None = None
    retrieval_queries: tuple[str, ...] = ()
    evidence_sufficiency: EvidenceSufficiencyResult | None = None


@dataclass(frozen=True)
class _MemoryEnrichment:
    """``_enrich_memory_context()`` 的返回结果，封装记忆富化阶段的所有产出。"""

    resolved_conversation_id: str
    memory_candidates: list[MemoryCandidate]
    memory_context: str
    user_message_recorded: bool
    persisted_history: list[dict[str, str]]


class DocumentQAService:
    """文档问答（RAG）核心服务。

    职责：
    1. 向量检索 uploaded 文档，组装带引用的 prompt
    2. 调用 LLM 生成答案，经 answer_guard 校验并可自动修复
    3. 提供 review_clause / check_conflict 等结构化法律审查能力
    4. 可选接入 MemoryService，注入用户长期记忆与对话历史

    Agent 与 ToolCalling 都依赖本类的 vector_store 与 chat_model。
    """

    def __init__(
        self,
        vector_store: DocumentVectorStore | None = None,
        chat_model: BaseChatModel | None = None,
        memory_service: MemoryService | None = None,
        tenant_id: str | None = None,
        skill_engine: SkillEngine | None = None,
    ) -> None:
        self.vector_store = vector_store or DocumentVectorStore()
        self.tenant_id = tenant_id or getattr(self.vector_store, "tenant_id", settings.default_tenant_id)
        self.base_prompt = load_base_legal_prompt()
        self.prompt = PromptTemplate.from_template(load_prompt("document_qa.txt"))
        self.general_prompt = PromptTemplate.from_template(load_prompt("general_chat.txt"))
        self.clause_review_prompt = PromptTemplate.from_template(load_prompt("clause_review.txt"))
        self.conflict_check_prompt = PromptTemplate.from_template(load_prompt("conflict_check.txt"))
        self.answer_repair_prompt = PromptTemplate.from_template(load_prompt("answer_repair.txt"))
        self.chat_model = chat_model or build_chat_model()
        self.memory_service = memory_service
        self.skill_engine = skill_engine if skill_engine is not None else build_skill_engine_from_settings()

    def ask(
        self,
        question: str,
        chat_history: list[dict[str, object]] | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        merge_persisted_history: bool = True,
        skill_names: tuple[str, ...] | None = None,
    ) -> QAAnswer:
        """同步问答：检索 → 组 prompt → 调 LLM → guard 校验（可自动 repair）。"""
        prepared = self.prepare_answer(
            question,
            chat_history=chat_history,
            user_id=user_id,
            conversation_id=conversation_id,
            task_id=task_id,
            merge_persisted_history=merge_persisted_history,
            skill_names=skill_names,
        )
        content = self._invoke_chat_messages(prepared.messages)
        return self.finalize_prepared_answer(prepared, content)

    async def aask(
        self,
        question: str,
        chat_history: list[dict[str, object]] | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        merge_persisted_history: bool = True,
        skill_names: tuple[str, ...] | None = None,
    ) -> QAAnswer:
        """异步版 ``ask``，适合 FastAPI 等 async 路由。"""
        prepared = await self.aprepare_answer(
            question,
            chat_history=chat_history,
            user_id=user_id,
            conversation_id=conversation_id,
            task_id=task_id,
            merge_persisted_history=merge_persisted_history,
            skill_names=skill_names,
        )
        content = await self._ainvoke_chat_messages(prepared.messages)
        return await asyncio.to_thread(self.finalize_prepared_answer, prepared, content)

    def finalize_prepared_answer(
        self,
        prepared: PreparedQAAnswer,
        content: str,
        *,
        repair: bool = True,
    ) -> QAAnswer:
        """答案后处理：guard 校验 → 必要时自动修复 → 记录对话 → 构建证据画像。"""
        guard_result = validate_answer(
            content,
            prepared.citations,
            has_retrieved_documents=prepared.has_retrieved_documents,
            verify_citation_semantics=self._citation_verification_enabled(prepared),
        )
        if repair and guard_result.needs_repair:
            content = self._repair_content(content, guard_result, prepared.citations)
            guard_result = validate_answer(
                content,
                prepared.citations,
                has_retrieved_documents=prepared.has_retrieved_documents,
                verify_citation_semantics=self._citation_verification_enabled(prepared),
            )
        self.record_prepared_answer(prepared, content)
        metadata: dict[str, Any] = {
            "evidence": build_evidence_profile(
                content,
                prepared.citations,
                guard_result.issues,
            ),
            "retrieval_queries": list(prepared.retrieval_queries),
        }
        if prepared.skill_context is not None:
            metadata.update(prepared.skill_context.metadata_payload())
        if prepared.evidence_sufficiency is not None:
            metadata["evidence_sufficiency"] = prepared.evidence_sufficiency.as_dict()
        if guard_result.citation_support is not None:
            metadata["citation_support"] = [
                {
                    "claim": check.claim,
                    "citation_ids": list(check.citation_ids),
                    "status": check.status,
                    "reason": check.reason,
                }
                for check in guard_result.citation_support.checks
            ]
        return QAAnswer(
            content=content,
            citations=prepared.citations,
            memories_used=prepared.memories_used,
            confidence=guard_result.confidence,
            guard_warnings=guard_result.issues,
            metadata=metadata,
        )

    # ── 记忆富化（同步，供 prepare / aprepare 共享） ──────────────────────

    def _enrich_memory_context(
        self,
        question: str,
        chat_history: list[dict[str, object]] | None,
        user_id: str | None,
        conversation_id: str | None,
        merge_persisted_history: bool,
    ) -> _MemoryEnrichment:
        """记忆富化：确保会话存在 → 加载历史 → 记录用户消息 → 检索相关记忆。

        失败时静默降级，不阻断主流程。
        """
        resolved_conversation_id = conversation_id
        memory_candidates: list[MemoryCandidate] = []
        memory_context = "No relevant user memory."
        user_message_recorded = False
        persisted_history: list[dict[str, str]] = []
        incoming_history = chat_history or []

        if self.memory_service and user_id:
            try:
                resolved_conversation_id = self.memory_service.ensure_context(
                    self.tenant_id,
                    user_id,
                    conversation_id,
                )
                if merge_persisted_history:
                    persisted_history = self.memory_service.load_conversation_history(
                        self.tenant_id,
                        user_id,
                        resolved_conversation_id,
                        limit=max(settings.chat_history_window, len(incoming_history)),
                    )
                message_id = self.memory_service.record_user_message(
                    tenant_id=self.tenant_id,
                    user_id=user_id,
                    conversation_id=resolved_conversation_id,
                    content=question,
                )
                user_message_recorded = True
                self.memory_service.write_memories_from_user_message(
                    tenant_id=self.tenant_id,
                    user_id=user_id,
                    conversation_id=resolved_conversation_id,
                    message_id=message_id,
                    content=question,
                )
                memory_candidates = self.memory_service.retrieve_relevant_memories(
                    tenant_id=self.tenant_id,
                    user_id=user_id,
                    query=question,
                )
                memory_context = self.memory_service.format_for_prompt(memory_candidates)
            except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
                logger.warning(
                    "Memory enrichment failed; continuing without memory context.",
                    extra={"tenant_id": self.tenant_id, "user_id": user_id, "memory_available": False},
                    exc_info=True,
                )
                memory_candidates = []
                memory_context = "No relevant user memory."

        return _MemoryEnrichment(
            resolved_conversation_id=resolved_conversation_id,
            memory_candidates=memory_candidates,
            memory_context=memory_context,
            user_message_recorded=user_message_recorded,
            persisted_history=persisted_history,
        )

    # ── 检索后组装（供 prepare / aprepare 共享） ──────────────────────────

    def _finalize_preparation(
        self,
        question: str,
        enrichment: _MemoryEnrichment,
        chat_history_text: str,
        documents: list[Document],
        task_id: str | None,
        user_id: str | None,
        skill_context: SkillRuntimeContext,
        retrieval_queries: list[str],
    ) -> PreparedQAAnswer:
        """从记忆富化结果 + 检索结果组装最终的 PreparedQAAnswer。"""
        memories_used = (
            self.memory_service.usages_from_candidates(enrichment.memory_candidates)
            if self.memory_service
            else []
        )
        evidence_assessment = (
            assess_evidence_sufficiency(question, documents)
            if "assess-evidence-sufficiency" in skill_context.selected_skills
            else None
        )
        runtime_guidance = self._runtime_guidance(skill_context, evidence_assessment)
        if not documents:
            task_prompt = self.general_prompt.format(
                question=question,
                chat_history=chat_history_text,
                user_memory=enrichment.memory_context,
            )
            if runtime_guidance:
                task_prompt = f"{task_prompt}\n\n{runtime_guidance}"
            return PreparedQAAnswer(
                messages=self._build_messages(task_prompt),
                citations=[],
                memories_used=memories_used,
                user_id=user_id,
                conversation_id=enrichment.resolved_conversation_id,
                user_message_recorded=enrichment.user_message_recorded,
                task_id=task_id,
                has_retrieved_documents=False,
                skill_context=skill_context,
                retrieval_queries=tuple(retrieval_queries),
                evidence_sufficiency=evidence_assessment,
            )

        context, citations = format_document_context(documents, prefix="S")
        task_prompt = self.prompt.format(
            question=question,
            context=context,
            chat_history=chat_history_text,
            user_memory=enrichment.memory_context,
        )
        if runtime_guidance:
            task_prompt = f"{task_prompt}\n\n{runtime_guidance}"
        return PreparedQAAnswer(
            messages=self._build_messages(task_prompt),
            citations=citations,
            memories_used=memories_used,
            user_id=user_id,
            conversation_id=enrichment.resolved_conversation_id,
            user_message_recorded=enrichment.user_message_recorded,
            task_id=task_id,
            has_retrieved_documents=True,
            skill_context=skill_context,
            retrieval_queries=tuple(retrieval_queries),
            evidence_sufficiency=evidence_assessment,
        )

    def prepare_answer(
        self,
        question: str,
        chat_history: list[dict[str, object]] | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        merge_persisted_history: bool = True,
        skill_names: tuple[str, ...] | None = None,
    ) -> PreparedQAAnswer:
        """问答准备阶段：加载记忆与历史 → 改写检索 query → 向量检索 → 组装 prompt。

        无检索结果时切换 general_prompt，避免模型假装有文档依据。
        记忆服务失败时静默降级，不阻断主流程。
        """
        enrichment = self._enrich_memory_context(
            question, chat_history, user_id, conversation_id, merge_persisted_history,
        )
        incoming_history = chat_history or []
        effective_chat_history = self._merge_chat_history(
            enrichment.persisted_history,
            incoming_history,
            max_messages=settings.chat_history_window,
        )
        chat_history_text = self._format_chat_history(
            effective_chat_history,
            max_messages=settings.chat_history_window,
        )
        retrieval_query = self._rewrite_query(question, chat_history_text)
        skill_context = self._prepare_skill_context(question, selected_names=skill_names)
        retrieval_queries = self._retrieval_queries(retrieval_query, skill_context)
        result_sets = [self.vector_store.search(query) for query in retrieval_queries]
        documents = fuse_retrieval_results(result_sets, top_k=settings.top_k)
        return self._finalize_preparation(
            question, enrichment, chat_history_text, documents, task_id, user_id,
            skill_context, retrieval_queries,
        )

    async def aprepare_answer(
        self,
        question: str,
        chat_history: list[dict[str, object]] | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        merge_persisted_history: bool = True,
        skill_names: tuple[str, ...] | None = None,
    ) -> PreparedQAAnswer:
        """异步版 prepare_answer：记忆富化与向量检索通过线程池执行。"""
        enrichment = await asyncio.to_thread(
            self._enrich_memory_context,
            question, chat_history, user_id, conversation_id, merge_persisted_history,
        )
        incoming_history = chat_history or []
        effective_chat_history = self._merge_chat_history(
            enrichment.persisted_history,
            incoming_history,
            max_messages=settings.chat_history_window,
        )
        chat_history_text = self._format_chat_history(
            effective_chat_history,
            max_messages=settings.chat_history_window,
        )
        retrieval_query = await asyncio.to_thread(
            self._rewrite_query, question, chat_history_text,
        )
        skill_context = self._prepare_skill_context(question, selected_names=skill_names)
        retrieval_queries = self._retrieval_queries(retrieval_query, skill_context)
        result_sets = await asyncio.gather(
            *(asyncio.to_thread(self.vector_store.search, query) for query in retrieval_queries)
        )
        documents = fuse_retrieval_results(result_sets, top_k=settings.top_k)
        return self._finalize_preparation(
            question, enrichment, chat_history_text, documents, task_id, user_id,
            skill_context, retrieval_queries,
        )

    def _prepare_skill_context(
        self,
        question: str,
        *,
        selected_names: tuple[str, ...] | None = None,
        phase: str = "qa",
    ) -> SkillRuntimeContext:
        if self.skill_engine is None:
            return SkillRuntimeContext()
        purpose = ""
        if _looks_like_document_question(question):
            purpose = "evidence grounded RAG answer citation support evidence sufficiency"
            if settings.skill_query_decomposition_enabled and is_complex_retrieval_query(question):
                purpose += " complex compound multi-hop retrieval subqueries"
        return self.skill_engine.prepare(
            question,
            phase=phase,
            purpose=purpose,
            selected_names=selected_names,
        )

    def prepare_skill_guidance(
        self,
        question: str,
        documents: list[Document],
        *,
        phase: str = "answer",
    ) -> tuple[SkillRuntimeContext, EvidenceSufficiencyResult | None, str]:
        """Select relevant skills and run the pre-generation evidence audit."""
        context = self._prepare_skill_context(question, phase=phase)
        assessment = (
            assess_evidence_sufficiency(question, documents)
            if "assess-evidence-sufficiency" in context.selected_skills
            else None
        )
        return context, assessment, self._runtime_guidance(context, assessment)

    @staticmethod
    def _retrieval_queries(
        retrieval_query: str,
        skill_context: SkillRuntimeContext,
    ) -> list[str]:
        if (
            settings.skill_query_decomposition_enabled
            and "decompose-retrieval-query" in skill_context.selected_skills
        ):
            return decompose_retrieval_query(
                retrieval_query,
                max_queries=settings.skill_max_retrieval_queries,
            )
        return [retrieval_query]

    @staticmethod
    def _runtime_guidance(
        skill_context: SkillRuntimeContext,
        evidence_assessment: EvidenceSufficiencyResult | None,
    ) -> str:
        blocks = [skill_context.rendered_instructions]
        if evidence_assessment is not None:
            blocks.append(render_evidence_assessment(evidence_assessment))
        return "\n\n".join(block for block in blocks if block)

    def stream_prepared_answer(self, prepared: PreparedQAAnswer) -> Iterator[str]:
        yield from self._stream_chat_messages(prepared.messages)

    def guard_streamed_answer(
        self,
        prepared: PreparedQAAnswer,
        content: str,
    ) -> AnswerGuardResult:
        return validate_answer(
            content,
            prepared.citations,
            has_retrieved_documents=prepared.has_retrieved_documents,
            verify_citation_semantics=self._citation_verification_enabled(prepared),
        )

    @staticmethod
    def _citation_verification_enabled(prepared: PreparedQAAnswer) -> bool:
        return bool(
            prepared.skill_context
            and "verify-citation-support" in prepared.skill_context.selected_skills
        )

    def record_prepared_answer(self, prepared: PreparedQAAnswer, content: str) -> None:
        self._record_assistant_message(
            user_id=prepared.user_id,
            conversation_id=prepared.conversation_id,
            task_id=prepared.task_id,
            content=content,
            user_message_recorded=prepared.user_message_recorded,
        )

    def review_clause(self, clause_type: str, top_k: int | None = None) -> QAAnswer:
        """按条款类型检索文档并输出结构化风险审查（高/中/低 + 理由 + 引用）。"""
        from doc_assistant.tools.legal_review import review_clause as _review

        return _review(self, clause_type, top_k)

    def check_conflict(
        self,
        contract_query: str,
        policy_query: str,
        top_k: int | None = None,
    ) -> QAAnswer:
        """分别检索合同与政策片段，比对义务/定义是否冲突并输出结构化结论。"""
        from doc_assistant.tools.legal_review import check_conflict as _check

        return _check(self, contract_query, policy_query, top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _build_messages(self, task_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.base_prompt},
            {"role": "user", "content": task_prompt},
        ]

    def repair_content(
        self,
        answer: str,
        guard_result: AnswerGuardResult,
        citations: list[Citation],
    ) -> str:
        """Public entry point for answer repair (e.g. from streaming / Agent paths)."""
        return self._repair_content(answer, guard_result, citations)

    def _repair_content(
        self,
        answer: str,
        guard_result: AnswerGuardResult,
        citations: list[Citation],
    ) -> str:
        if not guard_result.issues:
            return answer

        repaired, fixed_all = try_lightweight_repair(answer, guard_result, citations)
        if fixed_all:
            return repaired

        source_ids = ", ".join(c.source_id for c in citations) or "None"
        repair_prompt = self.answer_repair_prompt.format(
            issues="\n".join(f"- {issue}" for issue in guard_result.issues),
            source_ids=source_ids,
            answer=answer,
        )
        return self._invoke_chat_messages(self._build_messages(repair_prompt))

    def _rewrite_query(self, question: str, chat_history_text: str) -> str:
        if not settings.query_rewrite_enabled:
            return question
        if chat_history_text == "No previous messages.":
            return question
        normalized_question = question.casefold()
        if len(question) > 50 and not any(term in normalized_question for term in _VAGUE_QUERY_TERMS):
            return question
        if not any(term in normalized_question for term in _VAGUE_QUERY_TERMS):
            return question

        prompt = QUERY_REWRITE_PROMPT.format(
            chat_history=chat_history_text[-1000:],
            question=question,
        )
        try:
            rewritten = self._invoke_chat_messages(
                [
                    {"role": "system", "content": "You rewrite document retrieval queries."},
                    {"role": "user", "content": prompt},
                ]
            ).strip()
        except (OSError, RuntimeError, ValueError):
            logger.debug(
                "Query rewrite failed; using original question.",
                extra={"tenant_id": self.tenant_id},
                exc_info=True,
            )
            return question

        if not rewritten or len(rewritten) > 500:
            return question
        return rewritten

    def _invoke_chat_messages(self, messages: list[dict[str, str]]) -> str:
        return str(self.chat_model.invoke(messages).content or "")

    async def _ainvoke_chat_messages(self, messages: list[dict[str, str]]) -> str:
        response = await self.chat_model.ainvoke(messages)
        return str(response.content or "")

    def _stream_chat_messages(self, messages: list[dict[str, str]]) -> Iterator[str]:
        for chunk in self.chat_model.stream(messages):
            if chunk.content:
                yield str(chunk.content)

    def _record_assistant_message(
        self,
        *,
        user_id: str | None,
        conversation_id: str | None,
        task_id: str | None,
        content: str,
        user_message_recorded: bool,
    ) -> None:
        if not (self.memory_service and user_id and conversation_id and user_message_recorded):
            return
        try:
            self.memory_service.record_assistant_message(
                tenant_id=self.tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                content=content,
            )
        except (OSError, RuntimeError, ValueError):
            logger.warning(
                "Assistant message persistence failed.",
                extra={"tenant_id": self.tenant_id, "user_id": user_id},
                exc_info=True,
            )
            return

    @staticmethod
    def _format_chat_history(messages: list[dict[str, object]], max_messages: int = 12) -> str:
        return format_chat_history(messages, max_messages)

    @staticmethod
    def _merge_chat_history(
        persisted_history: list[dict[str, object]],
        incoming_history: list[dict[str, object]],
        *,
        max_messages: int,
    ) -> list[dict[str, object]]:
        return merge_chat_history(
            persisted_history,
            incoming_history,
            max_messages=max_messages,
        )


def _looks_like_document_question(question: str) -> bool:
    normalized = " ".join(question.split()).strip().casefold()
    if normalized in {
        "hello",
        "hi",
        "hey",
        "你好",
        "您好",
        "谢谢",
        "thanks",
        "thank you",
    }:
        return False
    return (
        "?" in normalized
        or "？" in normalized
        or len(normalized) >= 12
        or any(
            term in normalized
            for term in (
                "contract",
                "document",
                "clause",
                "agreement",
                "policy",
                "合同",
                "文档",
                "条款",
                "协议",
                "政策",
            )
        )
    )

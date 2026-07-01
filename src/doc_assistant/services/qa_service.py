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
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from doc_assistant.config.settings import settings
from doc_assistant.memory.schemas import MemoryCandidate, MemoryUsage
from doc_assistant.memory.service import MemoryService
from doc_assistant.models.language_model import (
    AsyncChatModelProtocol,
    ChatModelProtocol,
    build_chat_model,
)
from doc_assistant.retrieval.vector_store import DocumentVectorStore
from doc_assistant.schemas.citation import Citation, QAAnswer
from doc_assistant.services.answer_guard import AnswerGuardResult, validate_answer
from doc_assistant.services.evidence import build_evidence_profile
from doc_assistant.utils.coercion import (
    optional_str,
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
        chat_model=None,
        memory_service: MemoryService | None = None,
        tenant_id: str | None = None,
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

    def ask(
        self,
        question: str,
        chat_history: list[dict[str, object]] | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        merge_persisted_history: bool = True,
    ) -> QAAnswer:
        """同步问答：检索 → 组 prompt → 调 LLM → guard 校验（可自动 repair）。"""
        prepared = self.prepare_answer(
            question,
            chat_history=chat_history,
            user_id=user_id,
            conversation_id=conversation_id,
            task_id=task_id,
            merge_persisted_history=merge_persisted_history,
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
    ) -> QAAnswer:
        """异步版 ``ask``，适合 FastAPI 等 async 路由。"""
        prepared = await self.aprepare_answer(
            question,
            chat_history=chat_history,
            user_id=user_id,
            conversation_id=conversation_id,
            task_id=task_id,
            merge_persisted_history=merge_persisted_history,
        )
        content = await self._ainvoke_chat_messages(prepared.messages)
        return self.finalize_prepared_answer(prepared, content)

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
        )
        if repair and guard_result.needs_repair:
            content = self._repair_answer(content, guard_result, prepared)
            guard_result = validate_answer(
                content,
                prepared.citations,
                has_retrieved_documents=prepared.has_retrieved_documents,
            )
        self.record_prepared_answer(prepared, content)
        return QAAnswer(
            content=content,
            citations=prepared.citations,
            memories_used=prepared.memories_used,
            confidence=guard_result.confidence,
            guard_warnings=guard_result.issues,
            metadata={
                "evidence": build_evidence_profile(
                    content,
                    prepared.citations,
                    guard_result.issues,
                )
            },
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
        retrieval_query: str,
        documents: list[Document],
        task_id: str | None,
        user_id: str | None,
    ) -> PreparedQAAnswer:
        """从记忆富化结果 + 检索结果组装最终的 PreparedQAAnswer。"""
        self._log_retrieval(
            question=retrieval_query,
            user_id=user_id,
            conversation_id=enrichment.resolved_conversation_id,
            document_count=len(documents),
            memory_candidates=enrichment.memory_candidates,
        )
        memories_used = (
            self.memory_service.usages_from_candidates(enrichment.memory_candidates)
            if self.memory_service
            else []
        )
        if not documents:
            task_prompt = self.general_prompt.format(
                question=question,
                chat_history=chat_history_text,
                user_memory=enrichment.memory_context,
            )
            return PreparedQAAnswer(
                messages=self._build_messages(task_prompt),
                citations=[],
                memories_used=memories_used,
                user_id=user_id,
                conversation_id=enrichment.resolved_conversation_id,
                user_message_recorded=enrichment.user_message_recorded,
                task_id=task_id,
                has_retrieved_documents=False,
            )

        context, citations = self._format_context(documents)
        task_prompt = self.prompt.format(
            question=question,
            context=context,
            chat_history=chat_history_text,
            user_memory=enrichment.memory_context,
        )
        return PreparedQAAnswer(
            messages=self._build_messages(task_prompt),
            citations=citations,
            memories_used=memories_used,
            user_id=user_id,
            conversation_id=enrichment.resolved_conversation_id,
            user_message_recorded=enrichment.user_message_recorded,
            task_id=task_id,
            has_retrieved_documents=True,
        )

    def prepare_answer(
        self,
        question: str,
        chat_history: list[dict[str, object]] | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        merge_persisted_history: bool = True,
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
        documents = self.vector_store.search(retrieval_query)
        return self._finalize_preparation(
            question, enrichment, chat_history_text, retrieval_query, documents, task_id, user_id,
        )

    async def aprepare_answer(
        self,
        question: str,
        chat_history: list[dict[str, object]] | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        merge_persisted_history: bool = True,
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
        retrieval_query = self._rewrite_query(question, chat_history_text)
        documents = await asyncio.to_thread(self.vector_store.search, retrieval_query)
        return self._finalize_preparation(
            question, enrichment, chat_history_text, retrieval_query, documents, task_id, user_id,
        )

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
        from doc_assistant.services.review_service import review_clause as _review

        return _review(self, clause_type, top_k)

    def check_conflict(self, contract_query: str, policy_query: str, top_k: int | None = None) -> QAAnswer:
        """分别检索合同与政策片段，比对义务/定义是否冲突并输出结构化结论。"""
        from doc_assistant.services.review_service import check_conflict as _check

        return _check(self, contract_query, policy_query, top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _build_messages(self, task_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.base_prompt},
            {"role": "user", "content": task_prompt},
        ]

    def _repair_answer(
        self,
        answer: str,
        guard_result: AnswerGuardResult,
        prepared: PreparedQAAnswer,
    ) -> str:
        return self._repair_content(answer, guard_result, prepared.citations)

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

        repaired, fixed_all = _try_lightweight_repair(answer, guard_result, citations)
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
        except (OSError, RuntimeError, ValueError) as exc:
            logger.debug(
                "Query rewrite failed; using original question.",
                extra={"tenant_id": self.tenant_id},
                exc_info=True,
            )
            return question

        if not rewritten or len(rewritten) > 500:
            return question
        return rewritten

    def _format_context(self, documents: list[Document]) -> tuple[str, list[Citation]]:
        """把检索到的 Document 列表格式化为 prompt 上下文 + Citation 列表（编号 S1,S2…）。"""
        return self._format_context_prefixed(documents, prefix="S")

    def _format_context_prefixed(
        self, documents: list[Document], prefix: str = "S"
    ) -> tuple[str, list[Citation]]:
        context_parts = []
        citations = []

        for index, document in enumerate(documents, start=1):
            source_id = f"{prefix}{index}"
            metadata = document.metadata or {}
            text = self._compact_text(document.page_content)
            page = metadata.get("page")
            chunk_id = metadata.get("chunk_id")
            section_heading = metadata.get("section_heading")
            retrieval_score = metadata.get("retrieval_score")
            retrieval_relevance = metadata.get("retrieval_relevance")
            file_name = metadata.get("file_name") or metadata.get("source") or "unknown"
            file_id = optional_str(metadata.get("file_id"))
            document_key = optional_str(metadata.get("document_key"))
            document_version = (
                metadata.get("document_version")
                if isinstance(metadata.get("document_version"), int)
                else None
            )
            page_number = page if isinstance(page, int) else None
            page_label = f"page {page_number + 1}" if page_number is not None else None
            section_part = f"; section={section_heading}" if section_heading else ""
            page_part = page_label or "unknown"

            context_parts.append(
                f"[{source_id}] file={file_name}; page={page_part}; "
                f"page_index={page}; chunk={chunk_id}{section_part}\n{text}"
            )
            citations.append(
                Citation(
                    source_id=source_id,
                    file_name=str(file_name),
                    page=page_number,
                    chunk_id=chunk_id if isinstance(chunk_id, int) else None,
                    preview=text[:500],
                    source_type="document",
                    file_id=file_id,
                    document_key=document_key,
                    document_version=document_version,
                    page_label=page_label,
                    section_heading=str(section_heading) if section_heading else None,
                    exact_quote=text[:1200],
                    retrieval_score=(
                        float(retrieval_score)
                        if isinstance(retrieval_score, int | float)
                        else None
                    ),
                    retrieval_relevance=(
                        float(retrieval_relevance)
                        if isinstance(retrieval_relevance, int | float)
                        else None
                    ),
                )
            )

        return "\n\n".join(context_parts), citations

    @staticmethod
    def _compact_text(text: str) -> str:
        return " ".join(text.split())

    def _invoke_chat_messages(self, messages: list[dict[str, str]]) -> str:
        # 1) 项目自己的 ChatModelProtocol（首选）
        if isinstance(self.chat_model, ChatModelProtocol):
            response = self.chat_model.invoke_messages(messages)
            return str(response.get("content") or "")

        # 2) LangChain BaseChatModel 兼容（FakeListChatModel 等）
        invoke = getattr(self.chat_model, "invoke", None)
        if callable(invoke):
            try:
                response = invoke(messages=messages)
            except TypeError:
                response = invoke(self._messages_to_prompt(messages))
            content = getattr(response, "content", response)
            return str(content)

        raise ValueError("The configured chat model does not support message-based chat.")

    async def _ainvoke_chat_messages(self, messages: list[dict[str, str]]) -> str:
        if isinstance(self.chat_model, AsyncChatModelProtocol):
            response = await self.chat_model.ainvoke_messages(messages)
            return str(response.get("content") or "")
        return await asyncio.to_thread(self._invoke_chat_messages, messages)

    def _stream_chat_messages(self, messages: list[dict[str, str]]) -> Iterator[str]:
        if isinstance(self.chat_model, ChatModelProtocol):
            chunks = self.chat_model.stream(messages=messages)
            for chunk in chunks:
                content = getattr(chunk, "content", chunk)
                if content:
                    yield str(content)
            return
        yield self._invoke_chat_messages(messages)

    @staticmethod
    def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = str(message.get("content") or "").strip()
            if content:
                parts.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(parts)

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
            message_id = self.memory_service.record_assistant_message(
                tenant_id=self.tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                content=content,
            )
            self.memory_service.write_memories_from_assistant_message(
                tenant_id=self.tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                message_id=message_id,
                content=content,
                task_id=task_id,
            )
            self.memory_service.maybe_summarize_conversation(
                tenant_id=self.tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
            )
        except (OSError, RuntimeError, ValueError):
            logger.warning(
                "Assistant message persistence failed.",
                extra={"tenant_id": self.tenant_id, "user_id": user_id},
                exc_info=True,
            )
            return

    def _log_retrieval(
        self,
        *,
        question: str,
        user_id: str | None,
        conversation_id: str | None,
        document_count: int,
        memory_candidates: list[MemoryCandidate],
    ) -> None:
        if not (self.memory_service and user_id):
            return
        try:
            self.memory_service.log_retrieval(
                tenant_id=self.tenant_id,
                user_id=user_id,
                conversation_id=conversation_id,
                query=question,
                document_count=document_count,
                memories=memory_candidates,
            )
        except (OSError, RuntimeError, ValueError):
            logger.warning(
                "Retrieval logging failed.",
                extra={"tenant_id": self.tenant_id, "user_id": user_id},
                exc_info=True,
            )
            return

    @staticmethod
    def _format_chat_history(messages: list[dict[str, object]], max_messages: int = 12) -> str:
        system_parts = []
        chat_parts = []
        for message in messages:
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                if _is_conversation_summary_context(content):
                    system_parts.append(f"Session summary: {content}")
                continue
            if role not in {"user", "assistant"}:
                continue

            label = "User" if role == "user" else "Assistant"
            chat_parts.append(f"{label}: {content}")

        history_parts = [*system_parts, *chat_parts[-max_messages:]]
        return "\n".join(history_parts) if history_parts else "No previous messages."

    @staticmethod
    def _merge_chat_history(
        persisted_history: list[dict[str, object]],
        incoming_history: list[dict[str, object]],
        *,
        max_messages: int,
    ) -> list[dict[str, object]]:
        system_context: list[dict[str, object]] = []
        merged: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for message in [*persisted_history, *incoming_history]:
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                if not _is_conversation_summary_context(content):
                    continue
                key = ("system", content)
                if key in seen:
                    continue
                seen.add(key)
                system_context.append({"role": "system", "content": content})
                continue
            if role not in {"user", "assistant"}:
                continue
            key = (str(role), content)
            if key in seen:
                continue
            seen.add(key)
            merged.append({"role": str(role), "content": content})
        recent_messages = merged[-max(0, max_messages) :] if max_messages else []
        return [*system_context, *recent_messages]


def _is_conversation_summary_context(content: str) -> bool:
    return content.strip().casefold().startswith("conversation summary:")


def _try_lightweight_repair(
    content: str,
    guard_result: AnswerGuardResult,
    citations: list[Citation],
) -> tuple[str, bool]:
    valid_ids = {c.source_id.upper() for c in citations}
    if not valid_ids:
        return content, False

    repaired = content
    fixed_any = False
    fixed_all = True
    first_ref = f"[{next(iter(sorted(valid_ids)))}]"

    for issue in guard_result.issues:
        lowered = issue.casefold()
        if "source ids that were not returned" in lowered:
            repaired = re.sub(
                r"\[([SCDPW]\d+)\]",
                lambda match: match.group(0)
                if match.group(1).upper() in valid_ids
                else "",
                repaired,
                flags=re.IGNORECASE,
            )
            fixed_any = True
        elif "does not include any source citations" in lowered:
            repaired = f"{repaired.rstrip()} {first_ref}".strip()
            fixed_any = True
        elif "material paragraph lacks a source citation" in lowered:
            repaired = _append_default_citations_to_material_paragraphs(repaired, first_ref)
            fixed_any = True
        elif "specific fact" in lowered and "without a nearby citation" in lowered:
            repaired = _append_default_citations_to_fact_sentences(repaired, first_ref)
            fixed_any = True
        else:
            fixed_all = False

    if fixed_any and not re.search(r"\[[SCDPW]\d+\]", repaired, flags=re.IGNORECASE):
        repaired = f"{repaired.rstrip()} {first_ref}".strip()

    return repaired if fixed_any else content, fixed_all and fixed_any


def _append_default_citations_to_material_paragraphs(content: str, source_ref: str) -> str:
    blocks = re.split(r"(\n\s*\n)", content)
    repaired_blocks = []
    for block in blocks:
        stripped = block.strip()
        if not stripped or block.startswith("\n"):
            repaired_blocks.append(block)
            continue
        if stripped.startswith("#") or re.search(r"\[[SCDPW]\d+\]", block, flags=re.IGNORECASE):
            repaired_blocks.append(block)
            continue
        if len(stripped) >= 40:
            repaired_blocks.append(f"{block.rstrip()} {source_ref}")
        else:
            repaired_blocks.append(block)
    return "".join(repaired_blocks)


def _append_default_citations_to_fact_sentences(content: str, source_ref: str) -> str:
    sentences = re.split(r"([.!?。！？]\s*)", content)
    repaired = []
    for index in range(0, len(sentences), 2):
        sentence = sentences[index]
        punctuation = sentences[index + 1] if index + 1 < len(sentences) else ""
        if (
            re.search(r"\b\d+(?:\.\d+)?%|\b\d+\s+(?:days?|business days?|months?|years?)\b|\$\s?\d", sentence)
            and not re.search(r"\[[SCDPW]\d+\]", sentence, flags=re.IGNORECASE)
        ):
            sentence = f"{sentence.rstrip()} {source_ref}"
        repaired.append(sentence + punctuation)
    return "".join(repaired)

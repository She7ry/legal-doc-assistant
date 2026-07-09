"""工具调用聊天服务：LLM + 文档检索 / 条款审查 / 网页搜索 的 ReAct 循环。

与 ``DocumentQAService`` 的区别：模型可主动决定调用哪个 tool（LangGraph 状态机），
适合开放式对话；工具包括 search_documents、review_clause、web_search 等。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Annotated, Any

from langchain.tools import tool
from langchain_core.messages.utils import convert_to_messages
from langchain_core.tools import BaseTool, InjectedToolCallId
from pydantic import BaseModel, Field, field_validator

from doc_assistant.agent._helpers import _remap_metadata, _remap_source_refs
from doc_assistant.config.settings import settings
from doc_assistant.graphs.tool_calling import build_tool_calling_graph
from doc_assistant.grounding.evidence import build_evidence_profile
from doc_assistant.grounding.guard import validate_answer
from doc_assistant.memory.history import merge_chat_history
from doc_assistant.memory.schemas import MemoryCandidate, MemoryUsage
from doc_assistant.schemas.citation import Citation
from doc_assistant.services.qa_service import DocumentQAService
from doc_assistant.tools.document_search import DocumentSearchTool, SearchDocumentsInput
from doc_assistant.tools.web_search import (
    WebSearchClient,
    WebSearchInput,
    WebSearchTool,
    WebSource,
    build_web_search_client,
    web_source,
    web_source_citations,
)
from doc_assistant.utils.prompt_loader import load_base_legal_prompt, load_prompt

logger = logging.getLogger(__name__)


class ReviewClauseInput(BaseModel):
    clause_type: str = Field(min_length=1, max_length=120)
    top_k: int | None = Field(default=None, ge=1, le=10)
    tool_call_id: Annotated[str, InjectedToolCallId] = ""

    @field_validator("clause_type")
    @classmethod
    def clean_clause_type(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("clause_type is required")
        return value


class CheckConflictInput(BaseModel):
    contract_query: str = Field(min_length=1, max_length=500)
    policy_query: str = Field(min_length=1, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=10)
    tool_call_id: Annotated[str, InjectedToolCallId] = ""

    @field_validator("contract_query", "policy_query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        if not (value := value.strip()):
            raise ValueError("query is required")
        return value


def build_tool_system_prompt(user_memory: str | None = None) -> str:
    """组装 tool-calling 模式的系统 prompt：基础法律角色 + 工具说明 + 可选用户记忆。"""
    prompt = f"{load_base_legal_prompt()}\n\n{load_prompt('tool_calling_system.txt')}"
    if user_memory:
        prompt = f"{prompt}\n\n<user_memory>\n{user_memory}\n</user_memory>"
    return prompt


@dataclass(frozen=True)
class ToolCallTrace:
    """单次 tool 调用的审计记录：调用了什么、传了什么参数、返回了什么。"""

    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass(frozen=True)
class ToolCallingAnswer:
    """``ToolCallingChatService.ask()`` 的完整返回值。

    比 QAAnswer 多了 web_sources（网页引用）和 tool_calls（ReAct 轨迹），
    便于前端展示「模型调用了哪些工具」以及调试。
    """

    content: str
    citations: list[Citation] = field(default_factory=list)
    memories_used: list[MemoryUsage] = field(default_factory=list)
    web_sources: list[WebSource] = field(default_factory=list)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    confidence: str | None = None
    guard_warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ToolExecutionState:
    """一次 ask() 调用内的可变状态：累积引用、网页来源、tool 调用轨迹。

    仅在 LangGraph ToolNode 调用的工具中读写；每次 ask 新建实例。
    """

    citations: list[Citation] = field(default_factory=list)
    web_sources: list[WebSource] = field(default_factory=list)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    document_source_ids: dict[str, str] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)


@dataclass
class _ToolMemoryContext:
    """ask() 开始前准备好的记忆与对话上下文，贯穿整轮 tool-calling。"""

    user_id: str | None = None
    conversation_id: str | None = None
    task_id: str | None = None
    user_message_recorded: bool = False
    memory_candidates: list[MemoryCandidate] = field(default_factory=list)
    memory_context: str = "No relevant user memory."
    chat_history: list[dict[str, object]] = field(default_factory=list)


class ToolCallingChatService:
    """支持多轮工具调用的对话服务（ReAct 模式）。

    与 DocumentQAService 的区别：本类让 LLM **自主决定**何时检索文档、
    审查条款或搜索网页；QA 服务只负责执行具体 tool 逻辑。

    内部用 LangGraph（model ↔ tools 循环）替代手写 for 循环；
    每次 tool 调用会累积 Citation / WebSource，最终一并返回给前端。
    """

    def __init__(
        self,
        qa_service: DocumentQAService,
        web_search_client: WebSearchClient | None = None,
    ) -> None:
        self.qa_service = qa_service
        self.chat_model = qa_service.chat_model
        self.vector_store = qa_service.vector_store
        self.document_search_tool = DocumentSearchTool(
            self.vector_store,
            default_top_k=settings.top_k,
        )
        self.web_search_client = web_search_client or build_web_search_client()
        self.web_search_tool = WebSearchTool(
            self.web_search_client,
            default_max_results=settings.web_search_max_results,
        )

    def ask(
        self,
        question: str,
        *,
        chat_history: list[dict[str, object]] | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        enable_web_search: bool = False,
        max_tool_iterations: int | None = None,
    ) -> ToolCallingAnswer:
        """运行 LangGraph 工具调用循环：模型自主选 tool → 执行 → 合成带引用的答案。"""
        exec_state = _ToolExecutionState()
        memory_context = self._prepare_memory_context(
            question,
            chat_history=chat_history or [],
            user_id=user_id,
            conversation_id=conversation_id,
            task_id=task_id,
        )
        tools = self._build_tools(exec_state, enable_web_search=enable_web_search)
        messages = self._initial_messages(
            question,
            memory_context.chat_history,
            user_memory=memory_context.memory_context,
        )
        iterations = _clamp_int(
            max_tool_iterations or settings.tool_call_max_iterations,
            minimum=1,
            maximum=10,
        )

        graph = build_tool_calling_graph(
            llm=self.chat_model,
            tools=tools,
        )

        result = graph.invoke(
            {"messages": convert_to_messages(messages), "iteration": 0, "max_iterations": iterations},
            config={"recursion_limit": iterations * 2 + 5},
        )

        content = str(result["messages"][-1].content or "")
        return self._finalize_answer(content, exec_state, memory_context, question)

    def _finalize_answer(
        self,
        content: str,
        state: _ToolExecutionState,
        memory_context: _ToolMemoryContext,
        question: str,
    ) -> ToolCallingAnswer:
        guard_citations = state.citations + web_source_citations(state.web_sources)
        guard_result = validate_answer(
            content,
            guard_citations,
            has_retrieved_documents=bool(guard_citations),
        )
        if guard_result.needs_repair:
            content = self.qa_service.repair_content(content, guard_result, guard_citations)
            guard_result = validate_answer(
                content,
                guard_citations,
                has_retrieved_documents=bool(guard_citations),
            )

        self._record_memory_result(content, memory_context)
        memories_used = (
            self.qa_service.memory_service.usages_from_candidates(memory_context.memory_candidates)
            if self.qa_service.memory_service
            else []
        )
        return ToolCallingAnswer(
            content=content,
            citations=state.citations,
            memories_used=memories_used,
            web_sources=state.web_sources,
            tool_calls=state.tool_calls,
            confidence=guard_result.confidence,
            guard_warnings=guard_result.issues,
            metadata={
                "evidence": build_evidence_profile(content, guard_citations, guard_result.issues)
            },
        )

    def _initial_messages(
        self,
        question: str,
        chat_history: list[dict[str, object]],
        *,
        user_memory: str | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_tool_system_prompt(user_memory)}
        ]
        history_window = _clamp_int(settings.tool_call_history_window, minimum=0, maximum=100)
        system_history = []
        chat_messages = []
        for message in chat_history:
            role = message.get("role")
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                if content.casefold().startswith("conversation summary:"):
                    system_history.append({"role": "system", "content": content})
                continue
            if role in {"user", "assistant"}:
                chat_messages.append({"role": role, "content": content})
        messages.extend(system_history)
        for message in chat_messages[-history_window:] if history_window else []:
            messages.append(message)
        messages.append({"role": "user", "content": question})
        return messages

    def _prepare_memory_context(
        self,
        question: str,
        *,
        chat_history: list[dict[str, object]],
        user_id: str | None,
        conversation_id: str | None,
        task_id: str | None,
    ) -> _ToolMemoryContext:
        context = _ToolMemoryContext(user_id=user_id)
        memory_service = self.qa_service.memory_service
        if not (memory_service and user_id):
            context.chat_history = chat_history
            return context

        try:
            resolved_conversation_id = memory_service.ensure_context(
                self.qa_service.tenant_id,
                user_id,
                conversation_id,
            )
            persisted_history = memory_service.load_conversation_history(
                self.qa_service.tenant_id,
                user_id,
                resolved_conversation_id,
                limit=max(settings.tool_call_history_window, len(chat_history)),
            )
            message_id = memory_service.record_user_message(
                tenant_id=self.qa_service.tenant_id,
                user_id=user_id,
                conversation_id=resolved_conversation_id,
                content=question,
            )
            memory_service.write_memories_from_user_message(
                tenant_id=self.qa_service.tenant_id,
                user_id=user_id,
                conversation_id=resolved_conversation_id,
                message_id=message_id,
                content=question,
            )
            memory_candidates = memory_service.retrieve_relevant_memories(
                tenant_id=self.qa_service.tenant_id,
                user_id=user_id,
                query=question,
            )
            context.conversation_id = resolved_conversation_id
            context.task_id = task_id
            context.user_message_recorded = True
            context.memory_candidates = memory_candidates
            context.memory_context = memory_service.format_for_prompt(memory_candidates)
            context.chat_history = merge_chat_history(
                persisted_history,
                chat_history,
                max_messages=settings.tool_call_history_window,
            )
        except Exception:
            logger.warning(
                "Memory context preparation failed; continuing without memory.",
                extra={
                    "tenant_id": self.qa_service.tenant_id,
                    "user_id": user_id,
                    "memory_available": False,
                },
                exc_info=True,
            )
            context.chat_history = chat_history
        return context

    def _record_memory_result(
        self,
        content: str,
        memory_context: _ToolMemoryContext,
    ) -> None:
        memory_service = self.qa_service.memory_service
        if not (
            memory_service
            and memory_context.user_id
            and memory_context.conversation_id
            and memory_context.user_message_recorded
        ):
            return
        try:
            memory_service.record_assistant_message(
                tenant_id=self.qa_service.tenant_id,
                user_id=memory_context.user_id,
                conversation_id=memory_context.conversation_id,
                content=content,
            )
        except Exception:
            logger.warning(
                "Tool calling memory result recording failed.",
                extra={"tenant_id": self.qa_service.tenant_id, "user_id": memory_context.user_id},
                exc_info=True,
            )
            return

    def _build_tools(
        self,
        state: _ToolExecutionState,
        *,
        enable_web_search: bool,
    ) -> list[BaseTool]:
        @tool("search_documents", args_schema=SearchDocumentsInput)
        def search_documents(
            query: str,
            tool_call_id: Annotated[str, InjectedToolCallId],
            top_k: int | None = None,
        ) -> dict[str, Any]:
            """Search uploaded legal documents and return cited excerpts."""
            arguments = {"query": query, "top_k": top_k}
            return self._run_traced_tool(
                tool_call_id,
                "search_documents",
                arguments,
                state,
                lambda: self._search_documents(arguments, state),
            )

        @tool("review_clause", args_schema=ReviewClauseInput)
        def review_clause(
            clause_type: str,
            tool_call_id: Annotated[str, InjectedToolCallId],
            top_k: int | None = None,
        ) -> dict[str, Any]:
            """Review a clause type in the uploaded legal documents."""
            arguments = {"clause_type": clause_type, "top_k": top_k}
            return self._run_traced_tool(
                tool_call_id,
                "review_clause",
                arguments,
                state,
                lambda: self._review_clause(arguments, state),
            )

        @tool("check_conflict", args_schema=CheckConflictInput)
        def check_conflict(
            contract_query: str,
            policy_query: str,
            tool_call_id: Annotated[str, InjectedToolCallId],
            top_k: int | None = None,
        ) -> dict[str, Any]:
            """Compare contract excerpts with policy or compliance excerpts."""
            arguments = {
                "contract_query": contract_query,
                "policy_query": policy_query,
                "top_k": top_k,
            }
            return self._run_traced_tool(
                tool_call_id,
                "check_conflict",
                arguments,
                state,
                lambda: self._check_conflict(arguments, state),
            )

        tools: list[BaseTool] = [search_documents, review_clause, check_conflict]
        if enable_web_search:

            @tool("web_search", args_schema=WebSearchInput)
            def web_search(
                query: str,
                tool_call_id: Annotated[str, InjectedToolCallId],
                recency_days: int | None = None,
                domains: list[str] | None = None,
                max_results: int | None = None,
            ) -> dict[str, Any]:
                """Search public pages without exposing confidential document text."""
                arguments = {
                    "query": query,
                    "recency_days": recency_days,
                    "domains": domains or [],
                    "max_results": max_results,
                }
                return self._run_traced_tool(
                    tool_call_id,
                    "web_search",
                    arguments,
                    state,
                    lambda: self._web_search(arguments, state),
                )

            tools.append(web_search)
        return tools

    @staticmethod
    def _run_traced_tool(
        tool_call_id: str,
        name: str,
        arguments: dict[str, Any],
        state: _ToolExecutionState,
        handler,
    ) -> dict[str, Any]:
        try:
            result = handler()
        except Exception as exc:
            result = {"error": str(exc)}
            with state.lock:
                state.tool_calls.append(ToolCallTrace(tool_call_id, name, arguments, result))
            raise
        with state.lock:
            state.tool_calls.append(ToolCallTrace(tool_call_id, name, arguments, result))
        return result

    def _search_documents(
        self,
        arguments: dict[str, Any],
        state: _ToolExecutionState,
    ) -> dict[str, Any]:
        execution = self.document_search_tool.execute(arguments["query"], arguments.get("top_k"))

        results = []
        for hit in execution.hits:
            with state.lock:
                source_id = state.document_source_ids.get(hit.identity)
                is_new_source = source_id is None
                if source_id is None:
                    source_id = f"D{len(state.citations) + 1}"
                    state.document_source_ids[hit.identity] = source_id
                item = {"source_id": source_id, **hit.result}
                if is_new_source:
                    state.citations.append(
                        Citation(
                            source_id=source_id,
                            file_name=item["file_name"],
                            page=item["page"],
                            chunk_id=item["chunk_id"],
                            preview=item["content"][:500],
                            source_type="document",
                            file_id=item["file_id"],
                            document_key=item["document_key"],
                            document_version=item["document_version"],
                            page_label=item["page_label"],
                            section_heading=item["section_heading"],
                            exact_quote=item["content"][:1200],
                            retrieval_score=item["retrieval_score"],
                            retrieval_relevance=item["retrieval_relevance"],
                        )
                    )
            results.append(item)

        return {"query": execution.query, "result_count": len(results), "results": results}

    def _review_clause(
        self,
        arguments: dict[str, Any],
        state: _ToolExecutionState,
    ) -> dict[str, Any]:
        answer = self.qa_service.review_clause(
            str(arguments["clause_type"]),
            top_k=arguments.get("top_k"),
        )
        content, citations, metadata = _append_qa_answer(answer, state)
        return _qa_answer_tool_result(content, citations, answer.confidence, answer.guard_warnings, metadata)

    def _check_conflict(
        self,
        arguments: dict[str, Any],
        state: _ToolExecutionState,
    ) -> dict[str, Any]:
        answer = self.qa_service.check_conflict(
            str(arguments["contract_query"]),
            str(arguments["policy_query"]),
            top_k=arguments.get("top_k"),
        )
        content, citations, metadata = _append_qa_answer(answer, state)
        return _qa_answer_tool_result(content, citations, answer.confidence, answer.guard_warnings, metadata)

    def _web_search(
        self,
        arguments: dict[str, Any],
        state: _ToolExecutionState,
    ) -> dict[str, Any]:
        execution = self.web_search_tool.execute(
            arguments["query"],
            max_results=arguments.get("max_results"),
            recency_days=arguments.get("recency_days"),
            domains=arguments.get("domains"),
        )
        results = []
        for result in execution.results:
            with state.lock:
                source_id = f"W{len(state.web_sources) + 1}"
                source = web_source(source_id, result)
                state.web_sources.append(source)
            results.append(
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "url": source.url,
                    "snippet": source.snippet,
                    "published_at": source.published_at,
                    "source": source.source,
                }
            )

        return {"query": execution.query, "result_count": len(results), "results": results}


def _append_qa_answer(
    answer,
    state: _ToolExecutionState,
) -> tuple[str, list[Citation], dict[str, Any]]:
    mapping: dict[str, str] = {}
    remapped_citations: list[Citation] = []
    with state.lock:
        for citation in answer.citations:
            source_id = f"D{len(state.citations) + 1}"
            mapping[str(citation.source_id).upper()] = source_id
            remapped = replace(citation, source_id=source_id)
            state.citations.append(remapped)
            remapped_citations.append(remapped)
    return (
        _remap_source_refs(answer.content, mapping),
        remapped_citations,
        _remap_metadata(answer.metadata, mapping),
    )


def _qa_answer_tool_result(
    content: str,
    citations: list[Citation],
    confidence: str | None,
    guard_warnings: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "content": content,
        "citation_count": len(citations),
        "citations": [
            {
                "source_id": citation.source_id,
                "file_name": citation.file_name,
                "page": citation.page,
                "preview": citation.preview,
            }
            for citation in citations
        ],
        "confidence": confidence,
        "guard_warnings": guard_warnings,
        "metadata": metadata,
    }


def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))



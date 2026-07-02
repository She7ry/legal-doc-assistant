"""工具调用聊天服务：LLM + 文档检索 / 条款审查 / 网页搜索 的 ReAct 循环。

与 ``DocumentQAService`` 的区别：模型可主动决定调用哪个 tool（LangGraph 状态机），
适合开放式对话；工具包括 search_documents、review_clause、web_search 等。
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from doc_assistant.config.settings import settings
from doc_assistant.graphs.tool_calling import build_tool_calling_graph
from doc_assistant.memory.schemas import MemoryCandidate, MemoryUsage
from doc_assistant.models.langchain_adapter import ChatOpenAICompatible
from doc_assistant.schemas.citation import Citation
from doc_assistant.services.answer_guard import validate_answer
from doc_assistant.services.evidence import build_evidence_profile
from doc_assistant.services.qa_service import DocumentQAService
from doc_assistant.tools.web_search import (
    DisabledWebSearchClient,
    WebSearchClient,
    WebSearchResult,
    build_web_search_client,
)
from doc_assistant.utils.prompt_loader import load_base_legal_prompt, load_prompt

logger = logging.getLogger(__name__)


def build_tool_system_prompt(user_memory: str | None = None) -> str:
    """组装 tool-calling 模式的系统 prompt：基础法律角色 + 工具说明 + 可选用户记忆。"""
    prompt = f"{load_base_legal_prompt()}\n\n{load_prompt('tool_calling_system.txt')}"
    if user_memory:
        prompt = f"{prompt}\n\n<user_memory>\n{user_memory}\n</user_memory>"
    return prompt


@dataclass(frozen=True)
class WebSource:
    """网页搜索工具返回的单条外部来源（对应答案中的 [W1] 等编号）。"""

    source_id: str
    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None
    source: str | None = None


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

    仅在 LangGraph 的 execute_tools 回调中读写；每次 ask 新建实例。
    """

    citations: list[Citation] = field(default_factory=list)
    web_sources: list[WebSource] = field(default_factory=list)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    document_source_ids: dict[str, str] = field(default_factory=dict)


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
        from doc_assistant.models.language_model import MessageChatModelProtocol

        if not isinstance(self.chat_model, MessageChatModelProtocol):
            raise ValueError("The configured chat model does not support tool calling.")
        self.invoke_messages = self.chat_model.invoke_messages
        self._lc_chat_model = ChatOpenAICompatible(client=self.chat_model)
        self.vector_store = qa_service.vector_store
        self.web_search_client = web_search_client or build_web_search_client()
        self._tool_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool-call")

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
        tool_schemas = self._tool_schemas(enable_web_search=enable_web_search)
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

        def execute_tools_node(graph_state: dict) -> dict:
            last_message = graph_state["messages"][-1]
            tool_messages: list[ToolMessage] = []
            for tc in last_message.tool_calls:
                openai_tc = _lc_tool_call_to_openai(tc)
                result = self._execute_tool_call(openai_tc, exec_state, enable_web_search)
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                )
            return {"messages": tool_messages, "iteration": graph_state["iteration"] + 1}

        graph = build_tool_calling_graph(
            llm=self._lc_chat_model,
            tool_schemas=tool_schemas,
            execute_tools=execute_tools_node,
        )

        lc_messages = [_dict_to_lc_message(m) for m in messages]
        result = graph.invoke(
            {"messages": lc_messages, "iteration": 0, "max_iterations": iterations},
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
        guard_citations = state.citations + _web_source_citations(state.web_sources)
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

        self._record_memory_result(question, content, state, memory_context)
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
            context.chat_history = DocumentQAService._merge_chat_history(
                persisted_history,
                chat_history,
                max_messages=settings.tool_call_history_window,
            )
        except Exception:
            logger.warning(
                "Memory context preparation failed; continuing without memory.",
                extra={"tenant_id": self.qa_service.tenant_id, "user_id": user_id, "memory_available": False},
                exc_info=True,
            )
            context.chat_history = chat_history
        return context

    def _record_memory_result(
        self,
        question: str,
        content: str,
        state: _ToolExecutionState,
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
            message_id = memory_service.record_assistant_message(
                tenant_id=self.qa_service.tenant_id,
                user_id=memory_context.user_id,
                conversation_id=memory_context.conversation_id,
                content=content,
            )
            memory_service.write_memories_from_assistant_message(
                tenant_id=self.qa_service.tenant_id,
                user_id=memory_context.user_id,
                conversation_id=memory_context.conversation_id,
                message_id=message_id,
                content=content,
                task_id=memory_context.task_id,
            )
            memory_service.log_retrieval(
                tenant_id=self.qa_service.tenant_id,
                user_id=memory_context.user_id,
                conversation_id=memory_context.conversation_id,
                query=question,
                document_count=len(state.citations),
                memories=memory_context.memory_candidates,
            )
            memory_service.maybe_summarize_conversation(
                tenant_id=self.qa_service.tenant_id,
                user_id=memory_context.user_id,
                conversation_id=memory_context.conversation_id,
            )
        except Exception:
            logger.warning(
                "Tool calling memory result recording failed.",
                extra={"tenant_id": self.qa_service.tenant_id, "user_id": memory_context.user_id},
                exc_info=True,
            )
            return

    def _tool_schemas(self, *, enable_web_search: bool) -> list[dict[str, Any]]:
        return [schema for schema, _handler in self._enabled_tools(enable_web_search).values()]

    def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        state: _ToolExecutionState,
        enable_web_search: bool,
    ) -> dict[str, Any]:
        name = tool_call["function"]["name"]
        arguments = _parse_tool_arguments(tool_call)

        future = None
        try:
            future = self._tool_executor.submit(
                self._run_tool_call,
                name,
                arguments,
                state,
                enable_web_search,
            )
            result = future.result(timeout=max(1, settings.tool_call_timeout_seconds))
        except FutureTimeoutError:
            if future is not None:
                future.cancel()
            result = {
                "error": f"Tool execution timed out after {settings.tool_call_timeout_seconds} seconds."
            }
        except Exception as exc:
            result = {"error": str(exc)}

        state.tool_calls.append(
            ToolCallTrace(
                tool_call_id=tool_call["id"],
                name=name,
                arguments=arguments,
                result=result,
            )
        )
        return result

    def _enabled_tools(self, enable_web_search: bool) -> dict[str, tuple[dict[str, Any], Any]]:
        tools: dict[str, tuple[dict[str, Any], Any]] = {
            "search_documents": (SEARCH_DOCUMENTS_TOOL_SCHEMA, self._search_documents),
        }
        if enable_web_search:
            tools["web_search"] = (WEB_SEARCH_TOOL_SCHEMA, self._web_search)
        return tools

    def _run_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        state: _ToolExecutionState,
        enable_web_search: bool,
    ) -> dict[str, Any]:
        tools = self._enabled_tools(enable_web_search)
        match = tools.get(name)
        if match is None:
            if name == "web_search":
                raise RuntimeError("web_search was called but web search is not enabled.")
            return {"error": f"Unknown tool: {name}"}
        _schema, handler = match
        return handler(arguments, state)

    def _search_documents(
        self,
        arguments: dict[str, Any],
        state: _ToolExecutionState,
    ) -> dict[str, Any]:
        query = _required_string(arguments, "query", max_length=500)
        top_k = _clamp_int(int(arguments.get("top_k") or settings.top_k), minimum=1, maximum=10)
        documents = self.vector_store.search(query, k=top_k)

        results = []
        for document in documents:
            identity = _document_identity(document)
            source_id = state.document_source_ids.get(identity)
            is_new_source = source_id is None
            if source_id is None:
                source_id = f"D{len(state.citations) + 1}"
                state.document_source_ids[identity] = source_id
            item = _document_result(source_id, document)
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

        return {"query": query, "result_count": len(results), "results": results}

    def _web_search(
        self,
        arguments: dict[str, Any],
        state: _ToolExecutionState,
    ) -> dict[str, Any]:
        web_search_client = self.web_search_client
        if isinstance(web_search_client, DisabledWebSearchClient):
            raise RuntimeError("Web search is disabled. Set DOC_ASSISTANT_WEB_SEARCH_ENABLED=true.")

        query = _required_string(arguments, "query", max_length=300)
        top_k = _clamp_int(
            int(arguments.get("max_results") or settings.web_search_max_results),
            minimum=1,
            maximum=10,
        )
        recency_days = arguments.get("recency_days")
        if recency_days is not None:
            recency_days = _clamp_int(int(recency_days), minimum=1, maximum=365)
        domains = arguments.get("domains")
        if domains is not None and not isinstance(domains, list):
            raise ValueError("domains must be a list of domain strings.")
        clean_domains = [str(domain).strip() for domain in domains or [] if str(domain).strip()]

        search_results = web_search_client.search(
            query,
            max_results=top_k,
            recency_days=recency_days,
            domains=clean_domains,
        )
        results = []
        for result in search_results:
            source_id = f"W{len(state.web_sources) + 1}"
            source = _web_source(source_id, result)
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

        return {"query": query, "result_count": len(results), "results": results}


SEARCH_DOCUMENTS_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search uploaded/indexed legal documents and return cited excerpts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Focused search query for uploaded documents.",
                    "minLength": 1,
                    "maxLength": 500,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of document excerpts to retrieve.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

WEB_SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search public web pages for recent or external context. Do not include confidential "
            "contract excerpts in the query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Public web search query.",
                    "minLength": 1,
                    "maxLength": 300,
                },
                "recency_days": {
                    "type": "integer",
                    "description": "Optional recency window in days.",
                    "minimum": 1,
                    "maximum": 365,
                },
                "domains": {
                    "type": "array",
                    "description": "Optional domain filters such as sec.gov or court.gov.",
                    "items": {"type": "string"},
                    "maxItems": 5,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of web results to return.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def _normalise_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        return [_normalise_tool_call(call, index) for index, call in enumerate(tool_calls, start=1)]

    function_call = message.get("function_call")
    if function_call:
        return [
            {
                "id": "legacy_function_call_1",
                "type": "function",
                "function": function_call,
            }
        ]
    return []


def _normalise_tool_call(call: dict[str, Any], index: int) -> dict[str, Any]:
    function = call.get("function") or {}
    return {
        "id": call.get("id") or f"tool_call_{index}",
        "type": call.get("type") or "function",
        "function": {
            "name": function.get("name") or "",
            "arguments": function.get("arguments") or "{}",
        },
    }


def _parse_tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw_arguments = tool_call["function"].get("arguments") or "{}"
    if isinstance(raw_arguments, dict):
        return raw_arguments
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON arguments for {tool_call['function']['name']}.") from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"Arguments for {tool_call['function']['name']} must be a JSON object.")
    return arguments


def _document_result(source_id: str, document: Document) -> dict[str, Any]:
    metadata = document.metadata or {}
    content = _compact_text(document.page_content)[:1600]
    page = metadata.get("page")
    chunk_id = metadata.get("chunk_id")
    section_heading = metadata.get("section_heading")
    retrieval_score = metadata.get("retrieval_score")
    retrieval_relevance = metadata.get("retrieval_relevance")
    file_name = str(metadata.get("file_name") or metadata.get("source") or "unknown")
    page_number = page if isinstance(page, int) else None
    return {
        "source_id": source_id,
        "file_name": file_name,
        "file_id": _optional_string(metadata.get("file_id")),
        "document_key": _optional_string(metadata.get("document_key")),
        "document_version": (
            metadata.get("document_version")
            if isinstance(metadata.get("document_version"), int)
            else None
        ),
        "page": page_number,
        "page_label": f"page {page_number + 1}" if page_number is not None else None,
        "chunk_id": chunk_id if isinstance(chunk_id, int) else None,
        "section_heading": str(section_heading) if section_heading else None,
        "retrieval_score": retrieval_score if isinstance(retrieval_score, int | float) else None,
        "retrieval_relevance": (
            retrieval_relevance if isinstance(retrieval_relevance, int | float) else None
        ),
        "content": content,
    }


def _document_identity(document: Document) -> str:
    metadata = document.metadata or {}
    identity = {
        "file_id": _optional_string(metadata.get("file_id")),
        "document_key": _optional_string(metadata.get("document_key")),
        "document_version": metadata.get("document_version"),
        "page": metadata.get("page"),
        "chunk_id": metadata.get("chunk_id"),
        "source": _optional_string(metadata.get("source") or metadata.get("file_name")),
    }
    if any(value is not None for value in identity.values()):
        return json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return sha1(_compact_text(document.page_content).encode("utf-8")).hexdigest()


def _web_source(source_id: str, result: WebSearchResult) -> WebSource:
    return WebSource(
        source_id=source_id,
        title=result.title,
        url=result.url,
        snippet=result.snippet,
        published_at=result.published_at,
        source=result.source,
    )


def _web_source_citations(sources: list[WebSource]) -> list[Citation]:
    citations = []
    for source in sources:
        preview = source.snippet or source.title or source.url
        citations.append(
            Citation(
                source_id=source.source_id,
                file_name=source.title or source.url,
                preview=preview,
                source_type="web",
                exact_quote=preview,
            )
        )
    return citations


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_string(arguments: dict[str, Any], name: str, *, max_length: int) -> str:
    value = str(arguments.get(name) or "").strip()
    if not value:
        raise ValueError(f"{name} is required.")
    if len(value) > max_length:
        raise ValueError(f"{name} must be {max_length} characters or fewer.")
    return value


def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _compact_text(text: str) -> str:
    return " ".join(text.split())


def _dict_to_lc_message(msg: dict[str, Any]) -> BaseMessage:
    """Convert an OpenAI-style message dict to a LangChain BaseMessage."""
    role = msg.get("role", "user")
    content = str(msg.get("content") or "")
    if role == "system":
        return SystemMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    return HumanMessage(content=content)


def _lc_tool_call_to_openai(tc: dict[str, Any]) -> dict[str, Any]:
    """Convert a LangChain ToolCall dict back to OpenAI wire format."""
    args = tc.get("args", {})
    return {
        "id": tc["id"],
        "type": "function",
        "function": {
            "name": tc["name"],
            "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
        },
    }

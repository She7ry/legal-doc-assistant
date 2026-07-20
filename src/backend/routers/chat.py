from __future__ import annotations

import logging
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from ai.config.settings import settings
from ai.rag.qa_service import DocumentQAService, PreparedQAAnswer
from backend.dependencies import (
    MemoryServiceDep,
    QAServiceDep,
    ToolCallingServiceDep,
    UserIdDep,
)
from backend.routers.helpers import get_fields_set
from backend.schemas.requests import (
    AskRequest,
    ConversationCreateRequest,
    ConversationUpdateRequest,
    ToolChatRequest,
)
from backend.schemas.responses import (
    AskResponse,
    CitationOut,
    ConversationListResponse,
    ConversationMessageOut,
    ConversationMessagesResponse,
    ConversationOut,
    ToolCallOut,
    ToolChatResponse,
    WebSourceOut,
)
from backend.sse import SSE_HEADERS, format_sse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="List persisted conversations",
)
def list_conversations(
    memory_service: MemoryServiceDep,
    user_id: UserIdDep,
    status_filter: str | None = Query(default="active", alias="status"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> ConversationListResponse:
    """列出当前租户和用户下的持久化对话列表，支持按状态过滤（active/archived/all）和分页。"""
    resolved_status = None if status_filter in {"", "all"} else status_filter
    conversations = memory_service.list_conversations(
        user_id, status=resolved_status, limit=limit, offset=offset,
    )
    total = memory_service.count_conversations(
        user_id, status=resolved_status,
    )
    return ConversationListResponse(
        conversations=[ConversationOut.model_validate(c) for c in conversations],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.post(
    "/conversations",
    response_model=ConversationOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a persisted conversation",
)
def create_conversation(
    body: ConversationCreateRequest,
    memory_service: MemoryServiceDep,
    user_id: UserIdDep,
) -> ConversationOut:
    """创建一条新的持久化对话记录，可指定自定义 conversation_id 和标题。"""
    conversation = memory_service.create_conversation(
        user_id, conversation_id=body.conversation_id, title=body.title,
    )
    return ConversationOut.model_validate(conversation)


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationOut,
    summary="Update a persisted conversation",
)
def update_conversation(
    conversation_id: str,
    body: ConversationUpdateRequest,
    memory_service: MemoryServiceDep,
    user_id: UserIdDep,
) -> ConversationOut:
    """更新指定对话的标题或状态（如归档）。请求体至少需要提供一个待更新字段，若对话不存在则返回 404。"""
    fields_set = get_fields_set(body)
    if not fields_set:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="At least one conversation field must be provided.",
        )
    conversation = memory_service.update_conversation(
        user_id, conversation_id,
        title=body.title if "title" in fields_set else None,
        status=body.status if "status" in fields_set else None,
    )
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return ConversationOut.model_validate(conversation)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=ConversationMessagesResponse,
    summary="Load persisted conversation messages",
)
def get_conversation_messages(
    conversation_id: str,
    memory_service: MemoryServiceDep,
    user_id: UserIdDep,
    limit: int = Query(default=50, ge=1, le=200),
) -> ConversationMessagesResponse:
    """加载指定对话的历史消息记录，用于恢复对话上下文，支持 limit 控制返回条数。"""
    messages = memory_service.load_conversation_history(
        user_id, conversation_id, limit=limit, include_summary=False,
    )
    return ConversationMessagesResponse(
        conversation_id=conversation_id,
        messages=[ConversationMessageOut(**m) for m in messages],
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about indexed documents",
)
async def ask(body: AskRequest, qa_service: QAServiceDep, user_id: UserIdDep) -> AskResponse:
    """向已索引的法律文档提问（基于 RAG 检索增强生成），返回答案内容、引用来源及置信度。"""
    history = [{"role": m.role, "content": m.content} for m in body.chat_history]
    answer = await qa_service.aask(
        body.question,
        chat_history=history,
        user_id=user_id,
        conversation_id=body.conversation_id,
        task_id=body.task_id,
    )
    return AskResponse(
        content=answer.content,
        citations=[CitationOut.from_citation(c) for c in answer.citations],
        confidence=answer.confidence,
        guard_warnings=answer.guard_warnings,
        evidence=answer.metadata.get("evidence"),
    )


@router.post(
    "/tools",
    response_model=ToolChatResponse,
    summary="Ask a question with model-driven tool calling",
)
def ask_with_tools(
    body: ToolChatRequest,
    tool_service: ToolCallingServiceDep,
    user_id: UserIdDep,
) -> ToolChatResponse:
    """带工具调用能力的问答接口：模型可自主调用检索、网络搜索等工具获取补充信息后再生成回答。
    返回结果额外包含 web_sources 和 tool_calls 追踪信息。"""
    if body.enable_web_search and not settings.web_search_enabled:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Web search is disabled. Set DOC_ASSISTANT_WEB_SEARCH_ENABLED=true.",
        )

    history = [{"role": m.role, "content": m.content} for m in body.chat_history]
    answer = tool_service.ask(
        body.question,
        chat_history=history,
        user_id=user_id,
        conversation_id=body.conversation_id,
        task_id=body.task_id,
        enable_web_search=body.enable_web_search,
        max_tool_iterations=body.max_tool_iterations,
    )
    return ToolChatResponse(
        content=answer.content,
        citations=[CitationOut.from_citation(c) for c in answer.citations],
        web_sources=[WebSourceOut.model_validate(s) for s in answer.web_sources],
        tool_calls=[ToolCallOut.model_validate(t) for t in answer.tool_calls],
        confidence=answer.confidence,
        guard_warnings=answer.guard_warnings,
        evidence=answer.metadata.get("evidence"),
    )


@router.post(
    "/ask/stream",
    summary="Ask a question and stream the answer as server-sent events",
)
def ask_stream(body: AskRequest, qa_service: QAServiceDep, user_id: UserIdDep) -> StreamingResponse:
    """流式问答接口：通过 Server-Sent Events 逐块推送答案。
    流程：先发送引用元数据，再逐块发送 delta 文本，流结束后发送 guard 校验结果和完整答案。"""
    history = [{"role": m.role, "content": m.content} for m in body.chat_history]
    prepared = qa_service.prepare_answer(
        body.question,
        chat_history=history,
        user_id=user_id,
        conversation_id=body.conversation_id,
        task_id=body.task_id,
    )
    return StreamingResponse(
        _stream_answer_events(qa_service, prepared),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _stream_answer_events(
    qa_service: DocumentQAService,
    prepared: PreparedQAAnswer,
) -> Iterator[str]:
    """内部 SSE 事件生成器：将预处理好的答案转为 SSE 事件流。
    依次产出：metadata（引用）→ delta（增量文本块）→ guard_result（安全校验）→ done（最终完整答案）。
    若流式生成过程中出现异常，产出 error 事件并终止。"""
    yield format_sse(
        "metadata",
        {
            "citations": [CitationOut.from_citation(c) for c in prepared.citations],
        },
    )

    chunks: list[str] = []
    try:
        for chunk in qa_service.stream_prepared_answer(prepared):
            chunks.append(chunk)
            yield format_sse("delta", {"content": chunk})
    except Exception:
        logger.exception("Answer stream failed", extra={"task_id": prepared.task_id})
        yield format_sse(
            "error",
            {
                "code": "stream_error",
                "detail": "Answer stream failed.",
                "task_id": prepared.task_id,
            },
        )
        return

    content = "".join(chunks)
    guard_result = qa_service.guard_streamed_answer(prepared, content)
    yield format_sse(
        "guard_result",
        {
            "confidence": guard_result.confidence,
            "issues": guard_result.issues,
            "needs_repair": guard_result.needs_repair,
        },
    )
    answer = qa_service.finalize_prepared_answer(prepared, content)
    yield format_sse(
        "done",
        {
            "content": answer.content,
            "citations": [CitationOut.from_citation(c) for c in answer.citations],
            "confidence": answer.confidence,
            "guard_warnings": answer.guard_warnings,
            "evidence": answer.metadata.get("evidence"),
        },
    )

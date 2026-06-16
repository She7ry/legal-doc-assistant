from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from api.dependencies import (
    MemoryServiceDep,
    QAServiceDep,
    TenantIdDep,
    ToolCallingServiceDep,
    UserIdDep,
    require_api_key,
)
from api.schemas.requests import (
    AskRequest,
    ConversationCreateRequest,
    ConversationUpdateRequest,
    ToolChatRequest,
)
from api.schemas.responses import (
    AskResponse,
    CitationOut,
    ConversationListResponse,
    ConversationMessageOut,
    ConversationMessagesResponse,
    ConversationOut,
    MemoryUsageOut,
    ToolCallOut,
    ToolChatResponse,
    WebSourceOut,
)
from doc_assistant.config.settings import settings
from doc_assistant.services.qa_service import DocumentQAService, PreparedQAAnswer

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(require_api_key)])


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="List persisted conversations",
)
def list_conversations(
    memory_service: MemoryServiceDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
    status_filter: str | None = Query(default="active", alias="status"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> ConversationListResponse:
    resolved_status = None if status_filter in {"", "all"} else status_filter
    try:
        conversations = memory_service.list_conversations(
            tenant_id,
            user_id,
            status=resolved_status,
            limit=limit,
            offset=offset,
        )
        total = memory_service.count_conversations(
            tenant_id,
            user_id,
            status=resolved_status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return ConversationListResponse(
        conversations=[ConversationOut.from_conversation(conversation) for conversation in conversations],
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
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> ConversationOut:
    try:
        conversation = memory_service.create_conversation(
            tenant_id,
            user_id,
            conversation_id=body.conversation_id,
            title=body.title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return ConversationOut.from_conversation(conversation)


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationOut,
    summary="Update a persisted conversation",
)
def update_conversation(
    conversation_id: str,
    body: ConversationUpdateRequest,
    memory_service: MemoryServiceDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> ConversationOut:
    fields_set = getattr(body, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(body, "__fields_set__", set())
    if not fields_set:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one conversation field must be provided.",
        )
    try:
        conversation = memory_service.update_conversation(
            tenant_id,
            user_id,
            conversation_id,
            title=body.title if "title" in fields_set else None,
            status=body.status if "status" in fields_set else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return ConversationOut.from_conversation(conversation)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=ConversationMessagesResponse,
    summary="Load persisted conversation messages",
)
def get_conversation_messages(
    conversation_id: str,
    memory_service: MemoryServiceDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
    limit: int = Query(default=50, ge=1, le=200),
) -> ConversationMessagesResponse:
    messages = memory_service.load_conversation_history(
        tenant_id,
        user_id,
        conversation_id,
        limit=limit,
        include_summary=False,
    )
    return ConversationMessagesResponse(
        conversation_id=conversation_id,
        messages=[ConversationMessageOut(**message) for message in messages],
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about indexed documents",
)
async def ask(body: AskRequest, qa_service: QAServiceDep, user_id: UserIdDep) -> AskResponse:
    """
    Ask a question. If documents are indexed, answers are grounded in retrieved
    excerpts with [S1]/[S2] citations. Otherwise falls back to general chat.
    """
    history = [{"role": m.role, "content": m.content} for m in body.chat_history]
    try:
        answer = await qa_service.aask(
            body.question,
            chat_history=history,
            user_id=user_id,
            conversation_id=body.conversation_id,
            task_id=body.task_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return AskResponse(
        content=answer.content,
        citations=[CitationOut.from_citation(c) for c in answer.citations],
        memories_used=[MemoryUsageOut.from_usage(memory) for memory in answer.memories_used],
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
    """
    Let the model call controlled tools during generation.

    Tools:
    - search_documents: search uploaded/indexed documents and return [D#] sources
    - web_search: search public web pages and return [W#] sources when enabled
    """
    if body.enable_web_search and not settings.web_search_enabled:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Web search is disabled. Set DOC_ASSISTANT_WEB_SEARCH_ENABLED=true.",
        )

    history = [{"role": m.role, "content": m.content} for m in body.chat_history]
    try:
        answer = tool_service.ask(
            body.question,
            chat_history=history,
            user_id=user_id,
            conversation_id=body.conversation_id,
            task_id=body.task_id,
            enable_web_search=body.enable_web_search,
            max_tool_iterations=body.max_tool_iterations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return ToolChatResponse(
        content=answer.content,
        citations=[CitationOut.from_citation(c) for c in answer.citations],
        memories_used=[MemoryUsageOut.from_usage(memory) for memory in answer.memories_used],
        web_sources=[WebSourceOut.from_source(source) for source in answer.web_sources],
        tool_calls=[ToolCallOut.from_trace(trace) for trace in answer.tool_calls],
        confidence=answer.confidence,
        guard_warnings=answer.guard_warnings,
        evidence=answer.metadata.get("evidence"),
    )


@router.post(
    "/ask/stream",
    summary="Ask a question and stream the answer as server-sent events",
)
def ask_stream(body: AskRequest, qa_service: QAServiceDep, user_id: UserIdDep) -> StreamingResponse:
    """
    Stream answer tokens as SSE.

    Events:
    - metadata: citations and memories selected before generation
    - delta: incremental answer text
    - done: final answer payload
    - error: generation error after the stream has started
    """
    history = [{"role": m.role, "content": m.content} for m in body.chat_history]
    try:
        prepared = qa_service.prepare_answer(
            body.question,
            chat_history=history,
            user_id=user_id,
            conversation_id=body.conversation_id,
            task_id=body.task_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return StreamingResponse(
        _stream_answer_events(qa_service, prepared),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _stream_answer_events(
    qa_service: DocumentQAService,
    prepared: PreparedQAAnswer,
) -> Iterator[str]:
    yield _sse(
        "metadata",
        {
            "citations": [CitationOut.from_citation(citation) for citation in prepared.citations],
            "memories_used": [MemoryUsageOut.from_usage(memory) for memory in prepared.memories_used],
        },
    )

    chunks: list[str] = []
    try:
        for chunk in qa_service.stream_prepared_answer(prepared):
            chunks.append(chunk)
            yield _sse("delta", {"content": chunk})
    except Exception as exc:
        yield _sse("error", {"code": "stream_error", "detail": str(exc)})
        return

    content = "".join(chunks)
    guard_result = qa_service.guard_streamed_answer(prepared, content)
    yield _sse(
        "guard_result",
        {
            "confidence": guard_result.confidence,
            "issues": guard_result.issues,
            "needs_repair": guard_result.needs_repair,
        },
    )
    answer = qa_service.finalize_prepared_answer(prepared, content)
    yield _sse(
        "done",
        {
            "content": answer.content,
            "citations": [CitationOut.from_citation(citation) for citation in answer.citations],
            "memories_used": [
                MemoryUsageOut.from_usage(memory) for memory in answer.memories_used
            ],
            "confidence": answer.confidence,
            "guard_warnings": answer.guard_warnings,
            "evidence": answer.metadata.get("evidence"),
        },
    )


def _sse(event: str, data: object) -> str:
    payload = json.dumps(jsonable_encoder(data), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

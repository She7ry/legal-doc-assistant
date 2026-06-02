from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import QAServiceDep, UserIdDep, require_api_key
from api.schemas.requests import AskRequest
from api.schemas.responses import AskResponse, CitationOut, MemoryUsageOut

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(require_api_key)])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about indexed documents",
)
def ask(body: AskRequest, qa_service: QAServiceDep, user_id: UserIdDep) -> AskResponse:
    """
    Ask a question. If documents are indexed, answers are grounded in retrieved
    excerpts with [S1]/[S2] citations. Otherwise falls back to general chat.
    """
    history = [{"role": m.role, "content": m.content} for m in body.chat_history]
    try:
        answer = qa_service.ask(
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
    )

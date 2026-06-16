from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import MemoryServiceDep, TenantIdDep, UserIdDep, require_api_key
from api.schemas.requests import FeedbackCreateRequest
from api.schemas.responses import FeedbackResponse

router = APIRouter(
    prefix="/feedback",
    tags=["feedback"],
    dependencies=[Depends(require_api_key)],
)


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def create_feedback(
    body: FeedbackCreateRequest,
    memory_service: MemoryServiceDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> FeedbackResponse:
    try:
        event, adjustments = memory_service.record_feedback(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=body.conversation_id,
            message_id=body.message_id,
            rating=body.rating,
            memory_ids=body.memory_ids,
            comment=body.comment,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return FeedbackResponse.from_feedback(event, adjustments)

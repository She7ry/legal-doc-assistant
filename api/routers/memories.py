from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import MemoryServiceDep, TenantIdDep, UserIdDep, require_api_key
from api.schemas.requests import (
    MemoryBatchCreateRequest,
    MemoryBatchDeleteRequest,
    MemoryConversationSummaryRequest,
    MemoryCreateRequest,
    MemoryUpdateRequest,
)
from api.schemas.responses import (
    MemoryBatchDeleteResponse,
    MemoryListResponse,
    MemoryMaintenanceResponse,
    MemoryOut,
    MemoryStatsResponse,
)
from doc_assistant.memory.schemas import UNSET, MemoryUpdate

router = APIRouter(
    prefix="/memories",
    tags=["memories"],
    dependencies=[Depends(require_api_key)],
)


@router.get("", response_model=MemoryListResponse, summary="List user memories")
def list_memories(
    memory_service: MemoryServiceDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
    status_filter: str | None = Query(default="active", alias="status"),
    include_expired: bool = False,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> MemoryListResponse:
    memories = memory_service.list_memories(
        tenant_id,
        user_id,
        status=status_filter,
        include_expired=include_expired,
        limit=limit,
        offset=offset,
    )
    total = memory_service.count_memories(
        tenant_id,
        user_id,
        status=status_filter,
        include_expired=include_expired,
    )
    return MemoryListResponse(
        memories=[MemoryOut.from_memory(memory) for memory in memories],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/stats", response_model=MemoryStatsResponse, summary="Get memory system statistics")
def memory_stats(
    memory_service: MemoryServiceDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> MemoryStatsResponse:
    return MemoryStatsResponse(**memory_service.get_memory_stats(tenant_id, user_id))


@router.post("", response_model=MemoryOut, status_code=status.HTTP_201_CREATED, summary="Create memory")
def create_memory(
    body: MemoryCreateRequest,
    memory_service: MemoryServiceDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> MemoryOut:
    try:
        memory = memory_service.create_memory(
            tenant_id=tenant_id,
            user_id=user_id,
            scope=body.scope,
            type=body.type,
            key=body.key,
            content=body.content,
            value_json=body.value,
            source=body.source,
            confidence=body.confidence,
            expires_at=body.expires_at,
            visibility=body.visibility,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return MemoryOut.from_memory(memory)


@router.post("/batch", response_model=list[MemoryOut], status_code=status.HTTP_201_CREATED, summary="Create memories")
def create_memories(
    body: MemoryBatchCreateRequest,
    memory_service: MemoryServiceDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> list[MemoryOut]:
    try:
        memories = [
            memory_service.create_memory(
                tenant_id=tenant_id,
                user_id=user_id,
                scope=item.scope,
                type=item.type,
                key=item.key,
                content=item.content,
                value_json=item.value,
                source=item.source,
                confidence=item.confidence,
                expires_at=item.expires_at,
                visibility=item.visibility,
            )
            for item in body.memories
        ]
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return [MemoryOut.from_memory(memory) for memory in memories]


@router.patch("/{memory_id}", response_model=MemoryOut, summary="Update memory")
def update_memory(
    memory_id: str,
    body: MemoryUpdateRequest,
    memory_service: MemoryServiceDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> MemoryOut:
    try:
        memory = memory_service.update_memory(
            tenant_id,
            user_id,
            memory_id,
            _memory_update_from_body(body),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found.")
    return MemoryOut.from_memory(memory)


@router.post("/batch-delete", response_model=MemoryBatchDeleteResponse, summary="Delete memories")
def delete_memories(
    body: MemoryBatchDeleteRequest,
    memory_service: MemoryServiceDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> MemoryBatchDeleteResponse:
    deleted = []
    not_found = []
    for memory_id in body.memory_ids:
        memory = memory_service.delete_memory(tenant_id, user_id, memory_id)
        if memory is None:
            not_found.append(memory_id)
        else:
            deleted.append(MemoryOut.from_memory(memory))
    return MemoryBatchDeleteResponse(
        deleted=deleted,
        not_found=not_found,
        total_deleted=len(deleted),
    )


@router.post(
    "/maintenance",
    response_model=MemoryMaintenanceResponse,
    summary="Run memory maintenance",
)
def run_memory_maintenance(
    memory_service: MemoryServiceDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> MemoryMaintenanceResponse:
    expired = memory_service.cleanup_expired_memories(tenant_id, user_id)
    limit_stale = memory_service.enforce_memory_limit(tenant_id, user_id)
    vector_result = memory_service.repair_vector_index(tenant_id, user_id)
    return MemoryMaintenanceResponse(
        expired_stale=len(expired),
        limit_stale=len(limit_stale),
        vector_deleted=vector_result["deleted"],
        vector_upserted=vector_result["upserted"],
    )


@router.post(
    "/summarize-conversation",
    response_model=MemoryOut,
    status_code=status.HTTP_201_CREATED,
    summary="Summarize conversation into session memory",
)
def summarize_conversation(
    body: MemoryConversationSummaryRequest,
    memory_service: MemoryServiceDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> MemoryOut:
    memory = memory_service.summarize_conversation_to_memory(
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=body.conversation_id,
        limit=body.limit,
    )
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation history not found.")
    return MemoryOut.from_memory(memory)


@router.delete("/{memory_id}", response_model=MemoryOut, status_code=status.HTTP_200_OK, summary="Delete memory")
def delete_memory(
    memory_id: str,
    memory_service: MemoryServiceDep,
    tenant_id: TenantIdDep,
    user_id: UserIdDep,
) -> MemoryOut:
    memory = memory_service.delete_memory(tenant_id, user_id, memory_id)
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found.")
    return MemoryOut.from_memory(memory)


def _memory_update_from_body(body: MemoryUpdateRequest) -> MemoryUpdate:
    fields_set = getattr(body, "model_fields_set", getattr(body, "__fields_set__", set()))
    return MemoryUpdate(
        key=body.key,
        content=body.content,
        value_json=body.value if "value" in fields_set else UNSET,
        source=body.source,  # type: ignore[arg-type]
        confidence=body.confidence,
        expires_at=body.expires_at if "expires_at" in fields_set else UNSET,
        visibility=body.visibility,  # type: ignore[arg-type]
        status=body.status,  # type: ignore[arg-type]
    )

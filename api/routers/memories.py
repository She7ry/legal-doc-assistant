from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import MemoryServiceDep, TenantIdDep, UserIdDep, require_api_key
from api.schemas.requests import MemoryCreateRequest, MemoryUpdateRequest
from api.schemas.responses import MemoryListResponse, MemoryOut
from doc_assistant.memory.schemas import MemoryUpdate

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
) -> MemoryListResponse:
    memories = memory_service.list_memories(
        tenant_id,
        user_id,
        status=status_filter,
        include_expired=include_expired,
    )
    return MemoryListResponse(
        memories=[MemoryOut.from_memory(memory) for memory in memories],
        total=len(memories),
    )


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
            MemoryUpdate(
                key=body.key,
                content=body.content,
                value_json=body.value,
                source=body.source,  # type: ignore[arg-type]
                confidence=body.confidence,
                expires_at=body.expires_at,
                visibility=body.visibility,  # type: ignore[arg-type]
                status=body.status,  # type: ignore[arg-type]
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found.")
    return MemoryOut.from_memory(memory)


@router.delete("/{memory_id}", response_model=MemoryOut, summary="Delete memory")
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


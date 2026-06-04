from __future__ import annotations

from functools import lru_cache
import re
import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.jobs import IngestJobStore
from doc_assistant.config.settings import settings
from doc_assistant.memory.service import MemoryService
from doc_assistant.memory.store import MemoryStore
from doc_assistant.memory.vector_store import MemoryVectorStore
from doc_assistant.retrieval.vector_store import DocumentVectorStore
from doc_assistant.services.qa_service import DocumentQAService
from doc_assistant.services.tool_calling_service import ToolCallingChatService

_TENANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{0,126}$")
_bearer_scheme = HTTPBearer(auto_error=False)
_job_store = IngestJobStore()
_memory_store = MemoryStore()


def normalize_tenant_id(value: str | None) -> str:
    tenant_id = (value or settings.default_tenant_id).strip()
    if not _TENANT_ID_PATTERN.fullmatch(tenant_id):
        raise ValueError(
            "Invalid tenant id. Use 1-63 characters: letters, numbers, dot, underscore, or hyphen."
        )
    return tenant_id


def get_tenant_id(
    x_tenant_id: Annotated[str | None, Header(alias="X-Tenant-Id")] = None,
) -> str:
    try:
        return normalize_tenant_id(x_tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


def normalize_user_id(value: str | None) -> str:
    user_id = (value or "local-user").strip()
    if not _USER_ID_PATTERN.fullmatch(user_id):
        raise ValueError(
            "Invalid user id. Use 1-127 characters: letters, numbers, dot, underscore, at, or hyphen."
        )
    return user_id


def get_user_id(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> str:
    try:
        return normalize_user_id(x_user_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)] = None,
) -> None:
    if not settings.api_keys:
        return

    token = x_api_key
    if token is None and credentials and credentials.scheme.casefold() == "bearer":
        token = credentials.credentials

    if token and any(secrets.compare_digest(token, api_key) for api_key in settings.api_keys):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="A valid API key is required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


@lru_cache(maxsize=128)
def _vector_store(tenant_id: str | None = None) -> DocumentVectorStore:
    return DocumentVectorStore(tenant_id=normalize_tenant_id(tenant_id))


@lru_cache(maxsize=128)
def _memory_vector_store(tenant_id: str | None = None) -> MemoryVectorStore:
    return MemoryVectorStore(tenant_id=normalize_tenant_id(tenant_id))


@lru_cache(maxsize=128)
def _memory_service(tenant_id: str | None = None) -> MemoryService:
    return MemoryService(store=_memory_store, vector_store=_memory_vector_store(tenant_id))


@lru_cache(maxsize=128)
def _qa_service(tenant_id: str | None = None) -> DocumentQAService:
    normalized_tenant_id = normalize_tenant_id(tenant_id)
    return DocumentQAService(
        _vector_store(normalized_tenant_id),
        memory_service=_memory_service(normalized_tenant_id),
        tenant_id=normalized_tenant_id,
    )


@lru_cache(maxsize=128)
def _tool_calling_service(tenant_id: str | None = None) -> ToolCallingChatService:
    return ToolCallingChatService(_qa_service(tenant_id))


TenantIdDep = Annotated[str, Depends(get_tenant_id)]
UserIdDep = Annotated[str, Depends(get_user_id)]


def get_vector_store(tenant_id: TenantIdDep) -> DocumentVectorStore:
    return _vector_store(tenant_id)


def get_qa_service(tenant_id: TenantIdDep) -> DocumentQAService:
    return _qa_service(tenant_id)


def get_tool_calling_service(tenant_id: TenantIdDep) -> ToolCallingChatService:
    return _tool_calling_service(tenant_id)


def get_memory_service(tenant_id: TenantIdDep) -> MemoryService:
    return _memory_service(tenant_id)


def get_ingest_job_store() -> IngestJobStore:
    return _job_store


VectorStoreDep = Annotated[DocumentVectorStore, Depends(get_vector_store)]
QAServiceDep = Annotated[DocumentQAService, Depends(get_qa_service)]
ToolCallingServiceDep = Annotated[ToolCallingChatService, Depends(get_tool_calling_service)]
MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]
JobStoreDep = Annotated[IngestJobStore, Depends(get_ingest_job_store)]

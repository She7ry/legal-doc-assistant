from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from functools import lru_cache, partial
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from ai.agent.react_task import run_react_agent_task
from ai.agent.schemas import AgentTaskPause, AgentTaskResult
from ai.agent.tool_calling import ToolCallingChatService
from ai.config.settings import settings
from ai.memory.service import MemoryService
from ai.memory.store import MemoryStore
from ai.memory.vector_store import MemoryVectorStore
from ai.rag.qa_service import DocumentQAService
from ai.rag.retrieval.vector_store import DocumentVectorStore
from backend.agent_tasks import AgentTaskStore
from backend.auth_store import AuthStore, UserRecord
from backend.jobs import IngestJobStore

SESSION_COOKIE_NAME = "legal_doc_session"
_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_bearer_scheme = HTTPBearer(auto_error=False)
_auth_store = AuthStore(
    settings.auth_db_path,
    session_ttl_hours=settings.auth_session_ttl_hours,
)
_job_store = IngestJobStore(settings.ingest_jobs_db_path)
_agent_task_store = AgentTaskStore(settings.agent_tasks_db_path)
_agent_checkpoint_connection = sqlite3.connect(
    settings.agent_tasks_db_path,
    timeout=30,
    check_same_thread=False,
)
_agent_checkpointer = SqliteSaver(
    _agent_checkpoint_connection,
    serde=JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("ai.agent.schemas", "AgentStepResult"),
            ("ai.agent.schemas", "AgentTaskResult"),
            ("ai.rag.grounding.guard", "AnswerGuardResult"),
            ("ai.rag.schemas", "Citation"),
        ]
    ),
)
_memory_store = MemoryStore()


def normalize_user_id(value: str) -> str:
    user_id = value.strip()
    if not _USER_ID_PATTERN.fullmatch(user_id):
        raise ValueError("Invalid authenticated user ID.")
    return user_id


def get_auth_store() -> AuthStore:
    return _auth_store


AuthStoreDep = Annotated[AuthStore, Depends(get_auth_store)]


def get_session_token(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)] = None,
) -> str:
    token = credentials.credentials if credentials else request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


SessionTokenDep = Annotated[str, Depends(get_session_token)]


def get_current_user(token: SessionTokenDep, auth_store: AuthStoreDep) -> UserRecord:
    user = auth_store.resolve_session(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The session is invalid or expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


CurrentUserDep = Annotated[UserRecord, Depends(get_current_user)]


def get_user_id(current_user: CurrentUserDep) -> str:
    return normalize_user_id(current_user.user_id)


@lru_cache(maxsize=128)
def _vector_store(user_id: str) -> DocumentVectorStore:
    return DocumentVectorStore(user_id=normalize_user_id(user_id))


@lru_cache(maxsize=128)
def _memory_vector_store(user_id: str) -> MemoryVectorStore:
    return MemoryVectorStore(user_id=normalize_user_id(user_id))


@lru_cache(maxsize=128)
def _memory_service(user_id: str) -> MemoryService:
    return MemoryService(
        store=_memory_store,
        vector_store=_memory_vector_store(user_id),
    )


@lru_cache(maxsize=128)
def _qa_service(user_id: str) -> DocumentQAService:
    normalized_user_id = normalize_user_id(user_id)
    return DocumentQAService(
        _vector_store(normalized_user_id),
        memory_service=_memory_service(normalized_user_id),
        user_id=normalized_user_id,
    )


@lru_cache(maxsize=128)
def _tool_calling_service(user_id: str) -> ToolCallingChatService:
    return ToolCallingChatService(_qa_service(user_id))


AgentRunner = Callable[..., AgentTaskResult | AgentTaskPause]


@lru_cache(maxsize=128)
def _agent_service(user_id: str) -> AgentRunner:
    return partial(
        run_react_agent_task,
        _qa_service(user_id),
        checkpointer=_agent_checkpointer,
    )


UserIdDep = Annotated[str, Depends(get_user_id)]


def get_vector_store(user_id: UserIdDep) -> DocumentVectorStore:
    return _vector_store(user_id)


def get_qa_service(user_id: UserIdDep) -> DocumentQAService:
    return _qa_service(user_id)


def get_tool_calling_service(user_id: UserIdDep) -> ToolCallingChatService:
    return _tool_calling_service(user_id)


def get_agent_service(user_id: UserIdDep) -> AgentRunner:
    return _agent_service(user_id)


def get_memory_service(user_id: UserIdDep) -> MemoryService:
    return _memory_service(user_id)


def get_ingest_job_store() -> IngestJobStore:
    return _job_store


def get_agent_task_store() -> AgentTaskStore:
    return _agent_task_store


VectorStoreDep = Annotated[DocumentVectorStore, Depends(get_vector_store)]
QAServiceDep = Annotated[DocumentQAService, Depends(get_qa_service)]
ToolCallingServiceDep = Annotated[ToolCallingChatService, Depends(get_tool_calling_service)]
AgentRunnerDep = Annotated[AgentRunner, Depends(get_agent_service)]
MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]
JobStoreDep = Annotated[IngestJobStore, Depends(get_ingest_job_store)]
AgentTaskStoreDep = Annotated[AgentTaskStore, Depends(get_agent_task_store)]

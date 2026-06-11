from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import (
    _agent_service,
    _agent_task_store,
    _job_store,
    _matter_store,
    _memory_service,
    _qa_service,
    _vector_store,
)
from api.middleware.rate_limit import SlidingWindowRateLimiter
from api.routers import agent, chat, documents, matters, memories, review
from api.schemas.responses import HealthCheckOut, HealthResponse
from doc_assistant.config.settings import settings
from doc_assistant.ingestion.document_loader import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)
_rate_limiter = SlidingWindowRateLimiter(
    max_requests=settings.rate_limit_max_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm singletons on startup so the first request does not bear the
    # cost of initialising storage, Chroma connections, and embedding models.
    settings.ensure_directories()
    _vector_store(settings.default_tenant_id)
    _memory_service(settings.default_tenant_id)
    _qa_service(settings.default_tenant_id)
    _agent_service(settings.default_tenant_id)
    _recover_background_work()
    if not settings.api_keys:
        logger.warning("DOC_ASSISTANT_API_KEYS is not configured; API authentication is disabled.")
    yield


app = FastAPI(
    title="Legal Document Assistant API",
    description=(
        "Citation-first RAG API for contracts, policies, and compliance documents. "
        "Not a substitute for legal advice."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=settings.cors_allow_credentials and "*" not in settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(agent.router, prefix="/api/v1")
app.include_router(matters.router, prefix="/api/v1")
app.include_router(review.router, prefix="/api/v1")
app.include_router(memories.router, prefix="/api/v1")


def _recover_background_work() -> None:
    for record in _job_store.list_restartable():
        documents.enqueue_ingest_job(record, _vector_store(record.tenant_id), _job_store)
    for record in _agent_task_store.list_restartable():
        agent.enqueue_agent_task(
            record,
            _agent_service(record.tenant_id),
            _agent_task_store,
            _matter_store,
        )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-Id") or uuid4().hex
    start_time = perf_counter()
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Request failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round((perf_counter() - start_time) * 1000, 2),
            },
        )
        raise

    response.headers["X-Request-Id"] = request_id
    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round((perf_counter() - start_time) * 1000, 2),
        },
    )
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not settings.rate_limit_enabled or request.url.path == "/health":
        return await call_next(request)

    request.state.request_id = getattr(
        request.state,
        "request_id",
        request.headers.get("X-Request-Id") or uuid4().hex,
    )
    key = _rate_limit_key(request)
    if not _rate_limiter.is_allowed(key):
        return _error_response(
            request=request,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            code="rate_limited",
            detail="Too many requests. Please retry later.",
            headers={"Retry-After": str(settings.rate_limit_window_seconds)},
        )
    return await call_next(request)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return _error_response(
        request=request,
        status_code=exc.status_code,
        code=f"http_{exc.status_code}",
        detail=exc.detail,
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _error_response(
        request=request,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="validation_error",
        detail=exc.errors(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return _error_response(
        request=request,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="value_error",
        detail=str(exc),
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    return _error_response(
        request=request,
        status_code=status.HTTP_502_BAD_GATEWAY,
        code="upstream_error",
        detail=str(exc),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled request error", extra={"request_id": _request_id(request)})
    return _error_response(
        request=request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="internal_error",
        detail="Internal server error.",
    )


@app.get("/health", response_model=HealthResponse, tags=["health"], summary="Health check")
def health() -> HealthResponse:
    checks = [
        _path_check("uploads", settings.upload_dir),
        _path_check("vector_store", settings.vector_store_dir),
        _path_check("ingest_jobs", settings.ingest_jobs_db_path.parent),
        _path_check("agent_tasks", settings.agent_tasks_db_path.parent),
        _path_check("matters", settings.matter_db_path.parent),
        _path_check("memory_store", settings.memory_db_path.parent),
        _configuration_check(
            "chat_api_key",
            _chat_api_key_configured(),
            "Chat provider API key is configured.",
            "Chat provider API key is missing; question answering will fail until configured.",
        ),
        _configuration_check(
            "embedding_api_key",
            _embedding_api_key_configured(),
            "Embedding provider API key is configured.",
            "Embedding provider API key is missing; document ingestion and retrieval will fail until configured.",
        ),
    ]
    overall_status = "ok" if all(check.status == "ok" for check in checks) else "degraded"

    return HealthResponse(
        status=overall_status,
        version=app.version,
        auth_required=bool(settings.api_keys),
        default_tenant_id=settings.default_tenant_id,
        providers={
            "chat": {
                "provider": settings.chat_provider,
                "api": settings.chat_api,
                "model": settings.chat_model_name,
                "api_key_configured": _chat_api_key_configured(),
            },
            "embedding": {
                "provider": settings.embedding_provider,
                "model": settings.embedding_model_name,
                "api_key_configured": _embedding_api_key_configured(),
            },
        },
        features={
            "authentication": bool(settings.api_keys),
            "web_search": settings.web_search_enabled,
            "pdf_ocr": settings.pdf_ocr_enabled,
            "memory": True,
            "agent_tasks": True,
            "matters": True,
            "tenant_isolation": True,
        },
        limits={
            "max_upload_bytes": settings.max_upload_bytes,
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        },
        checks=checks,
    )


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    detail,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    response = JSONResponse(
        status_code=status_code,
        content={"code": code, "detail": jsonable_encoder(detail), "request_id": request_id},
        headers=headers,
    )
    response.headers["X-Request-Id"] = request_id
    return response


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or "unknown"


def _rate_limit_key(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    if authorization.casefold().startswith("bearer "):
        return f"bearer:{authorization[7:].strip()}"
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"api-key:{api_key}"
    tenant_id = request.headers.get("x-tenant-id", settings.default_tenant_id)
    user_id = request.headers.get("x-user-id", "anonymous")
    client_host = request.client.host if request.client else "unknown"
    return f"{tenant_id}:{user_id}:{client_host}"


def _path_check(name: str, path: Path) -> HealthCheckOut:
    if not path.exists():
        return HealthCheckOut(name=name, status="degraded", detail="Configured directory does not exist.")
    if not path.is_dir():
        return HealthCheckOut(name=name, status="degraded", detail="Configured path is not a directory.")
    if not os.access(path, os.W_OK):
        return HealthCheckOut(name=name, status="degraded", detail="Configured directory is not writable.")
    return HealthCheckOut(name=name, status="ok", detail="Configured directory is writable.")


def _configuration_check(
    name: str,
    configured: bool,
    ok_detail: str,
    degraded_detail: str,
) -> HealthCheckOut:
    return HealthCheckOut(
        name=name,
        status="ok" if configured else "degraded",
        detail=ok_detail if configured else degraded_detail,
    )


def _chat_api_key_configured() -> bool:
    provider = settings.chat_provider.strip().lower().replace("_", "-")
    if settings.chat_api_key:
        return True
    if provider in {"dashscope", "qwen", "tongyi"}:
        return bool(settings.dashscope_api_key)
    if provider == "deepseek":
        return bool(settings.deepseek_api_key)
    return False


def _embedding_api_key_configured() -> bool:
    provider = settings.embedding_provider.strip().lower().replace("_", "-")
    if provider == "dashscope":
        return bool(settings.embedding_api_key or settings.dashscope_api_key)
    return bool(settings.embedding_api_key)

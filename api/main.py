from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.dependencies import _memory_service, _qa_service, _vector_store
from api.routers import chat, documents, memories, review
from doc_assistant.config.settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm singletons on startup so the first request does not bear the
    # cost of initialising storage, Chroma connections, and embedding models.
    _vector_store(settings.default_tenant_id)
    _memory_service(settings.default_tenant_id)
    _qa_service(settings.default_tenant_id)
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
app.include_router(review.router, prefix="/api/v1")
app.include_router(memories.router, prefix="/api/v1")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


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


@app.get("/health", tags=["health"], summary="Health check")
def health() -> dict[str, str]:
    return {"status": "ok"}


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

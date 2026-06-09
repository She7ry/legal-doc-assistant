from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import QAServiceDep, require_api_key
from api.schemas.requests import ClauseReviewRequest, ConflictCheckRequest
from api.schemas.responses import CitationOut, ClauseReviewResponse, ConflictCheckResponse

router = APIRouter(prefix="/review", tags=["review"], dependencies=[Depends(require_api_key)])


@router.post(
    "/clause",
    response_model=ClauseReviewResponse,
    summary="Review a specific clause type across indexed documents",
)
def review_clause(body: ClauseReviewRequest, qa_service: QAServiceDep) -> ClauseReviewResponse:
    """
    Search indexed documents for a clause type (e.g. "termination clause",
    "non-compete") and return a risk assessment with source citations.
    """
    try:
        answer = qa_service.review_clause(clause_type=body.clause_type, top_k=body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return ClauseReviewResponse(
        content=answer.content,
        citations=[CitationOut.from_citation(c) for c in answer.citations],
        guard_warnings=answer.guard_warnings,
        **answer.metadata,
    )


@router.post(
    "/conflict",
    response_model=ConflictCheckResponse,
    summary="Check for conflicts between contract and policy excerpts",
)
def check_conflict(body: ConflictCheckRequest, qa_service: QAServiceDep) -> ConflictCheckResponse:
    """
    Retrieve contract excerpts (via contract_query) and policy excerpts
    (via policy_query) from the vector store, then identify potential conflicts.
    """
    try:
        answer = qa_service.check_conflict(
            contract_query=body.contract_query,
            policy_query=body.policy_query,
            top_k=body.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return ConflictCheckResponse(
        content=answer.content,
        citations=[CitationOut.from_citation(c) for c in answer.citations],
        guard_warnings=answer.guard_warnings,
        **answer.metadata,
    )

from __future__ import annotations

from fastapi import APIRouter, Depends

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
    answer = qa_service.review_clause(clause_type=body.clause_type, top_k=body.top_k)
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
    answer = qa_service.check_conflict(
        contract_query=body.contract_query,
        policy_query=body.policy_query,
        top_k=body.top_k,
    )
    return ConflictCheckResponse(
        content=answer.content,
        citations=[CitationOut.from_citation(c) for c in answer.citations],
        guard_warnings=answer.guard_warnings,
        **answer.metadata,
    )

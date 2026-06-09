import { apiRequest } from "./http";
import type {
  ClauseReviewRequest,
  ClauseReviewResponse,
  ConflictCheckRequest,
  ConflictCheckResponse,
} from "./types";

export function reviewClause(body: ClauseReviewRequest): Promise<ClauseReviewResponse> {
  return apiRequest<ClauseReviewResponse>({
    method: "POST",
    url: "/api/v1/review/clause",
    data: body,
  });
}

export function checkConflict(body: ConflictCheckRequest): Promise<ConflictCheckResponse> {
  return apiRequest<ConflictCheckResponse>({
    method: "POST",
    url: "/api/v1/review/conflict",
    data: body,
  });
}

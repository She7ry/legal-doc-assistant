import { apiRequest } from "./http";
import type { AnswerResponse, ClauseReviewRequest, ConflictCheckRequest } from "./types";

export function reviewClause(body: ClauseReviewRequest): Promise<AnswerResponse> {
  return apiRequest<AnswerResponse>({
    method: "POST",
    url: "/api/v1/review/clause",
    data: body,
  });
}

export function checkConflict(body: ConflictCheckRequest): Promise<AnswerResponse> {
  return apiRequest<AnswerResponse>({
    method: "POST",
    url: "/api/v1/review/conflict",
    data: body,
  });
}

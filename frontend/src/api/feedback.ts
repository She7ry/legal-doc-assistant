import { apiRequest } from "./http";
import type { FeedbackCreateRequest, FeedbackResponse } from "./types";

export function submitFeedback(body: FeedbackCreateRequest): Promise<FeedbackResponse> {
  return apiRequest<FeedbackResponse>({
    method: "POST",
    url: "/api/v1/feedback",
    data: body,
  });
}

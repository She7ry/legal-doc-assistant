import { apiRequest } from "./http";
import type { AnswerResponse, AskRequest } from "./types";

export function askQuestion(body: AskRequest): Promise<AnswerResponse> {
  return apiRequest<AnswerResponse>({
    method: "POST",
    url: "/api/v1/chat/ask",
    data: body,
  });
}

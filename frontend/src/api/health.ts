import { apiRequest } from "./http";
import type { HealthResponse } from "./types";

export function checkHealth(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>({
    method: "GET",
    url: "/health",
  });
}

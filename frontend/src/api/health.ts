import { apiRequest } from "./http";

export interface HealthResponse {
  status: string;
}

export function checkHealth(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>({
    method: "GET",
    url: "/health",
  });
}

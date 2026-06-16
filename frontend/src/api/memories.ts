import { apiRequest } from "./http";
import type {
  MemoryConversationSummaryRequest,
  MemoryCreateRequest,
  MemoryListResponse,
  MemoryMaintenanceResponse,
  MemoryRecord,
  MemoryStatsResponse,
  MemoryUpdateRequest,
} from "./types";

export function listMemories(): Promise<MemoryListResponse> {
  return apiRequest<MemoryListResponse>({
    method: "GET",
    url: "/api/v1/memories",
  });
}

export function getMemoryStats(): Promise<MemoryStatsResponse> {
  return apiRequest<MemoryStatsResponse>({
    method: "GET",
    url: "/api/v1/memories/stats",
  });
}

export function createMemory(body: MemoryCreateRequest): Promise<MemoryRecord> {
  return apiRequest<MemoryRecord>({
    method: "POST",
    url: "/api/v1/memories",
    data: body,
  });
}

export function updateMemory(memoryId: string, body: MemoryUpdateRequest): Promise<MemoryRecord> {
  return apiRequest<MemoryRecord>({
    method: "PATCH",
    url: `/api/v1/memories/${memoryId}`,
    data: body,
  });
}

export function deleteMemory(memoryId: string): Promise<MemoryRecord> {
  return apiRequest<MemoryRecord>({
    method: "DELETE",
    url: `/api/v1/memories/${memoryId}`,
  });
}

export function runMemoryMaintenance(): Promise<MemoryMaintenanceResponse> {
  return apiRequest<MemoryMaintenanceResponse>({
    method: "POST",
    url: "/api/v1/memories/maintenance",
  });
}

export function summarizeConversation(body: MemoryConversationSummaryRequest): Promise<MemoryRecord> {
  return apiRequest<MemoryRecord>({
    method: "POST",
    url: "/api/v1/memories/summarize-conversation",
    data: body,
  });
}

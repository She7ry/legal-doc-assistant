import { apiRequest } from "./http";
import type {
  MemoryCreateRequest,
  MemoryListResponse,
  MemoryRecord,
  MemoryUpdateRequest,
} from "./types";

export function listMemories(): Promise<MemoryListResponse> {
  return apiRequest<MemoryListResponse>({
    method: "GET",
    url: "/api/v1/memories",
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

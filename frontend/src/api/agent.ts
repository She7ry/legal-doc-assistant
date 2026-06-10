import { readRuntimeSettings } from "../config/runtime";
import { ApiError, apiRequest } from "./http";
import type {
  AgentTaskEvent,
  AgentTaskRecordResponse,
  AgentTaskRequest,
  AgentTaskResumeRequest,
} from "./types";

export function runAgentTask(body: AgentTaskRequest): Promise<AgentTaskRecordResponse> {
  return apiRequest<AgentTaskRecordResponse>({
    method: "POST",
    url: "/api/v1/agent/tasks",
    data: body,
  });
}

export function getAgentTask(taskId: string): Promise<AgentTaskRecordResponse> {
  return apiRequest<AgentTaskRecordResponse>({
    method: "GET",
    url: `/api/v1/agent/tasks/${taskId}`,
  });
}

export function resumeAgentTask(
  taskId: string,
  body: AgentTaskResumeRequest,
): Promise<AgentTaskRecordResponse> {
  return apiRequest<AgentTaskRecordResponse>({
    method: "POST",
    url: `/api/v1/agent/tasks/${taskId}/resume`,
    data: body,
  });
}

export interface AgentTaskEventHandlers {
  onEvent?: (event: AgentTaskEvent) => void;
  onError?: (message: string) => void;
}

export async function streamAgentTaskEvents(
  taskId: string,
  handlers: AgentTaskEventHandlers = {},
  afterEventId = 0,
): Promise<void> {
  const settings = readRuntimeSettings();
  const apiBaseUrl = settings.apiBaseUrl.replace(/\/+$/, "");
  const response = await fetch(
    `${apiBaseUrl}/api/v1/agent/tasks/${taskId}/events?after_event_id=${afterEventId}`,
    {
      method: "GET",
      headers: {
        ...(settings.apiKey ? { "X-API-Key": settings.apiKey } : {}),
        ...(settings.tenantId ? { "X-Tenant-Id": settings.tenantId } : {}),
        ...(settings.userId ? { "X-User-Id": settings.userId } : {}),
      },
    },
  );

  if (!response.ok) {
    throw await toFetchApiError(response);
  }
  if (!response.body) {
    throw new ApiError("当前浏览器不支持任务事件流。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSseBuffer(buffer);
    buffer = parsed.rest;
    parsed.events.forEach((event) => handleEvent(event, handlers));
  }

  buffer += decoder.decode();
  const parsed = parseSseBuffer(`${buffer}\n\n`);
  parsed.events.forEach((event) => handleEvent(event, handlers));
}

interface SseEvent {
  event: string;
  data: unknown;
}

function parseSseBuffer(buffer: string): { events: SseEvent[]; rest: string } {
  const events: SseEvent[] = [];
  let rest = buffer;
  let boundary = findSseBoundary(rest);

  while (boundary) {
    const block = rest.slice(0, boundary.index);
    rest = rest.slice(boundary.index + boundary.length);
    boundary = findSseBoundary(rest);

    const parsed = parseSseBlock(block);
    if (parsed) {
      events.push(parsed);
    }
  }

  return { events, rest };
}

function parseSseBlock(block: string): SseEvent | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  if (!dataLines.length) {
    return null;
  }

  let data: unknown;
  try {
    data = JSON.parse(dataLines.join("\n"));
  } catch {
    throw new ApiError("任务事件流格式错误。", { detail: dataLines.join("\n") });
  }
  return { event, data };
}

function findSseBoundary(value: string): { index: number; length: number } | null {
  const match = /\r?\n\r?\n/.exec(value);
  return match?.index === undefined ? null : { index: match.index, length: match[0].length };
}

function handleEvent(event: SseEvent, handlers: AgentTaskEventHandlers) {
  if (event.event === "heartbeat") {
    return;
  }
  if (event.event === "error") {
    const payload = event.data as { detail?: unknown };
    handlers.onError?.(String(payload.detail || "任务事件流错误。"));
    return;
  }
  handlers.onEvent?.(event.data as AgentTaskEvent);
}

async function toFetchApiError(response: Response): Promise<ApiError> {
  let payload: { code?: string; detail?: unknown; request_id?: string } | undefined;
  try {
    payload = await response.json();
  } catch {
    payload = undefined;
  }

  return new ApiError(formatDetail(payload?.detail) || response.statusText || "请求失败。", {
    status: response.status,
    code: payload?.code,
    requestId: payload?.request_id,
    detail: payload?.detail,
  });
}

function formatDetail(detail: unknown): string {
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }
        if (item && typeof item === "object" && "msg" in item) {
          return String(item.msg);
        }
        return JSON.stringify(item);
      })
      .join("; ");
  }
  if (detail && typeof detail === "object") {
    return JSON.stringify(detail);
  }
  return "";
}

import { apiRequest } from "./http";
import { ApiError } from "./http";
import { readRuntimeSettings } from "../config/runtime";
import type {
  AnswerResponse,
  AskRequest,
  Citation,
  ConversationCreateRequest,
  ConversationListResponse,
  ConversationMessagesResponse,
  ConversationRecord,
  ConversationUpdateRequest,
  MemoryUsage,
} from "./types";

export function askQuestion(body: AskRequest): Promise<AnswerResponse> {
  return apiRequest<AnswerResponse>({
    method: "POST",
    url: "/api/v1/chat/ask",
    data: body,
  });
}

export function fetchConversationMessages(
  conversationId: string,
  limit = 50,
): Promise<ConversationMessagesResponse> {
  return apiRequest<ConversationMessagesResponse>({
    method: "GET",
    url: `/api/v1/chat/conversations/${encodeURIComponent(conversationId)}/messages`,
    params: { limit },
  });
}

export function listChatConversations(params: {
  status?: string | null;
  limit?: number;
  offset?: number;
} = {}): Promise<ConversationListResponse> {
  return apiRequest<ConversationListResponse>({
    method: "GET",
    url: "/api/v1/chat/conversations",
    params: {
      status: params.status ?? "active",
      limit: params.limit ?? 50,
      offset: params.offset ?? 0,
    },
  });
}

export function createChatConversation(
  body: ConversationCreateRequest = {},
): Promise<ConversationRecord> {
  return apiRequest<ConversationRecord>({
    method: "POST",
    url: "/api/v1/chat/conversations",
    data: body,
  });
}

export function updateChatConversation(
  conversationId: string,
  body: ConversationUpdateRequest,
): Promise<ConversationRecord> {
  return apiRequest<ConversationRecord>({
    method: "PATCH",
    url: `/api/v1/chat/conversations/${encodeURIComponent(conversationId)}`,
    data: body,
  });
}

export interface StreamMetadata {
  citations: Citation[];
  memories_used?: MemoryUsage[];
}

export interface StreamDelta {
  content: string;
}

export interface StreamHandlers {
  onMetadata?: (metadata: StreamMetadata) => void;
  onDelta?: (delta: string) => void;
  onDone?: (answer: AnswerResponse) => void;
}

export async function askQuestionStream(
  body: AskRequest,
  handlers: StreamHandlers = {},
): Promise<AnswerResponse> {
  const settings = readRuntimeSettings();
  const apiBaseUrl = settings.apiBaseUrl.replace(/\/+$/, "");
  const response = await fetch(`${apiBaseUrl}/api/v1/chat/ask/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(settings.apiKey ? { "X-API-Key": settings.apiKey } : {}),
      ...(settings.tenantId ? { "X-Tenant-Id": settings.tenantId } : {}),
      ...(settings.userId ? { "X-User-Id": settings.userId } : {}),
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw await toFetchApiError(response);
  }
  if (!response.body) {
    throw new ApiError("当前浏览器不支持流式响应。");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer: AnswerResponse = { content: "", citations: [], memories_used: [] };

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSseBuffer(buffer);
    buffer = parsed.rest;
    for (const event of parsed.events) {
      answer = handleStreamEvent(event, answer, handlers);
    }
  }

  buffer += decoder.decode();
  const parsed = parseSseBuffer(`${buffer}\n\n`);
  for (const event of parsed.events) {
    answer = handleStreamEvent(event, answer, handlers);
  }

  return answer;
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

  const dataText = dataLines.join("\n");
  let data: unknown;
  try {
    data = JSON.parse(dataText);
  } catch (error) {
    throw new ApiError("流式响应格式错误。", { detail: dataText });
  }

  return {
    event,
    data,
  };
}

function findSseBoundary(value: string): { index: number; length: number } | null {
  const match = /\r?\n\r?\n/.exec(value);
  return match?.index === undefined ? null : { index: match.index, length: match[0].length };
}

function handleStreamEvent(
  event: SseEvent,
  answer: AnswerResponse,
  handlers: StreamHandlers,
): AnswerResponse {
  if (event.event === "metadata") {
    const metadata = event.data as StreamMetadata;
    answer = {
      ...answer,
      citations: metadata.citations,
      memories_used: metadata.memories_used ?? [],
    };
    handlers.onMetadata?.(metadata);
    return answer;
  }

  if (event.event === "delta") {
    const delta = event.data as StreamDelta;
    answer = { ...answer, content: answer.content + delta.content };
    handlers.onDelta?.(delta.content);
    return answer;
  }

  if (event.event === "done") {
    answer = event.data as AnswerResponse;
    handlers.onDone?.(answer);
    return answer;
  }

  if (event.event === "error") {
    const payload = event.data as { code?: string; detail?: unknown };
    throw new ApiError(String(payload.detail || "流式请求失败。"), {
      code: payload.code,
      detail: payload.detail,
    });
  }

  return answer;
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

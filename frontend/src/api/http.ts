import axios, { AxiosError, type AxiosRequestConfig } from "axios";

import { readRuntimeSettings } from "../config/runtime";
import type { ApiErrorPayload } from "./types";

export class ApiError extends Error {
  status?: number;
  code?: string;
  requestId?: string;
  detail?: unknown;

  constructor(message: string, options: Partial<ApiError> = {}) {
    super(message);
    this.name = "ApiError";
    this.status = options.status;
    this.code = options.code;
    this.requestId = options.requestId;
    this.detail = options.detail;
  }
}

export async function apiRequest<T>(config: AxiosRequestConfig): Promise<T> {
  const settings = readRuntimeSettings();

  try {
    const response = await axios.request<T>({
      ...config,
      baseURL: settings.apiBaseUrl,
      headers: {
        ...(settings.apiKey ? { "X-API-Key": settings.apiKey } : {}),
        ...(settings.tenantId ? { "X-Tenant-Id": settings.tenantId } : {}),
        ...(settings.userId ? { "X-User-Id": settings.userId } : {}),
        ...config.headers,
      },
    });
    return response.data;
  } catch (error) {
    throw toApiError(error);
  }
}

export function toApiError(error: unknown): ApiError {
  if (!axios.isAxiosError(error)) {
    return new ApiError(error instanceof Error ? error.message : "请求失败");
  }

  const axiosError = error as AxiosError<ApiErrorPayload>;
  const payload = axiosError.response?.data;
  const detail = payload?.detail;
  const message = formatDetail(detail) || axiosError.message || "请求失败";

  return new ApiError(message, {
    status: axiosError.response?.status,
    code: payload?.code,
    requestId: payload?.request_id,
    detail,
  });
}

export function formatApiError(error: unknown): string {
  if (error instanceof ApiError) {
    const requestId = error.requestId ? `（Request ID: ${error.requestId}）` : "";
    return `${error.message}${requestId}`;
  }

  return error instanceof Error ? error.message : "请求失败";
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

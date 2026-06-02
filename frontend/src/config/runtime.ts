export interface RuntimeSettings {
  apiBaseUrl: string;
  apiKey: string;
  tenantId: string;
  userId: string;
}

export const DEFAULT_RUNTIME_SETTINGS: RuntimeSettings = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  apiKey: "",
  tenantId: "default",
  userId: "local-user",
};

const STORAGE_KEYS = {
  apiBaseUrl: "legal-doc-assistant.apiBaseUrl",
  apiKey: "legal-doc-assistant.apiKey",
  tenantId: "legal-doc-assistant.tenantId",
  userId: "legal-doc-assistant.userId",
} as const;

export function readRuntimeSettings(): RuntimeSettings {
  const userId = localStorage.getItem(STORAGE_KEYS.userId) || createLocalUserId();
  return {
    apiBaseUrl: localStorage.getItem(STORAGE_KEYS.apiBaseUrl) || DEFAULT_RUNTIME_SETTINGS.apiBaseUrl,
    apiKey: localStorage.getItem(STORAGE_KEYS.apiKey) || DEFAULT_RUNTIME_SETTINGS.apiKey,
    tenantId: localStorage.getItem(STORAGE_KEYS.tenantId) || DEFAULT_RUNTIME_SETTINGS.tenantId,
    userId,
  };
}

export function writeRuntimeSettings(settings: RuntimeSettings): void {
  localStorage.setItem(STORAGE_KEYS.apiBaseUrl, settings.apiBaseUrl.trim());
  localStorage.setItem(STORAGE_KEYS.apiKey, settings.apiKey.trim());
  localStorage.setItem(STORAGE_KEYS.tenantId, settings.tenantId.trim());
  localStorage.setItem(STORAGE_KEYS.userId, settings.userId.trim() || DEFAULT_RUNTIME_SETTINGS.userId);
}

export function clearRuntimeSettings(): void {
  localStorage.removeItem(STORAGE_KEYS.apiBaseUrl);
  localStorage.removeItem(STORAGE_KEYS.apiKey);
  localStorage.removeItem(STORAGE_KEYS.tenantId);
  localStorage.removeItem(STORAGE_KEYS.userId);
}

function createLocalUserId(): string {
  const randomId =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2);
  const userId = `local-${randomId}`;
  localStorage.setItem(STORAGE_KEYS.userId, userId);
  return userId;
}

export interface RuntimeSettings {
  apiBaseUrl: string;
}

export const DEFAULT_RUNTIME_SETTINGS: RuntimeSettings = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
};

const API_BASE_URL_KEY = "legal-doc-assistant.apiBaseUrl";

export function readRuntimeSettings(): RuntimeSettings {
  return {
    apiBaseUrl: localStorage.getItem(API_BASE_URL_KEY) || DEFAULT_RUNTIME_SETTINGS.apiBaseUrl,
  };
}

export function writeRuntimeSettings(settings: RuntimeSettings): void {
  localStorage.setItem(API_BASE_URL_KEY, settings.apiBaseUrl.trim());
}

export function clearRuntimeSettings(): void {
  localStorage.removeItem(API_BASE_URL_KEY);
}

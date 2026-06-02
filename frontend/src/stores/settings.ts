import { defineStore } from "pinia";

import {
  clearRuntimeSettings,
  DEFAULT_RUNTIME_SETTINGS,
  readRuntimeSettings,
  type RuntimeSettings,
  writeRuntimeSettings,
} from "../config/runtime";

export const useSettingsStore = defineStore("settings", {
  state: (): RuntimeSettings => readRuntimeSettings(),
  getters: {
    hasApiKey: (state) => Boolean(state.apiKey),
    displayBaseUrl: (state) => state.apiBaseUrl.replace(/^https?:\/\//, ""),
  },
  actions: {
    save(settings: RuntimeSettings) {
      this.apiBaseUrl = settings.apiBaseUrl.trim();
      this.apiKey = settings.apiKey.trim();
      this.tenantId = settings.tenantId.trim() || DEFAULT_RUNTIME_SETTINGS.tenantId;
      this.userId = settings.userId.trim() || DEFAULT_RUNTIME_SETTINGS.userId;
      writeRuntimeSettings({
        apiBaseUrl: this.apiBaseUrl,
        apiKey: this.apiKey,
        tenantId: this.tenantId,
        userId: this.userId,
      });
    },
    reset() {
      clearRuntimeSettings();
      const defaults = readRuntimeSettings();
      this.apiBaseUrl = defaults.apiBaseUrl;
      this.apiKey = defaults.apiKey;
      this.tenantId = defaults.tenantId;
      this.userId = defaults.userId;
    },
  },
});
